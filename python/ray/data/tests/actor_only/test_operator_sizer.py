# ABOUTME: Unit tests for OperatorSizer warmup sizing (bootstrap + scale_how_many).
# ABOUTME: Uses fake APMO/pool/OpState doubles; no cluster needed, ray import only.

import logging
import sys
from collections import Counter

import pytest

from ray.data._internal.execution.interfaces.execution_options import (
    ExecutionResources,
)
from ray.data._internal.experimental.execution.actor_only_resource_reporter import (
    ActorOnlyResourceReporter,
)
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (
    ActorStatus,
    ExperimentalAPMO,
)
from ray.data._internal.experimental.execution.operators.placement_constraints import (
    PlacementConstraint,
)
from ray.data._internal.experimental.execution.sizer.operator_sizer import (
    OperatorSizer,
    PerOpSizingRequest,
    PlacedSizingRequest,
)
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.stats import _StatsManager
from ray.data.context import DataContext


class _FakeMetrics:
    def __init__(self):
        self.num_task_outputs_generated = 0
        # Inputs consumed per task; the demand target divides enqueued by this.
        self.average_num_inputs_per_task = 1


class _FakePool:
    def __init__(
        self,
        *,
        min_size,
        max_size,
        cpu=0.0,
        gpu=0.0,
        memory=0.0,
        running=0,
        pending=0,
        in_flight=0,
        concurrency=1,
        idle=None,
    ):
        self._min = min_size
        self._max = max_size
        self._per_actor = ExecutionResources(cpu=cpu, gpu=gpu, memory=memory)
        self.running = running
        self.pending = pending
        self.in_flight = in_flight
        self.concurrency = concurrency
        # Idle actors (no in-flight task). Defaults to one-task-per-actor
        # spreading; pass explicitly to model task concentration.
        self._idle = idle
        self.scaled = []

    def num_idle_actors(self):
        if self._idle is not None:
            return self._idle
        return max(0, self.running - self.in_flight)

    def scale(self, request):
        self.scaled.append(request)

    def current_size(self):
        return self.running + self.pending

    def serving_size(self):
        # The fake has no draining actors, so serving == current.
        return self.running + self.pending

    def min_size(self):
        return self._min

    def initial_size(self):
        return self._min

    def max_size(self):
        return self._max

    def num_pending_actors(self):
        return self.pending

    def num_running_actors(self):
        return self.running

    def num_terminating_actors(self):
        # No real draining in the fake; the global capacity baseline reads this.
        return 0

    def per_actor_resource_usage(self):
        return self._per_actor

    def num_tasks_in_flight(self):
        return self.in_flight

    def max_actor_concurrency(self):
        return self.concurrency


class _FakeAPMO(ExperimentalAPMO):
    """Duck-typed APMO exposing only what OperatorSizer reads.

    Deliberately does NOT call ExperimentalAPMO.__init__ — the sizer only
    touches the accessors overridden below.
    """

    def __init__(
        self,
        name,
        pool,
        *,
        completed=False,
        inputs_complete=False,
        throttling_disabled=False,
        execution_finished=False,
        logical_ops=(),
        inputs=(),
    ):
        self._fake_name = name
        # Back the inherited PhysicalOperator.id property (f"{name}_{_id}") so
        # per-operator telemetry has a stable tag without calling __init__.
        self._id = name
        # Attributes the sizer's support-check (validate_ray_remote_args)
        # reads; empty/falsy so the fake op validates as supported and
        # bootstrap proceeds.
        self._user_ray_remote_args = {}
        # Merged remote args placement_constraint() reads (empty == no
        # label_selector / PG). Distinct from _user_ray_remote_args.
        self._ray_remote_args = {}
        self._user_set_ray_remote_args_fn = None
        # placement_constraint() caches into this member (set by the real
        # __init__, which the fake skips); start uncached so it parses on demand.
        self._placement_constraint = None
        # Node the op pins itself to, read by the sizer for placement (None ==
        # the op is node-targeted by the sizer, the common case).
        self._placement_pinned_node = None
        self._data_context = DataContext.get_current()
        self._pool = pool
        self._completed = completed
        self._inputs_complete = inputs_complete
        self._throttling_disabled = throttling_disabled
        self._execution_finished = execution_finished
        self._fake_metrics = _FakeMetrics()
        self._logical_operators = list(logical_ops)
        self.out_backpressure = (0, 0)  # (blocked, active) for the fraction helper
        self._input_dependencies = list(inputs)
        self._output_dependencies = []
        for dep in inputs:
            dep._output_dependencies.append(self)

    @property
    def name(self):
        return self._fake_name

    @property
    def metrics(self):
        return self._fake_metrics

    @property
    def input_dependencies(self):
        return self._input_dependencies

    @property
    def output_dependencies(self):
        return self._output_dependencies

    def get_autoscaling_actor_pools(self):
        return [self._pool]

    def committed_usage_by_node(self):
        # No real per-node placement in the fake; the sizer's shrink-eviction
        # detection reads this (empty -> never over-allocated).
        return {}

    def num_reclaimable_terminating_actors(self):
        # No draining actors in the fake; the capacity ledger reads this to add
        # back revivable actors (0 -> behaves like the pre-reclaim accounting).
        return 0

    def wait_for_min_actors(self):
        # Real op blocks on pending actor refs when wait_for_min_actors_s is
        # configured; the fake just records that the sizer invoked it.
        self.waited_for_min_actors = True

    def placement_pinned_node(self):
        return self._placement_pinned_node

    def expire_stuck_pending_actors(self):
        # The real implementation walks the pool's pending bookkeeping; the
        # fake's pendings are plain counters, so expiry is a no-op here.
        return 0

    def process_draining_actors(self):
        # scale() calls this on each op to kill fully-drained actors; the fake
        # has no real pool to drain, so it's a no-op.
        return 0

    def apply_scale(self, req):
        # scale() routes downscale/upscale through the operator; the fake just
        # forwards to its pool double (matching the old direct-pool behavior).
        return self._pool.scale(req)

    def has_completed(self):
        return self._completed

    def throttling_disabled(self):
        return self._throttling_disabled

    def has_execution_finished(self):
        return self._execution_finished

    def output_backpressured_fraction(self):
        return self.out_backpressure

    def output_backpressured_actors(self):
        # Mirror the real contract: a pool with no running actors or with
        # pending (mid-resize) actors reports (0, 0). Fakes default to the
        # task tuple unless a test sets the actor-denominated pair.
        if self._pool.num_running_actors() == 0 or self._pool.num_pending_actors() > 0:
            return 0, 0
        return getattr(self, "out_backpressure_actors", self.out_backpressure)

    def min_max_resource_requirements(self):
        per = self._pool.per_actor_resource_usage()
        min_actors = self._pool.min_size()
        max_actors = self._pool.max_size()
        return (
            ExecutionResources(
                cpu=per.cpu * min_actors,
                gpu=per.gpu * min_actors,
                memory=per.memory * min_actors,
            ),
            ExecutionResources(
                cpu=per.cpu * max_actors,
                gpu=per.gpu * max_actors,
                memory=per.memory * max_actors,
                object_store_memory=float("inf"),
            ),
        )


class _FakeOpState:
    def __init__(self, enqueued_blocks=0):
        self._enqueued = enqueued_blocks

    def total_enqueued_input_blocks(self):
        return self._enqueued


class _FakeCoordinator:
    """Fake AutoscalingCoordinator exposing a fixed single-node allocation.

    The sizer reads cluster capacity from the coordinator (Train's HIGH-priority
    reservation already subtracted), so a fixed allocation lets sizing tests run
    with no cluster. Only the three methods the sizer calls are implemented.
    """

    def __init__(self, cpu, gpu):
        self._alloc = {"node0": {"CPU": float(cpu), "GPU": float(gpu)}}

    def request_resources(self, *args, **kwargs):
        pass

    def get_reserved_resources_by_node(self):
        return self._alloc

    def cancel_request(self, *args, **kwargs):
        pass


def _make_sizer(cpu=100.0, gpu=4.0):
    sizer = OperatorSizer(dataset_id="test-ds")
    # The sizer reads allocations through the reporter; wrap the fake
    # coordinator in one. Topology + resource_bank are unused for the per-node
    # allocation read the sizer makes, so they can be empty/None here.
    sizer._resource_reporter = ActorOnlyResourceReporter(
        topology={},
        autoscaling_coordinator=_FakeCoordinator(cpu=cpu, gpu=gpu),
        resource_bank=None,
    )
    # These tests exercise the global (non-constraint-aware) sizing path, which
    # needs no live cluster view. Constraint-aware placement is now the flag
    # default, so pin it off here; the constraint-aware path is covered with Ray
    # mocks in test_sizer_label_pg.
    sizer._constraint_aware = False
    return sizer


def _topology(*ops, enqueued=0):
    # Source-first insertion order, like build_streaming_topology.
    return {op: _FakeOpState(enqueued_blocks=enqueued) for op in ops}


def _linear_pipeline(running=1):
    """A -> B -> infer(GPU, fixed 4) -> write. Returns (ops dict, topology)."""
    a = _FakeAPMO(
        "ReadFiles", _FakePool(min_size=1, max_size=500, cpu=1, running=running)
    )
    b = _FakeAPMO(
        "FlatMap(decode)->Map(preprocess)",
        _FakePool(min_size=1, max_size=500, cpu=1, running=running),
        inputs=(a,),
    )
    infer = _FakeAPMO(
        "MapBatches(Infer)",
        _FakePool(min_size=4, max_size=4, gpu=1, running=4),
        inputs=(b,),
    )
    write = _FakeAPMO(
        "Write",
        _FakePool(min_size=1, max_size=500, cpu=1, running=running),
        inputs=(infer,),
        logical_ops=(Write.__new__(Write),),
    )
    return {"a": a, "b": b, "infer": infer, "write": write}, _topology(
        a, b, infer, write
    )


def test_bootstrap_returns_empty_request_and_resolves_critical_op():
    ops, topo = _linear_pipeline()
    sizer = _make_sizer()
    request = sizer.bootstrap(topology=topo)
    # bootstrap initializes state and returns no request; initial sizing
    # happens in initial_sizing_request, further warmup in scale_how_many.
    assert request is None
    assert sizer._warmup is True
    # Critical op = last op that is not a sink/write -> infer.
    assert sizer._critical_op is ops["infer"]


def test_initial_sizing_request_creates_untargeted_in_scarcity_order():
    # Pools no longer scale themselves to their initial size under the sizer
    # (see ExperimentalAPMO._scale_to_initial_size); the executor asks the
    # sizer right after bootstrap instead. Initial actors are created
    # UNTARGETED (Ray core places them; constrained actors carry their
    # label_selector), submitted in the sizer's scarcity order -- a
    # tiebreaker, not a placement guarantee (Ray core has no actor priority).
    # Zero-actor pools: nothing has been created yet at startup.
    read = _FakeAPMO("read", _FakePool(min_size=1, max_size=10, cpu=1))
    infer = _FakeAPMO("infer", _FakePool(min_size=4, max_size=4, gpu=1), inputs=[read])
    write = _FakeAPMO(
        "write",
        _FakePool(min_size=1, max_size=10, cpu=1),
        inputs=[infer],
        logical_ops=(Write.__new__(Write),),
    )
    topo = _topology(read, infer, write)
    sizer = _make_sizer()
    sizer.bootstrap(topology=topo)
    request = sizer.initial_sizing_request(topology=topo)
    assert request is None
    scaled_ops = [
        op
        for op in sizer._ordered_apmo_ops(topo)
        for pool in op.get_autoscaling_actor_pools()
        if pool.scaled
    ]
    # Every op got its initial actors, untargeted (no target nodes on the
    # request), in the sizer's ordering.
    assert scaled_ops == sizer._ordered_apmo_ops(topo)
    for op in scaled_ops:
        pool = op.get_autoscaling_actor_pools()[0]
        assert len(pool.scaled) == 1
        req = pool.scaled[0]
        assert req.delta == pool.initial_size()
        assert not getattr(req, "target_nodes_to_scale_actors_on", ())
    # The optional wait_for_min_actors_s wait runs here too (after all pools
    # scaled), since op.start() no longer creates the actors it would wait on.
    for op in (read, infer, write):
        assert getattr(op, "waited_for_min_actors", False)


def test_initial_sizing_request_skips_pools_at_size():
    # Pools already at (or above) their initial size get no scale call.
    _, topo = _linear_pipeline()  # fixture pools already run at initial size
    sizer = _make_sizer()
    sizer.bootstrap(topology=topo)
    assert sizer.initial_sizing_request(topology=topo) is None
    for op in topo:
        for pool in op.get_autoscaling_actor_pools():
            assert pool.scaled == []


def test_mixed_cpu_gpu_pool_allowed_and_classified_gpu():
    # A GPU pool that also reserves CPU (num_cpus=1 + num_gpus=1, which exposes
    # the GPU op's CPU footprint to Ray Data) is allowed and classified as
    # GPU-currency -- the resource-shape guard no longer rejects cpu>0 + gpu>0.
    mixed = _FakeAPMO("Infer", _FakePool(min_size=1, max_size=10, cpu=1, gpu=1))
    sizer = _make_sizer()
    sizer._assert_phase1_resource_shape(_topology(mixed))  # does not raise
    assert sizer._op_resource_type(mixed) == "gpu"


def test_phase1_shape_skips_ray_remote_args_fn_op():
    # A ray_remote_args_fn op (the vLLM "ray" backend) declares num_gpus=0 and
    # no num_cpus -- its real GPUs live in the fn's placement group. It's
    # self-placed, so the "requires CPU or GPU" guard must not fire on it.
    fn_op = _FakeAPMO(
        "vLLMEngineStageUDF", _FakePool(min_size=1, max_size=1, cpu=0, gpu=0)
    )
    fn_op._user_set_ray_remote_args_fn = True
    sizer = _make_sizer()
    sizer._assert_phase1_resource_shape(_topology(fn_op))  # does not raise


def test_bootstrap_allows_per_actor_memory():
    # The base actor-only backend enabled per-actor memory limits by default
    # (the sizer now models memory as a currency), so a per-actor memory
    # footprint is accepted at bootstrap rather than rejected.
    op = _FakeAPMO("Mem", _FakePool(min_size=1, max_size=10, cpu=1, memory=128))
    sizer = _make_sizer()
    sizer.bootstrap(topology=_topology(op))  # does not raise
    assert sizer._warmup is True


# NOTE: placement_group / placement_group_bundle_index are now ACCEPTED (the
# constraint-aware flag defaults on) -- their reject/accept behavior is covered
# by test_validate_rejects_pg_without_flag / test_validate_allows_pg_under_flag
# in test_sizer_label_pg. placement_group_capture_child_tasks is always rejected.
@pytest.mark.parametrize(
    "user_remote_args",
    [
        {"scheduling_strategy": "SPREAD"},
        {"scheduling_strategy": "PACK"},
        {"placement_group_capture_child_tasks": True},
        {"resources": {"custom": 1}},
    ],
)
def test_validate_ray_remote_args_rejects_placement_and_custom_resources(
    user_remote_args,
):
    op = _FakeAPMO("Bad", _FakePool(min_size=1, max_size=10, cpu=1))
    op._user_ray_remote_args = dict(user_remote_args)
    with pytest.raises(NotImplementedError):
        op.validate_ray_remote_args()


def test_validate_ray_remote_args_allows_ray_remote_args_fn_under_flag(monkeypatch):
    # On the PG-aware path a ray_remote_args_fn is allowed at validate: it's a
    # self-placed op (the fn assigns a PG per actor). The PG-only contract is
    # enforced later, in _merge_ray_remote_args (see test_experimental_actor_pool).
    import ray.data._internal.experimental.execution.operators.actor_pool_map_operator as apmo  # noqa: E501

    monkeypatch.setattr(apmo, "SIZER_CONSTRAINT_AWARE_PLACEMENT", True)
    op = _FakeAPMO("Ok", _FakePool(min_size=1, max_size=10, cpu=1))
    op._user_set_ray_remote_args_fn = True
    op.validate_ray_remote_args()  # does not raise under the flag


def test_validate_ray_remote_args_rejects_ray_remote_args_fn_without_flag(monkeypatch):
    import ray.data._internal.experimental.execution.operators.actor_pool_map_operator as apmo  # noqa: E501

    monkeypatch.setattr(apmo, "SIZER_CONSTRAINT_AWARE_PLACEMENT", False)
    op = _FakeAPMO("Bad", _FakePool(min_size=1, max_size=10, cpu=1))
    op._user_set_ray_remote_args_fn = True
    with pytest.raises(NotImplementedError, match="ray_remote_args_fn"):
        op.validate_ray_remote_args()


def test_validate_ray_remote_args_allows_resources_and_benign_knobs():
    # cpu/gpu/memory plus non-placement knobs the sizer doesn't care about
    # validate cleanly (memory shape is enforced separately at bootstrap).
    op = _FakeAPMO("Ok", _FakePool(min_size=1, max_size=10, cpu=1, gpu=1))
    op._user_ray_remote_args = {
        "num_cpus": 1,
        "num_gpus": 1,
        "memory": 100,
        "max_restarts": 0,
        "runtime_env": {"env_vars": {"X": "1"}},
    }
    op.validate_ray_remote_args()  # does not raise


def test_validate_ray_remote_args_allows_label_selector():
    # A label_selector is supported: the sizer honors it by placing the op
    # untargeted (Ray core resolves it). Both the DataContext-level selector
    # and a per-op ray_remote_args selector validate cleanly.
    op = _FakeAPMO("Ok", _FakePool(min_size=1, max_size=10, cpu=1))
    op._user_ray_remote_args = {"label_selector": {"ray.io/node-id": "n1"}}
    opts = op.data_context.execution_options
    prev = opts.label_selector
    opts.label_selector = {"region": "us"}
    try:
        op.validate_ray_remote_args()  # does not raise
    finally:
        opts.label_selector = prev


@pytest.mark.parametrize(
    "ray_remote_args,expected",
    [
        # NodeAffinity pin (RayTurbo local-scheme reads) -> its node.
        ("node_affinity", "a" * 56),
        # node-id label_selector (OSS local-scheme reads) -> its node.
        ({"label_selector": {"ray.io/node-id": "node-xyz"}}, "node-xyz"),
        # node-id present alongside other label keys -> still the node.
        (
            {"label_selector": {"ray.io/node-id": "n1", "region": "us"}},
            "n1",
        ),
        # unpinned (the common case) -> None.
        ({"num_cpus": 1}, None),
        # generic label selector with no node-id pin -> None.
        ({"label_selector": {"region": "us"}}, None),
    ],
)
def test_placement_pinned_node(ray_remote_args, expected):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if ray_remote_args == "node_affinity":
        ray_remote_args = {
            "scheduling_strategy": NodeAffinitySchedulingStrategy("a" * 56, soft=False)
        }
    # Build a bare op (skip __init__) exposing only what placement_pinned_node reads.
    op = ExperimentalAPMO.__new__(ExperimentalAPMO)
    op._data_context = DataContext.get_current()
    op._ray_remote_args = ray_remote_args
    assert op.placement_pinned_node() == expected


def test_bootstrap_rejects_unsupported_ray_remote_args():
    bad = _FakeAPMO("Bad", _FakePool(min_size=1, max_size=10, cpu=1))
    bad._user_ray_remote_args = {"scheduling_strategy": "SPREAD"}
    sizer = _make_sizer()
    with pytest.raises(NotImplementedError, match="scheduling_strategy"):
        sizer.bootstrap(topology=_topology(bad))


def test_apply_default_remote_args_drops_default_strategy_with_sizer(monkeypatch):
    import ray.data._internal.experimental.execution.operators.actor_pool_map_operator as m

    ctx = DataContext.get_current()

    # Sizer ON: the library-default SPREAD strategy is dropped (the sizer owns
    # placement), so a sizer-managed op carries no scheduling_strategy default.
    monkeypatch.setattr(m, "ENABLE_OPERATOR_SIZER", True)
    args = m.ExperimentalAPMO._apply_default_remote_args({}, ctx)
    assert "scheduling_strategy" not in args

    # A user-set strategy is preserved here (and rejected later at bootstrap).
    args = m.ExperimentalAPMO._apply_default_remote_args(
        {"scheduling_strategy": "PACK"}, ctx
    )
    assert args["scheduling_strategy"] == "PACK"

    # Sizer OFF: the default is applied as before.
    monkeypatch.setattr(m, "ENABLE_OPERATOR_SIZER", False)
    args = m.ExperimentalAPMO._apply_default_remote_args({}, ctx)
    assert args["scheduling_strategy"] == ctx.scheduling_strategy


def test_warmup_equal_share_cpu_targets():
    ops, topo = _linear_pipeline(running=1)
    sizer = _make_sizer(cpu=100.0, gpu=4.0)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    assert isinstance(result, PerOpSizingRequest)
    # Min_reserved_cpu = 3 (a, b, write at min 1 x 1 cpu); 3 autoscaled cpu ops.
    # ceiling = (100 - 3) * 0.5 / 3 = 16.16 -> target 16 -> delta 15 each.
    for key in ("a", "b", "write"):
        req = result.request[ops[key]]
        assert req.delta == 15, key
    # Fixed GPU pool at size: no resize request.
    assert ops["infer"] not in result.request
    # Sigma(created x per_actor_cpu) stays under the cluster CPU count.
    total_cpu = sum(
        (op._pool.current_size() + result.request[op].delta) * 1
        for op in (ops["a"], ops["b"], ops["write"])
    )
    assert total_cpu <= 100


def test_warmup_gpu_ceiling_and_floor():
    gpu_op = _FakeAPMO("GpuMap", _FakePool(min_size=1, max_size=40, gpu=1, running=1))
    sizer = _make_sizer(cpu=100.0, gpu=8.0)
    sizer.bootstrap(topology=_topology(gpu_op))
    result = sizer.scale_how_many(topology=_topology(gpu_op))
    # Min_reserved_gpu = 1; ceiling = (8 - 1) * 0.5 / 1 = 3.5 -> target 3 -> delta 2.
    assert result.request[gpu_op].delta == 2


def test_below_min_replenish_applies_to_fixed_pools():
    fixed = _FakeAPMO("Infer", _FakePool(min_size=4, max_size=4, gpu=1, running=2))
    sizer = _make_sizer()
    sizer.bootstrap(topology=_topology(fixed))
    result = sizer.scale_how_many(topology=_topology(fixed))
    req = result.request[fixed]
    assert req.delta == 2
    assert "min" in req.reason


def test_fixed_pool_at_size_never_resized():
    fixed = _FakeAPMO("Infer", _FakePool(min_size=4, max_size=4, gpu=1, running=4))
    sizer = _make_sizer()
    sizer.bootstrap(topology=_topology(fixed))
    result = sizer.scale_how_many(topology=_topology(fixed))
    assert fixed not in result.request


def test_below_min_grants_full_min_in_order_source_shortfall_is_unmet_demand():
    # The one-slot-per-unstarted-peer starter reservation was removed: each
    # below-min op is granted its FULL min (clamped to its placeable
    # capacity), processed in the sizer's ordering. On a full cluster a large
    # downstream pool processed first CAN consume all capacity and leave the
    # source at 0 granted actors -- that shortfall must then surface as unmet
    # demand so the cluster autoscaler (not a sizer-side reservation) resolves
    # it. NOTE(team discussion): on a FIXED-size cluster this scenario has no
    # autoscaler to grow it; whether the sizer should rebalance/reserve for
    # stranded mins is an open design question.
    # Repro: source (fixed 1) -> producer (min 30) -> consumer (fixed 1) on a
    # 16-CPU cluster; ordering is sink-most first (consumer, producer, src).
    src = _FakeAPMO("ReadRange", _FakePool(min_size=1, max_size=1, cpu=1, running=0))
    producer = _FakeAPMO(
        "MapBatches(producer)",
        _FakePool(min_size=30, max_size=30, cpu=1, running=0),
        inputs=(src,),
    )
    consumer = _FakeAPMO(
        "MapBatches(consumer)",
        _FakePool(min_size=1, max_size=1, cpu=1, running=0),
        inputs=(producer,),
    )
    sizer = _make_sizer(cpu=16)
    topo = _topology(src, producer, consumer)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    # Consumer (sink-most) then producer are granted; producer's min 30 is
    # clamped to the remaining capacity, exhausting the cluster.
    assert result.request[consumer].delta == 1
    assert result.request[producer].delta == 15
    # The source got nothing this tick...
    assert src not in result.request or result.request[src].delta == 0
    # ...but its min shortfall (plus the producer's remaining 15) is surfaced
    # as unmet demand for the cluster autoscaler.
    demand = sizer.get_unmet_demand()
    assert demand.cpu == 16, f"expected 16 CPUs of unmet demand, got {demand}"


def test_below_min_shortfall_reported_as_unmet_demand_at_zero_capacity():
    """Autoscale-from-zero: with no placeable capacity, the below-min starters
    can't be placed. That shortfall must surface via ``get_unmet_demand()`` so
    the cluster autoscaler has a scale-up signal -- otherwise a min_workers=0
    cluster deadlocks at 0 nodes (the sizer wants nothing it can't place, the
    autoscaler sees no demand, the cluster never grows)."""
    src = _FakeAPMO("ReadRange", _FakePool(min_size=1, max_size=1, cpu=1, running=0))
    mapper = _FakeAPMO(
        "Map(sleep_task)",
        _FakePool(min_size=4, max_size=8, cpu=1, running=0),
        inputs=(src,),
    )
    sizer = _make_sizer(cpu=0, gpu=0)
    topo = _topology(src, mapper)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    # Nothing can be placed...
    assert all(req.delta <= 0 for req in result.request.values())
    # ...so the min-size shortfall (src 1 + mapper 4 = 5 CPUs) is unmet demand.
    demand = sizer.get_unmet_demand()
    assert demand.cpu == 5, f"expected 5 CPUs of unmet demand, got {demand}"


def test_below_min_partial_placement_reports_remaining_shortfall():
    """When only part of the min-size shortfall fits the cluster, the placed
    part is requested and the rest is reported as unmet demand."""
    src = _FakeAPMO("ReadRange", _FakePool(min_size=1, max_size=1, cpu=1, running=0))
    mapper = _FakeAPMO(
        "Map(sleep_task)",
        _FakePool(min_size=4, max_size=8, cpu=1, running=0),
        inputs=(src,),
    )
    sizer = _make_sizer(cpu=2, gpu=0)
    topo = _topology(src, mapper)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    placed = sum(max(0, req.delta) for req in result.request.values())
    demand = sizer.get_unmet_demand()
    # Placed + unmet covers the full min shortfall (5), with 2 placeable.
    assert placed == 2
    assert demand.cpu == 3, f"expected 3 CPUs of unmet demand, got {demand}"


def test_terminal_release_fires_even_for_finished_op():
    done = _FakeAPMO(
        "ListFiles",
        _FakePool(min_size=1, max_size=500, cpu=1, running=10),
        completed=True,
        execution_finished=True,  # branch (1) terminal release must still fire
    )
    sizer = _make_sizer()
    sizer.bootstrap(topology=_topology(done))
    result = sizer.scale_how_many(topology=_topology(done))
    req = result.request[done]
    assert req.delta == -10
    assert req.force is True


def test_terminal_release_logs_only_when_attrition_progresses(caplog, propagate_logs):
    pool = _FakePool(min_size=1, max_size=500, cpu=1, running=10)
    done = _FakeAPMO(
        "ReadFiles",
        pool,
        completed=True,
        execution_finished=True,
    )
    topo = _topology(done)
    sizer = _make_sizer()
    sizer.bootstrap(topology=topo)
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            sizer._last_eval_t = None
            result = sizer.scale_how_many(topology=topo)
            # The request must keep firing so attrition continues.
            assert result.request[done].delta == -10
        pool.running = 4  # attrition removed 6 idle actors
        sizer._last_eval_t = None
        result = sizer.scale_how_many(topology=topo)
        assert result.request[done].delta == -4
    releases = [r for r in caplog.records if "releasing" in r.getMessage()]
    # One line at size 10 and one at size 4 -- not one per tick.
    assert len(releases) == 2


def test_warmup_pending_gate_is_idempotent():
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=1))
    topo = _topology(op)
    sizer = _make_sizer(cpu=100.0)
    sizer.bootstrap(topology=topo)
    first = sizer.scale_how_many(topology=topo)
    delta = first.request[op].delta
    assert delta > 0
    # Simulate the request landing as pending actors; advance past the cadence gate.
    op._pool.pending = delta
    sizer._last_eval_t -= sizer._cadence_s + 1
    second = sizer.scale_how_many(topology=topo)
    assert op not in second.request


def test_warmup_is_upscale_only():
    # current_size above the warmup target: warmup must not downscale.
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=90))
    topo = _topology(op)
    sizer = _make_sizer(cpu=100.0)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    assert op not in result.request


def test_latch_flips_on_critical_op_first_output():
    ops, topo = _linear_pipeline(running=1)
    sizer = _make_sizer()
    sizer.bootstrap(topology=topo)
    assert sizer._warmup is True
    ops["infer"]._fake_metrics.num_task_outputs_generated = 1
    result = sizer.scale_how_many(topology=topo)
    assert sizer._warmup is False
    # Step 2: post-latch holds sizes (warm loop arrives in step 3).
    assert result.request == {}


def test_cadence_gate_returns_empty_between_ticks():
    ops, topo = _linear_pipeline(running=1)
    sizer = _make_sizer()
    sizer.bootstrap(topology=topo)
    first = sizer.scale_how_many(topology=topo)
    assert first.request  # evaluated
    second = sizer.scale_how_many(topology=topo)  # immediately again
    assert second.request == {}


def test_scale_with_empty_request_is_noop():
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=1))
    sizer = _make_sizer()
    sizer.bootstrap(topology=_topology(op))
    where = sizer.scale_where(request=PerOpSizingRequest(request={}))
    assert where.request == {}
    sizer.scale(request=where)  # must not raise
    assert op._pool.scaled == []


def _warm_sizer(op_or_ops, cpu=1000.0, gpu=8.0):
    ops = op_or_ops if isinstance(op_or_ops, (list, tuple)) else [op_or_ops]
    topo = _topology(*ops)
    sizer = _make_sizer(cpu=cpu, gpu=gpu)
    sizer.bootstrap(topology=topo)
    sizer._warmup = False  # warm regime
    return sizer, topo


def _tick(sizer, topo):
    sizer._last_eval_t = None  # bypass the cadence gate
    return sizer.scale_how_many(topology=topo)


def test_outqueue_full_guards_mid_resize_op():
    # A pool with pending actors (mid-resize) must not read as output-full:
    # cross-op queries (_output_flow_controlled, _upstream_is_backpressured)
    # have no other guard, so a resizing neighbour would flip them.
    a = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=4, pending=2))
    a.out_backpressure = (8, 8)
    sizer = _make_sizer()
    full, blocked, running = sizer._op_outqueue_full(a)
    assert full is False


def test_outqueue_full_uses_actor_denominator():
    # 1 blocked actor out of 10 running (9 idle) must not read as full: idle
    # actors are unblocked capacity, not absent from the denominator.
    a = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=10))
    a.out_backpressure_actors = (1, 10)
    sizer = _make_sizer()
    full, blocked, running = sizer._op_outqueue_full(a)
    assert (full, blocked, running) == (False, 1, 10)
    # All 10 blocked -> full.
    a.out_backpressure_actors = (10, 10)
    assert sizer._op_outqueue_full(a)[0] is True


def test_empty_allocation_before_first_grant_skips_reclaim():
    # The coordinator client's first poll returns its INITIAL EMPTY cache while
    # the real RPC is in flight (and a bootstrap blocked on wait_for_min_actors
    # can also let the server-side request expire). An empty allocation that
    # has never been non-empty must not be treated as "allocation shrank" --
    # doing so shed a freshly-bootstrapped 40-actor min_size pool in prod.
    sizer = _make_sizer()
    sizer._resource_reporter._autoscaling_coordinator._alloc = {}
    sizer._resource_reporter.committed_by_node = lambda: {
        "node0": ExecutionResources(cpu=44)
    }
    sizer._last_eval_t = None
    assert sizer.prepare_tick(topology={}) is True
    assert sizer._nodes_to_reclaim == {}


def test_empty_allocation_needs_two_consecutive_ticks_to_reclaim():
    # After a real grant has been seen, a single empty read may be a transient
    # coordinator hiccup: evictions fire only on the second consecutive empty
    # tick, and a non-empty read in between resets the counter.
    sizer = _make_sizer(cpu=100.0)
    coord = sizer._resource_reporter._autoscaling_coordinator
    sizer._resource_reporter.committed_by_node = lambda: {
        "node0": ExecutionResources(cpu=44)
    }

    def tick():
        sizer._last_eval_t = None
        assert sizer.prepare_tick(topology={}) is True

    tick()  # non-empty grant seen
    coord._alloc = {}
    tick()  # first empty read: skeptical
    assert sizer._nodes_to_reclaim == {}
    coord._alloc = {"node0": {"CPU": 100.0}}
    tick()  # recovers; counter resets
    assert sizer._nodes_to_reclaim == {}
    coord._alloc = {}
    tick()
    assert sizer._nodes_to_reclaim == {}
    tick()  # second consecutive empty: the shrink is believed
    assert "node0" in sizer._nodes_to_reclaim


def test_warmup_sink_cap_limits_write_share():
    a = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=1))
    write = _FakeAPMO(
        "Write",
        _FakePool(min_size=1, max_size=500, cpu=1, running=1),
        inputs=(a,),
        logical_ops=(Write.__new__(Write),),
    )
    topo = _topology(a, write)
    sizer = _make_sizer(cpu=1000.0)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    # ceiling = (1000 - 2) * 0.5 / 2 = 249.5 -> A targets 249.
    assert result.request[a].delta == 249 - 1
    # Sink share capped (default 64), not the equal share.
    assert result.request[write].delta == 64 - 1


def test_warm_sink_upscale_hard_capped():
    # A sink can never be output-backpressured, so without a hard cap the
    # warm loop grows it until the cluster is full. The sink cap binds in
    # BOTH regimes.
    a = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=1))
    write = _FakeAPMO(
        "Write",
        _FakePool(min_size=1, max_size=500, cpu=1, running=60),
        inputs=(a,),
        logical_ops=(Write.__new__(Write),),
    )
    sizer, topo = _warm_sizer([a, write])
    topo[write]._enqueued = 1000  # trickle-fed forever at full scale
    write.out_backpressure = (0, 60)  # sinks never out_full
    result = _tick(sizer, topo)
    # want = ceil(0.5*60) = 30; clamped to cap(64) - 60 = 4, not max_size.
    assert result.request[write].delta == 4
    write._pool.running = 64  # at the cap
    result = _tick(sizer, topo)
    assert write not in result.request  # never grows past the cap


def test_warm_upscale_grows_multiplicatively():
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=10))
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 40  # in_nonempty; demand = ceil(40/1.75) = 23 (no clamp)
    op.out_backpressure = (0, 5)  # out not full
    result = _tick(sizer, topo)
    # want = ceil(0.5*10) = 5; demand(23), max_size, and cluster all leave it.
    assert result.request[op].delta == 5


def test_warm_demand_target_formula():
    # ceil((in_flight + ceil(enqueued/avg)) / (max_concurrency * 1.75)).
    op = _FakeAPMO(
        "A", _FakePool(min_size=1, max_size=500, cpu=1, running=10, in_flight=20)
    )
    op._fake_metrics.average_num_inputs_per_task = 2
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 40  # expected = 40/2 = 20 tasks; total = 40
    assert sizer._backlog_actor_target(op, op._pool, topo[op]) == 23  # ceil(40/1.75)


def test_warm_upscale_bounded_by_demand_below_want():
    # demand sits between cur and cur+want, so demand (not the +50% rate) caps.
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=10))
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 21  # demand = ceil(21/1.75) = 12; want = ceil(0.5*10) = 5
    op.out_backpressure = (0, 5)
    result = _tick(sizer, topo)
    # desired_step = min(want=5, demand-cur=2, max-cur) = 2.
    assert result.request[op].delta == 2


def test_warm_no_upscale_when_at_or_above_demand():
    # The cold-start failure mode: fed and NOT output-backpressured (blocked=0),
    # but already at/above the work it has -> must not grow.
    op = _FakeAPMO(
        "A", _FakePool(min_size=1, max_size=500, cpu=1, running=100, in_flight=10)
    )
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 5  # demand = ceil((10+5)/1.75) = 9 << cur=100
    op.out_backpressure = (0, 0)  # blocked=0: pre-demand-clamp this exploded
    result = _tick(sizer, topo)
    assert op not in result.request  # held at demand, no unbounded growth


def test_warm_upscale_clamped_by_max_size():
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=12, cpu=1, running=10))
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 40
    op.out_backpressure = (0, 5)
    result = _tick(sizer, topo)
    # want = 5; desired_step = min(5, 12-10) = 2.
    assert result.request[op].delta == 2


def test_warm_upscale_no_request_at_max_size():
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=10, cpu=1, running=10))
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 40
    op.out_backpressure = (0, 5)
    result = _tick(sizer, topo)
    # desired_step = min(want, 0) = 0 -> no upscale, no downscale arm either.
    assert op not in result.request


def test_warm_same_tick_grants_do_not_overcommit_cluster():
    # Two fed pools, 5 free CPU slots: downstream-first grant must debit the
    # tick-local budget so the second op cannot claim the same slots.
    a = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=10))
    b = _FakeAPMO(
        "B", _FakePool(min_size=1, max_size=500, cpu=1, running=10), inputs=(a,)
    )
    sizer, topo = _warm_sizer([a, b], cpu=25.0)  # committed 20, free 5
    for op in (a, b):
        topo[op]._enqueued = 40
        op.out_backpressure = (0, 5)
    result = _tick(sizer, topo)
    granted = sum(req.delta for req in result.request.values() if req.delta > 0)
    assert granted <= 5
    # Downstream-first: B gets its full step (want=5), A gets nothing.
    assert result.request[b].delta == 5
    assert a not in result.request


def test_warm_no_downscale_when_backpressured_but_backlogged():
    # Regression guard (heterogeneous gen_data drain): output fully backpressured
    # but a large input backlog (demand >> current) means the DOWNSTREAM op is the
    # bottleneck, not that this stage is over-provisioned. Draining the feeder would
    # starve the bottleneck, so the sizer must HOLD, never downscale.
    op = _FakeAPMO("Heavy", _FakePool(min_size=1, max_size=500, cpu=1, running=18))
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 200  # demand = ceil(200 / 1.75) = 115 >> current 18
    op.out_backpressure = (38, 38)  # fully output-backpressured
    for _ in range(sizer._downscale_hysteresis + 2):
        result = _tick(sizer, topo)
        assert op not in result.request or result.request[op].delta >= 0


def test_warm_no_shed_when_output_backpressured_but_all_actors_busy():
    # BK 7939 video flap: the GPU op was repeatedly killed as "over-provisioned
    # (output backpressured)" at util=1.5 / idle=0 — every actor mid-task, its
    # outputs merely alive downstream (per-actor output flow control). A pool
    # with zero idle actors has nothing to shed without killing in-flight work:
    # the sizer must HOLD, no matter how long the blocked streak runs.
    op = _FakeAPMO(
        "GPU",
        _FakePool(min_size=1, max_size=8, gpu=1, running=2, in_flight=2),
    )
    sizer, topo = _warm_sizer(op)
    # in_flight=2 -> has_work; demand = ceil(2 / 1.75) = 2 <= current 2 (the
    # over-provisioned arm); idle = 2 - 2 = 0.
    op.out_backpressure = (2, 2)  # all tasks pull-blocked -> out_full
    for _ in range(sizer._downscale_hysteresis + 3):
        result = _tick(sizer, topo)
        req = result.request.get(op)
        assert req is None or req.delta >= 0, f"shed a fully-busy pool: {req}"


def test_warm_rebalance_donor_requires_idle_actors():
    # Same topology as ..._sheds_backpressured_producer_when_demand_unplaced_
    # elsewhere, but the producer's actors are ALL busy (idle=0): donating
    # capacity would kill in-flight work, so the donor arm must hold.
    producer = _FakeAPMO(
        "Producer",
        _FakePool(min_size=1, max_size=500, cpu=1, running=50, in_flight=50),
    )
    consumer = _FakeAPMO(
        "Consumer",
        _FakePool(min_size=1, max_size=500, cpu=1, running=20),
        inputs=(producer,),
    )
    sizer, topo = _warm_sizer([producer, consumer], cpu=70.0)
    topo[consumer]._enqueued = 200
    consumer.out_backpressure = (0, 20)
    topo[producer]._enqueued = 200
    producer.out_backpressure = (50, 50)
    for _ in range(sizer._downscale_hysteresis + 2):
        result = _tick(sizer, topo)
        req = result.request.get(producer)
        assert req is None or req.delta >= 0, f"donated a fully-busy pool: {req}"


def test_warm_holds_backpressured_producer_when_consumer_cannot_grow():
    # The rebalance-donor arm must fire only when the producer's OWN consumer is
    # capacity-starved AND can grow. Here the consumer is a SATURATED bottleneck
    # (all 10 actors busy) pinned at its max_size, so it has no idle capacity and
    # no placeable unmet demand — draining OR growing the producer would just
    # starve/spill. The sizer must HOLD (the het feeder-drain shape: a capped,
    # busy bottleneck).
    producer = _FakeAPMO(
        "Producer", _FakePool(min_size=1, max_size=500, cpu=1, running=40)
    )
    consumer = _FakeAPMO(
        "Consumer",
        # at max AND saturated (in_flight 10 == running 10 * concurrency 1): the
        # busy capped bottleneck, not idle -> no grow, and no unmet demand -> no shed.
        _FakePool(min_size=10, max_size=10, cpu=1, running=10, in_flight=10),
        inputs=(producer,),
    )
    sizer, topo = _warm_sizer([producer, consumer], cpu=200.0)  # ample free CPU
    topo[consumer]._enqueued = 200  # backlogged, but max_size==current -> no unplaced
    consumer.out_backpressure = (0, 10)
    topo[producer]._enqueued = 200
    producer.out_backpressure = (40, 40)  # producer output-backpressured
    for _ in range(sizer._downscale_hysteresis + 1):
        result = _tick(sizer, topo)
        assert producer not in result.request, (
            "producer must hold (neither shed nor grow) when its consumer is a "
            "busy, capped bottleneck"
        )


def test_warm_no_grow_backpressured_producer_when_downstream_saturated():
    # Feeder-drain guard: the consumer is SATURATED (all actors busy) AND has a
    # genuine input backlog (queue deeper than its actor count) -- it truly
    # cannot absorb more, so growing the producer would just pile unconsumed
    # output. The producer must HOLD -- neither grow nor shed (the consumer has
    # no unmet demand, so the rebalance-donor path does not fire either). The
    # backlog is what distinguishes this from per-actor output flow control
    # (see ..._grows_flow_controlled_producer).
    producer = _FakeAPMO(
        "Producer", _FakePool(min_size=1, max_size=500, cpu=1, running=10)
    )
    consumer = _FakeAPMO(
        "Consumer",
        # saturated: 8 tasks in flight on 8 actors (concurrency 1) -> 0 idle.
        _FakePool(min_size=1, max_size=500, cpu=1, running=8, in_flight=8),
        inputs=(producer,),
    )
    sizer, topo = _warm_sizer([producer, consumer], cpu=1000.0)
    topo[producer]._enqueued = 200  # demand >> current, but downstream saturated
    topo[consumer]._enqueued = 50  # genuine backlog: 50 blocks >> 8 actors
    producer.out_backpressure = (10, 10)  # producer output-backpressured
    consumer.out_backpressure = (0, 8)
    result = _tick(sizer, topo)
    assert producer not in result.request, (
        "producer must not grow when its consumer is saturated (no idle capacity)"
    )


def test_warm_no_reclass_when_consumer_output_blocked():
    # If the consumer is itself output-blocked, the chain is genuinely stuck
    # downstream -- the producer's out_full must NOT be reclassified.
    producer = _FakeAPMO(
        "Producer", _FakePool(min_size=1, max_size=500, cpu=1, running=10)
    )
    consumer = _FakeAPMO(
        "Consumer",
        _FakePool(min_size=1, max_size=500, cpu=1, running=8, in_flight=8),
        inputs=(producer,),
    )
    sizer, topo = _warm_sizer([producer, consumer], cpu=1000.0)
    topo[producer]._enqueued = 200
    producer.out_backpressure = (10, 10)
    consumer.out_backpressure = (8, 8)  # consumer blocked too
    result = _tick(sizer, topo)
    req = result.request.get(producer)
    assert req is None or req.delta <= 0, (
        f"producer must not grow when its consumer is output-blocked, got {req}"
    )


def test_warm_no_bottleneck_lift_when_upstream_not_backpressured():
    # The lift must be gated on upstream backpressure. Same saturated, shallow-queue
    # op, but its producer is NOT output-backpressured -> the op is not the pipeline
    # sink, just keeping up -> hold, no phantom growth.
    producer = _FakeAPMO(
        "Producer",
        _FakePool(min_size=1, max_size=500, cpu=1, running=8, in_flight=8),
    )
    bottleneck = _FakeAPMO(
        "Bottleneck",
        _FakePool(
            min_size=1, max_size=500, cpu=1, running=10, in_flight=10, concurrency=1
        ),
        inputs=(producer,),
    )
    sizer, topo = _warm_sizer([producer, bottleneck], cpu=1000.0)
    topo[bottleneck]._enqueued = 2  # demand = 7 <= current 10
    bottleneck.out_backpressure = (0, 10)
    producer.out_backpressure = (0, 8)  # upstream NOT backpressured
    result = _tick(sizer, topo)
    assert bottleneck not in result.request  # held at demand, no lift


def test_warm_no_bottleneck_lift_when_not_saturated():
    # The lift must be gated on saturation (all actors busy). Upstream IS
    # backpressured, but the op has idle actors (in_flight < current * concurrency)
    # -> growing it wouldn't help, so no lift.
    producer = _FakeAPMO(
        "Producer",
        _FakePool(min_size=1, max_size=500, cpu=1, running=8, in_flight=8),
    )
    bottleneck = _FakeAPMO(
        "Bottleneck",
        _FakePool(
            min_size=1, max_size=500, cpu=1, running=10, in_flight=3, concurrency=1
        ),
        inputs=(producer,),
    )
    sizer, topo = _warm_sizer([producer, bottleneck], cpu=1000.0)
    topo[bottleneck]._enqueued = 2  # demand = ceil((3+2)/1.75) = 3 <= current 10
    bottleneck.out_backpressure = (0, 10)
    producer.out_backpressure = (8, 8)  # upstream backpressured
    result = _tick(sizer, topo)
    assert bottleneck not in result.request  # idle capacity -> no lift


def test_warm_bottleneck_lift_capped_at_max_size():
    # A saturated bottleneck already at max_size must stay held: the lift caps
    # demand at the ceiling, so there is no phantom growth and no request.
    producer = _FakeAPMO(
        "Producer",
        _FakePool(min_size=1, max_size=500, cpu=1, running=8, in_flight=8),
    )
    bottleneck = _FakeAPMO(
        "Bottleneck",
        _FakePool(
            min_size=1, max_size=10, cpu=1, running=10, in_flight=10, concurrency=1
        ),
        inputs=(producer,),
    )
    sizer, topo = _warm_sizer([producer, bottleneck], cpu=1000.0)
    topo[bottleneck]._enqueued = 2
    bottleneck.out_backpressure = (0, 10)
    producer.out_backpressure = (8, 8)
    result = _tick(sizer, topo)
    # demand = min(max(7, 15), max_size 10) = 10; desired_step = min(want, 10-10) = 0.
    assert bottleneck not in result.request


def test_warm_drain_on_idle_input_after_hysteresis():
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=20))
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 0  # input idle (but not inputs_complete)
    op.out_backpressure = (0, 0)
    for _ in range(sizer._downscale_hysteresis - 1):
        _tick(sizer, topo)
    result = _tick(sizer, topo)
    assert "drain" in result.request[op].reason


def test_warm_zero_active_tasks_is_not_output_full():
    # Gate 6: idle pool (zero active tasks) must not read as output-full.
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=5))
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 10
    op.out_backpressure = (0, 0)  # zero denominator
    result = _tick(sizer, topo)
    assert result.request[op].delta > 0  # upscales (in!=empty, out NOT full)


def test_warm_pending_gate_blocks_all_arms(caplog, propagate_logs):
    op = _FakeAPMO(
        "A", _FakePool(min_size=1, max_size=500, cpu=1, running=20, pending=3)
    )
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 0  # would otherwise build drain streak
    with caplog.at_level(logging.INFO):
        for _ in range(sizer._downscale_hysteresis + 2):
            result = _tick(sizer, topo)
    assert op not in result.request
    assert sizer._down_streak.get(op, 0) == 0  # frozen, not building
    # The gate is now visible in telemetry (it used to return silently).
    assert any("sizing in flight" in r.getMessage() for r in caplog.records)


def test_warm_upscale_clamped_by_free_cluster():
    # other -> op, so the topology has a single terminal (op). The point is that
    # `other` occupies 85 of the 100 cluster CPUs, leaving op only 5 placeable.
    other = _FakeAPMO("B", _FakePool(min_size=1, max_size=500, cpu=1, running=85))
    op = _FakeAPMO(
        "A", _FakePool(min_size=1, max_size=500, cpu=1, running=10), inputs=(other,)
    )
    sizer, topo = _warm_sizer([other, op], cpu=100.0)
    topo[op]._enqueued = 200
    op.out_backpressure = (0, 5)
    other.out_backpressure = (0, 40)
    topo[other]._enqueued = 0
    result = _tick(sizer, topo)
    # free = 100 - (10 + 85) = 5 -> placeable = min(want=5, free=5) = 5.
    assert result.request[op].delta == 5
    # Demand fit exactly within free capacity -> no unmet demand (no cluster
    # scale-up pressure).
    assert sizer.get_unmet_demand() == ExecutionResources.zero()


def test_unmet_demand_when_upscale_exceeds_free_cluster():
    # `other` occupies all 100 cluster CPUs, leaving op 0 placeable. op is fed
    # (backlog) and downstream absorbing, so the warm loop wants +want actors
    # but can place none -> the whole want is unmet demand.
    other = _FakeAPMO("B", _FakePool(min_size=1, max_size=500, cpu=1, running=90))
    op = _FakeAPMO(
        "A", _FakePool(min_size=1, max_size=500, cpu=1, running=10), inputs=(other,)
    )
    sizer, topo = _warm_sizer([other, op], cpu=100.0)
    topo[op]._enqueued = 200
    op.out_backpressure = (0, 5)
    other.out_backpressure = (0, 40)
    topo[other]._enqueued = 0
    result = _tick(sizer, topo)
    # free = 100 - (90 + 10) = 0 -> nothing placeable for op this tick.
    assert op not in result.request or result.request[op].delta == 0
    # want = ceil(0.5 * 10) = 5 actors, all unplaced -> 5 CPU of unmet demand.
    assert sizer.get_unmet_demand().cpu == 5


def test_warm_sink_over_cap_is_driven_down():
    # A fed sink already above the cap must be driven DOWN, not just held
    # (Cursor #3). Write at 100 with cap 64, fed, not out_full.
    a = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=1))
    write = _FakeAPMO(
        "Write",
        _FakePool(min_size=1, max_size=500, cpu=1, running=100),
        inputs=(a,),
        logical_ops=(Write.__new__(Write),),
    )
    sizer, topo = _warm_sizer([a, write])
    topo[write]._enqueued = 1000  # fed
    write.out_backpressure = (0, 50)  # never out_full
    result = _tick(sizer, topo)
    req = result.request[write]
    assert req.delta == -10  # 10% of 100, toward cap 64
    assert "over cap" in req.reason


def test_warm_in_flight_work_prevents_drain():
    # Empty input queue but busy in-flight tasks must NOT be treated as idle
    # (Cursor #5) -- the pool holds at demand, never enters the drain arm.
    op = _FakeAPMO(
        "A", _FakePool(min_size=1, max_size=500, cpu=1, running=20, in_flight=30)
    )
    sizer, topo = _warm_sizer(op)
    topo[op]._enqueued = 0  # queue drained...
    op.out_backpressure = (0, 30)  # ...but actors busy, not out_full
    # demand = ceil(30 / 1.75) = 18 < cur=20 -> hold at demand.
    for _ in range(sizer._downscale_hysteresis + 1):
        result = _tick(sizer, topo)
    assert op not in result.request  # never drained despite empty queue
    assert sizer._down_streak.get(op, 0) == 0  # drain streak never built


def test_source_op_not_released_at_startup():
    # A source op (_inputs_complete=True from construction) with an empty input
    # queue must NOT be force-released at startup -- it should warm up. Guards
    # against the old (_inputs_complete and queue==0) terminal-release clause
    # (Cursor: "Source ops wrongly mass-released").
    src = _FakeAPMO(
        "ReadFiles",
        _FakePool(min_size=1, max_size=500, cpu=1, running=4),
        inputs_complete=True,  # source: no upstream
    )
    topo = _topology(src)  # enqueued == 0
    sizer = _make_sizer(cpu=100.0)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    # Warms up (positive delta), never a forced terminal release.
    assert result.request[src].delta > 0
    assert not result.request[src].force


def test_replenish_clamped_by_cluster_headroom():
    # Two pools below min in one tick can't collectively over-request the
    # cluster (Cursor: "Replenish ignores cluster headroom"). cluster=10,
    # running 5+4=9, so only 1 CPU free for the two replenishes.
    a = _FakeAPMO("A", _FakePool(min_size=8, max_size=500, cpu=1, running=5))
    b = _FakeAPMO(
        "B", _FakePool(min_size=8, max_size=500, cpu=1, running=4), inputs=(a,)
    )
    topo = _topology(a, b)
    sizer = _make_sizer(cpu=10.0)
    sizer.bootstrap(topology=topo)
    result = sizer.scale_how_many(topology=topo)
    granted = sum(r.delta for r in result.request.values() if r.delta > 0)
    assert 5 + 4 + granted <= 10  # never over-commit the cluster


def test_loop_lag_warning_on_slow_tick(caplog, propagate_logs):
    # scale_how_many evaluates on its own cadence (default 2s). When the gap
    # between evaluations exceeds 1.5x cadence (>3s) -- the control loop running
    # slow -- it should emit a loop-lag warning; an on-cadence tick must not.
    op = _FakeAPMO("A", _FakePool(min_size=1, max_size=500, cpu=1, running=20))
    sizer, topo = _warm_sizer(op)
    clock = [100.0]
    sizer._clock = lambda: clock[0]
    sizer._last_eval_t = None

    with caplog.at_level(logging.WARNING):
        sizer.scale_how_many(topology=topo)  # first eval, no prior gap
        clock[0] = 104.0  # 4.0s gap > 1.5 * 2s cadence -> lag
        sizer.scale_how_many(topology=topo)
    lag = [r for r in caplog.records if "scheduling-loop lag" in r.getMessage()]
    assert len(lag) == 1, [r.getMessage() for r in caplog.records]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        clock[0] = 106.2  # 2.2s gap: > cadence (gate passes) but < 3s threshold
        sizer.scale_how_many(topology=topo)
    assert not [r for r in caplog.records if "scheduling-loop lag" in r.getMessage()]


# NOTE: The sizer's per-operator actor churn moved to OpRuntimeMetrics (re-emitted
# by the standard executor stats path) in the base actor-only backend (#3439),
# replacing the ``_sizer_actor_delta`` gauges these tests used to assert, so that
# half is now covered at the OpRuntimeMetrics layer. Per-phase tick wall time still
# goes to the _StatsActor, and is covered below.


def test_record_tick_emits_every_phase(monkeypatch):
    """An idle tick -- nothing sized, nothing placed -- still reports every phase.

    The optimizer does most of its work on ticks where no operator is resized, so
    a phase set that only appeared on sizing ticks would miss it entirely.
    """
    emitted = []

    def _capture(dataset_tag, tick_durations):
        emitted.append((dataset_tag, tick_durations))

    monkeypatch.setattr(_StatsManager, "update_sizer_metrics", _capture)

    sizer = _make_sizer()
    sizer.record_tick(
        PerOpSizingRequest(request={}),
        PlacedSizingRequest(request={}),
        optimize_s=0.001,
        how_many_s=0.002,
        where_s=0.004,
        apply_s=0.008,
        e2e_s=0.016,
    )

    assert emitted == [
        (
            "test-ds",
            {
                "optimize": 0.001,
                "how_many": 0.002,
                "where": 0.004,
                "apply": 0.008,
                "e2e": 0.016,
            },
        )
    ]


class _FakeActorState:
    def __init__(self, in_flight, ts=0.0, terminating=False, status=None, unconsumed=0):
        self.num_tasks_in_flight = in_flight
        self.num_unconsumed_outputs = unconsumed
        self.latest_task_submission_ts = ts
        self.is_terminating = terminating
        if status is None:
            status = ActorStatus.TERMINATING if terminating else ActorStatus.ACTIVE
        self.status = status


def test_placement_view_counts_only_matches_full_build():
    """need_victim_order=False must produce the same per-node counts as the
    full build — it only omits the sorted victim id lists."""
    from types import SimpleNamespace

    pool = SimpleNamespace(
        _node_to_actor_states={
            # n1: one idle, one busy, one terminating (excluded everywhere).
            "n1": {
                "a": _FakeActorState(0),
                "b": _FakeActorState(3),
                "t": _FakeActorState(1, terminating=True),
            },
            "n2": {"c": _FakeActorState(0), "d": _FakeActorState(0)},
            # n3: only terminating actors -> absent from every map.
            "n3": {"t2": _FakeActorState(2, terminating=True)},
        },
        pending_ids_by_target_node=lambda: {"n2": ["p1"]},
        _config=SimpleNamespace(
            per_actor_resource_usage=ExecutionResources(cpu=1),
            max_input_bytes_per_actor=None,
        ),
    )
    fake_op = SimpleNamespace(
        actor_pool=pool,
        id="op1",
        metrics=SimpleNamespace(
            bytes_inputs_received=0, bytes_task_outputs_generated=0
        ),
        # No APMO neighbors -> source op, no upstream/downstream collocation counts.
        input_dependency=None,
        output_dependencies=[],
        input_queue_bytes_by_node=lambda: {"n1": 42},
    )

    full = ExperimentalAPMO.build_placement_view(fake_op)
    counts = ExperimentalAPMO.build_placement_view(fake_op, need_victim_order=False)

    # Counts identical in both modes (pending "p1" counts toward n2).
    assert counts.actors_by_node == full.actors_by_node == {"n1": 2, "n2": 3}
    assert counts.idle_actors_by_node == full.idle_actors_by_node == {"n1": 1, "n2": 2}
    assert counts.pending_ids_by_node == full.pending_ids_by_node == {"n2": ["p1"]}
    # Only the full build carries the least-busy-first victim lists.
    assert full.actor_ids_by_node == {"n1": ["a", "b"], "n2": ["c", "d"]}
    assert counts.actor_ids_by_node == {}


def test_actor_counts_by_node_breakdown_feeds_collocation_counts():
    """``actor_counts_by_node`` is the per-node status breakdown the placement
    view collapses into alive collocation counts.

    Targeted pendings must appear (during warmup every neighbor is still
    pending, and alive-only counts leave placement signal-blind exactly when
    most placements happen), restarting actors must appear (they recover onto
    the same node, so dropping them breaks collocation), and terminating actors
    are surfaced so the view can exclude only them from the alive count.
    """
    from types import SimpleNamespace

    neighbor = SimpleNamespace(
        actor_pool=SimpleNamespace(
            _node_to_actor_states={
                "n1": {
                    "a": _FakeActorState(1, status=ActorStatus.ACTIVE),
                    "r": _FakeActorState(0, status=ActorStatus.RESTARTING),
                    "t": _FakeActorState(0, status=ActorStatus.TERMINATING),
                },
            },
            # 2 pendings targeted at n2; untargeted pendings carry no node and
            # are excluded by pending_ids_by_target_node upstream of this.
            pending_ids_by_target_node=lambda: {"n2": ["p1", "p2"]},
        )
    )
    counts = ExperimentalAPMO.actor_counts_by_node(neighbor)
    assert counts == {
        "n1": Counter(
            {
                ActorStatus.ACTIVE: 1,
                ActorStatus.RESTARTING: 1,
                ActorStatus.TERMINATING: 1,
            }
        ),
        "n2": Counter({ActorStatus.PENDING: 2}),
    }
    # The alive collocation count excludes only TERMINATING, so the restarting
    # actor on n1 stays counted alongside the active one.
    alive = {
        node: sum(n for status, n in c.items() if status != ActorStatus.TERMINATING)
        for node, c in counts.items()
    }
    assert alive == {"n1": 2, "n2": 2}


def test_reclaimable_node_ids_follow_revival_order():
    """reclaimable_node_ids(k) must return the nodes of the first k draining
    actors in the same (insertion) order reclaim_draining_actors revives them,
    so the sizer's pre-placement view adjustment matches what apply does."""
    from types import SimpleNamespace

    from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
        _ExperimentalActorPool,
        _TerminatingInfo,
    )

    pool = SimpleNamespace(
        _terminating_actors={
            "a1": _TerminatingInfo(since=0.0),
            "a2": _TerminatingInfo(since=0.0),
            "a3": _TerminatingInfo(since=0.0),
        },
        _running_actors={
            "a1": SimpleNamespace(actor_location="n1"),
            "a2": SimpleNamespace(actor_location="n2"),
            "a3": SimpleNamespace(actor_location="n1"),
        },
    )
    assert _ExperimentalActorPool.reclaimable_node_ids(pool, 2) == ["n1", "n2"]
    assert _ExperimentalActorPool.reclaimable_node_ids(pool, 99) == ["n1", "n2", "n1"]

    # Evicted (non-reclaimable) actors are skipped: a2 drops out entirely.
    pool._terminating_actors["a2"] = _TerminatingInfo(since=0.0, reclaimable=False)
    assert _ExperimentalActorPool.reclaimable_node_ids(pool, 99) == ["n1", "n1"]


def _placement_view(*, per_actor, actor_ids_by_node=None, pending_ids_by_node=None):
    """Minimal OpPlacementView carrying only what _resolve_evictions reads."""
    from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
        OpPlacementView,
    )

    return OpPlacementView(
        op_id="op",
        is_source=False,
        per_actor_usage=per_actor,
        input_bytes_per_actor=None,
        actors_by_node={},
        input_queue_bytes_by_node={},
        upstream_actors_by_node={},
        downstream_actors_by_node={},
        actor_ids_by_node=dict(actor_ids_by_node or {}),
        idle_actors_by_node={},
        pending_ids_by_node=dict(pending_ids_by_node or {}),
    )


def test_compute_nodes_to_reclaim_over_allocation():
    """Per node, reclaim = committed − alloc, clamped ≥ 0. A node absent from
    alloc (our slice lost it) is fully over-allocated; per-dimension clamped."""
    sizer = _make_sizer()
    op = _FakeAPMO("Map", _FakePool(min_size=1, max_size=10, cpu=1))
    op._placement_constraint = PlacementConstraint()
    op.committed_usage_by_node = lambda: {
        "n1": ExecutionResources(cpu=4),  # alloc 2 -> over 2
        "n2": ExecutionResources(cpu=1),  # alloc 4 -> under, dropped
        "n3": ExecutionResources(cpu=3),  # missing from alloc -> fully over
    }
    # committed_by_node reads the reporter's topology.
    sizer._resource_reporter._topology = _topology(op)
    alloc = {
        "n1": ExecutionResources(cpu=2),
        "n2": ExecutionResources(cpu=4),
    }
    reclaim = sizer._compute_nodes_to_reclaim(alloc)
    assert set(reclaim) == {"n1", "n3"}
    assert reclaim["n1"].cpu == 2
    assert reclaim["n3"].cpu == 3


def test_resolve_evictions_cheapest_first_cross_op():
    """Victims are chosen pending-then-running, across ops, only until each
    over-allocated node fits again; ops whose shape can't reduce the overage
    (wrong dimension) are skipped."""
    sizer = _make_sizer()
    # Over by 2 CPU on n1 (and only CPU).
    nodes_to_reclaim = {"n1": ExecutionResources(cpu=2)}

    cpu_op = _FakeAPMO("Cpu", _FakePool(min_size=1, max_size=10, cpu=1))
    gpu_op = _FakeAPMO("Gpu", _FakePool(min_size=1, max_size=10, gpu=1))
    # cpu_op has a pending (cheapest) + two running on n1.
    cpu_op.build_placement_view = lambda need_victim_order=False: _placement_view(
        per_actor=ExecutionResources(cpu=1),
        pending_ids_by_node={"n1": ["p1"]},
        actor_ids_by_node={"n1": ["r1", "r2"]},
    )
    # gpu_op is over n1 too but only frees GPU -> must NOT be touched (over is CPU).
    gpu_op.build_placement_view = lambda need_victim_order=False: _placement_view(
        per_actor=ExecutionResources(gpu=1),
        actor_ids_by_node={"n1": ["g1"]},
    )
    sizer._topology = _topology(cpu_op, gpu_op)

    victims = sizer._resolve_evictions(nodes_to_reclaim)
    # 2 CPU over -> pending p1 then running r1; r2 untouched; gpu_op skipped.
    assert victims == {cpu_op: ["p1", "r1"]}


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
