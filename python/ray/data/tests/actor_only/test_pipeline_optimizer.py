from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    OpPlacementView,
)
from ray.data._internal.experimental.execution.sizer.optimizer.capacity_transfer import (  # noqa: E501
    DONOR_HOLD_S,
    CapacityTransfer,
    TransferInFlight,
)
from ray.data._internal.experimental.execution.sizer.optimizer.context import (
    OpShortfall,
    OptimizationContext,
)
from ray.data._internal.experimental.execution.sizer.optimizer.silent_bottleneck import (  # noqa: E501
    SilentBottleneckGrant,
)
from ray.data._internal.experimental.execution.sizer.optimizer.windowed import (
    BoolStreak,
    WindowedSignal,
)


class _Metrics:
    def __init__(self):
        self.actions = []
        self.refusals = []
        self.transfers = []

    def record_action(self, optimization, recipient, result):
        self.actions.append((optimization, recipient, result))

    def record_refusal(self, optimization, reason):
        self.refusals.append((optimization, reason))

    def record_corrective_transfer(self, donor, recipient, num_actors):
        self.transfers.append((donor, recipient, num_actors))


class _Placement:
    def _locality_rank(self, _view, node):
        return (node,)


class _Pool:
    def __init__(
        self,
        *,
        current=1,
        min_size=1,
        pending=0,
        in_flight=0,
        concurrency=1,
        cpu=1,
    ):
        self._current = current
        self._min_size = min_size
        self._pending = pending
        self._in_flight = in_flight
        self._concurrency = concurrency
        self._per_actor = ExecutionResources(cpu=cpu)

    def current_size(self):
        return self._current

    def serving_size(self):
        # The fake has no draining actors, so serving == current.
        return self._current

    def min_size(self):
        return self._min_size

    def num_pending_actors(self):
        return self._pending

    def num_tasks_in_flight(self):
        return self._in_flight

    def max_actor_concurrency(self):
        return self._concurrency

    def per_actor_resource_usage(self):
        return self._per_actor

    def pending_ids_by_target_node(self):
        return {}


class _Op:
    def __init__(self, name, pool=None, view=None, output_dependencies=()):
        self.name = name
        self._pool = pool or _Pool()
        self._view = view or _view()
        self.output_dependencies = list(output_dependencies)

    def get_autoscaling_actor_pools(self):
        return [self._pool]

    def build_placement_view(self, need_victim_order=True):
        return self._view


class _State:
    def __init__(self, enqueued=0):
        self._enqueued = enqueued

    def total_enqueued_input_blocks(self):
        return self._enqueued


def _view(**overrides):
    base = dict(
        op_id="op",
        is_source=False,
        per_actor_usage=ExecutionResources(cpu=1),
        input_bytes_per_actor=None,
        actors_by_node={},
        input_queue_bytes_by_node={},
        upstream_actors_by_node={},
        downstream_actors_by_node={},
        actor_ids_by_node={},
        actor_drain_cost_by_node={},
        idle_actors_by_node={},
        pending_ids_by_node={},
    )
    base.update(overrides)
    return OpPlacementView(**base)


def _bool_streak(value):
    streak = BoolStreak(1)
    streak.record(value)
    return streak


def _signal(value):
    signal = WindowedSignal(1)
    signal.record(value)
    return signal


def _ctx(**overrides):
    base = dict(
        topology={},
        warmup=False,
        alloc={"node": ExecutionResources(cpu=1)},
        free={"node": ExecutionResources(cpu=0)},
        managed_ops=[],
        shortfall_streak={},
        shortfalls={},
        busy_signal={},
        productive_signal={},
        out_full_streak={},
        out_not_full_streak={},
        eligible_nodes=lambda _op: None,
        pool_cap=lambda _op: 100,
        growth_step=lambda _op: 1,
        seconds_since_allocation_change=999.0,
        now=1000.0,
    )
    base.update(overrides)
    return OptimizationContext(**base)


def test_capacity_transfer_does_not_reuse_donor_capacity_across_recipients():
    donor = _Op(
        "donor",
        pool=_Pool(current=2),
        view=_view(
            actors_by_node={"node": 1},
            actor_ids_by_node={"node": ["d1"]},
            actor_drain_cost_by_node={"node": [(0, 0)]},
            idle_actors_by_node={"node": 1},
        ),
    )
    r1 = _Op("r1")
    r2 = _Op("r2")
    metrics = _Metrics()
    rule = CapacityTransfer(metrics, _Placement(), clock=lambda: 1000.0)
    ctx = _ctx(
        managed_ops=[donor, r1, r2],
        shortfall_streak={r1: _bool_streak(True), r2: _bool_streak(True)},
        shortfalls={
            r1: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=1),
            r2: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=1),
        },
        productive_signal={donor: _signal(1)},
    )

    requests = rule.apply(ctx)

    donor_downscales = sum(
        -placed.request[donor].delta
        for placed in requests
        if donor in placed.request and placed.request[donor].delta < 0
    )
    recipient_upscales = sum(
        placed.request[op].delta
        for placed in requests
        for op in (r1, r2)
        if op in placed.request
    )
    assert donor_downscales == 1
    assert recipient_upscales == 1


def test_capacity_transfer_skips_non_managed_recipients():
    donor = _Op("donor", pool=_Pool(current=2))
    recipient = _Op("self_placed")
    metrics = _Metrics()
    rule = CapacityTransfer(metrics, _Placement(), clock=lambda: 1000.0)
    ctx = _ctx(
        managed_ops=[donor],
        shortfall_streak={recipient: _bool_streak(True)},
        shortfalls={
            recipient: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=1)
        },
        productive_signal={donor: _signal(1)},
    )

    assert rule.apply(ctx) == []


def test_silent_bottleneck_requires_downstream_output_unblocked():
    # Non-terminal consumer (it has its own downstream): its unblocked
    # streak is meaningful and required.
    consumer = _Op("consumer", pool=_Pool(current=2), output_dependencies=[_Op("sink")])
    bottleneck = _Op(
        "bottleneck",
        pool=_Pool(current=2, in_flight=2),
        output_dependencies=[consumer],
    )
    metrics = _Metrics()
    rule = SilentBottleneckGrant(metrics)
    base_ctx = dict(
        topology={consumer: _State(enqueued=0)},
        managed_ops=[bottleneck],
        free={"node": ExecutionResources(cpu=10)},
        busy_signal={bottleneck: _signal(2), consumer: _signal(1)},
        out_full_streak={bottleneck: _bool_streak(True)},
        pool_cap=lambda _op: 10,
        growth_step=lambda _op: 2,
    )

    blocked_ctx = _ctx(
        **base_ctx,
        out_not_full_streak={consumer: _bool_streak(False)},
    )
    assert rule.apply(blocked_ctx) == []

    unblocked_ctx = _ctx(
        **base_ctx,
        out_not_full_streak={consumer: _bool_streak(True)},
    )
    requests = rule.apply(unblocked_ctx)
    assert len(requests) == 1
    assert requests[0].request[bottleneck].delta == 2


def test_capacity_transfer_records_settling_skip_before_recipient_selection():
    # The skip must be visible while the cluster is still resizing, even
    # when no recipient has qualified yet.
    metrics = _Metrics()
    rule = CapacityTransfer(metrics, _Placement(), clock=lambda: 1000.0)

    assert rule.apply(_ctx(seconds_since_allocation_change=1.0)) == []
    assert (rule.name, "cluster_settling") in metrics.refusals


def test_donor_is_held_from_growing_after_a_transfer_closes():
    # A donor must not be regrown -- by another rule or normal sizing --
    # until its signals reflect the smaller pool.
    donor = _Op("donor", pool=_Pool(current=4))
    recipient = _Op("recipient")
    metrics = _Metrics()
    now = [1000.0]
    rule = CapacityTransfer(metrics, _Placement(), clock=lambda: now[0])
    transfer = TransferInFlight(
        recipient=recipient,
        donors=frozenset({donor}),
        nodes=("node",),
        started_t=now[0],
        pre_pending=frozenset(),
        claimed_ids=("claimed",),
    )
    rule._transfers[recipient] = transfer

    assert not rule.can_op_scale(donor)  # transfer in flight

    rule.advance(_ctx(now=now[0]))  # recipient has no pendings -> lands
    assert not rule._transfers
    assert not rule.can_op_scale(donor)  # still held after landing

    now[0] += DONOR_HOLD_S + 1
    assert rule.can_op_scale(donor)


def test_silent_bottleneck_skips_held_ops():
    consumer = _Op("consumer", pool=_Pool(current=2))
    bottleneck = _Op(
        "bottleneck",
        pool=_Pool(current=2, in_flight=2),
        output_dependencies=[consumer],
    )
    metrics = _Metrics()
    rule = SilentBottleneckGrant(metrics)
    ctx = _ctx(
        topology={consumer: _State(enqueued=0)},
        managed_ops=[bottleneck],
        free={"node": ExecutionResources(cpu=10)},
        busy_signal={bottleneck: _signal(2)},
        out_full_streak={bottleneck: _bool_streak(True)},
        out_not_full_streak={consumer: _bool_streak(True)},
        pool_cap=lambda _op: 10,
        growth_step=lambda _op: 2,
        can_op_scale=lambda op: op is not bottleneck,
    )

    assert rule.apply(ctx) == []


def test_silent_bottleneck_defers_while_consumer_is_resizing():
    # A consumer with pending actors is mid-resize: its absorption verdict
    # is stale, so no grant until the pendings land.
    consumer = _Op("consumer", pool=_Pool(current=2, pending=3))
    bottleneck = _Op(
        "bottleneck",
        pool=_Pool(current=2, in_flight=2),
        output_dependencies=[consumer],
    )
    rule = SilentBottleneckGrant(_Metrics())
    ctx = _ctx(
        topology={consumer: _State(enqueued=0)},
        managed_ops=[bottleneck],
        free={"node": ExecutionResources(cpu=10)},
        busy_signal={bottleneck: _signal(2)},
        out_full_streak={bottleneck: _bool_streak(True)},
        out_not_full_streak={consumer: _bool_streak(True)},
        pool_cap=lambda _op: 10,
        growth_step=lambda _op: 2,
    )

    assert rule.apply(ctx) == []


def test_capacity_transfer_refuses_when_no_donor_has_surplus():
    # Both ops starved: recipients exist but the donor list is empty, so
    # the rule refuses once with no_donor_surplus instead of planning.
    r1 = _Op("r1")
    r2 = _Op("r2")
    metrics = _Metrics()
    rule = CapacityTransfer(metrics, _Placement(), clock=lambda: 1000.0)
    ctx = _ctx(
        managed_ops=[r1, r2],
        shortfall_streak={r1: _bool_streak(True), r2: _bool_streak(True)},
        shortfalls={
            r1: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=1),
            r2: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=1),
        },
    )

    assert rule.apply(ctx) == []
    assert (rule.name, "no_donor_surplus") in metrics.refusals


def test_silent_bottleneck_busy_gate_needs_one_task_per_actor_not_full_slots():
    # A flow-controlled op's dispatch breathes: it cannot sustain every task
    # SLOT full at every window sample (hetero 8166: totals 20-28 against a
    # required 28 for the whole run). Busy means every actor holds at least
    # one task, not cur x max_actor_concurrency.
    consumer = _Op("consumer", pool=_Pool(current=2))  # terminal
    bottleneck = _Op(
        "bottleneck",
        pool=_Pool(current=2, in_flight=2, concurrency=2),
        output_dependencies=[consumer],
    )
    rule = SilentBottleneckGrant(_Metrics())
    ctx = _ctx(
        topology={consumer: _State(enqueued=0)},
        managed_ops=[bottleneck],
        free={"node": ExecutionResources(cpu=10)},
        busy_signal={bottleneck: _signal(2), consumer: _signal(1)},
        out_full_streak={bottleneck: _bool_streak(True)},
        pool_cap=lambda _op: 10,
        growth_step=lambda _op: 2,
    )
    requests = rule.apply(ctx)
    assert len(requests) == 1
    assert requests[0].request[bottleneck].delta == 2


def test_silent_bottleneck_records_shortfall_when_no_free_capacity():
    # Free-capacity-only granting plus a recipient that never asks leaves the
    # op invisible to CapacityTransfer. When the grant cannot place, the rule
    # must surface the unplaced want as a shortfall so the transfer rule can
    # qualify the op and tap donors.
    consumer = _Op("consumer", pool=_Pool(current=2))
    bottleneck = _Op(
        "bottleneck",
        pool=_Pool(current=2, in_flight=2),
        output_dependencies=[consumer],
    )
    recorded = []
    rule = SilentBottleneckGrant(_Metrics())
    ctx = _ctx(
        topology={consumer: _State(enqueued=0)},
        managed_ops=[bottleneck],
        free={"node": ExecutionResources(cpu=0)},
        busy_signal={bottleneck: _signal(2), consumer: _signal(1)},
        out_full_streak={bottleneck: _bool_streak(True)},
        pool_cap=lambda _op: 10,
        growth_step=lambda _op: 2,
        record_silent_shortfall=lambda op, sf: recorded.append((op, sf)),
    )
    assert rule.apply(ctx) == []
    assert recorded and recorded[0][0] is bottleneck
    assert recorded[0][1].num_actors == 2


def test_optimizer_merges_silent_shortfalls_into_sizing_results():
    # Silent shortfalls must flow through the SIZER's per-pass recording
    # (streak semantics), not be recorded directly -- a direct record would
    # interleave with the sizer's not-starved record and never form a streak.
    from ray.data._internal.experimental.execution.sizer.optimizer.optimizer import (
        PipelineOptimizer,
    )

    op = _Op("silent")
    optimizer = PipelineOptimizer(
        "ds",
        None,
        _Placement(),
        clock=lambda: 0.0,
        is_self_placed=lambda _op: False,
        outqueue_full=lambda _op: False,
    )
    sf = OpShortfall(resources=ExecutionResources(cpu=2), num_actors=2)
    for _ in range(3):
        optimizer._record_silent_shortfall(op, sf)
        optimizer.record_sizing_results([op], set(), {})
    assert optimizer._shortfall_streak[op].is_streak()
    assert optimizer._shortfalls[op].num_actors == 2
    # consumed per pass: no carry-over without a fresh record
    optimizer.record_sizing_results([op], set(), {})
    assert not optimizer._shortfall_streak[op].is_streak()


def test_capacity_transfer_serves_real_asks_before_silent_wants():
    # One donor actor, two starving recipients: the op that actually asked
    # and was clamped gets it; the bridged silent want waits.
    donor = _Op(
        "donor",
        pool=_Pool(current=2),
        view=_view(
            actors_by_node={"node": 1},
            actor_ids_by_node={"node": ["d1"]},
            actor_drain_cost_by_node={"node": [(0, 0)]},
            idle_actors_by_node={"node": 1},
        ),
    )
    silent_op = _Op("silent_op")
    real_op = _Op("real_op")
    metrics = _Metrics()
    rule = CapacityTransfer(metrics, _Placement(), clock=lambda: 1000.0)
    ctx = _ctx(
        managed_ops=[donor, silent_op, real_op],
        # Silent op listed first: without the priority sort it would win.
        shortfall_streak={
            silent_op: _bool_streak(True),
            real_op: _bool_streak(True),
        },
        shortfalls={
            silent_op: OpShortfall(
                resources=ExecutionResources(cpu=1), num_actors=1, silent=True
            ),
            real_op: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=1),
        },
        productive_signal={donor: _signal(1)},
    )

    requests = rule.apply(ctx)

    granted = [
        op for placed in requests for op, req in placed.request.items() if req.delta > 0
    ]
    assert granted == [real_op]


def test_shortfall_streak_freezes_on_backpressure_hold_ticks():
    # An op that oscillates between "asked and refused" and "held for
    # output backpressure with demand" must still complete the streak:
    # hold ticks freeze it instead of resetting it. Satisfied ticks reset.
    from ray.data._internal.experimental.execution.sizer.optimizer.optimizer import (
        PipelineOptimizer,
    )

    op = _Op("gpu_op")
    optimizer = PipelineOptimizer(
        dataset_id="test",
        resource_reporter=object(),
        placement=_Placement(),
        clock=lambda: 0.0,
        is_self_placed=lambda _op: False,
        outqueue_full=lambda _op: (False, 0, 0),
    )

    def tick(*, starved=False, neutral=False):
        optimizer.record_sizing_results(
            evaluated_ops=[op],
            starved_ops={op} if starved else set(),
            shortfalls={},
            neutral_ops={op} if neutral else set(),
        )

    # ask, ask, hold, ask -> streak completes (hold froze it)
    tick(starved=True)
    tick(starved=True)
    tick(neutral=True)
    tick(starved=True)
    assert optimizer._shortfall_streak[op].is_streak()

    # a satisfied tick still resets
    tick()
    assert not optimizer._shortfall_streak[op].is_streak()

    # holds alone can never BUILD a streak
    tick(neutral=True)
    tick(neutral=True)
    tick(neutral=True)
    assert not optimizer._shortfall_streak[op].is_streak()


def test_feeder_with_starved_consumer_donates_instead_of_competing():
    # Both ops read starved, but the feeder's consumer IS the starving op:
    # the feeder's backlog is phantom, so it must be skipped as a recipient
    # and used as the donor for its consumer.
    consumer = _Op("consumer")
    feeder = _Op(
        "feeder",
        pool=_Pool(current=2),
        view=_view(
            actors_by_node={"node": 1},
            actor_ids_by_node={"node": ["f1"]},
            actor_drain_cost_by_node={"node": [(0, 0)]},
            idle_actors_by_node={"node": 1},
        ),
        output_dependencies=[consumer],
    )
    metrics = _Metrics()
    rule = CapacityTransfer(metrics, _Placement(), clock=lambda: 1000.0)
    ctx = _ctx(
        managed_ops=[feeder, consumer],
        shortfall_streak={
            feeder: _bool_streak(True),
            consumer: _bool_streak(True),
        },
        shortfalls={
            feeder: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=31),
            consumer: OpShortfall(resources=ExecutionResources(cpu=1), num_actors=1),
        },
        productive_signal={feeder: _signal(1)},
    )

    requests = rule.apply(ctx)

    granted = [
        op for placed in requests for op, req in placed.request.items() if req.delta > 0
    ]
    assert granted == [consumer]
    donors = {
        op for placed in requests for op, req in placed.request.items() if req.delta < 0
    }
    assert donors == {feeder}


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
