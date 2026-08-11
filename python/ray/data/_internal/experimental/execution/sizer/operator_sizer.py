import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, AbstractSet, Dict, List, Optional, Set, Tuple

from ray.data._internal.actor_autoscaler import ActorPoolScalingRequest
from ray.data._internal.cached_ray_internals import get_draining_nodes
from ray.data._internal.execution.execution_flags import (
    ACTOR_MEMORY_PER_CPU_BYTES,
    SIZER_CONSTRAINT_AWARE_PLACEMENT,
    SIZER_ENABLE_OPTIMIZER,
    SIZER_ORDERING_POLICY,
    SIZER_RAY_CORE_SPREAD,
)
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.interfaces.common import NodeIdStr
from ray.data._internal.execution.operators.output_splitter import OutputSplitter
from ray.data._internal.execution.resource_bank import LogicalActorId
from ray.data._internal.execution.resource_manager import (
    terminal_operator_from_topology,
)
from ray.data._internal.execution.streaming_executor_state import Topology
from ray.data._internal.execution.util import locality_string
from ray.data._internal.experimental.execution.actor_only_resource_reporter import (
    ClusterView,
)
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (
    ExperimentalActorPoolScalingRequest,
    ExperimentalAPMO,
)
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    ActorPlacementStrategy,
    OpPlacementView,
    max_placeable_actors,
)
from ray.data._internal.experimental.execution.sizer.capacity_ledger import (
    CapacityLedger,
    ConstraintAwareCapacityLedger,
    GlobalCapacityLedger,
    TickSnapshot,
)
from ray.data._internal.experimental.execution.sizer.locality_actor_placement import (
    LocalityBasedActorPlacement,
)
from ray.data._internal.experimental.execution.sizer.operator_ordering import (
    OpOrderingSignals,
    make_ordering_policy,
)
from ray.data._internal.experimental.execution.sizer.optimizer import (
    OpShortfall,
    PipelineOptimizer,
)
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.stats import _StatsManager

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.experimental.execution.actor_only_resource_reporter import (
        ActorOnlyResourceReporter,
    )

logger = logging.getLogger(__name__)

# Throttle interval for emitting tick-duration telemetry from record_tick (which
# fires every scheduling-loop step). Matches the executor's op-metrics push
# cadence (StreamingExecutor.UPDATE_METRICS_INTERVAL_S) so the series stays alive
# in steady state without per-step _StatsActor RPCs.
_SIZER_TICK_EMIT_INTERVAL_S = 5.0


@dataclass(frozen=True)
class PerOpSizingRequest:
    # A mapping from actor operator, to how many to scale up/down
    # AMPO = ActorPoolMapOperator
    request: Dict[ExperimentalAPMO, ActorPoolScalingRequest]


@dataclass(frozen=True)
class PlacedSizingRequest:
    # A mapping from actor operator, to the placement-resolved scaling request.
    request: Dict[ExperimentalAPMO, ExperimentalActorPoolScalingRequest]


class OperatorSizer:
    """Responsible for sizing all actors in the topology.

    Three responsibilities, one per tick:

    - ``scale_how_many`` (HOW MANY): two-phase sizing.
      * Warmup (no feedback yet): feedforward equal-share reservations against
        the coordinator-allocated cluster capacity, re-evaluated each cadence
        tick. Ends when the critical op (last non-sink op) produces its first
        output.
      * Warm (backpressure-driven): binary queue signals (InQueue non-empty,
        OutQueue full via the per-actor output-cap fraction) drive rate-limited
        multiplicative upscale clamped to pool max and free cluster capacity,
        plus hysteretic drain / over-provisioned downscale.
    - ``scale_where`` (WHERE): resolves each per-op delta to specific nodes
      (upscale) or exact victim actors (downscale) using the coordinator's
      per-node allocation. See its docstring.
    - ``scale`` (APPLY): applies the placement-resolved requests and drives
      graceful terminations.

    ``bootstrap`` only initializes state and returns an empty request: a
    one-shot allocation cannot place onto an autoscaling cluster that is near
    zero at t=0, so all sizing happens in ``scale_how_many``.
    """

    def __init__(
        self,
        dataset_id: str,
        resource_reporter: Optional["ActorOnlyResourceReporter"] = None,
    ):
        # Set in bootstrap(), which runs once at execution start.
        self._topology: Optional[Topology] = None
        self._dataset_id = dataset_id
        self._placement: ActorPlacementStrategy = LocalityBasedActorPlacement()
        self._resource_reporter: Optional["ActorOnlyResourceReporter"] = (
            resource_reporter
        )

        # Constraint-aware placement: when on, the sizer resolves each op's
        # eligible node set (labels / pin / PG) and sizes/places against it
        # per-node (fragmentation-resilient). Off -> previous global behavior.
        self._constraint_aware: bool = SIZER_CONSTRAINT_AWARE_PLACEMENT
        # Shared operator-ordering policy for scale_how_many (grant order) and
        # scale_where (placement order); default reverse-topological == today.
        self._ordering = make_ordering_policy(SIZER_ORDERING_POLICY)
        # Bundle capacity (actors) of each PG op's placement group, computed once
        # in bootstrap() from the bundle shapes.
        self._pg_cap_by_op: Dict[ExperimentalAPMO, int] = {}
        # Per-tick snapshot built by scale_how_many, reused by scale_where
        # (constraint-aware path only; None on the default path).
        self._tick_snapshot: Optional[TickSnapshot] = None
        # Per-node resources by which our committed usage exceeds the current
        # allocation (the coordinator gave a slice to another dataset/train job,
        # or a node is draining). Computed each real tick by scale_how_many and
        # consumed by scale_where to evict actors off nodes we're losing.
        # Common to both the constraint-aware and default paths.
        self._nodes_to_reclaim: Dict[NodeIdStr, ExecutionResources] = {}

        self._warmup: bool = True
        self._critical_op: Optional["PhysicalOperator"] = None
        self._last_eval_t: Optional[float] = None
        self._clock = time.monotonic
        # Resources the warm loop wanted this eval tick but the cluster couldn't
        # place (per-op ``unplaced`` aggregated).
        self._pending_unmet_demand: ExecutionResources = ExecutionResources.zero()
        # Shaped unmet demand (label-aware path): (per-actor ResourceDict, count,
        # label_selector) per shortfall, so the autoscaler asks for the right node
        # type instead of the default. Empty on the default path.
        self._pending_unmet_bundles: List[
            Tuple[Dict[str, float], int, Optional[Dict[str, str]]]
        ] = []
        # Per-op unplaced ask from the last sizing pass; handed to the
        # optimizer at the end of each pass.
        self._op_shortfalls: Dict[ExperimentalAPMO, OpShortfall] = {}
        # Ops held for output backpressure this tick while demand remained:
        # neutral for the optimizer's shortfall streak (freeze, not reset),
        # so an op oscillating between asking and being braked can still
        # qualify for a transfer.
        self._held_with_demand: Set["PhysicalOperator"] = set()

        self._cadence_s = float(os.environ.get("RAY_DATA_SIZER_CADENCE_S", "2"))
        self._reservation_ratio = float(
            os.environ.get("RAY_DATA_SIZER_RESERVATION_RATIO", "0.8")
        )
        # Hard size cap for sink/write pools, both regimes. Equal-share warmup
        # massively over-provisions sinks (a Write pool needs few actors
        # relative to an equal share of the cluster), and the warm loop cannot
        # converge on them at all: a sink's output queue never backpressures
        # and its input rarely idles at scale, so without the cap a sink grows
        # until the cluster is full, starving every upstream op.
        self._sink_cap = int(os.environ.get("RAY_DATA_SIZER_SINK_CAP", "64"))
        # Warm-loop tunables
        self._upscale_factor = float(
            os.environ.get("RAY_DATA_SIZER_UPSCALE_FACTOR", "0.5")
        )
        self._downscale_factor = float(
            os.environ.get("RAY_DATA_SIZER_DOWNSCALE_FACTOR", "0.1")
        )
        self._downscale_hysteresis = int(
            os.environ.get("RAY_DATA_SIZER_DOWNSCALE_HYSTERESIS", "3")
        )
        self._outqueue_full_fraction = float(
            os.environ.get("RAY_DATA_SIZER_OUTQUEUE_FULL_FRACTION", "0.5")
        )
        # Demand-target threshold (default 1.75, matching the pre-sizer
        # autoscaler's RAY_DATA_DEFAULT_ACTOR_POOL_UTIL_UPSCALING_THRESHOLD):
        # the warm-loop upscale arm grows a fed pool only toward the
        # backlog-based actor count divided by this, so a cold-start empty
        # output queue (blocked=0) cannot drive unbounded +upscale_factor/tick
        # growth that fills the cluster. A sizer-owned knob (not a DataContext
        # read). Ported from prasad/operator-sizer (PR #3396).
        self._util_upscaling_threshold = float(
            os.environ.get("RAY_DATA_SIZER_UTIL_THRESHOLD", "1.75")
        )
        # Per-op consecutive-tick counter for the hysteretic downscale arm.
        self._down_streak: Dict["PhysicalOperator", int] = {}
        # Cross-operator corrections (capacity transfers, silent-bottleneck
        # grants) live in the optimizer; the sizer's own passes stay
        # per-operator.
        self._optimizer: Optional[PipelineOptimizer] = (
            PipelineOptimizer(
                dataset_id,
                resource_reporter,
                self._placement,
                self._clock,
                self._is_self_placed,
                self._op_outqueue_full,
            )
            if resource_reporter is not None
            else None
        )
        # Handshake with the executor's phase ordering: both optimize_pipeline
        # and scale_how_many run each tick; prepare_tick's cadence gate is
        # time-based, so scale_how_many consumes optimize_pipeline's verdict
        # instead of re-gating.
        self._tick_prepared: bool = False
        self._tick_active: bool = False
        # Empty-allocation skepticism state (see _reclaim_with_empty_alloc_skepticism).
        self._alloc_seen_nonempty = False
        self._consecutive_empty_alloc_ticks = 0
        # Last pool size logged by the terminal-release branch, per op.
        self._release_logged_size: Dict["PhysicalOperator", int] = {}
        # Last per-tick request summary logged (suppresses repeats).
        self._last_request_summary: Optional[str] = None
        # Last placement summary logged by scale_where (suppresses repeats).
        self._last_placement_summary: Optional[str] = None
        # Last capacity-map / shortfall summaries logged (suppress repeats).
        self._last_capacity_summary: Optional[str] = None
        self._last_shortfall_summary: Optional[str] = None

        # Per-tick telemetry: per-operator actor-scaling churn is accumulated on
        # op.metrics (OpRuntimeMetrics) and re-emitted by the standard executor
        # stats push; per-phase tick wall time goes to the dataset _StatsActor,
        # throttled (see record_tick). Last wall-clock time tick duration was
        # emitted, for that throttle.
        self._last_tick_emit_s: float = 0.0
        # Cumulative cross-node input bytes per op at the last tick, to log the
        # per-tick delta (see _log_locality).
        self._last_remote_bytes: Dict["PhysicalOperator", int] = {}

    def bootstrap(self, topology: Topology) -> Optional[PlacedSizingRequest]:
        """Bootstrap the actor operators on startup.

        This should only be called once per dataset execution

        Initializes sizer state and validates pool resource shapes. Returns an
        empty request: initial sizing happens in ``initial_sizing_request``
        (called by the executor right after this), and all further warmup
        allocation in ``scale_how_many``.
        """
        self._topology = topology
        # Fail on operators the sizer can't manage (custom placement or
        # custom resources).
        self._assert_all_ops_supported(topology)

        self._warmup = True
        self._critical_op = self._resolve_critical_op(topology)
        self._last_eval_t = None
        self._down_streak.clear()
        self._release_logged_size.clear()
        self._last_request_summary = None
        self._assert_phase1_resource_shape(topology)
        # Constraints are parsed lazily and cached on each op itself (see
        # ExperimentalAPMO.placement_constraint), so there's nothing to build
        # here -- we only touch them on the label/PG-aware path.
        assert self._resource_reporter is not None, (
            "OperatorSizer requires a resource_reporter (injected by the executor)."
        )
        # warm up the non-blocking allocation cache so early ticks read a
        # populated allocation instead of the cold-cache empty value (the
        # client submits an async RPC and serves the result on a later call).
        self._resource_reporter.get_reserved_resources_by_node()

        # Cap each PG op by its bundle capacity (per-bundle; shapes are fixed at
        # PG creation). The reporter fetches + caches the bundle shapes once; the
        # constraint reads them off the ClusterView. Above this cap Ray core
        # would queue pendings forever, so the sizer never asks for more.
        #
        # ray_remote_args_fn ops assign a fresh PG per actor that we can't
        # inspect here (calling the fn mints a real PG). They're self-placed
        # too, so route them through the same capacity ledger path (``grant``
        # decrements this counter, not the free map) but cap them by their user
        # concurrency (``max_size``) instead of a bundle count.
        self._pg_cap_by_op = {}
        if self._constraint_aware:
            pg_ops = [
                op
                for op in topology
                if isinstance(op, ExperimentalAPMO)
                and op.placement_constraint().has_placement_group
            ]
            self._resource_reporter.prefetch_pg_bundles(
                [op.placement_constraint().placement_group for op in pg_ops]
            )
            view = self._resource_reporter.get_cluster_view(include_pg_bundles=True)
            for op in pg_ops:
                pool = op.get_autoscaling_actor_pools()[0]
                constraint = op.placement_constraint()
                cap = constraint.pg_capacity(pool.per_actor_resource_usage(), view)
                self._pg_cap_by_op[op] = cap
            for op in topology:
                if (
                    isinstance(op, ExperimentalAPMO)
                    and op.uses_ray_remote_args_fn
                    and not op.placement_constraint().has_placement_group
                ):
                    self._pg_cap_by_op[op] = op.get_autoscaling_actor_pools()[
                        0
                    ].max_size()

        return None

    def initial_sizing_request(
        self, topology: Topology
    ) -> Optional[PlacedSizingRequest]:
        """Scale up operators to their initial (min) size, untargeted."""
        ordered_ops = self._ordered_apmo_ops(topology)
        for op in ordered_ops:
            for pool in op.get_autoscaling_actor_pools():
                delta = pool.initial_size() - pool.serving_size()
                if delta > 0:
                    pool.scale(
                        ActorPoolScalingRequest(
                            delta=delta,
                            reason="scaling to initial size (sizer-ordered)",
                        )
                    )
            op.wait_for_min_actors()
        return None

    def prepare_tick(self, topology: Topology) -> bool:
        """Per-tick shared setup, run once on the sizing cadence before the
        how-many and where phases."""
        # Cadence gate: sizing decisions are meaningless at the ~100ms
        # scheduling-step rate (queue signals are noise below ~1s and actor
        # startup dominates), so evaluate on our own clock.
        now = self._clock()
        if self._last_eval_t is not None and now - self._last_eval_t < self._cadence_s:
            return False
        # Loop-lag telemetry: prepare_tick runs once per scheduling-loop step, so
        # the gap between evaluations tracks the control-loop iteration time. Flag
        # when it exceeds 1.5x the intended cadence (>3s for a 2s tick): a slow
        # loop means the sizer reacts only this often, delaying every decision
        # (observed ballooning to ~23s under heavy actor counts).
        if self._last_eval_t is not None:
            since = now - self._last_eval_t
            if since > 1.5 * self._cadence_s:
                logger.warning(
                    "OperatorSizer: scheduling-loop lag -- %.1fs since last sizer "
                    "tick (cadence %.1fs); the sizer evaluates only this often, so "
                    "a slow control loop delays sizing decisions.",
                    since,
                    self._cadence_s,
                )
        self._last_eval_t = now

        # Warmup ends (one-way latch) when the critical op produces its first
        # output: the pipeline is flowing end-to-end.
        if (
            self._warmup
            and self._critical_op is not None
            and self._critical_op.metrics.num_task_outputs_generated > 0
        ):
            logger.info(
                "OperatorSizer: warmup complete (%s produced first output)",
                self._critical_op.name,
            )
            self._warmup = False

        # Reset per-eval; scale_how_many's below-min replenishment and warm loop
        # re-accumulate unmet demand.
        self._pending_unmet_demand = ExecutionResources.zero()
        self._pending_unmet_bundles = []

        # Shared per-tick snapshot. The constraint-aware path plans against a
        # per-node snapshot (fragmentation-resilient) that scale_where reuses; the
        # default path keeps the previous global-cluster-totals behavior and
        # builds its free map lazily in scale_where.
        if self._constraint_aware:
            self._tick_snapshot = self._build_tick_snapshot(topology)
        else:
            self._tick_snapshot = None

        # Detect where the coordinator shrank our slice (another dataset / Train
        # job took part of it) -- per node, the amount our committed usage now
        # exceeds our allocation by. This covers BOTH a fully-lost node (absent
        # from alloc -> whole committed amount is over) and a partial shrink (our
        # share on a still-shared node dropped but didn't vanish -> only the
        # excess is over); scale_where sheds just enough actors to erase the
        # per-node overage, not the whole node. Based on the raw coordinator
        # allocation (same source both paths). scale_where consumes and clears
        # this, so it acts only on the real tick that computed it.
        self._nodes_to_reclaim = self._reclaim_with_empty_alloc_skepticism(
            self._get_alloc_by_node()
        )
        return True

    def _reclaim_with_empty_alloc_skepticism(
        self, alloc: Dict[NodeIdStr, ExecutionResources]
    ) -> Dict[NodeIdStr, ExecutionResources]:
        """Gate the allocation-shrank eviction on the allocation being credible.

        An EMPTY allocation is routinely degenerate rather than a real
        revocation: the coordinator client's first poll returns its initial
        empty cache while the RPC is in flight, and a bootstrap blocked in
        wait_for_min_actors starves the request long enough to expire it
        server-side. Acting on one such read has shed a freshly-bootstrapped
        min_size pool. So: before the first non-empty grant, never evict on
        empty; afterwards, believe an empty allocation only when it persists
        for two consecutive ticks. Non-empty allocations pass straight through
        -- partial shrinks stay effective immediately.
        """
        if alloc:
            self._alloc_seen_nonempty = True
            self._consecutive_empty_alloc_ticks = 0
            return self._compute_nodes_to_reclaim(alloc)
        if not self._alloc_seen_nonempty:
            logger.debug(
                "OperatorSizer: allocation empty before first grant -- "
                "skipping reclaim this tick."
            )
            return {}
        self._consecutive_empty_alloc_ticks += 1
        if self._consecutive_empty_alloc_ticks < 2:
            logger.info(
                "OperatorSizer: allocation read empty after a prior grant -- "
                "holding evictions one tick in case it is transient."
            )
            return {}
        return self._compute_nodes_to_reclaim(alloc)

    def scale_how_many(self, topology: Topology) -> PerOpSizingRequest:
        """Recalculates the size for each ActorPoolMapOperator,
        and returns a mapping from operator -> how much it should scale up/down by.
        See PerOpSizingRequest for more information. Must respect user concurrency args
        """
        # prepare_tick owns the cadence gate + shared per-tick state; a False
        # return means we're inside the cadence window -> free no-op. When the
        # executor already ran the optimizer phase this tick, prepare_tick ran
        # there -- consume its verdict instead of re-gating (a second call in
        # the same instant would read as off-cadence and skip the tick).
        if self._tick_prepared:
            self._tick_prepared = False
            if not self._tick_active:
                return PerOpSizingRequest(request={})
        elif not self.prepare_tick(topology):
            return PerOpSizingRequest(request={})

        request: Dict[ExperimentalAPMO, ActorPoolScalingRequest] = {}
        starved_ops: Set["PhysicalOperator"] = set()
        self._held_with_demand.clear()

        # Capacity source for this tick, from the snapshot prepare_tick built.
        if self._constraint_aware:
            cap: CapacityLedger = ConstraintAwareCapacityLedger(self._tick_snapshot)
        else:
            cap = GlobalCapacityLedger(self._max_placeable_actors, self._pool_resources)
        # Processing order shared with scale_where so grants match placements.
        # Default policy is reverse-topological (sink-most first) -- identical to
        # the previous ``reversed(list(topology))`` order.
        ordered_ops = self._ordered_apmo_ops(topology)

        # Number of below-min ops still at zero actors -- each needs its first
        # actor this tick. The below-min replenishment reserves one placeable
        # slot per still-unstarted op so a large downstream pool (e.g. a
        # ``map_batches(concurrency=30)`` producer) can't consume the capacity a
        # peer -- especially the source op -- needs to start. A source op left at
        # zero actors produces nothing and deadlocks the whole dataset.
        starters_unfilled = 0
        for o in topology:
            if not isinstance(o, ExperimentalAPMO):
                continue
            o_pools = o.get_autoscaling_actor_pools()
            if (
                o_pools
                and not o.has_completed()
                and 0 == o_pools[0].serving_size() < o_pools[0].min_size()
            ):
                starters_unfilled += 1

        # PASS 1 -- desired warm/warmup deltas from signals only (capacity is not
        # consulted here, so iteration order does not matter). Completed and
        # below-min ops are resolved in pass 2 (trivial and capacity-dependent).
        desired: Dict[ExperimentalAPMO, Optional[ActorPoolScalingRequest]] = {}
        for op in ordered_ops:
            pool = op.get_autoscaling_actor_pools()[0]
            if (
                op.has_completed()
                or pool.serving_size() < pool.min_size()
                or pool.min_size() == pool.max_size()
            ):
                continue
            if self._warmup:
                desired[op] = self._warmup_delta(op, pool)
            else:
                desired[op] = self._warm_delta(op, pool, topology[op])

        # REVOKE -- return same-tick downscale/completed capacity to the ledger so
        # later grants this tick can reuse it (both paths; the ledger knows how
        # to credit its own baseline).
        for op in ordered_ops:
            pool = op.get_autoscaling_actor_pools()[0]
            if op.has_completed():
                freed = pool.serving_size()
            else:
                req = desired.get(op)
                freed = -req.delta if (req is not None and req.delta < 0) else 0
            if freed > 0:
                cap.revoke(op, pool, freed)

        # PASS 2 -- clamp desired upscales to what fits and grant, walking ops in
        # placement order. Grant order == placement order, so granted ==
        # placeable by construction (a single op's fit is a sum of per-node
        # capacities, order-independent).
        #
        # Rebuild the per-op unmet-demand map here (not in prepare_tick): PASS 1
        # already read it for its shed decisions off the PRIOR eval's values, so
        # clearing it earlier would blank the signal it depends on.
        self._op_shortfalls = {}
        for op in ordered_ops:
            pool = op.get_autoscaling_actor_pools()[0]

            # (1) Completed -> release everything immediately. Safe despite
            # -current_size: the pool's scale() removes only pending or idle
            # actors (active actors finish their in-flight work first), and
            # force=True bypasses the downscale debounce.
            if op.has_completed():
                if pool.serving_size() > 0:
                    if self._release_logged_size.get(op) != pool.serving_size():
                        logger.info(
                            "OperatorSizer: releasing %d actors of %s (completed)",
                            pool.serving_size(),
                            op.name,
                        )
                        self._release_logged_size[op] = pool.serving_size()
                    request[op] = ActorPoolScalingRequest.downscale(
                        delta=-pool.serving_size(),
                        force=True,
                        reason="operator completed",
                    )
                continue
            # (2) Below-min replenishment, fixed pools included.
            if pool.serving_size() < pool.min_size():
                if pool.serving_size() == 0:
                    starters_unfilled -= 1
                # max_placeable already includes reclaimable draining actors the
                # ledger knows can be revived without new capacity.
                placeable_total = cap.max_placeable(op, pool)
                # Clamp to what fits on the op's eligible nodes (minus what was
                # already granted this tick -- the ledger debits grants in the
                # shared processing order, so scarcer ops claim their capacity
                # first).
                placeable = placeable_total
                want = pool.min_size() - pool.serving_size()
                delta = min(want, placeable)
                if (
                    delta <= 0
                    and pool.serving_size() == 0
                    and starters_unfilled == 0
                    and placeable_total >= 1
                ):
                    delta = 1
                # The part the cluster can't hold is unmet demand (autoscaler
                # scale-up signal); a min_workers=0 cluster deadlocks otherwise.
                unplaced = want - max(0, delta)
                if unplaced > 0:
                    self._record_unmet_demand(op, pool, unplaced)
                if delta > 0:
                    logger.info(
                        "OperatorSizer: replenishing %s by %d (below min size)",
                        op.name,
                        delta,
                    )
                    request[op] = ActorPoolScalingRequest.upscale(
                        delta=delta, reason="below min size"
                    )
                    cap.grant(op, pool, delta)
                continue
            # (3) Fixed pool (min == max): respect user concurrency; never resized.
            if pool.min_size() == pool.max_size():
                continue

            # (4) Warm / warmup: clamp the pass-1 desired to eligible capacity.
            delta_request = desired.get(op)
            if delta_request is None:
                continue
            if delta_request.delta < 0:
                request[op] = delta_request
                continue
            desired_delta = delta_request.delta
            # Warmup is already bounded by its equal-share ceiling; clamp it by
            # eligible capacity only on the label-aware path (so a constrained op
            # never gets an all-cluster share it can't place). The default path
            # keeps the previous unclamped-by-free warmup behavior. max_placeable
            # already includes reclaimable draining actors.
            if self._warmup and not self._constraint_aware:
                delta = desired_delta
            else:
                delta = min(desired_delta, cap.max_placeable(op, pool))
            unplaced = desired_delta - max(0, delta)
            if unplaced > 0 and delta <= 0:
                # The op asked and got nothing; the optimizer's transfer rule
                # keys recipients off a streak of these.
                starved_ops.add(op)
            # Warm always signals its shortfall; warmup does so only on the
            # label-aware path (where the shaped selector drives the autoscaler).
            if unplaced > 0 and (not self._warmup or self._constraint_aware):
                self._record_unmet_demand(op, pool, unplaced)
            if delta > 0:
                request[op] = ActorPoolScalingRequest.upscale(
                    delta=delta, reason=delta_request.reason
                )
                cap.grant(op, pool, delta)

        if self._optimizer is not None:
            self._optimizer.record_sizing_results(
                evaluated_ops=list(ordered_ops),
                starved_ops=starved_ops,
                shortfalls=self._op_shortfalls,
                neutral_ops=self._held_with_demand,
            )

        if request:
            summary = "; ".join(
                f"{op.name}: {req.delta:+d} ({req.reason})"
                for op, req in request.items()
            )
            if summary != self._last_request_summary:
                logger.info("OperatorSizer: requests %s", summary)
                self._last_request_summary = summary

        return PerOpSizingRequest(request=request)

    def scale_where(self, request: PerOpSizingRequest) -> PlacedSizingRequest:
        """Resolve where each operator's scaling delta lands.

        Stateless across ticks: placement is re-derived from current state
        every call, so failed placements and slow drains retry through
        ``scale_how_many`` re-emitting its deltas.
        """

        # Nothing to size AND nothing to evict this tick.
        if not request.request and not self._nodes_to_reclaim:
            return PlacedSizingRequest(request={})

        if SIZER_RAY_CORE_SPREAD:
            return self._scale_where_spread(request)

        zero = ExecutionResources.zero()
        # On the label-aware path, reuse the snapshot scale_how_many already built
        # this tick (same free map, plus the constraint-match sets and per-node
        # scarcity pressure). Otherwise rebuild the free map exactly as before.
        if self._constraint_aware and self._tick_snapshot is not None:
            snap = self._tick_snapshot
            alloc = dict(snap.alloc_by_node)
            # Copy: place_upscale mutates free in place; keep the snapshot intact.
            free = {node_id: res.copy() for node_id, res in snap.free_by_node.items()}
            eligible_nodes_per_op = snap.eligible_nodes_per_op
            pressure = snap.overlapping_op_count_by_node
        else:
            alloc = self._get_alloc_by_node()
            # never place onto draining nodes.
            for node_id in get_draining_nodes():
                alloc.pop(node_id, None)
            # free = allocation − everything our pools occupy or have promised
            # (running + terminating at their location, pendings at their target),
            # across ALL ops including ones with no request.
            free = self._resource_reporter.free_by_node(alloc)
            eligible_nodes_per_op = {}
            pressure = {}

        placed: Dict[ExperimentalAPMO, ExperimentalActorPoolScalingRequest] = {}

        # --- Eviction: shed actors off nodes we're losing (allocation shrank or
        # node draining), common to both paths. These drain gracefully and are
        # marked non-reclaimable so a later upscale can't revive them. Eviction
        # takes precedence over this tick's normal sizing for the same op (which
        # re-requests next tick). Consume _nodes_to_reclaim so it acts only on
        # the real tick that computed it, not on cadence-skipped steps.
        evicted_ops: Set[ExperimentalAPMO] = set()
        for op, victims in self._resolve_evictions(self._nodes_to_reclaim).items():
            if not victims:
                continue
            placed[op] = ExperimentalActorPoolScalingRequest(
                delta=-len(victims),
                force=True,
                reason="allocation shrank",
                actor_ids_to_downscale=tuple(victims),
                reclaimable=False,
            )
            evicted_ops.add(op)
        self._nodes_to_reclaim = {}

        # All request ops are sizer-managed: unsupported ones (custom placement
        # / resources) were rejected at bootstrap. Only downscales need the
        # sorted per-node victim lists; upscale views are cheap counts-only. On
        # the label-aware path each view carries the op's constraint-match set and
        # the per-node scarcity pressure so the strategy places within eligibility.
        views = {}
        for op, req in request.request.items():
            view = op.build_placement_view(need_victim_order=req.delta < 0)
            if self._constraint_aware:
                view = replace(
                    view,
                    match_nodes=eligible_nodes_per_op.get(op),
                    overlapping_op_count_by_node=pressure,
                )
            views[op] = view

        favored = self._compute_favored_nodes(request.request, views, alloc, free)

        # --- Choose Downscales first ---
        for op, req in request.request.items():
            if req.delta >= 0 or op in evicted_ops:
                continue
            # The delta is the net change to the LIVE (non-terminating) pool
            # terminating actors are already excluded in this calculation
            view = views[op]
            if op.has_completed():
                # Completed op (execution finished, outputs taken): the whole
                # pool is torn down, so there's no placement trade-off to make.
                # Skip victim selection and target every actor (running +
                # pending).
                victims = [
                    aid for ids in view.actor_ids_by_node.values() for aid in ids
                ] + [aid for ids in view.pending_ids_by_node.values() for aid in ids]
            else:
                victims = self._placement.pick_downscale_victims(
                    view, -req.delta, favored
                )
            if not victims:
                continue
            placed[op] = ExperimentalActorPoolScalingRequest.from_base(
                req,
                delta=-len(victims),
                force=True,
                actor_ids_to_downscale=tuple(victims),
            )
            # Refund capacity for PENDING victims: they never landed, so
            # cancelling them frees their claimed capacity for this tick's
            # upscales. Running victims hold their capacity until dead.
            victim_set = set(victims)
            for node_id, pending_ids in view.pending_ids_by_node.items():
                if node_id not in free:
                    continue
                for actor_id in pending_ids:
                    if actor_id in victim_set:
                        free[node_id] = free[node_id].add(view.per_actor_usage)

        # --- Upscale ordering. On the label-aware path the pin is just a size-1
        # constraint-match set, so pinned ops go through the normal targeted
        # place_upscale (candidates ∩ eligible) -- no special untargeted pass --
        # and ops are ordered by the shared policy (scarcest / most-constrained
        # first) so the grant order in scale_how_many matches placement here. ---
        if self._constraint_aware:
            upscale_ops = [
                op
                for op in self._ordered_apmo_ops(self._topology)
                if (req := request.request.get(op)) is not None
                and req.delta > 0
                and op not in evicted_ops
            ]
        else:
            # Default path: self-pinned ops FIRST, placed untargeted (Ray core
            # honors the op's own pin) with the pinned node's capacity reserved
            # from `free` so the node-targeted pass sees it occupied.
            upscale_ops = [
                op
                for op in reversed(list(self._topology))
                if (req := request.request.get(op)) is not None
                and req.delta > 0
                and op not in evicted_ops
            ]
            for op in upscale_ops:
                pinned = op.placement_pinned_node()
                if pinned is None:
                    continue
                req = request.request[op]
                # Reclaim still-DRAINING actors first (free, no startup); the rest
                # are genuinely new and placed untargeted (the op's own
                # label_selector / NodeAffinity resolves placement).
                reclaim_n = min(req.delta, op.num_reclaimable_terminating_actors())
                new_n = req.delta - reclaim_n
                if pinned in free and new_n > 0:
                    free[pinned] = (
                        free[pinned]
                        .subtract(
                            self._pool_resources(
                                op.get_autoscaling_actor_pools()[0], new_n
                            )
                        )
                        .max(zero)
                    )
                if reclaim_n == 0 and new_n == 0:
                    continue
                placed[op] = ExperimentalActorPoolScalingRequest.from_base(
                    req,
                    delta=reclaim_n + new_n,
                    num_to_reclaim=reclaim_n,
                    target_nodes_to_scale_actors_on=(),
                )

        # --- Node-targeted upscales, in the processing order above (sink-most /
        # scarcest first, so the throughput-gating op gets first pick of
        # remaining capacity). ---
        for op in upscale_ops:
            if not self._constraint_aware and op.placement_pinned_node() is not None:
                continue  # already placed untargeted above (default path only)
            req = request.request[op]
            # PG op: capacity was already capped by bundle count in
            # scale_how_many; the placement itself is delegated to Ray core (the
            # op's own PlacementGroupSchedulingStrategy picks the bundle). Emit
            # untargeted -- no free-map math, no strategy -- reclaiming draining
            # actors first as usual.
            if self._is_self_placed(op):
                reclaim_n = min(req.delta, op.num_reclaimable_terminating_actors())
                new_n = req.delta - reclaim_n
                if reclaim_n == 0 and new_n == 0:
                    continue
                placed[op] = ExperimentalActorPoolScalingRequest.from_base(
                    req,
                    delta=reclaim_n + new_n,
                    num_to_reclaim=reclaim_n,
                    target_nodes_to_scale_actors_on=(),
                )
                continue
            # Reclaim still-DRAINING actors from a prior (unfinished) downscale
            # first: they already hold their node's resources and need no
            # startup, so reviving them is free and avoids the kill+recreate
            # churn of an oscillating op. Only the remainder needs new actors
            # (which consume free capacity via place_upscale).
            reclaim_n = min(req.delta, op.num_reclaimable_terminating_actors())
            new_n = req.delta - reclaim_n
            view = views[op]
            if reclaim_n > 0:
                # Reclaimed (revived) draining actors reappear on their nodes
                # alongside the new actors, but the view excludes terminating
                # actors: count them back in so placement doesn't double-serve
                # the locality demand they already satisfy.
                actors_by_node = dict(view.actors_by_node)
                for node_id in op.reclaimable_node_ids(reclaim_n):
                    actors_by_node[node_id] = actors_by_node.get(node_id, 0) + 1
                view = replace(view, actors_by_node=actors_by_node)
            nodes = self._placement.place_upscale(view, free, new_n)
            if len(nodes) < new_n:
                # The free map has no room for the remainder: drop it and let
                # scale_how_many re-request it next tick (~2s). Every placement
                # stays under the locality strategy -- no untargeted fallback.
                free_cpu = sum(r.cpu or 0 for r in free.values())
                free_gpu = sum(r.gpu or 0 for r in free.values())
                free_mem = sum(r.memory or 0 for r in free.values())
                pa = views[op].per_actor_usage
                shortfall_summary = (
                    f"{op.name}: placed {len(nodes)}/{new_n} new "
                    f"(reclaimed {reclaim_n}; free across {len(free)} nodes: "
                    f"{free_cpu:.0f} cpu, {free_gpu:g} gpu, "
                    f"{free_mem / 1e9:.0f} GiB mem; per-actor "
                    f"{(pa.cpu or 0):g}cpu/{(pa.gpu or 0):g}gpu/"
                    f"{(pa.memory or 0) / 1e9:.1f}GiBmem)"
                )
                if shortfall_summary != self._last_shortfall_summary:
                    logger.info(
                        "OperatorSizer placement shortfall: %s -- remainder "
                        "dropped, re-requested next tick.",
                        shortfall_summary,
                    )
                    self._last_shortfall_summary = shortfall_summary
            if reclaim_n == 0 and not nodes:
                continue
            placed[op] = ExperimentalActorPoolScalingRequest.from_base(
                req,
                delta=reclaim_n + len(nodes),
                num_to_reclaim=reclaim_n,
                target_nodes_to_scale_actors_on=tuple(nodes),
            )

        self._log_placement(alloc, free, placed)
        return PlacedSizingRequest(request=placed)

    def _scale_where_spread(self, request: PerOpSizingRequest) -> PlacedSizingRequest:
        """Ray-core-SPREAD baseline placement (no locality, no node math).

        Upscales are emitted untargeted (empty ``target_nodes_to_scale_actors_on``)
        so the actor pool creates them with ``scheduling_strategy=SPREAD`` and
        Ray core places them; still-DRAINING actors are reclaimed first (free,
        node-agnostic). Downscales pick the cheapest victims (pending < idle <
        busy) with no favored-node preference.
        """
        placed: Dict[ExperimentalAPMO, ExperimentalActorPoolScalingRequest] = {}
        for op, req in request.request.items():
            if req.delta < 0:
                view = op.build_placement_view()
                if op.has_completed():
                    victims = [
                        aid for ids in view.actor_ids_by_node.values() for aid in ids
                    ] + [
                        aid for ids in view.pending_ids_by_node.values() for aid in ids
                    ]
                else:
                    victims = self._placement.pick_downscale_victims(
                        view, -req.delta, favored_nodes=frozenset()
                    )
                if not victims:
                    continue
                placed[op] = ExperimentalActorPoolScalingRequest.from_base(
                    req,
                    delta=-len(victims),
                    force=True,
                    actor_ids_to_downscale=tuple(victims),
                )
            elif req.delta > 0:
                # Reclaim still-DRAINING actors first (free, no startup); the
                # rest are genuinely new and untargeted (Ray-core SPREAD).
                reclaim_n = min(req.delta, op.num_reclaimable_terminating_actors())
                placed[op] = ExperimentalActorPoolScalingRequest.from_base(
                    req,
                    delta=req.delta,
                    num_to_reclaim=reclaim_n,
                    target_nodes_to_scale_actors_on=(),
                )
        return PlacedSizingRequest(request=placed)

    def _capacity_map(self, alloc, free) -> str:
        """Compact per-node free/allocated resource string for logging. Gives a full picture of the
        capacity each placement decision saw."""

        def gb(v):
            return (v or 0) / 1e9

        parts = []
        for node_id in sorted(alloc):
            a, f = alloc[node_id], free.get(node_id)
            entry = f"{node_id[:8]}: {(f.cpu if f else 0) or 0:.0f}/{a.cpu or 0:.0f}cpu"
            if a.gpu:
                entry += f" {(f.gpu if f else 0) or 0:g}/{a.gpu:g}gpu"
            if a.memory:
                entry += f" {gb(f.memory if f else 0):.0f}/{gb(a.memory):.0f}GiBmem"
            if a.object_store_memory:
                entry += (
                    f" {gb(f.object_store_memory if f else 0):.0f}/"
                    f"{gb(a.object_store_memory):.0f}GiBobj"
                )
            parts.append(entry)
        return "; ".join(parts)

    def _log_placement(self, alloc, free, placed) -> None:
        """INFO visibility into placement decisions + the capacity map the
        strategy saw (both suppress repeats to keep the ~2s loop quiet)."""
        if not alloc:
            if self._last_capacity_summary != "":
                logger.info(
                    "OperatorSizer capacity: empty allocation -- no "
                    "placement-eligible nodes this tick."
                )
                self._last_capacity_summary = ""
            return
        if not placed:
            return
        breakdown = getattr(self._placement, "last_upscale_breakdown", {})
        parts = []
        for op, req in placed.items():
            if req.delta > 0:
                via = breakdown.get(op.id)
                via_str = f" via={via[0]}collocate/{via[1]}spread" if via else ""
                parts.append(
                    f"{op.name}: +{req.delta}@"
                    f"{dict(Counter(req.target_nodes_to_scale_actors_on))}{via_str}"
                )
            else:
                parts.append(
                    f"{op.name}: {req.delta} "
                    f"(victims={len(req.actor_ids_to_downscale)})"
                )
        summary = "; ".join(parts)
        cap = self._capacity_map(alloc, free)
        if (
            summary != self._last_placement_summary
            or cap != self._last_capacity_summary
        ):
            logger.info(
                "OperatorSizer placement: %s | capacity(%d nodes): %s",
                summary,
                len(alloc),
                cap,
            )
            self._last_placement_summary = summary
            self._last_capacity_summary = cap

    def scale(self, request: PlacedSizingRequest) -> None:
        """Apply the placement-resolved requests and drive draining."""
        # Downscales before upscales, deterministically.
        for op, req in sorted(
            request.request.items(), key=lambda entry: entry[1].delta
        ):
            logger.debug(
                "scale: %s delta=%d nodes=%s victims=%s",
                op.name,
                req.delta,
                req.target_nodes_to_scale_actors_on,
                req.actor_ids_to_downscale,
            )
            op.apply_scale(req)

        # Drive draining AFTER applying this tick's requests, so a same-tick
        # upscale's reclaim_draining_actors can revive a draining actor before
        # process_draining_actors would otherwise kill it. Runs for EVERY
        # actor-pool op
        for op in self._topology:
            if isinstance(op, ExperimentalAPMO):
                op.process_draining_actors()
                # A pending that can never become ready (failed-restart wedge,
                # stuck env install, ghost) would hold this op's sizing arms
                # forever via the pending gate; expire it so the freed claim is
                # re-placed with current cluster information next tick.
                op.expire_stuck_pending_actors()

    def record_tick(
        self,
        how_many: PerOpSizingRequest,
        where: PlacedSizingRequest,
        *,
        optimize_s: float,
        how_many_s: float,
        where_s: float,
        apply_s: float,
        e2e_s: float,
    ) -> None:
        """Emit per-tick sizer telemetry (called by the executor after scale()).

        Always sets the per-phase duration gauges (so ``e2e`` stays continuous),
        but only logs the verbose roll-up + locality on cadence ticks -- the
        ones where scale_how_many actually evaluated and produced a request.

        ``optimize`` covers observe() + optimize_pipeline(), and like the other
        phases is emitted above the idle-tick early return: the optimizer does
        most of its work on ticks where nothing is sized, so a phase recorded
        only on sizing ticks would miss it.
        """
        tick_durations = {
            "optimize": optimize_s,
            "how_many": how_many_s,
            "where": where_s,
            "apply": apply_s,
            "e2e": e2e_s,
        }

        # Emit tick-duration even on no-op ticks so the series stays alive in
        # steady state -- but record_tick fires every scheduling-loop step, so
        # throttle the _StatsActor RPC to the op-metrics push cadence
        # (UPDATE_METRICS_INTERVAL_S) to avoid per-step flooding.
        now = time.monotonic()
        if now - self._last_tick_emit_s >= _SIZER_TICK_EMIT_INTERVAL_S:
            self._last_tick_emit_s = now
            _StatsManager.update_sizer_metrics(self._dataset_id, tick_durations)

        if not how_many.request and not where.request:
            # Idle (non-cadence) tick: nothing was sized or placed.
            return

        # Per-operator actor-scaling churn, split by direction, accumulated on
        # ``op.metrics`` (OpRuntimeMetrics). The standard executor stats push
        # (every UPDATE_METRICS_INTERVAL_S) re-emits these like every other
        # per-operator Ray Data metric, so they reliably reach Grafana.
        for op, r in how_many.request.items():
            if r.delta > 0:
                op.metrics.sizer_actor_delta_how_many_up += r.delta
            elif r.delta < 0:
                op.metrics.sizer_actor_delta_how_many_down += -r.delta
        for op, r in where.request.items():
            if r.delta > 0:
                op.metrics.sizer_actor_delta_where_up += r.delta
            elif r.delta < 0:
                op.metrics.sizer_actor_delta_where_down += -r.delta

        up = down = 0
        per_op = []
        for op, r in where.request.items():
            if not r.delta:
                continue
            per_op.append(f"{op.name}:{r.delta:+d}")
            if r.delta > 0:
                up += r.delta
            else:
                down -= r.delta
        nodes = {
            node
            for r in where.request.values()
            for node in r.target_nodes_to_scale_actors_on
        }
        logger.info(
            "OperatorSizer tick: e2e=%.1fms (optimize=%.1f how_many=%.1f "
            "where=%.1f apply=%.1f) | scale_up=+%d scale_down=-%d across %d ops "
            "[%s]; placed on %d nodes",
            e2e_s * 1e3,
            optimize_s * 1e3,
            how_many_s * 1e3,
            where_s * 1e3,
            apply_s * 1e3,
            up,
            down,
            len(where.request),
            ", ".join(per_op),
            len(nodes),
        )
        self._log_locality()

    def _log_locality(self) -> None:
        """One per-operator line combining the existing local/remote block
        ratio with the new cross-node bytes (this-tick delta + cumulative)."""
        assert self._topology is not None
        for op in self._topology:
            if not isinstance(op, ExperimentalAPMO):
                continue
            total = op.metrics.bytes_remote_inputs_read
            delta = total - self._last_remote_bytes.get(op, 0)
            self._last_remote_bytes[op] = total
            logger.info(
                "OperatorSizer locality %s: %s +%.1f MiB cross-node this tick "
                "(total %.1f MiB, %d remote blocks)",
                op.name,
                locality_string(op._locality_hits, op._locality_misses),
                delta / 1024**2,
                total / 1024**2,
                op.metrics.num_remote_blocks_read,
            )

    @staticmethod
    def _assert_all_ops_supported(topology: Topology) -> None:
        """Reject operators the sizer can't manage (custom placement/resources).

        The sizer owns actor placement when enabled; it doesn't yet support
        operators that take over scheduling themselves or request custom
        resources. ``validate_ray_remote_args`` raises a clear per-op error
        rather than letting them be mis-placed.
        """
        for op in topology:
            if isinstance(op, ExperimentalAPMO):
                op.validate_ray_remote_args()

    def get_unmet_demand(self) -> ExecutionResources:
        """Resources the warm loop wanted this eval but the cluster couldn't place.

        Aggregated per-op ``unplaced`` from the last real ``scale_how_many``
        evaluation. ``>0`` means the dataset's actor demand exceeds its current
        cluster allocation -- the cluster autoscaler's scale-up signal. Zero when
        the sizer's demand fits (e.g. a serial/backpressured workload), so the
        cluster is not grown past what the work needs.
        """
        return self._pending_unmet_demand

    def _get_alloc_by_node(self) -> Dict[NodeIdStr, ExecutionResources]:
        assert self._resource_reporter is not None
        return self._resource_reporter.get_reserved_resources_by_node()

    # --- label/PG-aware tick snapshot + ordering (Phase 1) -------------------

    def _apmo_ops_with_pools(self, topology: Topology) -> List[ExperimentalAPMO]:
        return [
            op
            for op in topology
            if isinstance(op, ExperimentalAPMO) and op.get_autoscaling_actor_pools()
        ]

    def _is_self_placed(self, op: ExperimentalAPMO) -> bool:
        """True if ``op`` is placed by Ray core, not the sizer (PG-aware path).

        Two cases, both created untargeted and kept out of the sizer's free-map
        accounting:
          - a static placement group (``PlacementGroupSchedulingStrategy`` /
            ``placement_group`` in ray_remote_args) -- capped by bundle count;
          - a ``ray_remote_args_fn`` that assigns a PG per actor (the vLLM "ray"
            backend) -- capped by user concurrency (``max_size``). The fn's
            PG-only contract is enforced in ``_merge_ray_remote_args``.
        """
        if not self._constraint_aware:
            return False
        return (
            op.placement_constraint().has_placement_group or op.uses_ray_remote_args_fn
        )

    def _build_tick_snapshot(self, topology: Topology) -> TickSnapshot:
        """One per-tick view both phases share (constraint-aware path).

        Hoists scale_where's free-map construction to tick start and adds the
        constraint-match set + per-node constraint pressure. Labels come from the
        5s-cached ``ray.nodes()`` view (the allocation RPC carries none), so this
        adds only that cheap local read.
        """
        assert self._resource_reporter is not None
        cluster_view = self._resource_reporter.get_cluster_view(include_labels=True)
        alloc = dict(cluster_view.alloc_by_node)
        for node_id in get_draining_nodes():
            alloc.pop(node_id, None)
        free = self._resource_reporter.free_by_node(alloc)

        # matching_nodes returns nodes drawn from this ClusterView's allocation,
        # so a match never names a node we can't place on.
        view_for_match = ClusterView(
            alloc_by_node=alloc, labels_by_node=cluster_view.labels_by_node
        )
        eligible_nodes_per_op: Dict[ExperimentalAPMO, AbstractSet[NodeIdStr]] = {}
        per_actor_resources_by_op: Dict[ExperimentalAPMO, ExecutionResources] = {}
        for op in self._apmo_ops_with_pools(topology):
            constraint = op.placement_constraint()
            per_actor = op.get_autoscaling_actor_pools()[0].per_actor_resource_usage()
            per_actor_resources_by_op[op] = per_actor
            # Pass per_actor so nodes structurally lacking a resource the actor
            # requests (e.g. a GPU op on CPU-only nodes) are excluded from the
            # match set, not just nodes that fail the label selector.
            eligible_nodes_per_op[op] = constraint.matching_nodes(
                view_for_match, per_actor
            )

        # Per-node scarcity: how many *constrained* ops can use each node. Feeds
        # the ledger's grant order and the strategies' spread tie-break, so a
        # less-constrained op consumes contended nodes last, leaving them for the
        # ops that can only run there. An op is constrained if it has a
        # label_selector OR it requests GPU: GPU nodes are inherently scarce (few
        # of them, and they carry CPU a plain CPU op could otherwise camp on), so
        # a GPU op raises pressure on them just like a label op does. PG ops size
        # against bundles, not this node set, so they don't add pressure.
        pressure: Dict[NodeIdStr, int] = {}
        for op, match in eligible_nodes_per_op.items():
            if self._is_self_placed(op):
                continue
            constrained = (
                op.placement_constraint().has_label_selector
                or per_actor_resources_by_op[op].gpu > 0
            )
            if constrained:
                for node_id in match:
                    pressure[node_id] = pressure.get(node_id, 0) + 1

        # Remaining PG bundle capacity = cap − the actors holding a bundle.
        # ``current_size`` (not ``serving_size``): a draining actor keeps its
        # bundle reserved until it is actually dead, so the bundle is not free
        # to hand to a new actor yet.
        pg_remaining_by_op: Dict[ExperimentalAPMO, int] = {}
        for op, cap in self._pg_cap_by_op.items():
            if op in eligible_nodes_per_op:  # op belongs to this topology
                current = op.get_autoscaling_actor_pools()[0].current_size()
                pg_remaining_by_op[op] = max(0, cap - current)

        return TickSnapshot(
            free_by_node=free,
            alloc_by_node=alloc,
            eligible_nodes_per_op=eligible_nodes_per_op,
            per_actor_resources_by_op=per_actor_resources_by_op,
            overlapping_op_count_by_node=pressure,
            pg_remaining_by_op=pg_remaining_by_op,
        )

    def _ordered_apmo_ops(self, topology: Topology) -> List[ExperimentalAPMO]:
        """APMO ops in the tick's processing order (shared by both phases)."""
        topo_index = {op: idx for idx, op in enumerate(topology)}
        # Constraint specificity for within-tier ordering: how many nodes each
        # op's constraint currently matches, from the tick snapshot. None
        # outside a constraint-aware tick (bootstrap / initial sizing, or the
        # global path) -- the ordering then falls back to sink-most ties.
        eligible = (
            self._tick_snapshot.eligible_nodes_per_op
            if self._constraint_aware and self._tick_snapshot is not None
            else {}
        )
        features = {
            op: OpOrderingSignals.from_operator(
                op,
                topo_index=topo_index[op],
                # Only the constraint-aware path uses constraints for ordering;
                # skip parsing (and the remote-args read) when it's off.
                constraint=op.placement_constraint()
                if self._constraint_aware
                else None,
                constraint_aware=self._constraint_aware,
                memory_per_cpu_bytes=ACTOR_MEMORY_PER_CPU_BYTES,
                num_eligible_nodes=(len(eligible[op]) if op in eligible else None),
            )
            for op in self._apmo_ops_with_pools(topology)
        }
        return list(self._ordering.order(features))

    # --- shrinking-allocation eviction (both paths) --------------------------

    def _compute_nodes_to_reclaim(
        self, alloc: Dict[NodeIdStr, ExecutionResources]
    ) -> Dict[NodeIdStr, ExecutionResources]:
        """Per-node resources our committed usage exceeds the allocation by.

        A node missing from ``alloc`` (dropped entirely, or draining) is fully
        over-allocated. PG ops are excluded (their actors live in the PG
        reservation, not this dataset's allocation), same as ``committed``.
        """
        zero = ExecutionResources.zero()
        reclaim: Dict[NodeIdStr, ExecutionResources] = {}
        for node_id, used in self._resource_reporter.committed_by_node().items():
            over = used.subtract(alloc.get(node_id, zero)).max(zero)
            if not over.is_zero():
                reclaim[node_id] = over
        return reclaim

    @staticmethod
    def _reduces_over(per: ExecutionResources, over: ExecutionResources) -> bool:
        """True if an actor of shape ``per`` frees a dimension still over budget
        -- so we don't evict, say, a CPU-only op when only GPU is over."""
        return any(
            getattr(per, dim) > 0 and getattr(over, dim) > 0
            for dim in ("cpu", "gpu", "memory")
        )

    def _resolve_evictions(
        self, nodes_to_reclaim: Dict[NodeIdStr, ExecutionResources]
    ) -> Dict[ExperimentalAPMO, List[LogicalActorId]]:
        """Pick victim actors to evict off over-allocated nodes, cheapest-first
        (pending -> idle -> least-busy), across ops, until each node fits its
        allocation. PG ops are skipped (not in our allocation)."""
        zero = ExecutionResources.zero()
        remaining = dict(nodes_to_reclaim)
        victims: Dict[ExperimentalAPMO, List[LogicalActorId]] = {}
        for op in self._apmo_ops_with_pools(self._topology):
            if self._is_self_placed(op):
                continue
            if all(r.is_zero() for r in remaining.values()):
                break
            view = op.build_placement_view(need_victim_order=True)
            per = view.per_actor_usage
            for node_id, over in remaining.items():
                if over.is_zero() or not self._reduces_over(per, over):
                    continue
                # pending (cancel, cheapest) then running least-busy-first.
                candidates = list(view.pending_ids_by_node.get(node_id, [])) + list(
                    view.actor_ids_by_node.get(node_id, [])
                )
                for actor_id in candidates:
                    if remaining[node_id].is_zero():
                        break
                    victims.setdefault(op, []).append(actor_id)
                    remaining[node_id] = remaining[node_id].subtract(per).max(zero)
        return victims

    def observe(self, topology: Topology) -> None:
        """Sample the optimizer's per-op signals; cheap to call every
        scheduling-loop iteration (the optimizer gates recording)."""
        if self._optimizer is not None and SIZER_ENABLE_OPTIMIZER:
            self._optimizer.observe(topology)

    def optimize_pipeline(self, topology: Topology) -> None:
        """The optimizer phase: cross-operator corrections (capacity
        transfers, silent-bottleneck grants), applied before scale_how_many
        each tick. Claims are targeted pendings in committed accounting, so
        the sizing passes that follow already see the reserved space."""
        self._tick_active = self.prepare_tick(topology)
        self._tick_prepared = True
        if (
            not self._tick_active
            or not SIZER_ENABLE_OPTIMIZER
            or self._optimizer is None
        ):
            return
        requests = self._optimizer.optimize_pipeline(
            topology,
            warmup=self._warmup,
            eligible_nodes=self._constraint_nodes,
            pool_cap=self._op_pool_cap,
            growth_step=self._growth_step,
        )
        for placed in requests:
            self.scale(request=placed)
        if requests:
            self._optimizer.after_apply()

    def _constraint_nodes(self, op) -> Optional[AbstractSet[NodeIdStr]]:
        """Constraint-match node set for ``op`` (None = unconstrained)."""
        snap = self._tick_snapshot
        return snap.eligible_nodes_per_op.get(op) if snap else None

    def _op_pool_cap(self, op) -> int:
        return self._pool_cap(op, op.get_autoscaling_actor_pools()[0])

    def _pool_cap(self, op, pool) -> int:
        limit = pool.max_size()
        if self._is_sink_or_write(op):
            limit = min(limit, self._sink_cap)
        return limit

    def _growth_step(self, op) -> int:
        pool = op.get_autoscaling_actor_pools()[0]
        return max(1, math.ceil(self._upscale_factor * pool.serving_size()))

    def _record_unmet_demand(self, op: ExperimentalAPMO, pool, count: int) -> None:
        """Accumulate ``count`` unplaced actors of ``op`` as unmet demand.

        Always updates the single ``ExecutionResources`` aggregate (the
        autoscaler's scalar scale-up signal). On the label-aware path it also
        records a shaped bundle tagged with the op's ``label_selector`` so the
        autoscaler can request the matching node type (see
        ``get_unmet_demand_bundles``). PG ops are excluded -- the PG won't grow,
        so their shortfall is not an autoscaler signal.
        """
        unmet = self._pool_resources(pool, count)
        self._pending_unmet_demand = self._pending_unmet_demand.add(unmet)
        # Per-op record for the optimizer (recipient sizing, consumer checks).
        prior = self._op_shortfalls.get(
            op, OpShortfall(resources=ExecutionResources.zero(), num_actors=0)
        )
        self._op_shortfalls[op] = OpShortfall(
            resources=prior.resources.add(unmet),
            num_actors=prior.num_actors + count,
        )
        if not self._constraint_aware:
            return
        constraint = op.placement_constraint()
        if self._is_self_placed(op):
            # PG / ray_remote_args_fn ops don't grow via the autoscaler (the PG
            # won't grow, and a fn op is capped at its concurrency), so their
            # shortfall is not an autoscaler node-request signal.
            return
        selector = constraint.label_selector
        self._pending_unmet_bundles.append(
            (pool.per_actor_resource_usage().to_resource_dict(), count, selector)
        )

    def get_unmet_demand_bundles(
        self,
    ) -> List[Tuple[Dict[str, float], int, Optional[Dict[str, str]]]]:
        """Shaped unmet demand: (per-actor resources, count, label_selector).

        One entry per shortfall from the last real ``scale_how_many``. The
        autoscaler expands these into per-bundle resource requests + selectors so
        a constrained op grows the right node type. Empty on the default path
        (use ``get_unmet_demand`` there).
        """
        return list(self._pending_unmet_bundles)

    def _cluster_capacity(self) -> ExecutionResources:
        """Cluster capacity available to this dataset, from the coordinator.

        Sum of the per-node allocation. Train's HIGH-priority reservation is
        already subtracted by the coordinator, so this is our fair share. Used
        by ``scale_how_many`` to bound how many actors to request; the same
        allocation drives ``scale_where``'s per-node ``free``, so the two stay
        consistent (what we size for, we can place).
        """
        total = ExecutionResources.zero()
        for resources in self._get_alloc_by_node().values():
            total = total.add(resources)
        return total

    def _compute_favored_nodes(
        self,
        managed_request: Dict[ExperimentalAPMO, ActorPoolScalingRequest],
        views: Dict[ExperimentalAPMO, OpPlacementView],
        alloc: Dict[NodeIdStr, ExecutionResources],
        free: Dict[NodeIdStr, ExecutionResources],
    ) -> Set[NodeIdStr]:
        """Nodes capacity-starved upscaling ops want freed.

        Fed into ``pick_downscale_victims`` so victims are taken where the
        freed capacity benefits the upscalers (the design doc's two-set rule).
        ``managed_request`` excludes custom-placement ops (those aren't placed
        by the sizer).
        """
        # Nodes where a downscaling op has pending/idle victims — eviction
        # there is ~instant, so the "capacity free now" ranking applies; nodes
        # that can only free capacity via a slow drain of busy actors are
        # ranked by durable source/sink locality instead.
        fast_free_nodes: Set[NodeIdStr] = set()
        for op, req in managed_request.items():
            if req.delta < 0:
                view = views[op]
                fast_free_nodes.update(view.pending_ids_by_node)
                fast_free_nodes.update(view.idle_actors_by_node)

        # (view, unmet delta) per upscaling op, sink-most first. On the default
        # path, self-pinned ops place untargeted (Ray core resolves placement),
        # so they never need a node freed and are excluded. On the label-aware
        # path a pin is a normal size-1 eligible set, so pinned ops DO
        # participate -- and every op's starved test / favored nomination is
        # restricted to its eligible (constraint-match) subset of free, else an
        # op with plenty of *ineligible* free capacity never looks starved.
        starved: List[Tuple[OpPlacementView, int]] = []
        for op in reversed(list(self._topology)):
            req = managed_request.get(op)
            if req is None or req.delta <= 0:
                continue
            if not self._constraint_aware and op.placement_pinned_node() is not None:
                continue
            # PG ops consume the PG reservation, not the free map, so evicting a
            # non-PG actor never helps them -- they never nominate favored nodes.
            if self._is_self_placed(op):
                continue
            view = views[op]
            eligible_nodes = (
                free
                if view.match_nodes is None
                else [node_id for node_id in free if node_id in view.match_nodes]
            )
            fits = sum(
                max_placeable_actors(free[node_id], view.per_actor_usage)
                for node_id in eligible_nodes
            )
            if fits < req.delta:
                starved.append((view, req.delta - int(fits)))
        if not starved:
            return set()

        return set(
            self._placement.compute_favored_nodes(
                starved_views=starved,
                free=free,
                all_nodes=set(alloc),
                fast_free_nodes=fast_free_nodes,
            )
        )

    # --- scale_how_many: warmup sizing ---------------------------------------

    def _warmup_delta(
        self, op: ExperimentalAPMO, pool
    ) -> Optional[ActorPoolScalingRequest]:
        """Equal-share warmup target for one op, as an upscale-only delta.

        target = clamp(ceiling / per_actor, [min_size, max_size]). The delta is
        self-zeroing because serving_size() counts pending actors (a free
        pending-gate), and warmup never downscales (a shrunken ceiling just
        stops further growth). The grant is clamped to what the cluster can
        actually place (minus resources already granted this tick), so peers
        can't collectively push past the cluster even though warmup never
        downscales an already-over-target pool.
        """
        resource_type = self._op_resource_type(op)
        per_actor = getattr(pool.per_actor_resource_usage(), resource_type)
        if per_actor <= 0:
            # Resource shape is asserted at bootstrap; defensive only.
            return None
        ceiling = self._warmup_ceiling(resource_type)
        # An actor with max_actor_concurrency=k serves ~k tasks at once, so size
        # for k x fewer actors (matching _backlog_actor_target's warm-phase
        # division). No-op for the default concurrency of 1.
        concurrency = max(1, pool.max_actor_concurrency())
        target = int(
            min(
                max(ceiling / per_actor / concurrency, pool.min_size()),
                pool.max_size(),
            )
        )
        if self._is_sink_or_write(op):
            target = max(min(target, self._sink_cap), pool.min_size())
        delta = target - pool.serving_size()
        logger.info(
            "OperatorSizer: warmup %s resource_type=%s ceiling=%.1f target=%d "
            "current=%d delta=%d",
            op.name,
            resource_type,
            ceiling,
            target,
            pool.serving_size(),
            max(delta, 0),
        )
        if delta > 0:
            return ActorPoolScalingRequest.upscale(
                delta=delta, reason="warmup equal-share"
            )
        return None

    def _warmup_ceiling(self, resource_type: str) -> float:
        """Per-op warmup ceiling in the given resource type ("cpu" or "gpu").

        (capacity - Min_reserved) * ratio / num_autoscaled_ops, recomputed
        against live capacity every call so the allocation follows the cluster
        ramp. Min_reserved sums the floors of ALL eligible ops in this resource type
        (fixed pools included -- their whole demand is taken off the top);
        only the divisor excludes fixed pools, since a fixed pool never
        consumes a ceiling share.
        """
        limit = getattr(self._cluster_capacity(), resource_type)
        min_reserved = 0.0
        autoscaled_count = 0
        for op in self._eligible_ops_by_resource_type(resource_type):
            min_resources, _ = op.min_max_resource_requirements()
            min_reserved += getattr(min_resources, resource_type)
            pool = op.get_autoscaling_actor_pools()[0]
            if pool.min_size() != pool.max_size():
                autoscaled_count += 1
        if autoscaled_count == 0:
            return 0.0
        return (
            max(limit - min_reserved, 0.0) * self._reservation_ratio / autoscaled_count
        )

    def _eligible_ops_by_resource_type(
        self, resource_type: str
    ) -> List[ExperimentalAPMO]:
        assert self._topology is not None
        return [
            op
            for op in self._topology
            if isinstance(op, ExperimentalAPMO)
            and op.get_autoscaling_actor_pools()
            and self._is_op_eligible(op)
            and self._op_resource_type(op) == resource_type
        ]

    def _op_resource_type(self, op: ExperimentalAPMO) -> str:
        """ "gpu" if the op's actors each request GPU, else "cpu"."""
        for pool in op.get_autoscaling_actor_pools():
            if pool.per_actor_resource_usage().gpu > 0:
                return "gpu"
        return "cpu"

    @staticmethod
    def _is_op_eligible(op: "PhysicalOperator") -> bool:
        # Mirrors ResourceManager.is_op_eligible without requiring a
        # ResourceManager handle.
        return not op.throttling_disabled() and not op.has_execution_finished()

    # --- scale_how_many: warm loop (backpressure-driven) ---------------------

    def _warm_delta(
        self, op: ExperimentalAPMO, pool, op_state
    ) -> Optional[ActorPoolScalingRequest]:
        """Backpressure-driven sizing for one op.

        Binary signals gate the decision -- has-work (InQueue non-empty OR
        in-flight tasks) and OutQueue full ("can downstream absorb?", the
        per-actor output-cap fraction) -- and a backlog-based demand target
        bounds the upscale magnitude so a fed pool never grows past the work it
        actually has. The demand bound is load-bearing: without it, at cold
        start a pool's output queue is empty (blocked=0) and +upscale_factor/tick
        growth fills the whole cluster before any backpressure builds. Downscale
        arms are hysteretic, never go below min_size, and remove pending/idle
        actors only -- busy actors drain by attrition until the drain flag
        lands.

        Emits one telemetry line per op per tick recording the chosen path.
        Ported from prasad/operator-sizer (PR #3396): adds the demand bound and
        the over-cap-while-fed downscale arm. The shortfall (`unplaced`) is
        surfaced by scale_where's placement logging in this branch's backend.
        """
        cur = pool.serving_size()
        pending = pool.num_pending_actors()
        enqueued = op_state.total_enqueued_input_blocks()
        in_flight = pool.num_tasks_in_flight()
        # "Has work" considers in-flight tasks, not just the input queue: a
        # steady-state pool can have an empty queue but busy actors, and must
        # not be treated as idle (which would drain it mid-flight).
        has_work = enqueued > 0 or in_flight > 0
        out_full, blocked, active = self._op_outqueue_full(op)
        limit = self._pool_cap(op, pool)
        demand = self._backlog_actor_target(op, pool, op_state)

        request: Optional[ActorPoolScalingRequest] = None
        if pending > 0:
            # Sizing in flight; don't judge a pool mid-resize, so a paid-for
            # upscale isn't undone the tick its actors land.
            decision = f"hold (sizing in flight: {pending} pending)"
        elif has_work and not out_full:
            # Grow when fed and downstream is absorbing (not out_full).
            # Rate-limited and clamped to demand, the pool cap, and free
            # cluster capacity. An out_full op whose consumers ARE absorbing
            # is grown by the optimizer's silent-bottleneck rule -- this eval
            # stays local to ``op``.
            self._down_streak[op] = 0
            want = max(1, math.ceil(self._upscale_factor * cur))
            if cur > limit:
                # Already over the (sink) cap while fed -- the upscale arm would
                # otherwise just hold here forever. Drive it back down toward the
                # cap (bounded by _down_delta, which respects min_size).
                dd = min(self._down_delta(pool), cur - limit)
                if dd > 0:
                    request = ActorPoolScalingRequest.downscale(
                        delta=-dd, reason="over cap (fed)"
                    )
                    decision = f"downscale -{dd} (over cap, limit={limit})"
                else:
                    decision = f"hold (at min_size, over cap limit={limit})"
            else:
                # Desired upscale from signals only (capacity-independent). The
                # sizer's pass 2 clamps this to what fits and records the
                # shortfall as unmet demand -- so both the default and label-aware
                # paths converge desired->placeable at one place, in placement
                # order.
                desired_step = min(want, max(0, demand - cur), limit - cur)
                if self._optimizer is not None and not self._optimizer.can_op_scale(op):
                    # The optimizer needs this op held (e.g. its transfer
                    # victims are still draining).
                    decision = "hold (optimizer holds op)"
                elif desired_step > 0:
                    request = ActorPoolScalingRequest.upscale(
                        delta=desired_step,
                        reason="backpressure: fed, downstream absorbing",
                    )
                    decision = f"want +{desired_step} (want={want}, demand={demand})"
                elif demand <= cur:
                    decision = f"hold (at demand={demand})"
                else:
                    decision = "hold (at max size)"
        elif not has_work:
            # Fully idle: no queue, no in-flight work. Drain after the
            # hysteresis streak. Over-provisioned-but-working pools are the
            # optimizer's job (a transfer with a recipient in hand), not a
            # local shed.
            self._down_streak[op] = self._down_streak.get(op, 0) + 1
            if (
                self._down_streak[op] >= self._downscale_hysteresis
                and cur > pool.min_size()
            ):
                self._down_streak[op] = 0
                request = ActorPoolScalingRequest.downscale(
                    delta=-self._down_delta(pool), reason="drain (idle)"
                )
                decision = "downscale (drain: idle)"
            else:
                decision = (
                    f"hold (idle, streak "
                    f"{self._down_streak[op]}/{self._downscale_hysteresis})"
                )
        else:
            self._down_streak[op] = 0
            decision = "hold (output backpressured)"
            if demand > cur:
                self._held_with_demand.add(op)

        logger.info(
            "OperatorSizer: warm %s in_q=%d out_blocked=%d/%d demand=%d "
            "current=%d -> %s",
            op.name,
            enqueued,
            blocked,
            active,
            demand,
            cur,
            decision,
        )
        return request

    def _backlog_actor_target(self, op: ExperimentalAPMO, pool, op_state) -> int:
        """Actors needed for the current backlog (queued + in-flight work),
        ported from RayTurboActorAutoscaler (PR #3396). The in-flight term is
        load-bearing: a well-sized steady-state pool has a shallow queue but
        deep in-flight, so demand tracks real work instead of collapsing the
        moment the queue drains. Divided by the per-actor concurrent-task
        capacity scaled by the util threshold, so the target is "actors to run
        this many tasks at the configured utilization".
        """
        avg = op.metrics.average_num_inputs_per_task or 1
        enqueued = op_state.total_enqueued_input_blocks()
        expected = math.ceil(enqueued / avg)
        total_tasks = pool.num_tasks_in_flight() + expected
        return math.ceil(
            total_tasks
            / (pool.max_actor_concurrency() * self._util_upscaling_threshold)
        )

    def _op_outqueue_full(self, op: ExperimentalAPMO):
        """(full?, blocked, running) from the per-actor output-cap state.

        Fraction of RUNNING actors whose pull budget is exhausted: idle actors
        count as unblocked capacity rather than vanishing from the denominator
        (the task-denominated fraction saturates at 1.0 whenever every busy
        actor is blocked, however idle the pool is).
        """
        # output_backpressured_actors() owns the resize guards (no running
        # actors / pending actors -> (0, 0)), so a pool mid-resize reads as
        # not full here -- this matters for the optimizer's out_full streaks,
        # which judge ops against their neighbours.
        blocked, running = op.output_backpressured_actors()
        if running == 0:
            return False, blocked, running
        return (blocked / running) >= self._outqueue_full_fraction, blocked, running

    def _max_placeable_actors(self, pool, granted: ExecutionResources) -> int:
        """Max additional actors of this pool that fit in free cluster
        resources, floordiv'd across all nonzero per-actor resource fields.
        ``granted`` debits upscales already issued earlier in this tick (pool
        sizes do not reflect them until scale() applies).

        Terminating (draining) actors are counted as committed -- hence
        ``current_size`` rather than ``serving_size``: they hold their resources
        until actually dead, and ``scale_where``'s free map counts them too, so
        the two phases agree. The current op's reclaimable draining actors are
        added back on top by ``GlobalCapacityLedger.max_placeable`` (reviving
        them needs no new capacity) -- see that ledger.
        """
        assert self._topology is not None
        limits = self._cluster_capacity()
        committed_cpu = granted.cpu
        committed_gpu = granted.gpu
        committed_mem = granted.memory
        for other in self._topology:
            if not isinstance(other, ExperimentalAPMO):
                continue
            for p in other.get_autoscaling_actor_pools():
                per = p.per_actor_resource_usage()
                n = p.current_size()
                committed_cpu += per.cpu * n
                committed_gpu += per.gpu * n
                committed_mem += per.memory * n
        per = pool.per_actor_resource_usage()
        max_place = float("inf")
        for free, need in (
            (limits.cpu - committed_cpu, per.cpu),
            (limits.gpu - committed_gpu, per.gpu),
            (limits.memory - committed_mem, per.memory),
        ):
            if need and need > 0:
                max_place = min(max_place, max(free, 0.0) // need)
        return 0 if max_place == float("inf") else int(max_place)

    @staticmethod
    def _pool_resources(pool, num_actors: int) -> ExecutionResources:
        """Resources consumed by ``num_actors`` actors of this pool."""
        per = pool.per_actor_resource_usage()
        return ExecutionResources(
            cpu=per.cpu * num_actors,
            gpu=per.gpu * num_actors,
            memory=per.memory * num_actors,
        )

    def _down_delta(self, pool) -> int:
        return min(
            max(1, int(self._downscale_factor * pool.serving_size())),
            pool.serving_size() - pool.min_size(),
        )

    # --- scale_how_many: critical op / warmup latch --------------------------

    def _resolve_critical_op(self, topology: Topology) -> "PhysicalOperator":
        """Last op in the DAG, or the op before it while that op is a
        sink/write.

        Linear-pipeline scoped (Phase 1): ``terminal_operator_from_topology``
        raises ValueError on zero or multiple terminals.
        """
        op = terminal_operator_from_topology(topology)
        while self._is_sink_or_write(op) and len(op.input_dependencies) == 1:
            op = op.input_dependencies[0]
        return op

    @staticmethod
    def _is_sink_or_write(op: "PhysicalOperator") -> bool:
        if isinstance(op, OutputSplitter):
            return True
        if any(isinstance(lop, Write) for lop in op._logical_operators):
            return True
        return op.name == "Write"

    # --- scale_how_many: phase-1 guards --------------------------------------

    @staticmethod
    def _assert_phase1_resource_shape(topology: Topology) -> None:
        """Fail fast on pool resource shapes the sizing math does not cover.

        The warmup ceiling split sizes each op against a single scarce resource
        type -- GPU if the actor requests any GPU, else CPU (``_op_resource_type``).
        Beyond that, per-actor cpu/gpu/memory are all honored as placement
        constraints: ``max_placeable_actors``/``_fits`` and ``_max_placeable_actors``
        floordiv over all three. So mixed cpu+gpu and a per-actor memory
        reservation are supported -- the only unschedulable shape is an actor
        requesting neither CPU nor GPU.

        Self-placed ops (a static placement group, or a ``ray_remote_args_fn``
        that assigns one) are exempt: Ray core places them into their PG
        """
        for op in topology:
            if not isinstance(op, ExperimentalAPMO):
                continue
            if (
                op.placement_constraint().has_placement_group
                or op.uses_ray_remote_args_fn
            ):
                continue
            for pool in op.get_autoscaling_actor_pools():
                per_actor = pool.per_actor_resource_usage()
                if not (per_actor.cpu > 0 or per_actor.gpu > 0):
                    raise ValueError(
                        f"OperatorSizer requires each actor to request CPU or "
                        f"GPU; {op.name} requests neither "
                        f"(cpu={per_actor.cpu}, gpu={per_actor.gpu})."
                    )
