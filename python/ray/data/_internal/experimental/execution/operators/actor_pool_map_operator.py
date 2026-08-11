from __future__ import annotations

import logging
import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from typing_extensions import override

import ray
from ray.actor import ActorHandle
from ray.core.generated import gcs_pb2
from ray.data._internal.actor_autoscaler import AutoscalingActorPool
from ray.data._internal.actor_autoscaler.autoscaling_actor_pool import (
    ActorPoolScalingRequest,
    AutoscalingActorConfig,
)
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.block_ref_counter import BlockRefCounter
from ray.data._internal.execution.bundle_queue import (
    BaseBundleQueue,
    ReorderingBundleQueue,
)
from ray.data._internal.execution.bundle_queue.experimental import (
    DrainingActorPriorityBundleQueue,
)
from ray.data._internal.execution.execution_flags import (
    ACTOR_ONLY_DEBUG,
    BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO,
    CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD,
    ENABLE_DEFAULT_MEMORY_LIMITS,
    ENABLE_OPERATOR_SIZER,
    OBJECT_STORE_OUTPUT_FRACTION,
    PER_ACTOR_INPUT_BACKPRESSURE_LIMIT,
    PER_ACTOR_INPUT_BYTES_BACKPRESSURE_LIMIT,
    PER_ACTOR_OBJECT_STORE_MAX_BYTES,
    PER_ACTOR_OBJECT_STORE_MIN_BYTES,
    PER_ACTOR_OUTPUT_BACKPRESSURE_LIMIT,
    PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT,
    RAY_CORE_FAULT_TOLERANCE,
    READ_ACTOR_CONCURRENCY,
    REFRESH_ACTOR_STATE,
    SIZER_CONSTRAINT_AWARE_PLACEMENT,
    SIZER_DRAIN_STALL_WARN_S,
    SIZER_PENDING_ACTOR_EXPIRY_S,
    SIZER_EDGE_WEIGHTED_COLOCATION,
    SIZER_RAY_CORE_SPREAD,
    _usable_memory_per_cpu,
    actor_only_backend_enabled,
    core_actor_backpressure_enabled,
)
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    ExecutionResources,
    NodeIdStr,
    RefBundle,
    TaskContext,
)
from ray.data._internal.execution.interfaces.execution_options import safe_or
from ray.data._internal.execution.interfaces.physical_operator import TaskPullRequest
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator as _OSSActorPoolMapOperator,
    _MapWorker,
)
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.resource_bank import (
    LogicalActorId,
    ResourceBankBase,
    is_cross_node_transfer,
)
from ray.data._internal.execution.util import memory_string, merge_label_selector
from ray.data._internal.experimental.execution.operators.placement_constraints import (
    PlacementConstraint,
)
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    OpPlacementView,
)
from ray.data._internal.util import MiB
from ray.data._internal.utils.heapdict import heapdict
from ray.data.context import (
    DataContext,
)
from ray.types import ObjectRef
from ray.util.debug import log_once
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.data._internal.experimental.execution.actor_only_metrics import (
        ActorOnlyMetrics,
    )

_ACTOR_STATE_DEAD = gcs_pb2.ActorTableData.ActorState.DEAD
_ACTOR_STATE_ALIVE = gcs_pb2.ActorTableData.ActorState.ALIVE
_ACTOR_STATE_RESTARTING = gcs_pb2.ActorTableData.ActorState.RESTARTING


class ActorStatus(str, Enum):
    """Lifecycle state of an experimental actor-pool actor."""

    # Breakdown of alive states
    PENDING = "pending"
    IDLE = "idle"
    ACTIVE = "active"

    # Actor is restarting from some unexpected error
    RESTARTING = "restarting"

    # The actor is marked for graceful termination.
    TERMINATING = "terminating"


@dataclass(frozen=True)
class ExperimentalActorPoolScalingRequest(ActorPoolScalingRequest):
    """Placement-resolved scaling request for ``_ExperimentalActorPool``. Supports scaling actors on specific nodes and killing specific actors on scale down."""

    # Upscale: one node id PER NEW ACTOR, repeat node id if scaling up multiple
    # actors on the same node; len == delta - num_to_reclaim.
    target_nodes_to_scale_actors_on: Tuple[NodeIdStr, ...] = ()
    # Downscale: exact victim actors; len == -delta.
    actor_ids_to_downscale: Tuple[LogicalActorId, ...] = ()
    # Downscale: pick ``-delta`` victims ON THIS NODE at apply time from live
    # actor state (idle first, then fewest unconsumed outputs / least
    # in-flight), so a snapshot-time pick can't race task dispatch.
    victim_node: Optional[NodeIdStr] = None
    # revive this many still-DRAINING actors (from a
    # prior downscale that hasn't finished) back into the running pool instead
    # of creating fresh ones. They already hold their node's resources, so this
    # is free capacity-wise and skips actor startup. ``delta`` is the TOTAL
    # upscale; ``len(target_nodes_to_scale_actors_on) == delta - num_to_reclaim``
    # (only the genuinely-new actors carry target nodes).
    num_to_reclaim: int = 0
    # Downscale only: when False, the victims are being EVICTED because our
    # allocation shrank -- they drain gracefully but must never be revived by a
    # later upscale (their resource is no longer ours). Default True (ordinary
    # downscale victims stay revivable).
    reclaimable: bool = True

    @classmethod
    def from_base(
        cls, req: ActorPoolScalingRequest, **overrides
    ) -> "ExperimentalActorPoolScalingRequest":
        if isinstance(req, cls):
            return replace(req, **overrides) if overrides else req
        kwargs = dict(delta=req.delta, force=req.force, reason=req.reason)
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class _PendingActorInfo:
    """Placement bookkeeping for a not-yet-ready actor."""

    logical_id: LogicalActorId
    # Node the actor was pinned to via NodeAffinitySchedulingStrategy if passed, else none (i.e. Ray-core-scheduled).
    target_node_id: Optional[NodeIdStr]
    # When the actor was added as pending; drives the expiry watchdog.
    created_at: float = field(default_factory=time.time)


@dataclass
class _TerminatingInfo:
    """Bookkeeping for one draining actor."""

    # When the actor was marked terminating.
    since: float
    # Whether this actor may be revived by reclaim_draining_actors. False for
    # actors evicted because our allocation shrank -- their node is no longer
    # ours, so a later upscale must never revive them (they drain and die).
    reclaimable: bool = True


@dataclass
class _ExperimentalActorState:
    """Actor state. Not to be confused with Ray Core actor state that tracks
    DEAD, RESTARTING, or ALIVE statuses, but rather, tracks additional info
    in order to inform Ray Data scheduling decisions."""

    # Pool-assigned logical id (carried in actor labels). Stable for the
    # actor's lifetime; lets us go from state -> logical id without a
    # reverse lookup.
    logical_id: LogicalActorId

    # Number of tasks in flight per actor
    num_tasks_in_flight: int

    # Total bytes of inputs currently in flight on this actor.
    num_input_bytes_in_flight: int

    # Node id of each ready actor
    actor_location: str

    # Per-actor output backpressure limits.
    max_num_outputs: float
    max_num_output_bytes: float

    # Per-actor input backpressure limits
    max_input_bytes: float
    max_tasks_in_flight: int

    latest_task_submission_ts: float

    status: ActorStatus

    # Output blocks this actor's tasks have produced into the operator's output
    # queue that have NOT yet been consumed downstream. Incremented when an
    # output is queued, decremented when it is dequeued. A draining actor is only
    # killed once this reaches 0 (and ``num_tasks_in_flight == 0``), so we never
    # discard output blocks the actor produced.
    num_unconsumed_outputs: int = 0

    default_max_num_output_bytes: float = float("inf")

    # Node whose active-actor count this actor currently holds (recorded at
    # mark-active time). actor_location itself changes when Ray core restarts
    # the actor on another node, so the paired decrement must not re-read it.
    active_count_node: Optional[str] = None


# Scheduling rank used by both the per-node (locality) heap and the global
# fallback heap. Lower is preferred. Sort order: ascending
# ``num_tasks_in_flight``, ``latest_task_submission_ts`` (less-loaded actor first,
# where tiebreaks are handled by the earliest latest_task_submission_ts).
@dataclass(order=True)
class _ExperimentalActorRank:
    num_tasks_in_flight: int
    latest_task_submission_ts: float


class _ExperimentalActorPool(AutoscalingActorPool):
    """A pool of actors for map task execution.

    This class is in charge of tracking the number of in-flight tasks per actor,
    providing the least heavily loaded actor to the operator, and killing idle
    actors when the operator is done submitting work to the pool.
    """

    _ACTOR_POOL_SCALE_DOWN_DEBOUNCE_PERIOD_S = 3

    def __init__(
        self,
        create_actor_fn: "Callable[..., Tuple[ActorHandle, ObjectRef[Any], ExecutionResources]]",
        config: AutoscalingActorConfig,
        map_worker_cls_name: str = "MapWorker",
        debounce_period_s: int = _ACTOR_POOL_SCALE_DOWN_DEBOUNCE_PERIOD_S,
    ):
        """Initialize the actor pool.

        Args:
            create_actor_fn: Callable that takes key-value labels, a logical actor
                ID, and an optional scheduling strategy, then creates an actor.
                Returns the actor handle, a reference to the actor's node ID, and
                the actor's resource usage.
            config: Configuration for the autoscaling actor pool, including
                min/max/initial pool sizes, concurrency, and resource usage.
            map_worker_cls_name: Name of the map worker class for logging
                purposes.
            debounce_period_s: Debounce period for scaling down after scaling
                up.
        """
        super().__init__(config=config)

        self._resource_bank: Optional[ResourceBankBase] = None
        self._create_actor_fn = create_actor_fn
        self._map_worker_cls_name = map_worker_cls_name
        self._debounce_period_s = debounce_period_s
        # Timestamp of the last scale up action
        self._last_upscaled_at: Optional[float] = None
        # Timestamp of the last pool size change (scale up or scale down)
        self._last_scaled_at: Optional[float] = None
        self._last_downscaling_debounce_warning_ts: Optional[float] = None
        # Actors that have started running, including alive and restarting actors.
        self._running_actors: Dict[ActorHandle, _ExperimentalActorState] = {}
        # Actors that are not yet ready (still pending creation).
        self._pending_actors: Dict[ObjectRef, ActorHandle] = {}
        # Placement bookkeeping for pending actors, parallel to
        # ``_pending_actors``.
        self._pending_actor_info: Dict[ObjectRef, _PendingActorInfo] = {}
        # Graceful-termination state machines, keyed by actor. Entries also
        # remain in ``_running_actors`` (with ``status == ActorStatus.
        # TERMINATING``) until the actor is observed dead.
        self._terminating_actors: Dict[ActorHandle, _TerminatingInfo] = {}
        self._num_terminating_actors: int = 0
        # Bumped on every change to the set of terminating actors (add on
        # graceful-termination start, remove on reclaim/release). Lets callers
        # cheaply detect "did the draining set change?" and skip work otherwise
        # -- e.g. the output-queue priority sync only reconciles when this moves.
        self._terminating_generation: int = 0
        # Debounce clock for the stalled-drain WARNING (see
        # ``process_draining_actors``): at most one line per
        # ``SIZER_DRAIN_STALL_WARN_S`` for the whole pool.
        self._last_drain_stall_warn_ts: float = 0.0
        # Map from actor handle to its logical ID.
        self._actor_to_logical_id: Dict[ActorHandle, LogicalActorId] = {}
        # Per-actor resource usage, needed because ray_remote_args_fn can
        # produce different resources for each actor.
        self._actor_resource_usage: Dict[ActorHandle, ExecutionResources] = {}
        # Cached aggregate resource counters.
        self._total_usage = ExecutionResources.zero()
        self._pending_or_restarting_usage = ExecutionResources.zero()
        # Resources this pool occupies (or has promised) per node, maintained
        # incrementally as actors are created, become ready, and die (see
        # ``committed_usage_by_node``). Avoids re-iterating every running +
        # pending actor on each sizer tick. Running actors are counted at their
        # actual location; pending actors at their ``target_node_id`` (only when
        # one was set -- untargeted/core-scheduled pendings contribute nothing
        # until they become running).
        self._committed_usage_by_node: Dict[NodeIdStr, ExecutionResources] = {}
        # Cached values for actor / task counts
        self._num_restarting_actors: int = 0

        self._num_actors_with_tasks: int = 0
        self._total_num_tasks_in_flight: int = 0
        # Running actors satisfying the idle predicate: status != TERMINATING
        # and num_tasks_in_flight == 0 (RESTARTING with no in-flight counts as
        # idle). Maintained at every predicate-flipping transition so
        # ``num_idle_actors`` (called per task completion for metrics) is O(1)
        # instead of a scan over ``_running_actors``.
        self._num_idle_actors: int = 0

        # Global fallback heap of all alive actors with both flight capacity
        # (``num_tasks_in_flight < max_tasks_in_flight_per_actor``) AND under
        # the input bytes capacity (``num_input_bytes_in_flight`` roughly
        # less than ``max_input_bytes``)
        # NOTE: The input bytes capacity is a soft cap, ie, you can go above the
        # ``max_input_bytes`` but once you are over, you may not schedule more tasks
        # to that actor.
        self._available_actors_to_tasks_fallback_heap = heapdict[
            ActorHandle, _ExperimentalActorRank
        ]()

        # ALIVE node -> per-node heap of ALIVE actors with flight capacity
        # and input bytes capacity.
        self._alive_node_to_available_actor_heap: DefaultDict[
            NodeIdStr, heapdict[ActorHandle, _ExperimentalActorRank]
        ] = defaultdict(heapdict[ActorHandle, _ExperimentalActorRank])

        # NOTE: Only contains ALIVE actors.
        self._logical_id_to_actor: Dict[LogicalActorId, ActorHandle] = {}

        # NOTE: Only contains ALIVE actors.
        self._node_to_actor_states: Dict[
            NodeIdStr, Dict[LogicalActorId, _ExperimentalActorState]
        ] = {}
        # Nodes hosting at least one actor with ``num_tasks_in_flight > 0``.
        self._node_to_active_actor_count: Dict[NodeIdStr, int] = {}

    def set_resource_bank(self, resource_bank: ResourceBankBase) -> None:
        self._resource_bank = resource_bank

    @property
    def resource_bank(self) -> ResourceBankBase:
        assert self._resource_bank is not None
        return self._resource_bank

    @property
    def map_worker_cls_name(self) -> str:
        return self._map_worker_cls_name

    @property
    def last_scaled_at(self) -> Optional[float]:
        return self._last_scaled_at

    def _has_flight_capacity(self, state: _ExperimentalActorState) -> bool:
        """We want to limit the # of tasks in flight per-actor so that
        we can spread the workload across many actors.

        Also the graceful-termination gate: a terminating actor must never
        receive new submissions.
        """
        if state.status == ActorStatus.TERMINATING:
            return False

        return state.num_tasks_in_flight < state.max_tasks_in_flight

    def _has_input_bytes_capacity(self, state: _ExperimentalActorState) -> bool:
        """We want to limit the # of input bytes per actor to prevent spilling.
        NOTE: The actor can go above this capacity since bytes are approximate, but
        once the actor goes above, it is unavailable for scheduling. This means spilling
        can happen in scenarios where the input bytes is very large since we'll allow
        at least 1 task in flight per actor.
        """
        return state.num_input_bytes_in_flight < state.max_input_bytes

    def _remove_from_heaps(self, actor: ActorHandle, node_id: NodeIdStr) -> None:
        """Drop the actor from both scheduling heaps if present."""
        if actor in self._available_actors_to_tasks_fallback_heap:
            del self._available_actors_to_tasks_fallback_heap[actor]
        node_heap = self._alive_node_to_available_actor_heap.get(node_id)
        if node_heap is not None and actor in node_heap:
            del node_heap[actor]

    def _sync_actor_to_heaps(
        self, actor: ActorHandle, state: _ExperimentalActorState
    ) -> None:
        """Reconcile ``actor``'s membership in both scheduling heaps.

        Per-node (locality) heap: gated by the HARD primary cap.
        Global fallback (non-local) heap: gated by the soft caps too.

        Restarting actors stay out of both heaps regardless of capacity;
        ``refresh_actor_state`` re-syncs them once they are ALIVE again (see
        ``_update_rank``).
        """
        if state.status == ActorStatus.RESTARTING:
            self._remove_from_heaps(actor, state.actor_location)
            return
        if self._has_input_bytes_capacity(state) and self._has_flight_capacity(state):
            rank = _ExperimentalActorRank(
                num_tasks_in_flight=state.num_tasks_in_flight,
                latest_task_submission_ts=state.latest_task_submission_ts,
            )
            self._alive_node_to_available_actor_heap[state.actor_location][actor] = rank
            self._available_actors_to_tasks_fallback_heap[actor] = rank
        else:
            node_heap = self._alive_node_to_available_actor_heap.get(
                state.actor_location
            )
            if node_heap is not None and actor in node_heap:
                del node_heap[actor]
            if actor in self._available_actors_to_tasks_fallback_heap:
                del self._available_actors_to_tasks_fallback_heap[actor]

    def _assert_heap_invariant(self, label: str) -> None:
        """Validate that every actor in the scheduling heaps respects its cap
        and is in sync with its tracked ``_ExperimentalActorState``.

        Intended as debug instrumentation: called after every mutation of the
        scheduling heaps so that the first offending code path raises here
        rather than later in ``select_actors``.
        """

        if not ACTOR_ONLY_DEBUG:
            return

        def debug_assert(
            heap_name: str, rank: _ExperimentalActorRank, state: _ExperimentalActorState
        ) -> None:
            assert rank.num_tasks_in_flight == state.num_tasks_in_flight, (
                f"[{label}] flight rank out of sync with state: "
                f"actor={actor}, rank.tasks={rank.num_tasks_in_flight}, "
                f"state=(tasks={state.num_tasks_in_flight}, "
                f"input_bytes={state.num_input_bytes_in_flight})"
            )
            assert self._has_flight_capacity(state), (
                f"[{label}] actor over max_tasks_in_flight in {heap_name} heap: "
                f"actor={actor}, "
                f"tasks={state.num_tasks_in_flight}/{state.max_tasks_in_flight}, "
                f"input_bytes={state.num_input_bytes_in_flight}/"
                f"{self._config.max_input_bytes_per_actor}"
            )
            assert self._has_input_bytes_capacity(state), (
                f"[{label}] actor over a input bytes cap in {heap_name} heap: "
                f"actor={actor}, "
                f"tasks={state.num_tasks_in_flight}/{state.max_tasks_in_flight}, "
                f"input_bytes={state.num_input_bytes_in_flight}/"
                f"{self._config.max_input_bytes_per_actor}"
            )
            assert state.status != ActorStatus.RESTARTING, (
                f"[{label}] restarting actor in {heap_name} heap: actor={actor}"
            )

        for (
            actor,
            fallback_rank,
        ) in self._available_actors_to_tasks_fallback_heap.items():
            state = self._running_actors.get(actor)
            assert state is not None, (
                f"fallback heap has actor missing from _running_actors: actor={actor}"
            )
            debug_assert(heap_name="fallback", rank=fallback_rank, state=state)

        for node_id, node_heap in self._alive_node_to_available_actor_heap.items():
            for actor, rank in node_heap.items():
                state = self._running_actors.get(actor)
                assert state is not None, (
                    f"[{label}] node heap[{node_id}] has actor missing from "
                    f"_running_actors: actor={actor}"
                )
                assert state.actor_location == node_id, (
                    f"[{label}] actor in wrong node heap: actor={actor}, "
                    f"actor_location={state.actor_location}, node_heap_id={node_id}"
                )
                debug_assert(heap_name="node", rank=rank, state=state)

        idle_scan = sum(
            1
            for state in self._running_actors.values()
            if state.status != ActorStatus.TERMINATING
            and state.num_tasks_in_flight == 0
        )
        assert self._num_idle_actors == idle_scan, (
            f"[{label}] idle-actor counter out of sync: "
            f"counter={self._num_idle_actors}, scan={idle_scan}"
        )

        active_by_node: DefaultDict[NodeIdStr, int] = defaultdict(int)
        for state in self._running_actors.values():
            if state.num_tasks_in_flight > 0:
                active_by_node[state.actor_location] += 1
        assert dict(active_by_node) == self._node_to_active_actor_count, (
            f"[{label}] active-node counter out of sync: "
            f"counter={self._node_to_active_actor_count}, scan={dict(active_by_node)}"
        )

    def _mark_node_actor_active(self, state) -> None:
        """Record that one more actor on its node now has in-flight tasks.

        The node is captured on the state: ``actor_location`` changes when Ray
        core restarts the actor on another node, and the paired decrement must
        hit the node that was incremented, not wherever the actor lives now.
        """
        node_id = state.actor_location
        state.active_count_node = node_id
        self._node_to_active_actor_count[node_id] = (
            self._node_to_active_actor_count.get(node_id, 0) + 1
        )

    def _mark_node_actor_inactive(self, state) -> None:
        """Record that one actor no longer has in-flight tasks, decrementing
        the node captured at mark-active time."""
        node_id = state.active_count_node
        state.active_count_node = None
        if node_id is None:
            return
        count = self._node_to_active_actor_count[node_id] - 1
        if count == 0:
            del self._node_to_active_actor_count[node_id]
        else:
            self._node_to_active_actor_count[node_id] = count

    def nodes(self) -> Set[NodeIdStr]:
        """Nodes hosting any alive actor in this pool (idle or active)."""
        return set(self._node_to_actor_states.keys())

    def active_nodes(self) -> Set[NodeIdStr]:
        """Nodes hosting at least one actor with in-flight tasks."""
        return set(self._node_to_active_actor_count.keys())

    # === Overriding methods of AutoscalingActorPool ===

    @override
    def num_running_actors(self) -> int:
        # NOTE: Terminating actors are excluded — they no longer accept work
        # and are counted separately via ``num_terminating_actors`` so sizing
        # decisions see them as "going away but still holding resources".
        return len(self._running_actors) - self._num_terminating_actors

    @override
    def num_terminating_actors(self) -> int:
        """Actors draining or awaiting ``__ray_terminate__`` completion."""
        return self._num_terminating_actors

    @override
    def num_restarting_actors(self) -> int:
        """Restarting actors are all the running actors not in ALIVE state."""
        return self._num_restarting_actors

    @override
    def num_active_actors(self) -> int:
        """Active actors are all the running actors with inflight tasks, this included draining actors even though they will not be used to schedule new tasks"""
        return self._num_actors_with_tasks

    @override
    def num_idle_actors(self) -> int:
        # Schedulable-idle actors: non-terminating with no in-flight tasks. Overridden to exclude terminating actors from the count, since they are not schedulable.
        return self._num_idle_actors

    @override
    def num_pending_actors(self) -> int:
        return len(self._pending_actors)

    @override
    def num_tasks_in_flight(self) -> int:
        return self._total_num_tasks_in_flight

    @override
    def scale(self, req: ActorPoolScalingRequest) -> Optional[int]:
        # Normalize legacy requests (DefaultActorAutoscaler / start() with the
        # operator sizer disabled). Empty targets mean untargeted behavior.
        req = ExperimentalActorPoolScalingRequest.from_base(req)

        # Verify request could be applied
        if not self._can_apply_request(req):
            return 0

        map_worker_cls_name = self.map_worker_cls_name

        if req.delta > 0:
            # First revive still-DRAINING actors (free: they already hold their
            # node's resources and need no startup). The remaining genuinely-new
            # actors are created on the target nodes. ``delta`` is the total;
            # only the new actors carry target nodes.
            reclaimed = self.reclaim_draining_actors(req.num_to_reclaim)
            target_nodes = req.target_nodes_to_scale_actors_on
            # One node id per NEW actor; ids may repeat to place several actors
            # on the same node (e.g. ("n1", "n1", "n1") -> 3 actors on n1).
            # Empty means untargeted (Ray-core-scheduled) creation.
            num_new = len(target_nodes) if target_nodes else req.delta - reclaimed
            assert (
                not target_nodes or len(target_nodes) == req.delta - req.num_to_reclaim
            ), (
                f"Expected len(target_nodes) == delta - num_to_reclaim (one node "
                f"id per genuinely-new actor; repeats allowed): delta={req.delta}, "
                f"num_to_reclaim={req.num_to_reclaim}, target_nodes={target_nodes}"
            )

            logger.debug(
                f"Scaling up {map_worker_cls_name} actor pool by {req.delta} "
                f"(reclaimed={reclaimed}, new={num_new}, reason={req.reason}, "
                f"{self.get_actor_info()})"
            )

            for i in range(num_new):
                target_node_id = target_nodes[i] if target_nodes else None
                # Untargeted creation (no target node): default to Ray-core
                # scheduling, or SPREAD when the sizer's Ray-core-SPREAD baseline
                # is enabled. NodeAffinity is only used when the sizer resolved a
                # specific target node.
                scheduling_strategy = (
                    NodeAffinitySchedulingStrategy(target_node_id, soft=True)
                    if target_node_id is not None
                    else ("SPREAD" if SIZER_RAY_CORE_SPREAD else None)
                )
                actor, ready_ref, resource_usage = self._create_actor(
                    scheduling_strategy=scheduling_strategy
                )
                self._add_pending_actor(
                    actor, ready_ref, resource_usage, target_node_id=target_node_id
                )

            # Capture last scale up timestamp
            now = time.time()
            self._last_upscaled_at = now

            return req.delta

        elif req.delta < 0:
            num_released = 0
            target_num_actors = abs(req.delta)

            if req.actor_ids_to_downscale:
                assert len(req.actor_ids_to_downscale) == target_num_actors, (
                    f"Expected one victim per downscaled actor "
                    f"(delta={req.delta}, victims={req.actor_ids_to_downscale})"
                )
                for logical_id in req.actor_ids_to_downscale:
                    if self._downscale_actor_by_id(
                        logical_id, reclaimable=req.reclaimable
                    ):
                        num_released += 1
            elif req.victim_node is not None:
                # Live-state victim selection on the target node, cheapest
                # drain first. May release fewer than asked if the node's
                # pool shrank since planning; the transfer watchdog is the
                # backstop.
                states = self._node_to_actor_states.get(req.victim_node, {})
                ranked = sorted(
                    (
                        (
                            0 if state.num_tasks_in_flight == 0 else 1,
                            state.num_unconsumed_outputs,
                            state.num_tasks_in_flight,
                            logical_id,
                        )
                        for logical_id, state in states.items()
                        if state.status != ActorStatus.TERMINATING
                    ),
                )
                for _, _, _, logical_id in ranked[:target_num_actors]:
                    if self._downscale_actor_by_id(
                        logical_id, reclaimable=req.reclaimable
                    ):
                        num_released += 1
            else:
                for _ in range(target_num_actors):
                    if self._remove_inactive_actor():
                        num_released += 1

            if num_released > 0:
                logger.debug(
                    f"Scaled down {map_worker_cls_name} actor pool by {num_released} "
                    f"(reason={req.reason}; {self.get_actor_info()})"
                )
                self._last_scaled_at = time.time()

            return -num_released

        return None

    @override
    def refresh_actor_state(self):
        if REFRESH_ACTOR_STATE:
            self._alive_node_to_available_actor_heap.clear()
            self._available_actors_to_tasks_fallback_heap.clear()
            # NOTE: Copy the keys — observing a terminating actor as DEAD
            # releases it from ``_running_actors`` mid-iteration.
            for actor in list(self._running_actors):
                self._update_running_actor_state(actor)
            self._assert_heap_invariant("refresh_actor_state")

    @override
    def on_task_submitted(self, actor: ActorHandle, input_bundle: RefBundle):
        state = self._running_actors[actor]
        state.num_tasks_in_flight += 1
        state.num_input_bytes_in_flight += input_bundle.size_bytes()
        state.latest_task_submission_ts = time.perf_counter()
        self._total_num_tasks_in_flight += 1

        # Transfer ownership of primary (local) inputs to this actor by
        # withdrawing them from the ResourceBank now: the producing actor's
        # outstanding output stats drop immediately rather than waiting for
        # the bundle to be fully consumed when downstream task is done
        # The withdraw at task-done is idempotent, so it'll be a no-op
        # for these refs.
        target_node_id = state.actor_location
        for entry in input_bundle.blocks:
            if not is_cross_node_transfer(
                bm=entry.metadata, target_node_id=target_node_id
            ):
                self.resource_bank.on_block_consumed(input_ref=entry.ref)

        if state.num_tasks_in_flight == 1:
            self._num_actors_with_tasks += 1
            self._mark_node_actor_active(state)
            # 0 -> 1 in flight: leaves the idle set (unless already excluded
            # from it as TERMINATING).
            if state.status != ActorStatus.TERMINATING:
                self._num_idle_actors -= 1
            state.status = ActorStatus.ACTIVE

        # Selected by `select_actors`, so must have primary capacity and thus
        # be in the per-node heap (membership in the fallback heap is not
        # guaranteed -- the actor may have been selected via locality despite
        # being over its soft input-bytes cap).
        assert actor in self._alive_node_to_available_actor_heap[state.actor_location]
        self._sync_actor_to_heaps(actor=actor, state=state)
        self._assert_heap_invariant("on_task_submitted")

    @override
    def get_actor_location(self, actor: ActorHandle) -> NodeIdStr:
        return self._running_actors[actor].actor_location

    @override
    def shutdown(self, force: bool = False):
        """Kills all actors, including running/active actors.

        This is called once the operator is shutting down.
        """
        self._release_pending_actors(force=force)
        self._release_running_actors(force=force)

    @override
    def pending_to_running(self, ready_ref: ray.ObjectRef) -> Optional[ActorHandle]:
        """Mark the actor corresponding to the provided ready future as running, making
        the actor pickable.

        Args:
            ready_ref: The ready future for the actor that we wish to mark as running.

        Returns:
            The actor handle of the pending/now ready actor. Otherwise, returns `None`
            if actor has already been killed

        Raises:
            RayError: If the actor initialization failed. The actor is cleaned up
                from internal tracking before re-raising.
        """
        if ready_ref not in self._pending_actors:
            # The actor has been removed from the pool before becoming running.
            return None
        actor = self._pending_actors.pop(ready_ref)
        pending_info = self._pending_actor_info.pop(ready_ref, None)
        # Target node this pending actor was committed to (None if untargeted,
        # in which case it contributed nothing to the committed-usage map).
        pending_target_node = (
            pending_info.target_node_id if pending_info is not None else None
        )
        try:
            actor_location: NodeIdStr = ray.get(ready_ref)
            assert actor_location is not None
            logical_id = self.get_actor_logical_id(actor)
            # Derive this actor's per-actor output-bytes budget from its landing
            # node's capacity. Done inside this try so a failure here (e.g. the
            # actor landed off-target on a node absent from the coordinator
            # allocation, so node_capacity raises) is cleaned up exactly like an
            # init failure below, rather than leaking pending/committed usage.
            default_max_num_output_bytes = self._derive_default_max_num_output_bytes(
                actor_location
            )
        except Exception:
            # Actor init failed, or its per-actor output budget couldn't be
            # derived. Clean up the actor from _actor_to_logical_id (for all
            # exceptions, not just RayError) to prevent memory leaks where dead
            # actor handles remain, and roll back its pending/committed usage.
            usage = self._actor_resource_usage.pop(actor)
            self._total_usage = self._total_usage.subtract(usage)
            self._pending_or_restarting_usage = (
                self._pending_or_restarting_usage.subtract(usage)
            )
            if pending_target_node is not None:
                self._remove_committed_usage(pending_target_node, usage)
            self._actor_to_logical_id.pop(actor, None)
            raise
        state = _ExperimentalActorState(
            logical_id=logical_id,
            num_tasks_in_flight=0,
            num_input_bytes_in_flight=0,
            actor_location=actor_location,
            max_num_outputs=safe_or(
                self._config.max_num_outputs_per_actor, float("inf")
            ),
            max_num_output_bytes=default_max_num_output_bytes,
            default_max_num_output_bytes=default_max_num_output_bytes,
            max_input_bytes=safe_or(
                self._config.max_input_bytes_per_actor, float("inf")
            ),
            max_tasks_in_flight=self.default_max_tasks_in_flight_per_actor(),
            # NOTE: We assume any actor that goes from pending to running is IDLE
            status=ActorStatus.IDLE,
            latest_task_submission_ts=0,
        )
        self._running_actors[actor] = state
        # New running actors join the pool IDLE with nothing in flight.
        self._num_idle_actors += 1
        # Mirror the new actor into the logical-id and node indexes. These
        # two maps are the source of truth for "which actors / nodes does
        # the pool currently keep alive" -- maintained explicitly here
        # (not via defaultdict-on-access) so cluster autoscaling and the
        # output-limit upgrader can trust their keysets.
        self._logical_id_to_actor[logical_id] = actor
        self._node_to_actor_states.setdefault(actor_location, {})[logical_id] = state
        # Actor is no longer pending — subtract from pending usage.
        self._pending_or_restarting_usage = self._pending_or_restarting_usage.subtract(
            self._actor_resource_usage[actor]
        )
        # Re-attribute committed usage from the (optional) pending target node to
        # the actual landing node now that placement is resolved.
        usage = self._actor_resource_usage[actor]
        if pending_target_node is not None:
            self._remove_committed_usage(pending_target_node, usage)
        self._add_committed_usage(actor_location, usage)
        self._sync_actor_to_heaps(actor, state)
        self._assert_heap_invariant("pending_to_running")
        return actor

    @override
    def get_pending_actor_refs(self) -> List[ray.ObjectRef]:
        return list(self._pending_actors.keys())

    @override
    def select_actors(
        self,
        bundle: Optional[RefBundle] = None,
        actor_locality_enabled: bool = False,
    ) -> Optional[ActorHandle]:
        """Select an actor to process the given bundle.

        When ``bundle`` is ``None``, returns any available actor with spare
        capacity (used by ``can_schedule_task`` to probe schedulability).
        When ``bundle`` is provided, returns the best actor for that bundle
        (considering locality when ``actor_locality_enabled`` is True).

        This method peeks (does not pop) from the heap, so
        ``on_task_submitted()`` must be called for the returned actor before
        the next ``select_actors()`` call.  Otherwise the same
        actor will be selected repeatedly.  The caller in
        ``ActorPoolMapOperator._dispatch_tasks()`` upholds this contract.

        Args:
            bundle: The bundle to schedule. When ``None``, returns any
                available actor with spare capacity.
            actor_locality_enabled: Whether to consider data locality
                when selecting an actor.

        Returns:
            An actor handle if an actor with capacity is available, otherwise
            ``None``.
        """
        self._assert_heap_invariant("select_actors:enter")

        # Locality path first: it draws from the per-node heap (gated only by
        # the hard primary cap), so it must be attempted BEFORE the
        # fallback-heap emptiness check below. An actor that is over the soft
        # input-bytes cap is absent from the fallback heap but still present in
        # its per-node heap.
        if bundle is not None and actor_locality_enabled:
            target_actor = self._find_actor_with_locality(bundle)
            if target_actor is not None:
                return target_actor

        # Fallback (non-local) path: pick the actor with the lowest rank.
        # Membership in this heap enforces both flight capacity:
        # (``max_tasks_in_flight_per_actor``) and the soft
        # input-bytes cap (``max_input_bytes_per_actor``);
        if not self._available_actors_to_tasks_fallback_heap:
            # No actor has non-local (fallback) capacity right now.
            return None
        (
            fallback_actor,
            fallback_rank,
        ) = self._available_actors_to_tasks_fallback_heap.peekitem()
        state = self._running_actors[fallback_actor]
        assert fallback_rank.num_tasks_in_flight < state.max_tasks_in_flight
        return fallback_actor

    @override
    def on_task_completed(self, actor: ActorHandle, input_bundle: RefBundle):
        """Called when a task completes. Returns the provided actor to the pool."""
        if actor not in self._running_actors:
            # the actor was already released while this task was
            # still in flight (e.g. an external death -- OOM / preemption / lost
            # node). Releasing the actor already reversed its in-flight
            # accounting, so there is nothing more to do here.
            return
        state = self._running_actors[actor]
        assert state.num_tasks_in_flight > 0
        state.num_tasks_in_flight -= 1
        state.num_input_bytes_in_flight -= input_bundle.size_bytes()
        assert state.num_input_bytes_in_flight >= 0, (
            f"num_input_bytes_in_flight underflow on actor={actor}: "
            f"{state.num_input_bytes_in_flight}"
        )

        self._total_num_tasks_in_flight -= 1
        if state.num_tasks_in_flight == 0:
            self._num_actors_with_tasks -= 1
            self._mark_node_actor_inactive(state)
            # 1 -> 0 in flight: rejoins the idle set, unless draining -- a
            # TERMINATING actor finishing its last task stays excluded.
            if state.status != ActorStatus.TERMINATING:
                self._num_idle_actors += 1
            # An actor could be in TERMINATING status, so
            # only convert to IDLE if it was ACTIVE
            if state.status == ActorStatus.ACTIVE:
                state.status = ActorStatus.IDLE

        # A task can complete (as a failure) while its actor is RESTARTING --
        # the accounting above must still drain so the ALIVE transition derives
        # the right status from num_tasks_in_flight. The heap sync below keeps
        # restarting actors out of the scheduling heaps.

        # Reconcile both heaps. The just-completed task strictly decreases
        # ``num_tasks_in_flight``, so the actor always regains primary
        # capacity here (re-added to the per-node heap). Soft caps may still
        # be exceeded (e.g., remaining bundles still hold lots of input
        # bytes), in which case ``_sync_actor_to_heaps`` will keep the actor
        # out of the fallback heap until the soft cap clears too.
        self._sync_actor_to_heaps(actor, state)
        self._assert_heap_invariant("on_task_completed")

    def on_output_produced(self, actor: ActorHandle) -> None:
        """Record that a task on this actor queued an output block (not yet
        consumed downstream)."""
        state = self._running_actors.get(actor)
        if state is not None:
            state.num_unconsumed_outputs += 1

    def on_output_consumed(self, actor: ActorHandle) -> None:
        """Record that one of this actor's queued output blocks was consumed
        downstream."""
        state = self._running_actors.get(actor)
        if state is None:
            return
        state.num_unconsumed_outputs -= 1
        assert state.num_unconsumed_outputs >= 0, (
            f"num_unconsumed_outputs underflow on actor "
            f"{state.logical_id}: {state.num_unconsumed_outputs}"
        )

    @override
    def get_actor_logical_id(self, actor: ActorHandle) -> LogicalActorId:
        return self._actor_to_logical_id[actor]

    @override
    def get_logical_ids(self) -> List[LogicalActorId]:
        """Get the logical IDs for pending and running actors in the actor pool.

        We can't use Ray Core actor IDs because we need to identify actors by labels,
        but labels must be set before creation, and actor IDs aren't available until
        after.
        """
        return list(self._actor_to_logical_id.values())

    @override
    def current_logical_usage(self) -> ExecutionResources:
        return self._total_usage

    @override
    def pending_logical_usage(self) -> ExecutionResources:
        return self._pending_or_restarting_usage

    # === End of overriding methods of AutoscalingActorPool ===

    def _can_apply_request(self, req: ActorPoolScalingRequest) -> bool:
        """Returns whether Actor Pool is able to execute scaling request"""

        if req.delta < 0:
            # To prevent bouncing back and forth, we disallow scale down for
            # a "cool-off" period after the most recent scaling up, with an intention
            # to allow application to actually utilize newly provisioned resources
            # before making decisions on subsequent actions.
            #
            # Note that this action is unidirectional and doesn't apply to
            # scaling up, ie if actor pool just scaled down, it'd still be able
            # to scale back up immediately.
            if (
                not req.force
                and self._last_upscaled_at is not None
                and (time.time() <= self._last_upscaled_at + self._debounce_period_s)
            ):
                # NOTE: To avoid spamming logs unnecessarily, debounce log is produced once
                #       per upscaling event
                if self._last_upscaled_at != self._last_downscaling_debounce_warning_ts:
                    logger.debug(
                        f"Ignoring scaling down request (request={req}; reason=debounced from scaling up at {self._last_upscaled_at})"
                    )
                    self._last_downscaling_debounce_warning_ts = self._last_upscaled_at

                return False

        return True

    def _create_actor(
        self,
        scheduling_strategy: NodeAffinitySchedulingStrategy | str | None = None,
    ) -> Tuple[ActorHandle, ObjectRef, ExecutionResources]:
        logical_actor_id = str(uuid.uuid4())
        labels = {self.get_logical_id_label_key(): logical_actor_id}
        actor, ready_ref, resource_usage = self._create_actor_fn(
            labels, logical_actor_id, scheduling_strategy=scheduling_strategy
        )
        self._actor_to_logical_id[actor] = logical_actor_id
        return actor, ready_ref, resource_usage

    def _update_running_actor_state(self, actor: ActorHandle):
        """Update running actor state. This is called for every actor
        in `refresh_actor_state`.

        Args:
            actor: The running actor that needs state update.
        """
        actor_state = actor._get_local_state()

        # 1) Check if actor is restarting
        running_actor_state = self._running_actors[actor]
        died: bool = False
        released: bool = False
        if actor_state in (None, _ACTOR_STATE_DEAD):
            # actor._get_local_state can return None if the state is Unknown.
            died = True
            if running_actor_state.status == ActorStatus.TERMINATING:
                self._release_running_actor(actor)
                released = True
        elif actor_state != _ACTOR_STATE_ALIVE:
            # The actors can be either ALIVE or RESTARTING here because they will
            # be restarted indefinitely until execution finishes.
            assert actor_state == _ACTOR_STATE_RESTARTING, actor_state
            # TERMINATING is sticky: a draining actor whose process restarts
            # must still drain and die. Flipping it to RESTARTING would strand
            # its terminating count/state and re-arm an evicted actor for
            # revival on a node the pool no longer owns.
            if running_actor_state.status not in (
                ActorStatus.RESTARTING,
                ActorStatus.TERMINATING,
            ):
                self._num_restarting_actors += 1
                self._pending_or_restarting_usage = (
                    self._pending_or_restarting_usage.add(
                        self._actor_resource_usage[actor]
                    )
                )
                running_actor_state.status = ActorStatus.RESTARTING
        else:
            if running_actor_state.status == ActorStatus.RESTARTING:
                self._num_restarting_actors -= 1
                self._pending_or_restarting_usage = (
                    self._pending_or_restarting_usage.subtract(
                        self._actor_resource_usage[actor]
                    )
                )
                running_actor_state.status = (
                    ActorStatus.IDLE
                    if running_actor_state.num_tasks_in_flight == 0
                    else ActorStatus.ACTIVE
                )

        if not released:
            self._update_rank(actor=actor, state=running_actor_state, died=died)

    def _update_rank(
        self, actor: ActorHandle, state: _ExperimentalActorState, died: bool
    ):
        """Update the scheduling rank for an actor after a state refresh.

        Per-node (locality) heap membership requires the actor to be alive and
        have flight capacity (see ``_has_flight_capacity``). The fallback
        heap additionally requires the soft input-bytes cap to be under its
        limit (see ``_has_secondary_capacity``). Restarting/dead actors are
        removed from both heaps; the per-node heap is cleared by
        ``refresh_actor_state`` before this method runs, so the explicit
        removal there is defensive.
        """
        if state.status == ActorStatus.RESTARTING or died:
            self._remove_from_heaps(actor, state.actor_location)
            return
        self._sync_actor_to_heaps(actor, state)

    def _add_pending_actor(
        self,
        actor: ActorHandle,
        ready_ref: ObjectRef,
        resource_usage: ExecutionResources,
        target_node_id: Optional[NodeIdStr] = None,
    ):
        """Adds a pending actor to the pool.

        This actor won't be pickable until it is marked as running via a
        pending_to_running() call.

        Args:
            actor: The not-yet-ready actor to add as pending to the pool.
            ready_ref: The ready future for the actor.
            resource_usage: The actual resource usage for this actor.
            target_node_id: Node the actor was pinned to via
                NodeAffinitySchedulingStrategy, or None if core-scheduled.
        """
        self._pending_actors[ready_ref] = actor
        self._pending_actor_info[ready_ref] = _PendingActorInfo(
            logical_id=self._actor_to_logical_id[actor],
            target_node_id=target_node_id,
        )
        self._actor_resource_usage[actor] = resource_usage
        self._total_usage = self._total_usage.add(resource_usage)
        self._pending_or_restarting_usage = self._pending_or_restarting_usage.add(
            resource_usage
        )
        # A pending actor is committed to its target node (if one was set);
        # untargeted/core-scheduled pendings contribute nothing until ready.
        if target_node_id is not None:
            self._add_committed_usage(target_node_id, resource_usage)

    def _remove_inactive_actor(self) -> bool:
        """Kills a single pending or idle actor, if any actors are pending/idle.

        Returns whether an inactive actor was actually released.
        """
        # We prioritize killing pending actors over idle actors to reduce actor starting
        # churn.
        released = self._try_remove_pending_actor()
        if not released:
            # If no pending actor was released, so kill actor.
            released = self._try_remove_idle_actor()
        return released

    def _try_remove_pending_actor(self) -> bool:
        if self._pending_actors:
            # At least one pending actor, so kill first one.
            ready_ref = next(iter(self._pending_actors.keys()))
            self._remove_pending_actor_by_ref(ready_ref)
            return True
        # No pending actors, so indicate to the caller that no actors were killed.
        return False

    def _remove_pending_actor_by_ref(self, ready_ref: ObjectRef) -> None:
        """Drop a specific pending actor and its bookkeeping, then tear it down."""
        actor = self._pending_actors.pop(ready_ref)
        info = self._pending_actor_info.pop(ready_ref, None)
        usage = self._actor_resource_usage.pop(actor)
        self._total_usage = self._total_usage.subtract(usage)
        self._pending_or_restarting_usage = self._pending_or_restarting_usage.subtract(
            usage
        )
        if info is not None and info.target_node_id is not None:
            self._remove_committed_usage(info.target_node_id, usage)
        del self._actor_to_logical_id[actor]
        if not RAY_CORE_FAULT_TOLERANCE:
            ray.kill(actor)

    def _try_remove_idle_actor(self) -> bool:
        for actor, state in self._running_actors.items():
            if (
                state.num_tasks_in_flight == 0
                and state.status != ActorStatus.TERMINATING
            ):
                # At least one idle actor: mark it draining (stop new
                # submissions now; ``process_draining_actors`` kills it once it
                # is fully drained -- no in-flight tasks, no unconsumed outputs).
                self.begin_graceful_termination(actor)
                return True
        # No idle actors, so indicate to the caller that no actors were killed.
        return False

    def release_expired_pending_actors(
        self, expiry_s: float, now: Optional[float] = None
    ) -> int:
        """Release pending actors older than ``expiry_s``, returning the count.

        A pending whose ready-ref never resolves -- an actor wedged in failed
        restarts after losing its constructor args (ray#53727), a stuck
        runtime-env install, or a ghost -- otherwise holds the pool's pending
        count (and with it every sizer arm) until the job dies. The handle is
        force-killed: a never-ready actor cannot be gracefully released, and
        killing it also breaks a failed-restart loop. The freed claim lets the
        next sizing tick re-place with current cluster information.
        """
        now = time.time() if now is None else now
        expired = [
            ready_ref
            for ready_ref, info in self._pending_actor_info.items()
            if now - info.created_at > expiry_s
        ]
        for ready_ref in expired:
            actor = self._pending_actors.pop(ready_ref)
            info = self._pending_actor_info.pop(ready_ref)
            usage = self._actor_resource_usage.pop(actor)
            self._total_usage = self._total_usage.subtract(usage)
            self._pending_or_restarting_usage = (
                self._pending_or_restarting_usage.subtract(usage)
            )
            if info.target_node_id is not None:
                self._remove_committed_usage(info.target_node_id, usage)
            self._actor_to_logical_id.pop(actor, None)
            logger.warning(
                "Releasing pending actor %s: not ready after %.0fs "
                "(target_node=%s). Killing the handle and freeing its claim so "
                "the next sizing tick can re-place it.",
                info.logical_id,
                now - info.created_at,
                info.target_node_id,
            )
            ray.kill(actor)
        return len(expired)

    def _release_pending_actors(self, force: bool):
        # Release pending actors from the set of pending ones
        pending = dict(self._pending_actors)
        # Capture target-node bookkeeping before clearing so we can release each
        # pending actor's committed-usage contribution.
        pending_info = dict(self._pending_actor_info)
        self._pending_actors.clear()
        self._pending_actor_info.clear()
        for ready_ref, actor in pending.items():
            usage = self._actor_resource_usage.pop(actor)
            self._total_usage = self._total_usage.subtract(usage)
            self._pending_or_restarting_usage = (
                self._pending_or_restarting_usage.subtract(usage)
            )
            info = pending_info.get(ready_ref)
            if info is not None and info.target_node_id is not None:
                self._remove_committed_usage(info.target_node_id, usage)
            self._actor_to_logical_id.pop(actor, None)

        if force:
            for actor in pending.values():
                # NOTE: Actors can't be brought back after being ``ray.kill``-ed,
                #       hence we're only doing that if this is a forced release
                ray.kill(actor)

    def _release_running_actors(self, force: bool):
        running = list(self._running_actors.keys())

        for actor in running:
            self._release_running_actor(actor)

        # NOTE: Actors can't be brought back after being ``ray.kill``-ed,
        #       hence we're only doing that if this is a forced release
        if force:
            for actor in running:
                ray.kill(actor)

    def _release_running_actor(self, actor: ActorHandle):
        """Remove the given actor from the pool by dropping all pool references."""
        # NOTE: By default, we remove references to the actor and let ref counting
        # garbage collect the actor, instead of using ray.kill.
        #
        # Otherwise, actor cannot be reconstructed for the purposes of produced
        # object's lineage reconstruction.
        if actor not in self._running_actors:
            return

        # Update cached statistics before removing the actor
        actor_state = self._running_actors[actor]

        # Update total tasks in flight
        self._total_num_tasks_in_flight -= actor_state.num_tasks_in_flight

        # Update active actors count. ``_num_actors_with_tasks`` counts any running
        # actor (incl. terminating) with in-flight tasks, so decrement whenever
        # the released actor still had tasks in flight (e.g. forced shutdown).
        if actor_state.num_tasks_in_flight > 0:
            self._num_actors_with_tasks -= 1
            self._mark_node_actor_inactive(actor_state)
        # An actor released while satisfying the idle predicate leaves the
        # idle set (a TERMINATING or task-holding actor was never in it).
        elif actor_state.status != ActorStatus.TERMINATING:
            self._num_idle_actors -= 1

        # Update restarting actors count
        if actor_state.status == ActorStatus.RESTARTING:
            self._num_restarting_actors -= 1

        # Retire the graceful-termination state machine, if any.
        if actor_state.status == ActorStatus.TERMINATING:
            self._terminating_actors.pop(actor, None)
            self._num_terminating_actors -= 1
            self._terminating_generation += 1

        self._remove_from_heaps(actor, actor_state.actor_location)

        # Drop logical-id and node-keyed views in lockstep with
        # ``_running_actors`` so the two maps never reference a freed
        # actor (or a node that no longer hosts any). Drop the node
        # entry entirely once its last actor is gone -- the keyset of
        # ``_node_to_actor_states`` is meant to reflect "nodes the pool
        # is currently keeping alive".
        node_id = actor_state.actor_location
        logical_id = actor_state.logical_id
        self._logical_id_to_actor.pop(logical_id, None)
        node_states = self._node_to_actor_states.get(node_id)
        if node_states is not None:
            node_states.pop(logical_id, None)
            if not node_states:
                del self._node_to_actor_states[node_id]

        del self._running_actors[actor]
        del self._actor_to_logical_id[actor]

        # Tell ResourceBank to drop this actor from the live-actor
        # indexes. Outputs the actor produced may still be in flight
        # downstream
        self.resource_bank.deregister_actor(actor_id=logical_id)

        usage = self._actor_resource_usage.pop(actor)
        self._total_usage = self._total_usage.subtract(usage)
        # Release this running actor's committed usage at its node.
        self._remove_committed_usage(node_id, usage)
        if actor_state.status == ActorStatus.RESTARTING:
            self._pending_or_restarting_usage = (
                self._pending_or_restarting_usage.subtract(usage)
            )
        self._assert_heap_invariant("_release_running_actor")

    def _teardown_drained_actor(self, actor: ActorHandle) -> None:
        """Tear down a fully-drained actor (no in-flight tasks, no unconsumed
        outputs)."""
        if not RAY_CORE_FAULT_TOLERANCE:
            ray.kill(actor)
        self._release_running_actor(actor)

    def begin_graceful_termination(
        self, actor: ActorHandle, reclaimable: bool = True
    ) -> None:
        """Mark an actor for draining. While draining, no new tasks can be submitted to it.

        The actor is removed from both scheduling heaps immediately (no new
        task submissions; ``_has_flight_capacity`` keeps it out) but stays in
        ``_running_actors`` and the capacity maps as it occupies its node's
        resources until it is actually dead. ``process_draining_actors`` kills
        it once it is fully drained (no in-flight tasks AND no unconsumed outputs).

        ``reclaimable=False`` marks an eviction (our allocation shrank): the
        actor drains and dies like any other, but ``reclaim_draining_actors``
        will never revive it, so a later upscale can't resurrect an actor on a
        node we no longer own."""
        state = self._running_actors[actor]
        assert state.status != ActorStatus.TERMINATING, (
            f"Actor {state.logical_id} already terminating"
        )
        # A RESTARTING victim leaves the restarting accounting here: its
        # counter/usage are otherwise only released on an ALIVE transition or
        # on release-while-still-RESTARTING, and TERMINATING supersedes both.
        if state.status == ActorStatus.RESTARTING:
            self._num_restarting_actors -= 1
            self._pending_or_restarting_usage = (
                self._pending_or_restarting_usage.subtract(
                    self._actor_resource_usage[actor]
                )
            )
        # Entering TERMINATING excludes the actor from the idle set (the
        # prior status is never TERMINATING here, per the assert above).
        if state.num_tasks_in_flight == 0:
            self._num_idle_actors -= 1
        state.status = ActorStatus.TERMINATING
        self._num_terminating_actors += 1
        self._remove_from_heaps(actor, state.actor_location)
        self._terminating_actors[actor] = _TerminatingInfo(
            since=time.time(), reclaimable=reclaimable
        )
        self._terminating_generation += 1
        self._assert_heap_invariant("begin_graceful_termination")

    def num_reclaimable_terminating_actors(self) -> int:
        """Draining actors that can still be revived -- excludes evicted
        (``reclaimable=False``) actors, which must not be resurrected."""
        return sum(1 for info in self._terminating_actors.values() if info.reclaimable)

    def draining_actor_ids(self) -> Set[LogicalActorId]:
        """Logical ids of actors currently draining (marked terminating, not yet
        killed). Used to keep the output queue's prioritized set in sync."""
        return {
            self._running_actors[actor].logical_id for actor in self._terminating_actors
        }

    def terminating_generation(self) -> int:
        """Monotonic counter bumped whenever the draining set changes. A caller
        that saw the same value since its last check knows the draining set is
        unchanged and can skip re-reconciling."""
        return self._terminating_generation

    def reclaimable_node_ids(self, count: int) -> List[NodeIdStr]:
        """Nodes (repeats allowed) of the first ``count`` draining actors, in
        the same order ``reclaim_draining_actors`` revives them. Lets the sizer
        count soon-to-be-revived actors into the placement view BEFORE placing
        new actors, so placement doesn't double-serve the same locality demand.
        """
        nodes: List[NodeIdStr] = []
        for actor, info in self._terminating_actors.items():
            if len(nodes) >= count:
                break
            if not info.reclaimable:
                continue
            nodes.append(self._running_actors[actor].actor_location)
        return nodes

    def reclaim_draining_actors(self, count: int) -> int:
        """Revive up to ``count`` draining actors back into the running pool
        (inverse of ``begin_graceful_termination``). Returns the number revived.
        Saves actor startup + capacity churn vs killing and recreating."""
        if count <= 0:
            return 0
        reclaimed = 0
        for actor, info in list(self._terminating_actors.items()):
            if reclaimed >= count:
                break
            if not info.reclaimable:
                continue
            state = self._running_actors[actor]
            state.status = (
                ActorStatus.IDLE
                if state.num_tasks_in_flight == 0
                else ActorStatus.ACTIVE
            )
            # Leaving TERMINATING re-admits the actor to the idle set if it
            # has nothing in flight.
            if state.num_tasks_in_flight == 0:
                self._num_idle_actors += 1
            self._num_terminating_actors -= 1
            del self._terminating_actors[actor]
            self._terminating_generation += 1
            # Re-admit to the scheduling heaps (it now passes the terminating
            # gate in _has_flight_capacity again).
            self._sync_actor_to_heaps(actor, state)
            reclaimed += 1
        if reclaimed:
            logger.debug(
                "Reclaimed %d draining %s actor(s) back into the running pool",
                reclaimed,
                self._map_worker_cls_name,
            )
        self._assert_heap_invariant("reclaim_draining_actors")
        return reclaimed

    def process_draining_actors(self) -> int:
        """kill every draining actor that is now fully drained.

        Called once per scheduling tick. An actor is "fully drained" once it has
        no in-flight tasks AND no unconsumed output blocks.

        Returns:
            The number of actors killed (fully released) this call.
        """
        if not self._terminating_actors:
            return 0

        now = time.time()
        num_finalized = 0
        stalled = []
        for actor in list(self._terminating_actors):
            state = self._running_actors[actor]
            if state.num_tasks_in_flight == 0 and state.num_unconsumed_outputs == 0:
                self._teardown_drained_actor(actor)
                num_finalized += 1
            elif (
                now - self._terminating_actors[actor].since >= SIZER_DRAIN_STALL_WARN_S
            ):
                stalled.append(
                    (state.num_tasks_in_flight, state.num_unconsumed_outputs)
                )

        if stalled and now - self._last_drain_stall_warn_ts >= SIZER_DRAIN_STALL_WARN_S:
            max_age = max(
                now - info.since for info in self._terminating_actors.values()
            )
            total_in_flight = sum(in_flight for in_flight, _ in stalled)
            total_unconsumed = sum(unconsumed for _, unconsumed in stalled)
            logger.warning(
                f"{len(stalled)} draining {self._map_worker_cls_name} actor(s) have not "
                f"finished draining after {SIZER_DRAIN_STALL_WARN_S}s "
                f"(longest {max_age:.0f}s; {total_in_flight} task(s) still in flight, "
                f"{total_unconsumed} unconsumed output block(s)). They hold their node "
                "capacity until their tasks finish and their outputs are consumed "
                "downstream."
            )
            self._last_drain_stall_warn_ts = now

        return num_finalized

    def _downscale_actor_by_id(
        self, logical_id: LogicalActorId, reclaimable: bool = True
    ) -> bool:
        """Downscale one exact victim chosen by the placement policy.

        ``reclaimable=False`` evicts (allocation shrank): the actor drains but is
        never revived. A pending victim is simply cancelled either way.
        """
        for ready_ref, info in self._pending_actor_info.items():
            if info.logical_id == logical_id:
                self._remove_pending_actor_by_ref(ready_ref)
                return True

        actor = self._logical_id_to_actor.get(logical_id)
        if (
            actor is None
            or self._running_actors[actor].status == ActorStatus.TERMINATING
        ):
            logger.debug(
                f"Skipping downscale of actor {logical_id}: not found or "
                "already terminating."
            )
            return False
        self.begin_graceful_termination(actor, reclaimable=reclaimable)
        return True

    def has_actor(self, actor_id: LogicalActorId) -> bool:
        """Whether the pool still holds a running actor with this logical id.

        False once the actor has been released (downscaled/killed or reaped),
        even if in-flight task refs for it are still resolving.
        """
        return actor_id in self._logical_id_to_actor

    def get_actor_by_logical_id(
        self, logical_id: LogicalActorId
    ) -> Optional[ActorHandle]:
        """The running actor with this logical id, or None if it has been
        released."""
        return self._logical_id_to_actor.get(logical_id)

    def is_actor_terminating(self, actor: ActorHandle) -> bool:
        """Whether this (still-running) actor is draining. False if released."""
        state = self._running_actors.get(actor)
        return state is not None and state.status == ActorStatus.TERMINATING

    def pending_ids_by_target_node(self) -> Dict[NodeIdStr, List[LogicalActorId]]:
        """Logical ids of pending actors, grouped by their target node."""
        result: DefaultDict[NodeIdStr, List[LogicalActorId]] = defaultdict(list)
        for info in self._pending_actor_info.values():
            if info.target_node_id is not None:
                result[info.target_node_id].append(info.logical_id)
        return dict(result)

    def committed_usage_by_node(self) -> Dict[NodeIdStr, ExecutionResources]:
        """Resources this pool occupies (or has promised) per node.

        Running, pending, and restarting actors all count as committed usage since they
        all hold onto their resources until they are actually dead.

        Maintained incrementally (``_add_committed_usage`` /
        ``_remove_committed_usage`` at the create/ready/death sites) rather than
        recomputed, so the sizer can read it once per op per tick cheaply.
        """
        return self._committed_usage_by_node

    def _add_committed_usage(
        self, node_id: NodeIdStr, usage: ExecutionResources
    ) -> None:
        """Credit ``usage`` to ``node_id`` in the committed-usage map."""
        self._committed_usage_by_node[node_id] = self._committed_usage_by_node.get(
            node_id, ExecutionResources.zero()
        ).add(usage)

    def _remove_committed_usage(
        self, node_id: NodeIdStr, usage: ExecutionResources
    ) -> None:
        """Debit ``usage`` from ``node_id``, dropping the entry once it hits zero
        so the map never accumulates stale zero-valued nodes (matching the
        empty-node-omitting semantics callers expect)."""
        remaining = self._committed_usage_by_node.get(
            node_id, ExecutionResources.zero()
        ).subtract(usage)
        if remaining.is_zero():
            self._committed_usage_by_node.pop(node_id, None)
        else:
            self._committed_usage_by_node[node_id] = remaining

    def _find_actor_with_locality(self, bundle: RefBundle) -> Optional[ActorHandle]:
        """Find an alive actor on the most-preferred node for the bundle.

        Preferred nodes are visited in descending order of on-node bytes; for
        each node, the per-node heap's peek gives the locality-best actor
        (see ``_ExperimentalActorRank``: ascending ``num_tasks_in_flight``). The per-node
        heap is gated only by the hard primary cap, so the soft input-bytes
        cap does NOT block local scheduling here -- it only blocks non-local
        fallback scheduling.

        Args:
            bundle: The bundle to find an actor for.

        Returns:
            The locality-preferred actor, or None if no preferred node has any
            alive actor with primary capacity.
        """
        preferred_locs = bundle.get_preferred_object_locations_from_metadata()
        if not preferred_locs:
            return None

        for node_id, _total_bytes in sorted(
            preferred_locs.items(), key=lambda item: (-item[1], item[0])
        ):
            node_heap = self._alive_node_to_available_actor_heap.get(node_id)
            if not node_heap:
                continue
            actor, rank = node_heap.peekitem()
            state = self._running_actors[actor]
            assert rank.num_tasks_in_flight < state.max_tasks_in_flight
            return actor

        return None

    def _derive_default_max_num_output_bytes(self, actor_location: NodeIdStr) -> int:
        """Derive the per-actor output-bytes budget for an actor on ``actor_location``.

        Uses an explicit ``max_num_output_bytes_per_actor`` if configured;
        otherwise derives a CPU-proportional share of the node's usable object
        store and clamps it into the system [MIN, MAX] bounds.
        """
        if self._config.max_num_output_bytes_per_actor is not None:
            return self._config.max_num_output_bytes_per_actor

        res = self._resource_bank.node_capacity(node_id=actor_location)
        per_actor_usage = self._config.per_actor_resource_usage
        usable_object_store_for_outputs_only = (
            res.object_store_memory * OBJECT_STORE_OUTPUT_FRACTION
        )
        # NOTE: Normalize the object-store budget on CPU alone. We deliberately
        # do NOT also normalize on memory/GPU since normalizing across multiple
        # dimensions makes the budget hard to reason about (e.g. a num_cpus=0,
        # num_gpus=1 actor on a single-GPU node would take the entire object
        # store).
        node_cpu = res.cpu
        actor_cpu = per_actor_usage.cpu
        if node_cpu > 0 and actor_cpu > 0:
            obj_store_per_actor = (
                int(usable_object_store_for_outputs_only * (actor_cpu / node_cpu) / MiB)
                * MiB
            )
        else:
            # Default to 0 when node_cpu == 0. This can happen with
            # custom compute configs.
            obj_store_per_actor = 0

        # Clamp into the system [MIN, MAX] so high-cpu / low-cpu actors don't
        # get unbounded or tiny budgets. Clamping up to MIN can cause spilling
        # if the node's object store cannot hold the floor.
        default_max_num_output_bytes = min(
            PER_ACTOR_OBJECT_STORE_MAX_BYTES,
            max(PER_ACTOR_OBJECT_STORE_MIN_BYTES, obj_store_per_actor),
        )
        if default_max_num_output_bytes != obj_store_per_actor and log_once(
            f"actor_only_output_bytes_budget_clamped_{id(self)}"
        ):
            logger.warning(
                f"{self._map_worker_cls_name}: derived per-actor "
                f"output-bytes budget {memory_string(obj_store_per_actor)} "
                f"is outside the system range "
                f"[{memory_string(PER_ACTOR_OBJECT_STORE_MIN_BYTES)}, "
                f"{memory_string(PER_ACTOR_OBJECT_STORE_MAX_BYTES)}]; "
                f"clamping to {memory_string(default_max_num_output_bytes)}. "
                "Clamping up may cause spilling if the node's object store "
                "cannot hold the floor. Override with "
                "RAY_DATA_PER_ACTOR_OBJECT_STORE_MIN_BYTES / "
                "RAY_DATA_PER_ACTOR_OBJECT_STORE_MAX_BYTES or set "
                "max_num_output_bytes_per_actor explicitly. Logged once "
                "per actor pool."
            )
        return default_max_num_output_bytes

    def reset_output_limit(self):
        default_max_outputs = safe_or(
            self._config.max_num_outputs_per_actor, float("inf")
        )
        for state in self._running_actors.values():
            # Restore each actor's derived (or configured) output-bytes budget,
            # not an unbounded default -- otherwise the block-size sizing in
            # _create_task_context sees an infinite limit.
            state.max_num_output_bytes = state.default_max_num_output_bytes
            state.max_num_outputs = default_max_outputs

    def upgrade_output_limit(
        self, node_id: NodeIdStr, delta_bytes: int, delta_outputs: int = 1
    ):
        """Raise output bp caps on the most-constrained actor on ``node_id``.

        Picks the actor on the given node with the lowest current
        ``(max_num_output_bytes, max_num_outputs)`` tuple and bumps both
        of its caps in place on the ``_ExperimentalActorState`` (which is the same
        object held in ``_running_actors``, so no second write is
        needed). The caller (``MapOperator``) has already chosen
        ``node_id`` based on cluster-wide signals; we only need to fan
        that decision out to a specific actor.
        """
        node_states = self._node_to_actor_states.get(node_id)
        assert node_states, (
            f"upgrade_output_limit called for node={node_id} with no "
            f"running actors; pool has nodes={list(self._node_to_actor_states)}"
        )

        # TODO(Justin): This can be a bit more sophisticated by getting the node stats
        # and getting the ratio of max_num_output_bytes / node_object_store (for hetero clusters)
        candidate_states = [
            s for s in node_states.values() if s.status != ActorStatus.TERMINATING
        ] or list(node_states.values())
        best_state = min(
            candidate_states,
            key=lambda s: (s.max_num_output_bytes, s.max_num_outputs),
        )
        best_state.max_num_output_bytes += delta_bytes
        best_state.max_num_outputs += delta_outputs
        # NOTE: ``inf + delta`` is still ``inf`` -- the unset dimension
        # stays unset after the bump, which is fine: the tuple key above
        # will still discriminate via the finite dimension on the next call.

        logger.info(
            f"\nname={self._map_worker_cls_name} ({self.get_actor_info()})\n"
            f"Increasing the capacity for actor={best_state.logical_id} "
            f"located on {node_id} by {memory_string(delta_bytes)} / "
            f"{delta_outputs} output,\n"
            f"new limit is {memory_string(best_state.max_num_output_bytes)} / "
            f"{best_state.max_num_outputs} outputs\n"
        )

    def output_bytes_limit(self, actor_id: LogicalActorId) -> float:
        actor = self._logical_id_to_actor.get(actor_id)
        if actor is None:
            # The actor was released (killed / died) while one of its data
            # tasks lingers; callers like the output-backpressure path may still
            # ask for its limit -- impose no constraint rather than KeyError.
            return float("inf")
        return self._running_actors[actor].max_num_output_bytes

    def average_bytes_per_output(self) -> float | None:
        """Estimated bytes per output block, or None if there are no running
        actors. Computed as the mean per-actor ``max_num_output_bytes`` scaled by
        ``BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO`` (i.e. the per-actor target block
        size), matching how ``_create_task_context`` sizes output blocks."""
        total: float = 0.0
        count: int = 0
        for state in self._running_actors.values():
            limit = state.max_num_output_bytes
            assert math.isfinite(limit)
            total += limit
            count += 1
        if count == 0:
            return None
        return total * BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO / count

    def output_count_limit(self, actor_id: LogicalActorId) -> float:
        actor = self._logical_id_to_actor.get(actor_id)
        if actor is None:
            return float("inf")
        return self._running_actors[actor].max_num_outputs


class ExperimentalAPMO(_OSSActorPoolMapOperator):
    """Actor-only-backend ``ActorPoolMapOperator``. See module docstring."""

    # Scheduling strategy for the actor currently being created, stashed by
    # ``_start_actor`` for ``_merge_ray_remote_args`` (both run synchronously
    # on the single-threaded scheduling loop). Class-level default so the
    # merge override is safe even during ``__init__``.
    _next_scheduling_strategy: NodeAffinitySchedulingStrategy | str | None = None

    def __init__(self, *args, **kwargs):
        self._user_ray_remote_args: Dict[str, Any] = dict(
            kwargs.get("ray_remote_args") or {}
        )
        self._user_set_ray_remote_args_fn = kwargs.get("ray_remote_args_fn") is not None

        super().__init__(*args, **kwargs)
        self._input_queue_bytes_by_node: DefaultDict[NodeIdStr, int] = defaultdict(int)
        # Output-consumption attribution. ``_task_logical_id`` maps a task index
        # to the logical id of the actor that ran it (captured at submit time);
        # it lets a dequeued output be attributed back to its producer so the
        # producer's ``num_unconsumed_outputs`` can be decremented. Unlike the
        # input queue key, an output may be consumed AFTER its task finished, so
        # the entry is retired only once the task is done AND its last queued
        # output has been consumed -- tracked by ``_unconsumed_by_task`` (queued-
        # not-yet-consumed count per task) and ``_done_tasks`` (tasks whose
        # generator has finished).
        self._task_logical_id: Dict[int, LogicalActorId] = {}
        self._unconsumed_by_task: DefaultDict[int, int] = defaultdict(int)
        self._done_tasks: Set[int] = set()
        # Pool terminating-generation last reconciled into the output queue's
        # prioritized set. -1 forces the first sync. See _sync_output_priority.
        self._last_synced_terminating_generation: int = -1
        self._task_output_count: Dict[int, int] = {}
        self._task_ray_id: Dict[int, ray.TaskID] = {}
        # Re-dispatch count per still-queued input bundle (object identity),
        # capped by SIZER_TERMINATE_MAX_BLOCK_RETRIES.
        self._bundle_retry_count: Dict[int, int] = {}
        # Lazily-parsed scheduling constraint, cached for the op's lifetime
        # (constraints are fixed once bootstrapped).
        self._placement_constraint: Optional[PlacementConstraint] = None
        # TODO(Justin): This is not clean, clean it up later.
        self._actor_only_metrics: Optional["ActorOnlyMetrics"] = None
        self._metrics_op_tag: Optional[str] = None

    def set_actor_only_metrics(self, metrics: "ActorOnlyMetrics", op_tag: str) -> None:
        self._actor_only_metrics = metrics
        self._metrics_op_tag = op_tag

    @property
    def uses_ray_remote_args_fn(self) -> bool:
        """True if the user supplied a ``ray_remote_args_fn``."""
        return self._user_set_ray_remote_args_fn

    @property
    def actor_pool(self) -> _ExperimentalActorPool:
        assert isinstance(self._actor_pool, _ExperimentalActorPool)
        return self._actor_pool

    @property
    def resource_bank(self) -> ResourceBankBase:
        assert self._resource_bank is not None
        return self._resource_bank

    def validate_ray_remote_args(self) -> None:
        """Reject ``ray_remote_args`` the operator sizer cannot reason about.

        The sizer owns actor placement and models cluster capacity in
        cpu/gpu/memory only. Users may set ``num_cpus`` / ``num_gpus`` /
        ``memory`` (and benign non-placement knobs like ``max_restarts`` or
        ``runtime_env``); anything that dictates placement Ray-core-side or
        introduces resources the capacity model doesn't track is rejected up
        front (at ``OperatorSizer.bootstrap``) rather than silently mis-placed.

        A ``label_selector`` is the exception: the sizer honors it by placing
        the operator's actors untargeted (so Ray core resolves the selector)
        and ahead of node-targeted operators -- see
        ``OperatorSizer.scale_where`` and ``placement_pinned_node``. This is
        how local-scheme reads (pinned to the driver node) flow through.

        Raises ``NotImplementedError`` when the user supplied any of:
          - ``ray_remote_args_fn`` -- may set per-actor placement/resources we
            can't reason about;
          - a ``scheduling_strategy`` other than a
            ``NodeAffinitySchedulingStrategy`` -- the sizer owns actor placement
            (it places via NodeAffinity), so a user strategy is never honored.
            A ``NodeAffinitySchedulingStrategy`` is the exception: it's how
            local-scheme reads pin to the driver node, the one placement the
            sizer must respect
          - a placement group (``placement_group`` /
            ``placement_group_bundle_index`` /
            ``placement_group_capture_child_tasks``);
          - custom ``resources`` -- the capacity model only tracks
            cpu/gpu/memory.
        """
        args = self._user_ray_remote_args
        if self._user_set_ray_remote_args_fn and not SIZER_CONSTRAINT_AWARE_PLACEMENT:
            # On the PG-aware path a ray_remote_args_fn is allowed *if* it only
            # assigns a placement group (the vLLM "ray" backend does exactly
            # this). We can't call the fn here to check (each call mints a real
            # PG), so the "PG only, no resource/placement takeover" contract is
            # enforced per-actor in ``_merge_ray_remote_args``. Such ops are
            # self-placed: the sizer creates them untargeted and caps them at
            # their user concurrency, never consuming its free map.
            self._raise_unsupported(
                "a ray_remote_args_fn (may set per-actor placement/resources)"
            )
        strategy = args.get("scheduling_strategy")
        allowed_strategies = (NodeAffinitySchedulingStrategy,)
        if SIZER_CONSTRAINT_AWARE_PLACEMENT:
            # A PlacementGroupSchedulingStrategy is allowed on the PG-aware path:
            # the sizer caps the pool by bundle capacity and delegates the bundle
            # placement to Ray core (the strategy is passed through untouched).
            allowed_strategies += (PlacementGroupSchedulingStrategy,)
        if strategy is not None and not isinstance(strategy, allowed_strategies):
            # NodeAffinity is allowed (local-scheme reads pin to the driver);
            # any other strategy fights the sizer's placement.
            self._raise_unsupported(
                "a custom scheduling_strategy (the sizer owns actor placement)"
            )
        # Capturing Data's internal tasks into the user's PG is not allowed.
        if args.get("placement_group_capture_child_tasks"):
            self._raise_unsupported(
                "placement_group_capture_child_tasks (Data's internal tasks must "
                "not be captured into the user's placement group)"
            )
        # A placement_group / bundle index is only understood on the PG-aware
        # path (the sizer caps the pool by bundle capacity and delegates the
        # bundle placement to Ray core). Otherwise it's still unsupported.
        if not SIZER_CONSTRAINT_AWARE_PLACEMENT and any(
            key in args for key in ("placement_group", "placement_group_bundle_index")
        ):
            self._raise_unsupported("a placement_group")
        if args.get("resources"):
            self._raise_unsupported(
                "custom resources (the capacity model only tracks cpu/gpu/memory)"
            )

    def _raise_unsupported(self, reason: str) -> None:
        raise NotImplementedError(
            f"The Ray Data operator sizer (RAY_DATA_ENABLE_OPERATOR_SIZER) does "
            f"not support {self.name}: it specifies {reason}. Set only "
            f"num_cpus / num_gpus / memory in ray_remote_args, or disable the "
            f"operator sizer. (ray_remote_args={self._user_ray_remote_args}, "
            f"ray_remote_args_fn={self._user_set_ray_remote_args_fn})"
        )

    @override
    def _create_actor_pool(
        self, compute_strategy: ActorPoolStrategy
    ) -> "AutoscalingActorPool":
        config = self._create_actor_pool_config(compute_strategy)
        return _ExperimentalActorPool(
            create_actor_fn=self._start_actor,
            config=config,
            map_worker_cls_name=self._map_worker_cls_name,
        )

    @override
    def start(
        self,
        options: ExecutionOptions,
        block_ref_counter: BlockRefCounter,
        resource_bank: Optional[ResourceBankBase] = None,
    ) -> None:
        assert resource_bank is not None
        self.actor_pool.set_resource_bank(resource_bank)
        super().start(options, block_ref_counter, resource_bank=resource_bank)

    @override
    def _scale_to_initial_size(self) -> None:
        """No-op under the sizer: initial sizing happens in the sizer instead."""
        if ENABLE_OPERATOR_SIZER:
            return
        super()._scale_to_initial_size()

    @override
    def _start_actor(
        self,
        labels: Dict[str, str],
        logical_actor_id: LogicalActorId,
        scheduling_strategy: NodeAffinitySchedulingStrategy | str | None = None,
    ) -> Tuple[ActorHandle, ObjectRef, ExecutionResources]:
        # TODO(sampan): super()._start_actor uses ``actor.get_location()`` as
        # the readiness ref. For sizer-placed actors the node is already known
        # (the soft=False NodeAffinity target), so get_location is now really
        # just a liveness/readiness probe -- replace it with a purpose-named
        # actor health-check method. (Untargeted/custom-placement ops still
        # genuinely need the actual location.)
        # Stash the strategy for _merge_ray_remote_args, which
        # super()._start_actor invokes synchronously below.
        self._next_scheduling_strategy = scheduling_strategy
        try:
            return super()._start_actor(labels, logical_actor_id)
        finally:
            self._next_scheduling_strategy = None

    @override
    def _merge_ray_remote_args(self) -> Dict[str, Any]:
        remote_args = super()._merge_ray_remote_args()
        if (
            self._user_set_ray_remote_args_fn
            and ENABLE_OPERATOR_SIZER
            and SIZER_CONSTRAINT_AWARE_PLACEMENT
        ):
            # Sizer PG-aware path: a ray_remote_args_fn is allowed ONLY if it
            # assigns a placement group and nothing else. A fn that overrides
            # resources or pins placement would break the sizer's accounting
            # (the pool is self-placed and capped at its concurrency, not sized
            # by these args), so reject it here
            self._assert_ray_remote_args_fn_pg_only(remote_args)
        if self._next_scheduling_strategy is not None:
            # The sizer places via NodeAffinity, superseding the op's default
            # scheduling_strategy (a string). User-supplied strategies and
            # placement groups are rejected up front
            # (``validate_ray_remote_args``); self-placed (PG / ray_remote_args_fn)
            # ops are created untargeted, so _next_scheduling_strategy is None for
            # them -- this never silently overrides genuine user placement intent.
            remote_args["scheduling_strategy"] = self._next_scheduling_strategy
        return remote_args

    def _assert_ray_remote_args_fn_pg_only(self, merged: Dict[str, Any]) -> None:
        """Enforce that ``ray_remote_args_fn`` only assigns a placement group.

        Compares the fn's realized output (``merged``) against the static
        ``_ray_remote_args``: the fn may only add a
        ``PlacementGroupSchedulingStrategy``; it may not change the resource
        footprint the sizer accounts for (num_cpus / num_gpus / memory /
        resources) or set any other scheduling strategy. Raises
        ``NotImplementedError`` otherwise.
        """
        strategy = merged.get("scheduling_strategy")
        if not isinstance(strategy, PlacementGroupSchedulingStrategy):
            self._raise_unsupported(
                "a ray_remote_args_fn that does not assign a placement group "
                f"(resolved scheduling_strategy={strategy!r}); the sizer only "
                "supports a fn that assigns a PlacementGroupSchedulingStrategy"
            )
        for key in ("num_cpus", "num_gpus", "memory", "resources"):
            if merged.get(key) != self._ray_remote_args.get(key):
                self._raise_unsupported(
                    f"a ray_remote_args_fn that overrides {key!r} "
                    f"({self._ray_remote_args.get(key)!r} -> {merged.get(key)!r}); "
                    "the fn may only assign a placement group, not change the "
                    "actor's resource footprint (it would break sizer accounting)"
                )

    def num_terminating_actors(self) -> int:
        return self.actor_pool.num_terminating_actors()

    def num_reclaimable_terminating_actors(self) -> int:
        return self.actor_pool.num_reclaimable_terminating_actors()

    def reclaimable_node_ids(self, count: int) -> List[NodeIdStr]:
        return self.actor_pool.reclaimable_node_ids(count)

    def process_draining_actors(self) -> int:
        # Reconcile output prioritization with the pool's draining set before
        # advancing terminations, so a draining actor's already-queued blocks are
        # served first and it reaches "fully drained" (and gets killed) sooner.
        self._sync_output_priority()
        return self.actor_pool.process_draining_actors()

    def apply_scale(self, req: "ExperimentalActorPoolScalingRequest") -> Optional[int]:
        """Apply a placement-resolved scaling request to the pool. Output-queue
        prioritization is reconciled in ``process_draining_actors`` (which the
        sizer runs right after applying this tick's requests)."""
        return self.actor_pool.scale(req)

    def _sync_output_priority(self) -> None:
        """Keep the actor-priority output queue's prioritized set equal to the
        pool's currently-draining actors: newly-draining actors are prioritized
        (their already-queued blocks jump the line), reclaimed/killed actors are
        deprioritized. No-op unless the output queue is an
        ``DrainingActorPriorityBundleQueue`` (actor-only backend, ``preserve_order`` off).
        """
        queue = self._output_queue
        if not isinstance(queue, DrainingActorPriorityBundleQueue):
            return
        # Fast path: skip entirely (no set build/diff, no queue scan) unless the
        # pool's draining set has actually changed since our last reconcile.
        generation = self.actor_pool.terminating_generation()
        if generation == self._last_synced_terminating_generation:
            return
        self._last_synced_terminating_generation = generation
        draining = self.actor_pool.draining_actor_ids()
        prioritized = queue.priority_actors()
        # Only newly-draining actors trigger the O(queue) prioritize scan;
        # reclaimed/gone actors are dropped in O(1).
        for actor_id in draining - prioritized:
            queue.prioritize_actor(actor_id)
        for actor_id in prioritized - draining:
            queue.deprioritize_actor(actor_id)

    @override
    def _create_output_queue(self, preserve_order: bool) -> BaseBundleQueue:
        # The actor-only backend serves a draining actor's outputs first so it can
        # be killed sooner -- unless the user requires ordered output, which needs
        # the (non-actor-aware) reordering queue.
        if actor_only_backend_enabled() and not preserve_order:
            return DrainingActorPriorityBundleQueue()
        return super()._create_output_queue(preserve_order)

    @override
    def _output_add_kwargs(self, task_index: int) -> Dict[str, Any]:
        if isinstance(self._output_queue, DrainingActorPriorityBundleQueue):
            # Tag the bundle with its producing actor so the queue can serve
            # draining actors first and hand the id back on consume.
            return {"actor_id": self._task_logical_id.get(task_index)}
        return super()._output_add_kwargs(task_index)

    @override
    def _get_next_inner(self) -> RefBundle:
        queue = self._output_queue
        if isinstance(queue, DrainingActorPriorityBundleQueue):
            bundle, actor_id = queue.get_next_with_actor_id()
            self._record_output_dequeued(bundle)
            self._decrement_unconsumed_outputs(actor_id)
            return bundle
        if isinstance(queue, ReorderingBundleQueue):
            # preserve_order path: the queue is keyed by task index; map it to the
            # producing actor via ``_task_logical_id`` (kept alive until consumed).
            bundle, task_index = queue.get_next_with_key()
            self._record_output_dequeued(bundle)
            self._decrement_unconsumed_outputs(self._task_logical_id.get(task_index))
            self._retire_task_attribution(task_index)
            return bundle
        return super()._get_next_inner()

    def _decrement_unconsumed_outputs(self, actor_id: Optional[LogicalActorId]) -> None:
        """Decrement the producing actor's unconsumed-output count so the drain
        gate can release it once all its outputs are consumed downstream."""
        if actor_id is None:
            return
        actor = self.actor_pool.get_actor_by_logical_id(actor_id)
        if actor is not None:
            self.actor_pool.on_output_consumed(actor)

    def _retire_task_attribution(self, task_index: int) -> None:
        """Reordering-queue path only: drop the task->actor entry once the task is
        done and its last queued output has been consumed. (The actor-priority
        queue carries the actor id on each bundle, so it needs no per-task map.)"""
        self._unconsumed_by_task[task_index] -= 1
        if self._unconsumed_by_task[task_index] <= 0 and task_index in self._done_tasks:
            self._unconsumed_by_task.pop(task_index, None)
            self._done_tasks.discard(task_index)
            self._task_logical_id.pop(task_index, None)

    def committed_usage_by_node(self) -> Dict[NodeIdStr, ExecutionResources]:
        return self.actor_pool.committed_usage_by_node()

    def input_queue_bytes_by_node(self) -> Dict[NodeIdStr, int]:
        return {
            node_id: num_bytes
            for node_id, num_bytes in self._input_queue_bytes_by_node.items()
            if num_bytes > 0
        }

    def placement_pinned_node(self) -> Optional[NodeIdStr]:
        """The node this op pins its own actors to, or None if unpinned.

        Local-scheme reads pin to the driver via a
        ``NodeAffinitySchedulingStrategy`` (RayTurbo readers) or a
        ``ray.io/node-id`` ``label_selector`` (OSS readers, merged with any
        DataContext-level selector). When an op is pinned, the sizer must not
        assign nodes for it, it places the op untargeted.
        """
        strategy = self._ray_remote_args.get("scheduling_strategy")
        if isinstance(strategy, NodeAffinitySchedulingStrategy):
            return strategy.node_id
        merged = merge_label_selector(
            self._ray_remote_args, self.data_context.execution_options.label_selector
        )
        node = (merged.get("label_selector") or {}).get(ray._raylet.RAY_NODE_ID_KEY)
        return node if isinstance(node, str) else None

    @override
    def placement_constraint(self) -> "PlacementConstraint":
        """This op's scheduling constraints as a label_selector (+ optional PG).

        Cached on the operator: constraints are fixed for the op's lifetime, and
        the op is the natural owner (any consumer -- not just the sizer -- gets
        the parsed constraint without re-merging the selector each call).
        """
        if self._placement_constraint is not None:
            return self._placement_constraint
        merged = merge_label_selector(
            self._ray_remote_args, self.data_context.execution_options.label_selector
        )
        selector = dict(merged.get("label_selector") or {})
        strategy = self._ray_remote_args.get("scheduling_strategy")
        if isinstance(strategy, NodeAffinitySchedulingStrategy):
            selector[ray._raylet.RAY_NODE_ID_KEY] = strategy.node_id

        pg = self._user_ray_remote_args.get("placement_group")
        bundle_index = self._user_ray_remote_args.get("placement_group_bundle_index")
        pg_strategy = self._user_ray_remote_args.get("scheduling_strategy")
        if pg is None and isinstance(pg_strategy, PlacementGroupSchedulingStrategy):
            pg = pg_strategy.placement_group
            bundle_index = pg_strategy.placement_group_bundle_index

        constraint = PlacementConstraint(
            label_selector=selector or None,
            placement_group=pg,
            placement_group_bundle_index=bundle_index,
        )
        self._placement_constraint = constraint
        return constraint

    def actor_counts_by_node(self) -> Dict[NodeIdStr, Counter[ActorStatus]]:
        """Breakdown of actor counts per node"""
        counts: DefaultDict[NodeIdStr, Counter[ActorStatus]] = defaultdict(Counter)
        for node_id, states in self.actor_pool._node_to_actor_states.items():
            for state in states.values():
                counts[node_id][state.status] += 1
        for (
            node_id,
            pending_ids,
        ) in self.actor_pool.pending_ids_by_target_node().items():
            counts[node_id][ActorStatus.PENDING] += len(pending_ids)
        return dict(counts)

    def expire_stuck_pending_actors(self) -> int:
        """Release pendings older than the expiry bound (see the pool method).
        Called by the sizer every tick, next to draining maintenance."""
        return self.actor_pool.release_expired_pending_actors(
            expiry_s=SIZER_PENDING_ACTOR_EXPIRY_S
        )

    def build_placement_view(self, need_victim_order: bool = True) -> OpPlacementView:
        """Snapshot everything the placement strategy needs for this tick.

        Args:
            need_victim_order: When True, build ``actor_ids_by_node`` — the
                per-node actor ids sorted least-busy-first — which only
                downscale victim selection consumes. Upscale placement reads
                per-node *counts* only, so upscale requests pass False to skip
                the per-actor tuple construction and sorts (an O(actors)
                counting pass instead of O(actors log actors)).

        Returns:
            The operator's placement snapshot; ``actor_ids_by_node`` is empty
            when ``need_victim_order`` is False.
        """
        pool = self.actor_pool

        actor_ids_by_node: Dict[NodeIdStr, List[LogicalActorId]] = {}
        actor_drain_cost_by_node: Dict[NodeIdStr, List[Tuple[int, int]]] = {}
        actors_by_node: Dict[NodeIdStr, int] = {}
        idle_actors_by_node: Dict[NodeIdStr, int] = {}
        for node_id, states in pool._node_to_actor_states.items():
            if need_victim_order:
                # Cheapest drain first: least queued work, then fewest
                # unconsumed output blocks (both must reach 0 before a
                # draining victim can exit), then least-recent submission.
                ranked = sorted(
                    (
                        (
                            state.num_tasks_in_flight,
                            state.num_unconsumed_outputs,
                            state.latest_task_submission_ts,
                            logical_id,
                        )
                        for logical_id, state in states.items()
                        if state.status != ActorStatus.TERMINATING
                    ),
                )
                if ranked:
                    actor_ids_by_node[node_id] = [entry[3] for entry in ranked]
                    actor_drain_cost_by_node[node_id] = [
                        (entry[0], entry[1]) for entry in ranked
                    ]
                    actors_by_node[node_id] = len(ranked)
                    num_idle = sum(1 for entry in ranked if entry[0] == 0)
                    if num_idle:
                        idle_actors_by_node[node_id] = num_idle
            else:
                num_alive = 0
                num_idle = 0
                for state in states.values():
                    if state.status == ActorStatus.TERMINATING:
                        continue
                    num_alive += 1
                    if state.num_tasks_in_flight == 0:
                        num_idle += 1
                if num_alive:
                    actors_by_node[node_id] = num_alive
                    if num_idle:
                        idle_actors_by_node[node_id] = num_idle

        pending_ids_by_node = pool.pending_ids_by_target_node()
        for node_id, pending_ids in pending_ids_by_node.items():
            actors_by_node[node_id] = actors_by_node.get(node_id, 0) + len(pending_ids)

        def _alive_count(inner_counter: "Counter[ActorStatus]") -> int:
            # Everything except actors marked for graceful termination (i.e.
            # status != ActorStatus.TERMINATING), so RESTARTING actors
            # (recovering from a transient crash and returning to the same
            # node) stay counted for co-location.
            return sum(
                inner_counter[status]
                for status in ActorStatus
                if status != ActorStatus.TERMINATING
            )

        alive_upstream_count_by_node: DefaultDict[NodeIdStr, int] = defaultdict(int)
        is_source_op = False
        if isinstance(self.input_dependency, ExperimentalAPMO):
            for (
                node_id,
                inner_counter,
            ) in self.input_dependency.actor_counts_by_node().items():
                alive_upstream_count_by_node[node_id] += _alive_count(inner_counter)
        else:
            is_source_op = True
        alive_downstream_count_by_node: DefaultDict[NodeIdStr, int] = defaultdict(int)
        for output_dep in self.output_dependencies:
            if not isinstance(output_dep, ExperimentalAPMO):
                continue
            for node_id, inner_counter in output_dep.actor_counts_by_node().items():
                alive_downstream_count_by_node[node_id] += _alive_count(inner_counter)

        up_w = down_w = 0.5
        if SIZER_EDGE_WEIGHTED_COLOCATION:
            bytes_in = self.metrics.bytes_inputs_received
            bytes_out = self.metrics.bytes_task_outputs_generated
            total = bytes_in + bytes_out
            if total > 0:
                up_w = bytes_in / total
                down_w = bytes_out / total

        return OpPlacementView(
            op_id=self.id,
            # TODO(Justin): This is a little funky, circle back to what defines a source op?
            is_source=is_source_op,
            per_actor_usage=pool._config.per_actor_resource_usage,
            input_bytes_per_actor=pool._config.max_input_bytes_per_actor,
            actors_by_node=actors_by_node,
            input_queue_bytes_by_node=self.input_queue_bytes_by_node(),
            upstream_actors_by_node=alive_upstream_count_by_node,
            downstream_actors_by_node=alive_downstream_count_by_node,
            actor_ids_by_node=actor_ids_by_node,
            actor_drain_cost_by_node=actor_drain_cost_by_node,
            idle_actors_by_node=idle_actors_by_node,
            pending_ids_by_node=pending_ids_by_node,
            upstream_edge_weight=up_w,
            downstream_edge_weight=down_w,
        )

    @override
    def _maybe_dispatch_next_bundle(self) -> None:
        # Pull-based input protocol: instead of self-dispatching a task when a
        # full bundle is ready (the OSS push model), enqueue the bundle onto the
        # operator's internal task-submission queue. The scheduling loop later
        # pulls work from this queue via can_submit_task() / launch_task().
        if self._block_ref_bundler.peek_next() is not None:
            self._add_to_inner_bundle_queue()

    @override
    def has_next(self) -> bool:
        # Skip the OSS ActorPoolMapOperator bundle-queue dispatch on input
        # completion; under the actor-only backend the scheduling loop drives
        # task launches via launch_task().
        return super(_OSSActorPoolMapOperator, self).has_next()

    @override
    def all_inputs_done(self):
        self._block_ref_bundler.finalize()

        # Drain any remaining bundles into the internal task-submission queue.
        while self._block_ref_bundler.has_next():
            self._add_to_inner_bundle_queue()

        assert self._block_ref_bundler.estimate_size_bytes() == 0, (
            f"Bundler in {self} must be empty (got {self._block_ref_bundler.num_blocks()} blocks)"
        )

        # Skip the OSS ActorPoolMapOperator.all_inputs_done body (min-actors
        # warning + bundler drain-dispatch) and go straight to the generic
        # operator finalization.
        super(MapOperator, self).all_inputs_done()

    @staticmethod
    @override
    def _apply_default_remote_args(
        ray_remote_args: Dict[str, Any], data_context: DataContext
    ) -> Dict[str, Any]:
        user_set_strategy = "scheduling_strategy" in (ray_remote_args or {})
        ray_remote_args = _OSSActorPoolMapOperator._apply_default_remote_args(
            ray_remote_args, data_context
        )
        if ENABLE_OPERATOR_SIZER and not user_set_strategy:
            # The operator sizer owns actor placement (NodeAffinity, or an
            # untargeted placement honoring a label_selector).
            ray_remote_args.pop("scheduling_strategy", None)
        if core_actor_backpressure_enabled():
            if "_actor_generator_backpressure_num_objects" not in ray_remote_args:
                buffer = data_context._max_num_blocks_in_streaming_gen_buffer
                assert buffer is not None
                # Object count: one grouped yield emits block + metadata.
                ray_remote_args["_actor_generator_backpressure_num_objects"] = (
                    CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD * buffer
                )
                logger.info(
                    "Actor pool backpressure for %s: "
                    "_actor_generator_backpressure_num_objects=%s, "
                    "_num_objects_per_yield=%s on MapWorker.submit",
                    "MapWorker",
                    ray_remote_args["_actor_generator_backpressure_num_objects"],
                    CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD,
                )
        return ray_remote_args

    @override
    def _create_actor_pool_config(
        self, compute_strategy: ActorPoolStrategy
    ) -> "AutoscalingActorConfig":
        # Run read-source actors at READ_ACTOR_CONCURRENCY (c8/mtf8): set
        # max_concurrency on the remote args -- read both here and when the
        # actor is spawned -- so one actor serves that many concurrent reads.
        read_concurrency = (
            READ_ACTOR_CONCURRENCY
            if READ_ACTOR_CONCURRENCY > 1 and self.is_read_op
            else 1
        )
        if read_concurrency > 1:
            self._ray_remote_args["max_concurrency"] = read_concurrency

        max_actor_concurrency = self._ray_remote_args.get("max_concurrency", 1)

        max_tasks_in_flight_per_actor: int = (
            compute_strategy.max_tasks_in_flight_per_actor
            or self.data_context.max_tasks_in_flight_per_actor
            or max_actor_concurrency * PER_ACTOR_INPUT_BACKPRESSURE_LIMIT
        )

        max_num_output_bytes_per_actor: Optional[int] = (
            compute_strategy.max_num_output_bytes_per_actor
            or PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT
        )

        if read_concurrency > 1:
            # Match tasks-in-flight to the concurrency, and scale the output
            # budget by it so the actor's concurrent producers don't trip output
            # backpressure and pin the pool at a single actor.
            max_tasks_in_flight_per_actor = read_concurrency
            if max_num_output_bytes_per_actor is not None:
                max_num_output_bytes_per_actor *= read_concurrency

        max_num_outputs_per_actor: Optional[int] = (
            compute_strategy.max_num_outputs_per_actor
            or PER_ACTOR_OUTPUT_BACKPRESSURE_LIMIT
        )

        max_input_bytes_per_actor: Optional[int] = (
            compute_strategy.max_input_bytes_per_actor
            or PER_ACTOR_INPUT_BYTES_BACKPRESSURE_LIMIT
        )

        # Reserve one CPU's worth of usable heap per concurrent slot. This lets
        # the sizer (and Ray Core) place the actor by memory instead of packing
        # nodes by CPU alone and OOMing their RAM. Gated behind the
        # ENABLE_DEFAULT_MEMORY_LIMITS flag; an explicit user memory= always wins.
        #
        # This is a deliberately conservative static estimate. We do NOT use the
        # runtime per-output estimate (op.metrics.average_bytes_per_output, what
        # ConfigureMapTaskMemoryUsingOutputSize uses) as it measures only the
        # output block and omits in-flight input + transient/decode buffers .
        # TODO(sampan): add actor-specific memory estimation once a real
        # per-actor memory signal exists (measured RSS, or an input+output+
        # transient model).
        # Skip the default reservation for placement-group ops: the actor is
        # scheduled into a user-defined bundle, and if we reserve memory the
        # bundle didn't declare, neither Ray core (bundle can't fit the actor)
        # nor the sizer (per-bundle max_placeable_actors hits 0 on the memory dim) can
        # place it -- the pool stalls at 0. Mirrors ConfigureMapTaskMemoryRule.
        #
        # A ray_remote_args_fn counts too: it returns per-actor remote args
        # (e.g. the vLLM "ray" backend returns a PlacementGroupSchedulingStrategy
        # for each engine replica), so the actor may land in a bundle we can't
        # inspect here. Treat it as PG-backed and let the fn own placement.
        uses_placement_group = (
            "placement_group" in self._user_ray_remote_args
            or isinstance(
                self._user_ray_remote_args.get("scheduling_strategy"),
                PlacementGroupSchedulingStrategy,
            )
            or self._user_set_ray_remote_args_fn
        )
        reserved_memory = self._ray_remote_args.get("memory")
        if (
            reserved_memory is None
            and ENABLE_DEFAULT_MEMORY_LIMITS
            and not uses_placement_group
        ):
            num_cpus = self._ray_remote_args.get("num_cpus") or 1
            reserved_memory = int(
                _usable_memory_per_cpu() * num_cpus * max_actor_concurrency
            )
            # Write back so Ray Core admission-controls on it at scheduling, not
            # just the sizer. Safe: _create_actor_pool_config (called from
            # __init__ via _create_actor_pool) runs before start() builds
            # ray.remote(**self._ray_remote_args).
            self._ray_remote_args["memory"] = reserved_memory
        per_actor_resource_usage = ExecutionResources(
            cpu=self._ray_remote_args.get("num_cpus"),
            gpu=self._ray_remote_args.get("num_gpus"),
            memory=reserved_memory,
        )

        logger.info(
            "Actor pool backpressure limits for %s: max_actor_concurrency=%s "
            "max_tasks_in_flight_per_actor=%s max_input_bytes_per_actor=%s "
            "max_num_output_bytes_per_actor=%s max_num_outputs_per_actor=%s",
            self.name,
            max_actor_concurrency,
            max_tasks_in_flight_per_actor,
            max_input_bytes_per_actor,
            max_num_output_bytes_per_actor,
            max_num_outputs_per_actor,
        )

        config = AutoscalingActorConfig(
            min_size=compute_strategy.min_size,
            max_size=compute_strategy.max_size,
            initial_size=compute_strategy.initial_size,
            max_tasks_in_flight_per_actor=max_tasks_in_flight_per_actor,
            max_input_bytes_per_actor=max_input_bytes_per_actor,
            max_num_output_bytes_per_actor=max_num_output_bytes_per_actor,
            max_num_outputs_per_actor=max_num_outputs_per_actor,
            max_actor_concurrency=max_actor_concurrency,
            per_actor_resource_usage=per_actor_resource_usage,
        )
        return config

    def _add_to_inner_bundle_queue(self):
        # The ref bundler combines one or more `RefBundle`s into a new
        # `RefBundle`. To update metrics appropriately, we need to deque
        # original input bundles.
        (
            bundled_input,
            input_refs,
        ) = self._block_ref_bundler.get_next_with_original()
        # NOTE: These are the 2 same bundle

        for input_ref in input_refs:
            self._metrics.on_input_dequeued(input_ref, input_index=0)
        self._bundle_queue.add(bundled_input)
        for (
            node_id,
            num_bytes,
        ) in bundled_input.get_preferred_object_locations_from_metadata().items():
            self._input_queue_bytes_by_node[node_id] += num_bytes
        self._metrics.on_input_queued(bundled_input, input_index=0)
        # Notify first input for deferred initialization (e.g., Iceberg schema evolution).
        # Enqueue input bundle
        self._notify_first_input(bundled_input)

    def _pick_next_actor(self) -> Tuple[ActorHandle[_MapWorker], RefBundle] | None:
        bundle = self._bundle_queue.peek_next()
        if bundle is None:
            return None

        if self._actor_locality_enabled is None:
            actor_locality_enabled = True
        else:
            actor_locality_enabled = self._actor_locality_enabled

        next_actor = self.actor_pool.select_actors(
            bundle=bundle, actor_locality_enabled=actor_locality_enabled
        )
        if next_actor is None:
            return None

        return next_actor, bundle

    def can_submit_task(self) -> bool:
        """NOTE: PLEASE READ CAREFULLY

        This method has to abide by the following contract to guarantee Operator's
        ability to handle all provided inputs (liveness):

            - This method should only return `True` when operator is guaranteed
            to be able to launch a task, meaning that subsequent `op.add_input(...)`
            should be able to launch a task.

        """

        return self._pick_next_actor() is not None

    @override
    def _create_task_context(self, actor: ActorHandle) -> TaskContext:
        """Build TaskContext, sizing blocks from this actor's output budget.

        Skips the bytes-based override when the map transformer sizes by row
        count or has block shaping disabled (e.g. StreamingRepartition) — those
        transforms reject ``target_max_block_size`` overrides.
        """
        # Ops that size by rows / disable shaping cannot accept a bytes override.
        last_transform = self.get_map_transformer().get_transform_fns()[-1]
        block_opt = last_transform.output_block_size_option
        can_override_block_size = block_opt is None or (
            not block_opt.disable_block_shaping
            and block_opt.target_num_rows_per_block is None
        )

        if not can_override_block_size:
            target_max_block_size = self.target_max_block_size_override
        else:
            target_max_block_size = (
                self.target_max_block_size_override
                or self.data_context.target_max_block_size
            )
            # None means the user disabled chopping — leave blocks whole.
            if target_max_block_size is not None:
                output_limit = self.actor_pool.output_bytes_limit(
                    self.actor_pool.get_actor_logical_id(actor)
                )
                assert math.isfinite(output_limit) and output_limit > 0
                target_max_block_size = min(
                    int(output_limit * BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO),
                    target_max_block_size,
                )

        return TaskContext(
            task_idx=self._next_data_task_idx,
            op_name=self.name,
            target_max_block_size_override=target_max_block_size,
        )

    @override
    def launch_task(self):
        """Try to dispatch tasks from the internal queue. Returns the # of tasks submitted"""

        actor_bundle = self._pick_next_actor()
        assert actor_bundle is not None
        actor, bundle = actor_bundle
        actor_node_id = self.actor_pool.get_actor_location(actor)
        actor_logical_id = self.actor_pool.get_actor_logical_id(actor)

        self.resource_bank.maybe_register_actor(
            op=self,
            actor_id=actor_logical_id,
            new_node_id=actor_node_id,
        )

        self._bundle_queue.remove(bundle)
        for (
            node_id,
            num_bytes,
        ) in bundle.get_preferred_object_locations_from_metadata().items():
            self._input_queue_bytes_by_node[node_id] -= num_bytes

        self._metrics.on_input_dequeued(bundle, input_index=0)
        input_blocks = [entry.ref for entry in bundle.blocks]
        self.actor_pool.on_task_submitted(actor, input_bundle=bundle)
        # _submit_data_task below assigns this task index, then increments.
        # Capture it (and the actor it ran on) so a later RayActorError can be
        # attributed even after the actor has been released.
        task_index = self._next_data_task_idx
        # Attribute this task's outputs back to the actor that ran it, so a
        # dequeued output can decrement the producer's unconsumed-output count.
        self._task_logical_id[task_index] = actor_logical_id
        self._task_output_count[task_index] = 0

        ctx = self._create_task_context(actor)
        actor_task_args = dict(self._ray_actor_task_remote_args)
        extra_labels = actor_task_args.pop("_labels", None) or {}
        gen = actor.submit.options(
            num_returns="streaming",
            _labels={self._OPERATOR_ID_LABEL_KEY: self.id, **extra_labels},
            **actor_task_args,
        ).remote(
            self._data_context_ref,
            ctx,
            *input_blocks,
            slices=bundle.slices,
            **self.get_map_task_kwargs(),
        )

        def _task_output_ready_callback(output: RefBundle):
            if isinstance(self._output_queue, ReorderingBundleQueue):
                self._unconsumed_by_task[task_index] += 1
            # Track that this task emitted output: a task that already streamed
            # blocks downstream cannot be safely re-dispatched (it would
            # duplicate them), so this gates the retry below.
            self._task_output_count[task_index] = (
                self._task_output_count.get(task_index, 0) + 1
            )
            # Count streamed output as drain progress so a long streaming task
            # on a draining actor isn't mistaken for a stalled drain.
            self.actor_pool.on_output_produced(actor)
            ray_task_id = self._task_ray_id.get(task_index)
            target_block_size = ctx.target_max_block_size_override
            output_bytes_limit = self.actor_pool.output_bytes_limit(
                actor_id=actor_logical_id
            )
            for entry in output.blocks:
                size_bytes = entry.metadata.size_bytes or 0
                if (
                    target_block_size is not None
                    and target_block_size > 0
                    and size_bytes > 1.5 * target_block_size
                ):
                    self._actor_only_metrics.record_block_size_overshoot(
                        self._metrics_op_tag
                    )
                    if log_once(f"actor_only_block_size_overshoot_{self.id}"):
                        logger.warning(
                            f"{self.name}: actor={actor_logical_id} produced an "
                            f"output of {memory_string(size_bytes)}, which is "
                            f"more than 150% of its target block size "
                            f"{memory_string(target_block_size)}. This warning "
                            "is logged once per operator; see the "
                            "data_actor_only_operator_output_overshoots metric "
                            "for the full count."
                        )
                self.resource_bank.on_new_output(
                    op=self,
                    ref=entry.ref,
                    bm=entry.metadata,
                    task_id=ray_task_id,
                )
                actor_live = self.resource_bank.live_object_store(
                    actor_id=actor_logical_id
                )
                total_output_bytes = (
                    actor_live.num_pulled_output_bytes
                    + actor_live.num_dangling_output_bytes
                    + actor_live.num_prebuffered_output_bytes
                )
                if (
                    math.isfinite(output_bytes_limit)
                    and output_bytes_limit > 0
                    and total_output_bytes > 1.5 * output_bytes_limit
                ):
                    self._actor_only_metrics.record_output_limit_overshoot(
                        self._metrics_op_tag
                    )
                    if log_once(f"actor_only_output_bytes_overshoot_{self.id}"):
                        logger.warning(
                            f"{self.name}: actor={actor_logical_id} has "
                            f"{memory_string(total_output_bytes)} of outstanding "
                            f"output, which is more than 150% of its output-bytes "
                            f"limit {memory_string(output_bytes_limit)}. This "
                            "warning is logged once per operator; see the "
                            "data_actor_only_operator_output_overshoots metric "
                            "for the full count."
                        )

        def _task_done_callback(
            inputs: RefBundle,
            task_index: int,
            exception: Optional[Exception],
            actor_to_return,
        ):
            # The task's generator has finished (success or failure): no further
            # outputs will be queued under this task index.
            # Reverse the in-flight accounting this task held.
            self._task_output_count.pop(task_index, 0)
            ray_task_id = self._task_ray_id.pop(task_index)

            # Reverse the in-flight accounting this task held. Both calls are
            # tolerant of an already-released actor (the force-kill path releases
            # the actor before its in-flight tasks surface RayActorError).
            self.resource_bank.on_task_completed(op=self, task_id=ray_task_id)
            self.actor_pool.on_task_completed(actor_to_return, input_bundle=inputs)
            for entry in inputs.blocks:
                self.resource_bank.on_block_consumed(input_ref=entry.ref)

            # Output attribution cleanup. The actor-priority queue carries the
            # actor id on each bundle, so its task->actor entry can be dropped
            # now (all outputs were queued before the task finished). The
            # reordering path may still consume this task's outputs later, so
            # defer to _retire_task_attribution once its last block is consumed.
            if isinstance(self._output_queue, ReorderingBundleQueue):
                self._done_tasks.add(task_index)
                if self._unconsumed_by_task.get(task_index, 0) <= 0:
                    self._unconsumed_by_task.pop(task_index, None)
                    self._done_tasks.discard(task_index)
                    self._task_logical_id.pop(task_index, None)
            else:
                self._task_logical_id.pop(task_index, None)

        self._submit_data_task(
            gen=gen,
            inputs=bundle,
            task_output_ready_callback=_task_output_ready_callback,
            task_done_callback=partial(_task_done_callback, actor_to_return=actor),
        )
        data_task = self._data_tasks[task_index]
        ray_task_id = data_task.get_task_id()
        self._task_ray_id[task_index] = ray_task_id
        self.resource_bank.on_task_submitted(
            actor_id=actor_logical_id,
            task_id=ray_task_id,
            input_bms=list(bundle.metadata),
        )

        # Update locality metrics
        if actor_node_id in bundle.get_preferred_object_locations_from_metadata():
            self._locality_hits += 1
        else:
            self._locality_misses += 1
        # Cross-node object transfer: any input block whose producing node is not
        # the actor's node will be fetched over the network. Use the same node
        # attribution as get_preferred_object_locations_from_metadata (skip
        # blocks without exec_stats) so this stays consistent with the
        # hit/miss accounting above.
        remote_bytes = 0
        remote_blocks = 0
        for bm in bundle.metadata:
            if bm.exec_stats is not None and bm.exec_stats.node_id != actor_node_id:
                remote_bytes += bm.size_bytes or 0
                remote_blocks += 1
        if remote_blocks:
            self._metrics.bytes_remote_inputs_read += remote_bytes
            self._metrics.num_remote_blocks_read += remote_blocks

    def max_output_bytes_for_actor(self, actor_id: LogicalActorId) -> float:
        """The (soft) maximum number of bytes that we can pull for this actor
        Actor's can handle multiple tasks, so we apply limit at a per-actor level.
        It's a soft maximum because we cannot be precise in how many bytes an actor
        can produce"""
        return self.actor_pool.output_bytes_limit(actor_id=actor_id)

    def max_outputs_for_actor(self, actor_id: LogicalActorId) -> float:
        """The (hard) maximum number of outputs that we can pull for this actor
        Actor's can handle multiple tasks, so we apply limit at a per-actor level.
        It's a hard maximum because ray core provides a way to pull by # of blocks"""
        return self.actor_pool.output_count_limit(actor_id=actor_id)

    @override
    def refresh_state(self):
        """Updates internal state"""
        info = self.get_actor_info()
        if info.pending > 0:
            self.actor_pool.reset_output_limit()
        return super().refresh_state()

    def upgrade_output_limit(self) -> bool:
        """In the scenario that
            - Upstream operator cannot produce more outputs (Output Backpressured)
            - Downstream operator cannot launch task due to some requirement (like batch_size)
        We increase the output limit for an actor whose node has the most available
        object store. We increase it by one block.
        """

        # Prefer nodes that currently host in-flight work: upgrading an idle
        # node's caps does not relieve the backpressured producer.
        pool_nodes = self.actor_pool.active_nodes()
        if not pool_nodes:
            return False

        most_free_obj_store: float = float("-inf")
        best_node: NodeIdStr | None = None

        for node_id in pool_nodes:
            obj_store_usage = self.resource_bank.live_object_store(node_id=node_id)
            # TODO(Justin): This needs to go through the DefautlAutoscalingCoordinator.
            obj_store_limit = self.resource_bank.node_capacity(
                node_id=node_id
            ).object_store_memory
            obj_store_free = (
                obj_store_limit
                - obj_store_usage.num_pulled_output_bytes
                - obj_store_usage.num_prebuffered_output_bytes
                - obj_store_usage.num_dangling_output_bytes
                - obj_store_usage.num_input_bytes
            )

            if obj_store_free > most_free_obj_store:
                most_free_obj_store = obj_store_free
                best_node = node_id

        mean_output_bytes = (
            self.resource_bank.cumulative_object_store(op=self).output_bytes.mean
            or self.actor_pool.average_bytes_per_output()
            or (128 << 20)
        )
        delta_bytes = max(1 << 20, int(math.ceil(mean_output_bytes)))
        assert best_node is not None
        self.actor_pool.upgrade_output_limit(
            node_id=best_node,
            delta_bytes=delta_bytes,
            delta_outputs=1,
        )
        return True

    def task_pull_request(self, task_idx: int) -> TaskPullRequest:
        """Return how many bytes to pull from streaming generators."""
        ray_task_id = self._task_ray_id.get(task_idx)
        actor_id = self.resource_bank.get_actor_id_from_task_id(
            op=self, task_id=ray_task_id
        )
        assert actor_id is not None
        # NOTE: the bytes_to_read are a soft upper bound, because we can't
        # read the exact bytes from a streaming generator.
        return self._actor_pull_budget(actor_id)

    def _actor_pull_budget(self, actor_id: LogicalActorId) -> TaskPullRequest:
        """Remaining per-actor pull budget: limit minus pulled, dangling and
        prebuffered outputs, clamped at zero. All tasks on an actor share it."""
        actor_object_store = self.resource_bank.live_object_store(actor_id=actor_id)
        max_output_bytes_per_actor = self.max_output_bytes_for_actor(actor_id=actor_id)
        max_bytes_to_read = max(
            0,
            max_output_bytes_per_actor
            - actor_object_store.num_pulled_output_bytes
            - actor_object_store.num_dangling_output_bytes
            - actor_object_store.num_prebuffered_output_bytes,
        )

        max_outputs_per_actor = self.max_outputs_for_actor(actor_id=actor_id)
        # NOTE: This can go below 0 when the user specified a max_output_per_actor
        # less than _generator_backpressure_num_objects
        max_blocks_to_read = max(
            0,
            max_outputs_per_actor
            - actor_object_store.num_pulled_output_blocks
            - actor_object_store.num_dangling_output_blocks
            - actor_object_store.num_prebuffered_output_blocks,
        )
        return TaskPullRequest(
            bytes_to_read=max_bytes_to_read,
            blocks_to_read=max_blocks_to_read,
        )

    @override
    def backpressure_progress_str(self) -> str:
        """Report the share of actors actually blocked on output.

        The base class reports output backpressure as the policy name that
        tripped, which is a boolean: it says a pull budget somewhere is
        exhausted, not how much of the pool that costs.  For an actor pool the
        actionable quantity is what fraction of serving actors had a ready output
        they could not pull, so that replaces the ``outputs(...)`` tag here.
        Submission backpressure is a separate axis and is still reported as-is.

        This is the raw per-iteration observation, so expect it to move between
        progress-bar refreshes.
        """
        parts = []
        if self._in_task_submission_backpressure:
            policy = self._task_submission_backpressure_policy or ""
            parts.append(f"backpressured:tasks({policy})")
        blocked, _ = self.output_backpressured_actors()
        if blocked > 0:
            parts.append(f"blocked={blocked}/{self.actor_pool.current_size()}")
        if not parts:
            return ""
        return f"[{', '.join(parts)}]"

    def is_output_backpressured(self) -> bool:
        """Output backpressured means that the actor has the means
        to physically produce more outputs, but because Ray Data wants to avoid
        spilling, we intentionally do not pull outputs from the actor's
        streaming generators."""
        info = self.get_actor_info()
        if info.running == 0 or info.pending > 0:
            return False
        if not self._data_tasks:
            return False
        # TASK denomination, deliberately: this is the deadlock hatch's trigger
        # (detect_if_idle), which asks "is every ACTIVE task blocked on output
        # budget?". At the pipeline tail that is a couple of blocked tasks on a
        # mostly-idle pool; the actor-denominated pair reads blocked < running
        # there and would disable the hatch at exactly the state it exists for.
        blocked, active = self.output_backpressured_fraction()
        return blocked == active

    def output_backpressured_fraction(self) -> Tuple[int, int]:
        """(blocked, active) data-task counts for output backpressure.

        A task is blocked when any field of its pull budget is exhausted
        (bytes or block count). ``is_output_backpressured`` is the
        blocked == active special case; the OperatorSizer uses the fraction
        as its operator-level OutQueue-full signal.
        """
        zeros = TaskPullRequest.zero()
        blocked = 0
        active = 0
        for task in self._data_tasks.values():
            actor_id = self.resource_bank.get_actor_id_from_task_id(
                op=self, task_id=task.get_task_id()
            )
            if actor_id is None or not self.actor_pool.has_actor(actor_id):
                # Skip tasks whose actor has been released (killed / died): the ref
                # is a lingering orphan that process_completed_tasks will reap and
                # it carries no meaningful backpressure signal.
                continue
            active += 1
            spec = self.task_pull_request(task_idx=task.task_index())
            if not spec.all_fields_gt(zeros):
                blocked += 1
        return blocked, active

    def output_backpressured_actors(self) -> Tuple[int, int]:
        """(blocked, running) ACTOR counts for output backpressure.

        An actor is blocked when its remaining pull budget is exhausted in any
        field. Unlike ``output_backpressured_fraction`` the denominator is the
        whole running pool, so idle actors count as unblocked capacity rather
        than vanishing from the ratio. An idle actor with an exhausted budget
        (consumed by dangling/prebuffered outputs) counts as blocked:
        dispatching to it produces blocks it cannot emit.
        """
        info = self.get_actor_info()
        if info.running == 0 or info.pending > 0:
            return 0, 0
        zeros = TaskPullRequest.zero()
        blocked = 0
        for actor_id in self.actor_pool.get_logical_ids():
            if not self.actor_pool.has_actor(actor_id):
                continue
            if not self._actor_pull_budget(actor_id).all_fields_gt(zeros):
                blocked += 1
        return blocked, info.running

    def has_enough_budget_for_prebuffered_outputs(self) -> bool:
        """With low object store, the operator may not have enough budget
        for prebuffered outputs. Returns True if at least one actor has more
        budget than prebuffered outputs. False if all actors are stalled on
        prebuffered outputs"""
        for actor_id in self.actor_pool.get_logical_ids():
            actor_object_store = self.resource_bank.live_object_store(actor_id=actor_id)
            max_outputs_per_actor = self.max_outputs_for_actor(actor_id=actor_id)
            max_output_bytes_per_actor = self.max_output_bytes_for_actor(
                actor_id=actor_id
            )
            limit = TaskPullRequest(
                bytes_to_read=max_output_bytes_per_actor,
                blocks_to_read=max_outputs_per_actor,
            )
            usage = TaskPullRequest(
                bytes_to_read=actor_object_store.num_prebuffered_output_bytes,
                blocks_to_read=actor_object_store.num_prebuffered_output_blocks,
            )
            if limit.all_fields_gt(usage):
                return True
        return False
