from __future__ import annotations

import logging
import math
import threading
from abc import ABC
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from typing_extensions import override

import ray
from ray.data._internal.cached_ray_internals import get_alive_nodes_uncached
from ray.data._internal.execution.execution_flags import (
    CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD,
    RAY_DATA_MOVE_SEMANTIC,
    actor_only_backend_enabled,
)
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    NodeIdStr,
    PhysicalOperator,
)
from ray.data._internal.execution.interfaces.common import LogicalActorId
from ray.data._internal.execution.interfaces.distribution_tracker import (
    DistributionTracker,
)
from ray.data.block import Block, BlockMetadata
from ray.types import ObjectRef

if TYPE_CHECKING:
    from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
        ExperimentalAPMO,
    )

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, repr=False, eq=False)
class UniqueBlockMetadata:
    # The operator that produced this block
    op: PhysicalOperator
    bm: BlockMetadata
    task_id: ray.TaskID | None = None


@dataclass(kw_only=True, slots=True, repr=False, eq=False)
class CumulativeObjStore:
    """These object store stats should be non-decreasing.
    Fine-grained breakdown of how the object store is being used
    """

    # Outputs (blocks produced from tasks)
    output_bytes: DistributionTracker = field(default_factory=DistributionTracker)
    output_rows: DistributionTracker = field(default_factory=DistributionTracker)

    # Inputs (blocks consumed in tasks)
    input_bytes: DistributionTracker = field(default_factory=DistributionTracker)
    input_rows: DistributionTracker = field(default_factory=DistributionTracker)

    # breakdown of which nodes are local, which are remote
    input_bytes_local: DistributionTracker = field(default_factory=DistributionTracker)
    input_bytes_remote: DistributionTracker = field(default_factory=DistributionTracker)

    def on_input_submitted(
        self,
        bm: BlockMetadata,
        remote: bool,
    ):
        size_bytes = bm.size_bytes or 0
        if remote:
            self.input_bytes_remote.add_sample(size_bytes)
        else:
            self.input_bytes_local.add_sample(size_bytes)

        self.input_rows.add_sample(bm.num_rows or 0)
        self.input_bytes.add_sample(size_bytes)

    def on_new_output(
        self,
        bm: BlockMetadata,
    ):
        self.output_bytes.add_sample(bm.size_bytes or 0)
        self.output_rows.add_sample(bm.num_rows or 0)

    @staticmethod
    def _merged_tracker(
        left: DistributionTracker, right: DistributionTracker
    ) -> DistributionTracker:
        tracker = DistributionTracker()
        tracker.merge(left)
        tracker.merge(right)
        return tracker

    def __add__(self, other: "CumulativeObjStore") -> "CumulativeObjStore":
        return CumulativeObjStore(
            output_bytes=self._merged_tracker(self.output_bytes, other.output_bytes),
            output_rows=self._merged_tracker(self.output_rows, other.output_rows),
            input_bytes=self._merged_tracker(self.input_bytes, other.input_bytes),
            input_rows=self._merged_tracker(self.input_rows, other.input_rows),
            input_bytes_local=self._merged_tracker(
                self.input_bytes_local, other.input_bytes_local
            ),
            input_bytes_remote=self._merged_tracker(
                self.input_bytes_remote, other.input_bytes_remote
            ),
        )


@dataclass(kw_only=True, slots=True, repr=False, eq=False)
class LiveObjStore:
    """The current snapshot of each object store, INCLUDING
    prebuffered bytes from _generator_backpressure_num_objects"""

    # Pulled outputs (blocks pulled from generators, not yet consumed downstream)
    num_pulled_output_blocks: int = 0
    num_pulled_output_bytes: int = 0
    num_pulled_output_rows: int = 0

    # Prebuffered outputs (generated but not yet pulled from generators)
    num_prebuffered_output_blocks: int = 0
    num_prebuffered_output_bytes: int = 0
    num_prebuffered_output_rows: int = 0

    # Inputs (blocks being consumed in tasks, aka in-flight)
    num_input_blocks: int = 0
    num_input_bytes: int = 0

    # Dangling Outputs
    num_dangling_output_blocks: int = 0
    num_dangling_output_bytes: int = 0
    num_dangling_output_rows: int = 0

    def on_input_submitted(
        self,
        bm: BlockMetadata,
    ):
        self.num_input_bytes += bm.size_bytes or 0
        self.num_input_blocks += 1

    def on_new_output(
        self,
        bm: BlockMetadata,
    ):
        self.num_pulled_output_blocks += 1
        self.num_pulled_output_bytes += bm.size_bytes or 0
        self.num_pulled_output_rows += bm.num_rows or 0

    def on_input_dropped(
        self,
        task_info: TaskInfo,
    ):
        self.num_input_blocks -= task_info.live.num_input_blocks
        self.num_input_bytes -= task_info.live.num_input_bytes

    def on_output_consumed(
        self,
        bm: BlockMetadata,
    ):
        """When an input is consumed from a downstream worker, it frees up space for the
        upstream actor. That's why we decrement the output, since it's
        also known as the input to the downstream worker."""
        size_bytes = bm.size_bytes or 0
        num_rows = bm.num_rows or 0
        if self.num_dangling_output_blocks > 0:
            self.num_dangling_output_blocks -= 1
            self.num_dangling_output_bytes -= size_bytes
            self.num_dangling_output_rows -= num_rows
        else:
            self.num_pulled_output_blocks -= 1
            self.num_pulled_output_bytes -= size_bytes
            self.num_pulled_output_rows -= num_rows

    def all_outputs_consumed(self) -> bool:
        return (
            self.num_pulled_output_blocks
            + self.num_prebuffered_output_blocks
            + self.num_dangling_output_blocks
            == 0
        )

    def __add__(self, other: "LiveObjStore") -> "LiveObjStore":
        return LiveObjStore(
            num_pulled_output_blocks=self.num_pulled_output_blocks
            + other.num_pulled_output_blocks,
            num_pulled_output_bytes=self.num_pulled_output_bytes
            + other.num_pulled_output_bytes,
            num_pulled_output_rows=self.num_pulled_output_rows
            + other.num_pulled_output_rows,
            num_prebuffered_output_blocks=self.num_prebuffered_output_blocks
            + other.num_prebuffered_output_blocks,
            num_prebuffered_output_bytes=self.num_prebuffered_output_bytes
            + other.num_prebuffered_output_bytes,
            num_prebuffered_output_rows=self.num_prebuffered_output_rows
            + other.num_prebuffered_output_rows,
            num_input_blocks=self.num_input_blocks + other.num_input_blocks,
            num_input_bytes=self.num_input_bytes + other.num_input_bytes,
            num_dangling_output_blocks=self.num_dangling_output_blocks
            + other.num_dangling_output_blocks,
            num_dangling_output_bytes=self.num_dangling_output_bytes
            + other.num_dangling_output_bytes,
            num_dangling_output_rows=self.num_dangling_output_rows
            + other.num_dangling_output_rows,
        )

    def total_bytes(self) -> int:
        return self.input_bytes() + self.output_bytes()

    def total_blocks(self) -> int:
        return self.input_blocks() + self.output_blocks()

    def __repr__(self) -> str:
        num_outputs = (
            self.num_pulled_output_blocks
            + self.num_prebuffered_output_blocks
            + self.num_dangling_output_blocks
        )
        return (
            f"input={self.num_input_blocks}, outputs={num_outputs}, "
            f"prebuffered={self.num_prebuffered_output_blocks}, "
            f"dangling={self.num_dangling_output_blocks}"
        )

    def output_bytes(self) -> int:
        return (
            self.num_pulled_output_bytes
            + self.num_prebuffered_output_bytes
            + self.num_dangling_output_bytes
        )

    def output_blocks(self) -> int:
        return (
            self.num_pulled_output_blocks
            + self.num_prebuffered_output_blocks
            + self.num_dangling_output_blocks
        )

    def input_bytes(self) -> int:
        return self.num_input_bytes

    def input_blocks(self) -> int:
        return self.num_input_blocks


@runtime_checkable
class TracksObjStore(Protocol):
    def on_task_submitted(
        self,
        target_actor_id: LogicalActorId | None,
        task_id: ray.TaskID | None,
        target_node_id: NodeIdStr,
        bms: List[BlockMetadata],
    ):
        ...

    def on_new_output(self, bm: BlockMetadata, task_id: ray.TaskID | None):
        ...

    def on_task_completed(self, task_id: ray.TaskID):
        ...

    def on_output_consumed(self, bm: BlockMetadata, task_id: ray.TaskID | None):
        ...


@dataclass(kw_only=True, slots=True, repr=False, eq=False)
class TaskInfo:
    # None for actorless ops (read/input/limit). They generate
    # output through ray.put, or other means
    actor_id: LogicalActorId | None
    task_id: ray.TaskID | None
    node_id: NodeIdStr
    # Set once the task's generator is exhausted. The task -> actor mapping is
    # only safe to drop once the task has completed AND every output block it
    # produced has been consumed downstream.
    completed: bool = False

    live: LiveObjStore = field(default_factory=LiveObjStore)


@dataclass(kw_only=True, slots=True, repr=False, eq=False)
class ActorInfo:
    # TODO(Justin): Relocation
    node_id: NodeIdStr
    # NOTE: This is supposed to act like a immutable
    # reference, so don't update it in this class
    op_stats: "PerOpStats"

    downscaled: bool

    # Number of tasks currently running on this actor.
    num_tasks_in_flight: int = 0

    live: LiveObjStore = field(default_factory=LiveObjStore)

    def clone(self) -> "ActorInfo":
        return ActorInfo(
            node_id=self.node_id,
            op_stats=self.op_stats,
            downscaled=self.downscaled,
        )

    def to_downscaled(self):
        self.downscaled = True
        self.live.num_dangling_output_blocks = self.live.num_pulled_output_blocks
        self.live.num_dangling_output_bytes = self.live.num_pulled_output_bytes
        self.live.num_dangling_output_rows = self.live.num_pulled_output_rows
        self.live.num_pulled_output_blocks = 0
        self.live.num_pulled_output_bytes = 0
        self.live.num_pulled_output_rows = 0

    def prebuffered_obj_store_estimate(self) -> LiveObjStore:
        """The estimated object store contribution of this actor's
        prebuffered (generated-but-not-yet-pulled) outputs."""
        if self.num_tasks_in_flight == 0:
            return LiveObjStore()

        prebuffered = self.op_stats.estimate_npo_per_actor()
        from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
            ExperimentalAPMO,
        )

        assert isinstance(self.op_stats.op, ExperimentalAPMO)
        bpo = (
            self.op_stats.average_bytes_per_output()
            or self.op_stats.op.actor_pool.average_bytes_per_output()
            or (128 << 20)
        )
        rpo = self.op_stats.average_rows_per_output() or 1
        return LiveObjStore(
            num_prebuffered_output_blocks=prebuffered,
            num_prebuffered_output_bytes=int(prebuffered * bpo),
            num_prebuffered_output_rows=int(prebuffered * rpo),
        )

    def live_obj_store_estimate(self) -> LiveObjStore:
        """This actor's estimated live object store usage: its pulled ``.live``
        plus its prebuffered estimate (see ``prebuffered_obj_store_estimate``)."""

        return self.live + self.prebuffered_obj_store_estimate()


@dataclass(kw_only=True)
class ActorRegistry(TracksObjStore):
    """Registry of the actors tracked (plus their tasks)."""

    # Currently-registered actors. An entry leaves `live` either when the
    # actor is downscaled (moved to `downscaled`).
    actor_info_map: Dict[LogicalActorId, ActorInfo] = field(default_factory=dict)
    # NOTE: Although ExperimentalAPMO should not support downscaling an actor
    # until its inputs are dropped, that concept is something that is bound to
    # Ray Data, meaning it is perfectly feasible to have a downscaled actor with
    # unconsumed, dangling outputs. In rare cases where actors die unexpectedly,
    # we track them here.
    downscaled_actors: Dict[LogicalActorId, ActorInfo] = field(default_factory=dict)

    # When None, technically there is no task , but it's to represent
    # the input operators that use ray.put (InputDataBuffer)
    task_info_map: Dict[ray.TaskID | None, TaskInfo] = field(default_factory=dict)

    def on_register_actor(
        self,
        actor_id: LogicalActorId,
        actor_info: ActorInfo,
    ):
        assert actor_id not in self.actor_info_map
        assert actor_id not in self.downscaled_actors
        self.actor_info_map[actor_id] = actor_info

    def on_deregister_actor(
        self,
        actor_id: LogicalActorId,
    ):
        if actor_id in self.actor_info_map:
            actor_info = self.actor_info_map.pop(actor_id)
            actor_info.to_downscaled()
            self.downscaled_actors[actor_id] = actor_info

    @override
    def on_task_submitted(
        self,
        target_actor_id: LogicalActorId | None,
        task_id: ray.TaskID | None,
        target_node_id: NodeIdStr,
        bms: List[BlockMetadata],
    ):
        live = LiveObjStore()
        if target_actor_id is not None:
            for bm in bms:
                live.on_input_submitted(bm=bm)
                self.actor_info_map[target_actor_id].live.on_input_submitted(bm=bm)
            self.actor_info_map[target_actor_id].num_tasks_in_flight += 1

        self.task_info_map[task_id] = TaskInfo(
            actor_id=target_actor_id,
            task_id=task_id,
            node_id=target_node_id,
            live=live,
        )

    @override
    def on_new_output(self, bm: BlockMetadata, task_id: ray.TaskID | None):
        task_info = self.task_info_map[task_id]
        task_info.live.on_new_output(bm=bm)
        actor_id = bm.get_actor_id() or task_info.actor_id
        if actor_id is not None:
            actor_info = self.get_actor_info(actor_id)
            if actor_info is None:
                # A restarted (relocated) actor can produce retried-task
                # outputs on its new node before a fresh submission
                # re-registers it there. Task/node-level accounting above
                # still applies; there is just no actor entry to charge yet.
                return
            actor_info.live.on_new_output(bm=bm)

    @override
    def on_task_completed(self, task_id: ray.TaskID):
        task_info = self.task_info_map[task_id]
        task_info.completed = True
        if task_info.actor_id is not None:
            actor_info = self.get_actor_info(task_info.actor_id)
            if actor_info is not None:
                actor_info.live.on_input_dropped(task_info=task_info)
                # Relocation resets the actor's in-flight count to 0 ("assume
                # 0 tasks running" -- core retries them), so a pre-relocation
                # task's completion must not drive it negative.
                actor_info.num_tasks_in_flight = max(
                    0, actor_info.num_tasks_in_flight - 1
                )
            # else: the actor relocated off this node (its entry was dropped
            # with the dead node); nothing to charge here.
        task_info.live.on_input_dropped(task_info=task_info)
        if task_info.live.all_outputs_consumed():
            self.task_info_map.pop(task_id, None)

    @override
    def on_output_consumed(self, bm: BlockMetadata, task_id: ray.TaskID | None):
        task_info = self.task_info_map.get(task_id)
        if task_info is None:
            return
        task_info.live.on_output_consumed(bm=bm)
        actor_id = bm.get_actor_id() or task_info.actor_id
        if actor_id is not None:
            actor_info = self.get_actor_info(actor_id)
            if actor_info is not None:
                actor_info.live.on_output_consumed(bm=bm)

        if task_info.live.all_outputs_consumed() and (
            task_info.completed or actor_id is None
        ):
            self.task_info_map.pop(task_info.task_id, None)

    def get_task_info(self, task_id: ray.TaskID | None) -> TaskInfo | None:
        if task_id is None:
            return None
        return self.task_info_map.get(task_id)

    def has_task_info(self, task_id: ray.TaskID | None) -> bool:
        return task_id is not None and task_id in self.task_info_map

    def get_actor_info(self, actor_id: LogicalActorId) -> ActorInfo | None:
        actor_info = self.actor_info_map.get(actor_id, None)
        if actor_info is None:
            return self.downscaled_actors.get(actor_id, None)
        return actor_info

    def task_to_actor(self, task_id: ray.TaskID | None) -> LogicalActorId | None:
        task_info = self.get_task_info(task_id=task_id)
        if task_info is None:
            return None
        return task_info.actor_id


@dataclass(kw_only=True)
class PerScopeStats(TracksObjStore):
    """A scope (an operator, a node, or an operator on a node) that aggregates
    object-store usage and owns an actor registry and a task registry.
    """

    # Registry of this scope's actors (plus the tasks those actors run).
    actors: ActorRegistry = field(default_factory=ActorRegistry)

    cumulative: CumulativeObjStore = field(default_factory=CumulativeObjStore)
    live: LiveObjStore = field(default_factory=LiveObjStore)

    @override
    def on_task_submitted(
        self,
        target_actor_id: LogicalActorId | None,
        task_id: ray.TaskID | None,
        target_node_id: NodeIdStr,
        bms: List[BlockMetadata],
    ):
        self.actors.on_task_submitted(
            target_actor_id=target_actor_id,
            task_id=task_id,
            target_node_id=target_node_id,
            bms=bms,
        )
        for bm in bms:
            remote = is_cross_node_transfer(bm=bm, target_node_id=target_node_id)
            self.cumulative.on_input_submitted(bm=bm, remote=remote)
            self.live.on_input_submitted(bm=bm)

    @override
    def on_new_output(self, bm: BlockMetadata, task_id: ray.TaskID | None):
        if self.actors.has_task_info(task_id):
            self.actors.on_new_output(bm=bm, task_id=task_id)
        self.cumulative.on_new_output(bm=bm)
        self.live.on_new_output(bm=bm)

    @override
    def on_task_completed(
        self,
        task_id: ray.TaskID,
    ):
        task_info = self.actors.get_task_info(task_id=task_id)
        assert task_info is not None, f"ray.TaskID is {task_id}"
        self.live.on_input_dropped(task_info=task_info)
        self.actors.on_task_completed(task_id=task_id)

    @override
    def on_output_consumed(
        self,
        bm: BlockMetadata,
        task_id: ray.TaskID | None,
    ):
        self.live.on_output_consumed(bm=bm)
        if self.actors.has_task_info(task_id):
            self.actors.on_output_consumed(bm=bm, task_id=task_id)

    def actor_ids(self) -> Set[LogicalActorId]:
        """All actor ids tracked by this scope (live + downscaled)."""
        return self.actors.actor_info_map.keys() | self.actors.downscaled_actors.keys()

    def actor_info(self) -> List[ActorInfo]:
        """All actor rows tracked by this scope (live + downscaled)."""
        return [
            *self.actors.actor_info_map.values(),
            *self.actors.downscaled_actors.values(),
        ]

    def average_bytes_per_output(self) -> float | None:
        if self.cumulative.output_bytes.num_samples > 0:
            return self.cumulative.output_bytes.mean
        return None

    def average_rows_per_output(self) -> float | None:
        if self.cumulative.output_rows.num_samples > 0:
            return self.cumulative.output_rows.mean
        return None

    def all_outputs_consumed(self) -> bool:
        return (
            self.live.num_pulled_output_blocks + self.live.num_prebuffered_output_blocks
            == 0
        )

    def live_obj_store_estimate(self) -> LiveObjStore:
        """This scope's estimated live object store usage. It's an estimate
        because we are assuming the prebuffer is full."""
        estimate = LiveObjStore() + self.live
        for actor_info in self.actor_info():
            estimate += actor_info.prebuffered_obj_store_estimate()
        return estimate


@dataclass(kw_only=True)
class PerNodeStats(PerScopeStats):

    node_id: NodeIdStr


@dataclass(kw_only=True)
class PerOpStats(PerScopeStats):

    op: PhysicalOperator
    # num_prebuffered_objects, static
    npo: int

    def estimate_npo_per_actor(self) -> int:
        if self.npo >= 0:
            return self.npo
        # TODO(Justin): Probably can unify parts of OpRuntimeMetrics
        num_blocks = self.op.metrics.num_task_outputs_generated
        num_tasks_submitted = self.op.metrics.num_tasks_submitted
        if num_tasks_submitted > 0:
            return max(1, math.ceil(num_blocks / num_tasks_submitted))
        return 1

    @override
    def live_obj_store_estimate(self) -> LiveObjStore:
        """This scope's estimated live object store usage. It's an estimate
        because we are assuming the prebuffer is full.

        Unlike the base implementation, this doesn't loop over every actor:
        every actor of this op shares the same npo/bpo estimate, so the
        aggregate prebuffer contribution is just
        ``num_active_actors * npo * bpo``. ``num_active_actors`` is read
        directly off the actor pool's own bookkeeping instead of duplicating
        it here."""
        from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
            ExperimentalAPMO,
        )

        if not isinstance(self.op, ExperimentalAPMO):
            # Non task submitters (read/input/limit) have no actor pool and no
            # prebuffer contribution
            return self.live
        num_active_actors = self.op._actor_pool.num_active_actors()
        npo = self.estimate_npo_per_actor()
        bpo = (
            self.average_bytes_per_output()
            or self.op.actor_pool.average_bytes_per_output()
            or (128 << 20)
        )
        rpo = self.average_rows_per_output() or 1
        return self.live + LiveObjStore(
            num_prebuffered_output_blocks=num_active_actors * npo,
            num_prebuffered_output_bytes=int(num_active_actors * npo * bpo),
            num_prebuffered_output_rows=int(num_active_actors * npo * rpo),
        )


@dataclass(kw_only=True)
class PerOpNodeStats(PerOpStats):
    """One operator's slice of one node.

    Subclasses (PerOpStats) rather than (PerScopeStats) so the prebuffer
    inputs methods can be reused. This scope is the operator scope narrowed
    to a single node, and its actor registry holds exactly that operator's
    actors on node_id.
    """

    node_id: NodeIdStr

    @override
    def live_obj_store_estimate(self) -> LiveObjStore:
        """This scope's estimated live object store usage.

        Same shape as ``PerOpStats``' estimate, but the actor count comes from
        ``actor_counts_by_node()`` so only this node's actors contribute. Note
        that count is ACTIVE-only, whereas the op-wide estimate uses the pool's
        ``num_active_actors()``, which also counts terminating actors that still
        have tasks in flight -- so a draining node can be understated for as long
        as its actors take to drain. Both are estimates; neither is authoritative.
        """
        from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
            ActorStatus,
            ExperimentalAPMO,
        )

        if not isinstance(self.op, ExperimentalAPMO):
            # Non task submitters (read/input/limit) have no actor pool and no
            # prebuffer contribution.
            return self.live
        counts = self.op.actor_counts_by_node().get(self.node_id)
        num_active_actors = 0
        if counts:
            num_active_actors = (
                counts[ActorStatus.ACTIVE] + counts[ActorStatus.TERMINATING]
            )
        if num_active_actors == 0:
            return self.live
        npo = self.estimate_npo_per_actor()
        bpo = (
            self.average_bytes_per_output()
            or self.op.actor_pool.average_bytes_per_output()
            or (128 << 20)
        )
        rpo = self.average_rows_per_output() or 1
        return self.live + LiveObjStore(
            num_prebuffered_output_blocks=num_active_actors * npo,
            num_prebuffered_output_bytes=int(num_active_actors * npo * bpo),
            num_prebuffered_output_rows=int(num_active_actors * npo * rpo),
        )


@dataclass(kw_only=True)
class ClusterStats:

    per_node: Dict[NodeIdStr, PerNodeStats] = field(default_factory=dict)
    per_operator: Dict[PhysicalOperator, PerOpStats] = field(default_factory=dict)
    per_op_node: Dict[Tuple[PhysicalOperator, NodeIdStr], PerOpNodeStats] = field(
        default_factory=dict
    )
    actor_to_operator: Dict[LogicalActorId, PerOpStats] = field(default_factory=dict)

    def on_register_node(self, node_id: NodeIdStr) -> PerNodeStats:
        node_stats = self.per_node.get(node_id)
        if node_stats is None:
            node_stats = PerNodeStats(node_id=node_id)
            self.per_node[node_id] = node_stats
        return node_stats

    def on_register_op(self, op: PhysicalOperator | ExperimentalAPMO) -> PerOpStats:
        op_stats = self.per_operator.get(op)
        if op_stats is None:
            op_stats = PerOpStats(
                op=op,
                npo=num_prebuffered_objects(op=op),
            )
            self.per_operator[op] = op_stats
        return op_stats

    def on_register_op_node(
        self, op: PhysicalOperator | ExperimentalAPMO, node_id: NodeIdStr
    ) -> PerOpNodeStats:
        key = (op, node_id)
        op_node_stats = self.per_op_node.get(key)
        if op_node_stats is None:
            op_node_stats = PerOpNodeStats(
                op=op,
                node_id=node_id,
                npo=num_prebuffered_objects(op=op),
            )
            self.per_op_node[key] = op_node_stats
        return op_node_stats

    def on_maybe_register_actor(
        self,
        node_id: NodeIdStr,
        op: ExperimentalAPMO,
        actor_id: LogicalActorId,
    ):

        node_stats = self.on_register_node(node_id=node_id)
        op_stats = self.on_register_op(op=op)
        op_node_stats = self.on_register_op_node(op=op, node_id=node_id)
        actor_info = op_stats.actors.get_actor_info(actor_id=actor_id)
        self.actor_to_operator[actor_id] = op_stats

        if actor_info is None:
            # 1) First sighting -> register the actor on its node.
            actor_info = ActorInfo(
                node_id=node_id,
                op_stats=op_stats,
                downscaled=False,
            )
            op_stats.actors.on_register_actor(actor_id=actor_id, actor_info=actor_info)
            node_stats.actors.on_register_actor(
                actor_id=actor_id, actor_info=actor_info.clone()
            )
            op_node_stats.actors.on_register_actor(
                actor_id=actor_id, actor_info=actor_info.clone()
            )
            return

        old_node_id = actor_info.node_id

        # 2) Already registered
        if old_node_id == node_id:
            return

        # 3) Otherwise, we need to remove the stats from the old node,
        # and migrate them to the new node? NOTE: Currently, the only way
        # an actor can be relocated onto another node is if the node died.
        # If the node dies, then the objects go through lineage reconstruction
        # tasks, and the "prior" running tasks go through task retries. TODO(Justin):
        # we are waiting on Ayush + David changes to make sure this works properly.
        # Since core handles both task FT paths, we will just assume the # of tasks
        # running on that actor is the same (to account for retry tasks). We cannot
        # model lineage reconstruction tasks.
        actor_info.node_id = node_id
        actor_info.live = LiveObjStore()

        # Migrate the per-node bookkeeping to match the op-level view above:
        # drop the actor from the (now-dead) old node's registry entirely and
        # register a fresh clone on the new node. The clone starts with
        # an empty object store (see ActorInfo.clone).
        #
        # The (op, node) scope migrates identically. Only the actor ROW moves;
        # the old scope's own `live`/`cumulative` counters stay put, which is
        # what leaves that node holding the blocks the actor already produced
        # there (dangling or not) -- those objects do not follow the actor.
        for old_scope in (
            self.per_node.get(old_node_id),
            self.per_op_node.get((op_stats.op, old_node_id)),
        ):
            if old_scope is not None:
                old_scope.actors.actor_info_map.pop(actor_id, None)
                old_scope.actors.downscaled_actors.pop(actor_id, None)
        for new_scope in (node_stats, op_node_stats):
            new_scope.actors.on_register_actor(
                actor_id=actor_id, actor_info=actor_info.clone()
            )

        logger.warning(
            f"Actor on old_node_id={old_node_id} is relocating to new_node_id={node_id}. "
            "This can only happen if the node dies or is preempted, so if that did not happen "
            "then please contact the ray team with the full logs."
        )

    def on_task_submitted(
        self,
        actor_id: LogicalActorId,
        task_id: ray.TaskID,
        input_bms: List[BlockMetadata],
    ):
        # Bind this task_id to the actor that will run it.
        op_stats = self.actor_to_operator[actor_id]
        actor_info = op_stats.actors.get_actor_info(actor_id=actor_id)
        assert actor_info is not None
        # TODO(Justin): Relocation
        node_stats = self.per_node[actor_info.node_id]

        op_node_stats = self.on_register_op_node(
            op=op_stats.op, node_id=actor_info.node_id
        )

        # Per-task bookkeeping (in-flight count, registries) -- once per scope.
        for scope in (op_stats, node_stats, op_node_stats):
            scope.on_task_submitted(
                target_actor_id=actor_id,
                task_id=task_id,
                target_node_id=actor_info.node_id,
                bms=input_bms,
            )

    def on_new_output(
        self,
        op: PhysicalOperator,
        bm: BlockMetadata,
        task_id: ray.TaskID | None,
    ):
        node_id = bm.get_node_id()
        node_stats = self.on_register_node(node_id=node_id)
        op_stats = self.on_register_op(op=op)
        op_node_stats = self.on_register_op_node(op=op, node_id=node_id)
        op_stats.on_new_output(bm=bm, task_id=task_id)
        node_stats.on_new_output(bm=bm, task_id=task_id)
        op_node_stats.on_new_output(bm=bm, task_id=task_id)

    def maybe_release_actor(
        self,
        actor_id: LogicalActorId,
        actor_info: ActorInfo,
        op_stats: PerOpStats,
        node_stats: PerNodeStats,
        op_node_stats: PerOpNodeStats,
    ):
        """Drop a downscaled actor's tracking entries once it is fully drained."""
        if (
            actor_info.downscaled
            and actor_info.num_tasks_in_flight == 0
            and actor_info.live.all_outputs_consumed()
        ):
            op_stats.actors.downscaled_actors.pop(actor_id, None)
            node_stats.actors.downscaled_actors.pop(actor_id, None)
            op_node_stats.actors.downscaled_actors.pop(actor_id, None)
            # Drop the reverse map entry too, otherwise it grows unbounded as
            # actors are churned by autoscaling.
            self.actor_to_operator.pop(actor_id, None)

    def on_task_completed(
        self, op: ExperimentalAPMO | PhysicalOperator, task_id: ray.TaskID
    ):
        op_stats = self.per_operator[op]
        task_info = op_stats.actors.get_task_info(task_id=task_id)
        assert task_info is not None
        node_stats = self.per_node[task_info.node_id]
        op_node_stats = self.per_op_node[(op, task_info.node_id)]

        # Each scope's ActorRegistry.on_task_completed sets
        # task_info.completed on its own TaskInfo object.
        op_stats.on_task_completed(task_id=task_id)
        node_stats.on_task_completed(task_id=task_id)
        op_node_stats.on_task_completed(task_id=task_id)

        actor_id = task_info.actor_id
        if actor_id is None:
            return

        actor_info = op_stats.actors.get_actor_info(actor_id=actor_id)
        if actor_info is None:
            # The actor's tracking entries were already dropped (e.g. a fully
            # drained downscaled actor); nothing left to release.
            return
        self.maybe_release_actor(
            actor_id=actor_id,
            actor_info=actor_info,
            op_stats=op_stats,
            node_stats=node_stats,
            op_node_stats=op_node_stats,
        )

    def on_block_consumed(
        self,
        bm: BlockMetadata,
        op: PhysicalOperator,
        task_id: ray.TaskID | None,
    ):
        node_id = bm.get_node_id()
        node_stats = self.per_node[node_id]
        op_stats = self.per_operator[op]
        task_info = op_stats.actors.get_task_info(task_id=task_id)
        actor_id = bm.get_actor_id() or (
            task_info.actor_id if task_info is not None else None
        )

        # Independent per-scope object store + registry GC.
        op_node_stats = self.per_op_node[(op, node_id)]
        op_stats.on_output_consumed(bm=bm, task_id=task_id)
        node_stats.on_output_consumed(bm=bm, task_id=task_id)
        op_node_stats.on_output_consumed(bm=bm, task_id=task_id)

        if actor_id is not None:
            actor_info = op_stats.actors.get_actor_info(actor_id=actor_id)
            if actor_info is not None:
                self.maybe_release_actor(
                    actor_id=actor_id,
                    actor_info=actor_info,
                    op_stats=op_stats,
                    node_stats=self.per_node[actor_info.node_id],
                    op_node_stats=op_node_stats,
                )

    def on_deregister_actor(self, actor_id: LogicalActorId):
        """Mark the actor into the deregistered state. However, we may continue
        to track the downscaled actor, because its outputs may still be alive. In this
        case, we rely on maybe_release_actor to fully stop tracking that actor"""
        op_stats = self.actor_to_operator.get(actor_id)
        actor_info = (
            op_stats.actors.get_actor_info(actor_id=actor_id)
            if op_stats is not None
            else None
        )
        if actor_info is None:
            # The actor never ran a task (e.g. it was scaled down while idle, so
            # it was never registered) or has already been fully released. There
            # is nothing tracked to deregister.
            return

        assert op_stats is not None and op_stats is actor_info.op_stats
        node_id = actor_info.node_id
        node_stats = self.per_node[node_id]
        node_stats.actors.on_deregister_actor(actor_id=actor_id)
        op_stats.actors.on_deregister_actor(actor_id=actor_id)
        op_node_stats = self.per_op_node[(op_stats.op, node_id)]
        op_node_stats.actors.on_deregister_actor(actor_id=actor_id)

        # The actor may already be fully drained (all tasks done, all outputs
        # consumed) at downscale time, in which case nothing else will trigger
        # cleanup -- so attempt it here too.
        self.maybe_release_actor(
            actor_id=actor_id,
            actor_info=actor_info,
            op_stats=op_stats,
            node_stats=node_stats,
            op_node_stats=op_node_stats,
        )

    def clear(self, ops: Set[PhysicalOperator]):
        """Remove all tracked information for this dataset execution."""
        self.per_node.clear()
        self.per_operator.clear()
        self.actor_to_operator.clear()
        self.per_op_node.clear()


class ResourceBankBase(ABC):
    """Interface for the object-store accounting bank that the streaming
    executor and actor pools depend on.

    Extracted so collaborators (and their tests) can depend on the contract
    rather than the concrete ``ResourceBank``. Every method defaults to raising
    ``NotImplementedError``, so a test double can subclass this and override only
    the methods it actually exercises -- an unexpected call then surfaces loudly
    instead of being silently absorbed the way a bare mock would, without each
    fake having to restate every stub.
    """

    def maybe_register_actor(
        self,
        op: ExperimentalAPMO,
        actor_id: LogicalActorId,
        new_node_id: NodeIdStr,
    ) -> None:
        raise NotImplementedError

    def on_task_submitted(
        self,
        actor_id: LogicalActorId,
        task_id: ray.TaskID,
        input_bms: List[BlockMetadata],
    ) -> None:
        raise NotImplementedError

    def on_new_output(
        self,
        op: PhysicalOperator,
        ref: ObjectRef[Block],
        bm: BlockMetadata,
        task_id: ray.TaskID | None = None,
    ) -> None:
        raise NotImplementedError

    def on_task_completed(self, op: ExperimentalAPMO, task_id: ray.TaskID) -> None:
        raise NotImplementedError

    def on_block_consumed(self, input_ref: ObjectRef[Block]) -> None:
        raise NotImplementedError

    def drain_consumed_blocks(self) -> None:
        raise NotImplementedError

    def deregister_actor(self, actor_id: LogicalActorId) -> None:
        raise NotImplementedError

    def get_actor_id_from_task_id(
        self, op: PhysicalOperator, task_id: ray.TaskID | None
    ) -> LogicalActorId | None:
        raise NotImplementedError

    def node_capacity(self, node_id: NodeIdStr) -> ExecutionResources:
        raise NotImplementedError

    def set_node_view(self, node_view: Dict[NodeIdStr, ExecutionResources]) -> None:
        raise NotImplementedError

    def live_object_store(
        self,
        op: PhysicalOperator | None = None,
        node_id: NodeIdStr | None = None,
        actor_id: LogicalActorId | None = None,
    ) -> LiveObjStore:
        raise NotImplementedError

    def cumulative_object_store(
        self,
        op: PhysicalOperator | None = None,
        node_id: NodeIdStr | None = None,
    ) -> CumulativeObjStore:
        raise NotImplementedError

    def finalize_dataset(self, ops: Set[PhysicalOperator]) -> None:
        raise NotImplementedError

    def node_ids(self) -> Set[NodeIdStr]:
        raise NotImplementedError

    def op_node_ids(self) -> Set[Tuple[PhysicalOperator, NodeIdStr]]:
        raise NotImplementedError


class ResourceBank(ResourceBankBase):
    """In this class, we model 3 entities:
        - operators
        - nodes
        - operators on a node (the join of the two)
    so we can provide statistics for each entity. Supports only actors.

    Each instance is scoped to a single dataset execution (owned by the
    streaming executor).
    """

    def __init__(self):
        self.stats: ClusterStats = ClusterStats()
        # Key by ObjectRef hex string so ResourceBank accounting never keeps an
        # extra Python ObjectRef alive and delays Ray Core ref-count cleanup.
        self.live_block_refs: Dict[str, UniqueBlockMetadata] = {}
        # This class should only be double-threaded, one thread for the
        # streaming executor loop, the other is the consumer thread (get_output_blocking)
        # Rather than creating a lock to guard each public method of the ResourceBank,
        # I decided to only protect the consumed_refs, since it's the only method that
        # should be called in the consumer thread.
        self._consumed_refs: List[str] = []
        self._consumed_refs_lock: threading.Lock = threading.Lock()
        # To have 1 source of truth of the shape of the cluster, we set this value
        # at the start of each scheduling loop.
        self._node_view: Optional[Dict[NodeIdStr, ExecutionResources]] = None

    @override
    def maybe_register_actor(
        self,
        op: ExperimentalAPMO,
        actor_id: LogicalActorId,
        new_node_id: NodeIdStr,
    ) -> None:
        """Register ``actor_id` on first sighting, or relocate it to ``new_node_id``.

        First sighting -- record the actor's existence on ``new_node_id``, seed
        its stats.

        Already registered -- TODO(Justin): We currently do not support actor's relocated
        on different nodes.

        A same/unknown node is a no-op.
        """
        if not actor_only_backend_enabled():
            raise NotImplementedError

        self.stats.on_maybe_register_actor(
            node_id=new_node_id,
            actor_id=actor_id,
            op=op,
        )

    @override
    def on_task_submitted(
        self,
        actor_id: LogicalActorId,
        task_id: ray.TaskID,
        input_bms: List[BlockMetadata],
    ) -> None:
        """Keep track of which task belongs to which actor."""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        self.stats.on_task_submitted(
            actor_id=actor_id,
            task_id=task_id,
            input_bms=input_bms,
        )

    @override
    def on_new_output(
        self,
        op: PhysicalOperator,
        ref: ObjectRef[Block],
        bm: BlockMetadata,
        task_id: ray.TaskID | None = None,
    ):
        """Called whenever there is new output from a task."""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        ref_key = ref.hex()
        assert ref_key not in self.live_block_refs
        self.live_block_refs[ref_key] = UniqueBlockMetadata(
            op=op,
            bm=bm,
            task_id=task_id,
        )
        self.stats.on_new_output(op=op, bm=bm, task_id=task_id)

    @override
    def on_task_completed(self, op: ExperimentalAPMO, task_id: ray.TaskID):
        """Called whenever task_id is completed"""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        op_stats = self.stats.per_operator.get(op)
        if op_stats is None or not op_stats.actors.has_task_info(task_id):
            return
        self.stats.on_task_completed(op=op, task_id=task_id)

    @override
    def on_block_consumed(self, input_ref: ObjectRef[Block]):
        """Report that ``input_ref`` is no longer needed downstream.

        This is the only ResourceBank entrypoint invoked off the executor
        thread (by consumer threads in ``OpState.get_output_blocking``), so it
        does the minimum possible. The actual updates are applied later
        on the executor thread in ``drain_consumed_blocks``.

        Idempotent: refs may be reported more than once (e.g. an early withdrawal
        when a downstream actor takes ownership of a primary input at submit time,
        and again at task-done). Duplicate / unknown refs are ignored when drained.
        """
        if not actor_only_backend_enabled():
            raise NotImplementedError

        with self._consumed_refs_lock:
            self._consumed_refs.append(input_ref.hex())

    @override
    def drain_consumed_blocks(self) -> None:
        """Apply buffered block-consumption reports on the executor thread.

        MUST only be called from the executor (scheduling-loop) thread. It swaps
        out the consumed-refs buffer under the queue lock, then applies each
        report with no stats lock -- the executor is the sole mutator of
        ``stats`` and ``live_block_refs``.
        """
        if not actor_only_backend_enabled():
            raise NotImplementedError

        with self._consumed_refs_lock:
            if not self._consumed_refs:
                return
            refs = self._consumed_refs
            self._consumed_refs = []

        for ref_key in refs:
            unique_bm = self.live_block_refs.pop(ref_key, None)
            if unique_bm is None:
                continue
            self.stats.on_block_consumed(
                bm=unique_bm.bm, op=unique_bm.op, task_id=unique_bm.task_id
            )

    @override
    def deregister_actor(self, actor_id: LogicalActorId) -> None:
        """Drop a downscaled actor from the live indexes."""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        self.stats.on_deregister_actor(actor_id=actor_id)

    @override
    def get_actor_id_from_task_id(
        self, op: PhysicalOperator, task_id: ray.TaskID | None
    ) -> LogicalActorId | None:
        """Given a task_id, returns the actor_id that generated that task_id"""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        op_stats = self.stats.per_operator.get(op)
        if op_stats is None:
            return None
        return op_stats.actors.task_to_actor(task_id=task_id)

    @override
    def set_node_view(self, node_view: Dict[NodeIdStr, ExecutionResources]) -> None:
        """Set this tick's per-node allocation snapshot (source of truth).

        Called once at the top of each scheduling-loop step with this dataset's
        coordinator allocation, so resource_bank the sizer agree on the
        node view for the whole tick.
        """
        self._node_view = node_view

    @override
    def node_capacity(self, node_id: NodeIdStr) -> ExecutionResources:
        if not actor_only_backend_enabled():
            raise NotImplementedError

        if self._node_view is not None:
            res = self._node_view.get(node_id)
            if res is not None:
                return res

        # Missing from coordinator allocation (cold cache, soft affinity, or
        # untargeted initial_size actors) or sizer-off: use a fresh cluster
        # snapshot so a just-joined landing node is visible.
        return get_alive_nodes_uncached().get(node_id, ExecutionResources.zero())

    @override
    def live_object_store(
        self,
        op: PhysicalOperator | None = None,
        node_id: NodeIdStr | None = None,
        actor_id: LogicalActorId | None = None,
    ) -> LiveObjStore:
        """Returns the object store based on the scope.
        Will go from most scoped (actor_id) to least scoped (op).
        This function does not validate requests (ie actor_id on wrong node_id)"""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        if actor_id is not None:
            op_stats = self.stats.actor_to_operator.get(actor_id)
            # Protect against pending actors.
            if op_stats is None:
                return LiveObjStore()
            actor_info = op_stats.actors.get_actor_info(actor_id=actor_id)
            if actor_info is None:
                return LiveObjStore()
            return actor_info.live_obj_store_estimate()
        elif node_id is not None and op is not None:
            op_node_stats = self.stats.per_op_node.get((op, node_id))
            if op_node_stats is None:
                return LiveObjStore()
            return op_node_stats.live_obj_store_estimate()
        elif node_id is not None:
            node_stats = self.stats.per_node.get(node_id)
            if node_stats is None:
                return LiveObjStore()
            return node_stats.live_obj_store_estimate()
        elif op is not None:
            if op in self.stats.per_operator:
                # TODO(justin): May need to have this check for entities too.
                return self.stats.per_operator[op].live_obj_store_estimate()
            return LiveObjStore()
        else:
            assert op is None and node_id is None and actor_id is None
            bytes_output = LiveObjStore()
            for _, stats in self.stats.per_operator.items():
                bytes_output += self.live_object_store(op=stats.op)
            return bytes_output

    @override
    def cumulative_object_store(
        self,
        op: PhysicalOperator | None = None,
        node_id: NodeIdStr | None = None,
    ) -> CumulativeObjStore:
        """Returns the object store based on the scope.
        Will go from most scoped (actor_id) to least scoped (op or node_id).
        This function does not validate requests (ie actor_id on wrong node_id)"""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        if node_id is not None and op is not None:
            op_node_stats = self.stats.per_op_node.get((op, node_id))
            if op_node_stats is None:
                return CumulativeObjStore()
            return op_node_stats.cumulative
        elif node_id is not None:
            node_stats = self.stats.per_node.get(node_id)
            if node_stats is None:
                return CumulativeObjStore()
            # TODO(Justin): Should we clone?
            return node_stats.cumulative
        elif op is not None:
            op_stats = self.stats.per_operator.get(op)
            if op_stats is None:
                return CumulativeObjStore()
            # TODO(Justin): Should we clone?
            return op_stats.cumulative
        else:
            assert op is None and node_id is None
            bytes_output = CumulativeObjStore()
            for _, stats in self.stats.per_operator.items():
                bytes_output += stats.cumulative
            return bytes_output

    @override
    def finalize_dataset(self, ops: Set[PhysicalOperator]) -> None:
        """Clear the space in the ResourceBank"""
        if not actor_only_backend_enabled():
            raise NotImplementedError

        # Apply any buffered consumptions before tearing down. Runs on the
        # executor thread, same as everything below.
        self.drain_consumed_blocks()
        # ResourceBank is scoped to one dataset execution, so finalization drops
        # all tracked stats and refs regardless of which operators are passed in.
        self.stats.clear(ops=ops)
        self.live_block_refs = {}
        # Drop any reports that arrived after the drain above (e.g. a straggler
        # consumer); they reference now-released refs and would otherwise
        # accumulate unbounded across executions.
        with self._consumed_refs_lock:
            self._consumed_refs = []

    @override
    def node_ids(self) -> Set[NodeIdStr]:
        return set(self.stats.per_node.keys())

    @override
    def op_node_ids(self) -> Set[Tuple[PhysicalOperator, NodeIdStr]]:
        """Every (operator, node) pair that has been observed this execution.

        Sparse: a pair appears once the operator first places an actor or lands
        an output on that node, and then stays for the rest of the execution.
        """
        return set(self.stats.per_op_node.keys())


def is_cross_node_transfer(bm: BlockMetadata, target_node_id: str) -> bool:
    """If the block meta were to transfer to actor_node_id, would it be
    a locality hit? Returns True if non-local, False for local"""
    if RAY_DATA_MOVE_SEMANTIC:
        return False
    return bm.get_node_id() != target_node_id


def _prebuffer_blocks_from_object_cap(raw: Any) -> int:
    """Convert a Ray generator backpressure cap from object units to blocks.

    Both ``_actor_generator_backpressure_num_objects`` and
    ``_generator_backpressure_num_objects`` are stored in Ray Core as object
    counts. Map actors with grouped yields emit one logical block as
    ``objects_per_yield`` objects (block + metadata).
    """
    if raw is None:
        return -1
    raw = int(raw)
    if raw < 0:
        return -1
    objects_per_yield = CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD
    assert raw % objects_per_yield == 0, (
        f"generator backpressure cap ({raw}) must be a multiple of "
        f"{objects_per_yield} when grouped yields are enabled"
    )
    return raw // objects_per_yield


def num_prebuffered_objects(op: PhysicalOperator | ExperimentalAPMO) -> int:
    from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
        ExperimentalAPMO,
    )

    ray_remote_args = {}
    ray_actor_task_remote_args = {}
    max_concurrency = 0
    if isinstance(op, ExperimentalAPMO):
        ray_remote_args = op._ray_remote_args
        ray_actor_task_remote_args = op._ray_actor_task_remote_args
        max_concurrency = op._actor_pool.max_actor_concurrency()

    agbno = _prebuffer_blocks_from_object_cap(
        ray_remote_args.get("_actor_generator_backpressure_num_objects")
    )
    gbno = _prebuffer_blocks_from_object_cap(
        ray_actor_task_remote_args.get("_generator_backpressure_num_objects")
    )

    if agbno >= 0 and gbno >= 0:
        return min(agbno, gbno * max_concurrency)
    elif agbno >= 0:
        return agbno
    elif gbno >= 0:
        return gbno * max_concurrency

    return -1
