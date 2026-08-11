import time
from typing import Optional, Tuple, Type

from typing_extensions import override

from ray.data._internal.cluster_autoscaler import (
    ClusterAutoscaler,
    DefaultAutoscalingCoordinator,
    create_cluster_autoscaler,
)
from ray.data._internal.cluster_autoscaler.resource_utilization_gauge import (
    RollingLogicalUtilizationGauge,
)
from ray.data._internal.execution.execution_flags import (
    ENABLE_OPERATOR_SIZER,
)
from ray.data._internal.execution.interfaces import ExecutionResources, PhysicalOperator
from ray.data._internal.execution.operators.base_physical_operator import (
    AllToAllOperator,
)
from ray.data._internal.execution.operators.hash_aggregate import (
    HashAggregateOperator,
)
from ray.data._internal.execution.operators.hash_shuffle import (
    HashShuffleOperator,
)
from ray.data._internal.execution.operators.join import JoinOperator
from ray.data._internal.execution.operators.shuffle_operators.shuffle_map_operator import (
    ShuffleMapOp,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_reduce_operator import (
    ShuffleReduceOp,
)
from ray.data._internal.execution.operators.zip_operator import ZipOperator
from ray.data._internal.execution.streaming_executor import (
    StreamingExecutor,
)
from ray.data._internal.execution.streaming_executor_state import (
    OpState,
    Topology,
)
from ray.data._internal.experimental.execution.actor_only_metrics import (
    ActorOnlyMetrics,
)
from ray.data._internal.experimental.execution.actor_only_resource_reporter import (
    ActorOnlyResourceReporter,
)
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (
    ExperimentalAPMO,
)
from ray.data._internal.experimental.execution.sizer.operator_sizer import (
    OperatorSizer,
)
from ray.data._internal.experimental.execution.streaming_executor_state import (
    detect_if_idle,
    process_completed_tasks,
)

__all__ = ["ExperimentalStreamingExecutor"]

# Shuffle / all-to-all physical ops that the actor-only backend does not support.
_UNSUPPORTED_ACTOR_ONLY_OPS: Tuple[Type[PhysicalOperator], ...] = (
    AllToAllOperator,
    HashShuffleOperator,
    HashAggregateOperator,
    JoinOperator,
    ShuffleMapOp,
    ShuffleReduceOp,
    ZipOperator,
)


class ExperimentalStreamingExecutor(StreamingExecutor):
    """Actor-only-backend streaming executor. See module docstring."""

    MIN_IDLE_DETECTION_INTERVAL_S: float = 0.1
    MAX_IDLE_DETECTION_INTERVAL_S: float = 2.0
    PROGRESS_REFRESH_INTERVAL_S: float = 1.0
    _sizer: Optional[OperatorSizer] = None
    # Actor-only resource view shared by the cluster autoscaler's utilization
    # gauge and ``_report_current_usage``. Set in ``bootstrap`` when the sizer
    # is enabled; ``None`` on the sizer-off fallback path.
    _resource_reporter: Optional[ActorOnlyResourceReporter] = None
    _actor_only_metrics: Optional[ActorOnlyMetrics] = None

    def _validate_actor_only_supported_topology(self) -> None:
        """Reject configs / operators the actor-only backend cannot run."""
        assert self._topology is not None
        if self._options.preserve_order:
            raise ValueError(
                "preserve_order=True is not supported for actor only backend"
            )
        for op in self._topology:
            if isinstance(op, _UNSUPPORTED_ACTOR_ONLY_OPS):
                raise ValueError(
                    f"{type(op).__name__} is not supported when actor only "
                    "backend is enabled"
                )

    @override
    def _create_cluster_autoscaler(self) -> ClusterAutoscaler:
        """Build the autoscaler over an actor-only resource reporter.

        Same factory (and so the same RAY_DATA_CLUSTER_AUTOSCALER selection) as
        the classic path, but with a coordinator shared with the sizer
        """
        if not ENABLE_OPERATOR_SIZER:
            return super()._create_cluster_autoscaler()

        assert self._topology is not None
        assert self._resource_bank is not None
        coordinator = DefaultAutoscalingCoordinator(
            requester_id=self._dataset_id,
            subcluster_selector=self._data_context.execution_options.label_selector,
        )
        # Built here (rather than in ``bootstrap``) because the autoscaler's
        # utilization gauge reads through it. ``bootstrap`` hands the same
        # instance to the sizer.
        self._resource_reporter = ActorOnlyResourceReporter(
            self._topology, coordinator, self._resource_bank
        )
        return create_cluster_autoscaler(
            self._topology,
            self._resource_manager,
            self._data_context,
            execution_id=self._dataset_id,
            autoscaling_coordinator=coordinator,
            resource_utilization_calculator=RollingLogicalUtilizationGauge(
                self._resource_reporter, execution_id=self._dataset_id
            ),
        )

    @override
    def bootstrap(self):
        self._validate_actor_only_supported_topology()
        self._since_last_detection: float = 0.0
        self._since_last_progress_refresh: float = 0.0
        self._idle_detection_interval_s: float = 1.0
        if ENABLE_OPERATOR_SIZER:
            assert self._topology is not None
            # ``_create_cluster_autoscaler`` already built the shared coordinator
            # and this reporter over it. The sizer reads allocations through the
            # reporter; the cluster autoscaler owns the coordinator's
            # request/cancel lifecycle, so the sizer never touches it directly.
            assert self._resource_reporter is not None
            self._sizer = OperatorSizer(
                dataset_id=self._dataset_id,
                resource_reporter=self._resource_reporter,
            )

            request = self._sizer.bootstrap(topology=self._topology)
            if request is not None:
                self._sizer.scale(request=request)
            # Scale up operators to their min size
            self._sizer.initial_sizing_request(topology=self._topology)

        self._actor_only_metrics = ActorOnlyMetrics(
            dataset_id=self._dataset_id, update_interval=self.UPDATE_METRICS_INTERVAL_S
        )
        # TODO(Justin): This is not clean, clean it up later.
        for op, op_state in self._topology.items():
            if isinstance(op, ExperimentalAPMO):
                op.set_actor_only_metrics(self._actor_only_metrics, op_state.op_tag())

    @override
    def _scheduling_loop_step(self, topology: Topology) -> bool:
        """Run one step of the actor-only scheduling loop.

        Args:
            topology: The execution topology to schedule.

        Returns:
            True if we should continue running the scheduling loop.
        """
        step_t0 = time.perf_counter()
        assert self._resource_bank is not None
        if self._resource_reporter is not None:
            # Establish a single source of truth for per-node capacity
            self._resource_bank.set_node_view(
                self._resource_reporter.get_reserved_resources_by_node()
            )
        self._resource_bank.drain_consumed_blocks()
        # TODO(Prasad): We need to remove this for actor-only backend. It's here now
        # because we don't have an operator sizer yet.
        if not ENABLE_OPERATOR_SIZER:
            self._resource_manager.update_usages()

        # Note: calling process_completed_tasks() is expensive since it incurs
        # ray.wait() overhead, so make sure to allow multiple dispatch per call
        # for greater parallelism.
        num_errored_blocks = process_completed_tasks(
            topology=topology,
            max_errored_blocks=self._max_errored_blocks,
            metadata_fetcher=self._metadata_fetcher,
        )

        now = time.perf_counter()
        if now - self._since_last_detection >= self._idle_detection_interval_s:
            assert self._output_node is not None
            idle_op = detect_if_idle(
                output_operator=self._output_node[0],
                resource_bank=self._resource_bank,
            )
            if idle_op is not None:
                op_state = self._topology[idle_op]
                if self._actor_only_metrics is not None:
                    self._actor_only_metrics.record_output_limit_upgrade(
                        op_state.op_tag()
                    )
                self._idle_detection_interval_s = max(
                    self.MIN_IDLE_DETECTION_INTERVAL_S,
                    self._idle_detection_interval_s / 2,
                )
            else:
                self._idle_detection_interval_s = min(
                    self.MAX_IDLE_DETECTION_INTERVAL_S,
                    self._idle_detection_interval_s * 2,
                )
            self._since_last_detection = now

        if self._max_errored_blocks > 0:
            self._max_errored_blocks -= num_errored_blocks
        self._num_errored_blocks += num_errored_blocks

        sizer_timing = None
        if ENABLE_OPERATOR_SIZER:
            # t0 starts BEFORE observe(): signal sampling is optimizer work, so
            # timing it with the optimizer keeps all of it inside a named phase.
            t0 = time.perf_counter()
            # Sample the optimizer's per-op signals (cheap; recording is
            # gated internally).
            self._sizer.observe(topology=self._topology)
            # Optimizer phase first: cross-operator corrections apply their
            # own fully-placed requests; their claims enter committed
            # accounting, so the sizing passes below already see them.
            self._sizer.optimize_pipeline(topology=self._topology)
            t1 = time.perf_counter()
            how_many = self._sizer.scale_how_many(topology=self._topology)
            t2 = time.perf_counter()
            where = self._sizer.scale_where(request=how_many)
            t2b = time.perf_counter()
            self._sizer.scale(request=where)
            t3 = time.perf_counter()
            sizer_timing = (how_many, where, t1 - t0, t2 - t1, t2b - t2, t3 - t2b)
        else:
            self._resource_manager.update_usages()

        self._resource_bank.drain_consumed_blocks()
        self._launch_tasks()

        self._actor_only_metrics.maybe_update(
            topology=topology,
            resource_bank=self._resource_bank,
        )

        self._report_current_usage()

        # Cluster (node-level) autoscaling runs in both paths. In the sizer
        # path this is the RAY_DATA_CLUSTER_AUTOSCALER-selected autoscaler wired
        # up in bootstrap(), which shares the sizer's coordinator.
        self._cluster_autoscaler.try_trigger_scaling()
        if not ENABLE_OPERATOR_SIZER:
            # Actor-pool autoscaling: sizer-off fallback only. With
            # ENABLE_OPERATOR_SIZER, OperatorSizer.scale_where/scale own actor
            # placement (NodeAffinity).
            self._actor_autoscaler.try_trigger_scaling()

        if sizer_timing is not None:
            # e2e ~ the whole step (everything but _finalize, negligible), so it
            # covers non-sizer work too and the named phases don't sum to it.
            how_many, where, optimize_s, how_many_s, where_s, apply_s = sizer_timing
            self._sizer.record_tick(
                how_many,
                where,
                optimize_s=optimize_s,
                how_many_s=how_many_s,
                where_s=where_s,
                apply_s=apply_s,
                e2e_s=time.perf_counter() - step_t0,
            )

        return self._finalize_scheduling_loop_step(topology)

    def _launch_tasks(self):
        """Dispatch as many operators as we can for completed tasks."""
        assert self._topology is not None
        for op in self._topology:
            if not isinstance(op, ExperimentalAPMO):
                continue
            while op.can_submit_task():
                op.launch_task()
                now = time.perf_counter()
                if (
                    now - self._since_last_progress_refresh
                    >= self.PROGRESS_REFRESH_INTERVAL_S
                ):
                    self._refresh_progress_manager(self._topology)
                    self._since_last_progress_refresh = now

    @override
    def shutdown(self, force: bool, exception: Optional[Exception] = None):
        should_finalize = (
            self._topology is not None
            and self._execution_started
            and not self._shutdown
        )
        try:
            if (
                self._actor_only_metrics is not None
                and self._topology is not None
                and self._resource_bank is not None
            ):
                self._actor_only_metrics.maybe_update(
                    topology=self._topology,
                    resource_bank=self._resource_bank,
                    force=True,
                )
            return super().shutdown(force, exception)
        finally:
            if should_finalize:
                assert self._resource_bank is not None
                self._resource_bank.finalize_dataset(ops=set(self._topology.keys()))
            # super().shutdown() calls cluster_autoscaler.on_executor_shutdown(),
            # which cancels the shared coordinator allocation (owned by the
            # autoscaler, not the sizer). The sizer holds no coordinator
            # lifecycle, so we just drop it.
            self._sizer = None

    @override
    def _get_operator_progress_extra(self, op_state: OpState) -> str:
        assert self._resource_bank is not None
        return str(self._resource_bank.live_object_store(op=op_state.op))

    @override
    def _report_current_usage(self) -> None:
        # running_usage is the amount of resources that have been requested but
        # not necessarily available
        # TODO(sofian) https://github.com/ray-project/ray/issues/47520
        # We need to split the reported resources into running, pending-scheduling,
        # pending-node-assignment.
        assert self._topology is not None
        assert self._resource_bank is not None

        # Block-count summary for the status line (needed on both paths).
        running_usage = ExecutionResources.zero()
        pending_usage = ExecutionResources.zero()
        num_inputs: int = 0
        num_pulled_outputs: int = 0
        num_prebuffered_outputs: int = 0
        num_dangling_outputs: int = 0
        for op in self._topology:
            object_store = self._resource_bank.live_object_store(op=op)
            num_inputs += object_store.num_input_blocks
            num_pulled_outputs += object_store.num_pulled_output_blocks
            num_prebuffered_outputs += object_store.num_prebuffered_output_blocks
            num_dangling_outputs += object_store.num_dangling_output_blocks
            if self._resource_reporter is None:
                total_object_store = object_store.total_bytes()
                running_usage = running_usage.add(
                    ExecutionResources(object_store_memory=total_object_store)
                )
                running_usage = running_usage.add(op.running_logical_usage())
                pending_usage = pending_usage.add(op.pending_logical_usage())

        if self._resource_reporter is not None:
            # Sizer path: usage/limits from the shared reporter, so this status
            # line and the cluster autoscaler's scale-up signal agree by
            # construction and limits reflect the coordinator allocation.
            running_usage = self._resource_reporter.get_global_usage()
            pending_usage = self._resource_reporter.get_global_pending_usage()
            limits = self._resource_reporter.get_global_limits()
        else:
            limits = self._resource_manager.get_global_limits()

        resources_status = (
            f"Active & requested resources: "
            f"{running_usage.cpu:.4g}/{limits.cpu:.4g} CPU, "
        )
        if running_usage.memory > 0:
            resources_status += (
                f"{running_usage.memory_str()}/{limits.memory_str()} memory, "
            )
        if running_usage.gpu > 0:
            resources_status += f"{running_usage.gpu:.4g}/{limits.gpu:.4g} GPU, "

        num_outputs = (
            num_pulled_outputs + num_prebuffered_outputs + num_dangling_outputs
        )
        resources_status += (
            f"{running_usage.object_store_memory_str()}/"
            f"{limits.object_store_memory_str()} object store "
            f"(input={num_inputs}, outputs={num_outputs}, "
            f"prebuffered={num_prebuffered_outputs}, dangling={num_dangling_outputs})"
        )

        # Only include pending section when there are pending resources.
        pending_parts = []
        if pending_usage.cpu:
            pending_parts.append(f"{pending_usage.cpu:.4g} CPU")
        if pending_usage.memory:
            pending_parts.append(f"{pending_usage.memory_str()} memory")
        if pending_usage.gpu:
            pending_parts.append(f"{pending_usage.gpu:.4g} GPU")
        if pending_parts:
            resources_status += f" (pending: {', '.join(pending_parts)})"

        self._progress_manager.update_total_resource_status(resources_status)
