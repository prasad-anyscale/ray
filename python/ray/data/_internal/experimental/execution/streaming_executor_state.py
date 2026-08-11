import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import ray
import ray.exceptions
from ray.data._internal.execution.interfaces.physical_operator import (
    DataOpTask,
    MetadataOpTask,
    OpTask,
    PhysicalOperator,
    Waitable,
)
from ray.data._internal.execution.metadata_fetcher import MetadataFetcher
from ray.data._internal.execution.operators.base_physical_operator import (
    InternalQueueOperatorMixin,
)
from ray.data._internal.execution.resource_bank import ResourceBankBase
from ray.data._internal.execution.streaming_executor_state import (
    WAIT_FOR_TASK_COMPLETION_TIMEOUT_S,
    OpState,
    Topology,
    _handle_data_op_task_exception,
)
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (
    ExperimentalAPMO,
)

logger = logging.getLogger(__name__)

__all__ = ["process_completed_tasks"]


def process_completed_tasks(
    topology: Topology,
    max_errored_blocks: int,
    metadata_fetcher: MetadataFetcher,
) -> int:
    """Process any newly completed tasks. To update operator
    states, call `update_operator_states()` afterwards.

    Args:
        topology: The topology of operators.
        max_errored_blocks: Max number of errored blocks to allow,
            unlimited if negative.
        metadata_fetcher: Resolves pulled (block_ref, meta_ref) pairs into
            emitted RefBundles. The threaded fetcher defers metadata fetches to
            a background thread (emitting in per-op order as they become ready);
            the inline fetcher emits synchronously.
    Returns:
        The number of errored blocks.
    """

    # All active tasks, keyed by their waitables.
    active_tasks: Dict[Waitable, Tuple[OpState, OpTask]] = {}
    for op, state in topology.items():
        for task in op.get_active_tasks():
            active_tasks[task.get_waitable()] = (state, task)

    # Process completed Ray tasks and notify operators.
    num_errored_blocks = 0
    if active_tasks:
        ready, _ = ray.wait(
            list(active_tasks.keys()),
            num_returns=len(active_tasks),
            fetch_local=False,
            timeout=WAIT_FOR_TASK_COMPLETION_TIMEOUT_S,
        )

        # Organize tasks by the operator they belong to, and sort them by task index.
        # So that we'll process them in a deterministic order.
        # This is because backpressure policies may limit the number of blocks to read
        # per operator. In this case, we want to have fewer tasks finish quickly and
        # yield resources, instead of having all tasks output blocks together.
        ready_tasks_by_op: DefaultDict[OpState, List[OpTask]] = defaultdict(list)
        for ref in ready:
            state, task = active_tasks[ref]
            ready_tasks_by_op[state].append(task)

        for op, state in reversed(topology.items()):
            ready_tasks = ready_tasks_by_op[state]
            if ready_tasks:
                assert isinstance(op, ExperimentalAPMO)
                op_data_tasks: List[DataOpTask] = []
                # Process MetadataOpTasks (actor readiness) before DataOpTasks
                # so a block-level abort cannot leave pending actors stranded.
                try:
                    for task in ready_tasks:
                        if not isinstance(task, MetadataOpTask):
                            continue
                        try:
                            task.on_task_finished()
                        except Exception:
                            # Actor-readiness ref (get_location) from
                            # _start_actor. A RayError here means the actor
                            # failed to START (e.g. its NodeAffinity target
                            # died). pending_to_running already cleaned up the
                            # pending state — don't crash the loop; the next
                            # sizing tick re-places.
                            logger.warning(
                                f"An actor of {op.name} failed to start; "
                                "reconciling on the next sizing tick.",
                                exc_info=True,
                            )

                    for task in ready_tasks:
                        if not isinstance(task, DataOpTask):
                            continue
                        try:
                            soft_upper_bound = op.task_pull_request(
                                task_idx=task.task_index()
                            )
                            _response = task.on_data_ready(
                                max_to_read=soft_upper_bound,
                                metadata_fetcher=metadata_fetcher,
                            )
                            op_data_tasks.append(task)
                        except Exception as e:
                            # Block-level / task failure: account against
                            # max_errored_blocks (may re-raise and abort).
                            num_errored_blocks = _handle_data_op_task_exception(
                                e,
                                op.name,
                                max_errored_blocks,
                                num_errored_blocks,
                            )
                finally:
                    # Hand this op's just-deferred pairs to the fetcher, and register
                    # any end-of-stream tasks for a postponed done-callback (a no-op
                    # in inline mode, where the pairs already emitted above). In a
                    # ``finally`` so a thrown error can't strand pairs already
                    # deferred into the fetcher this iteration.
                    metadata_fetcher.submit(state, op_data_tasks)

    # Emit whatever's ready, in per-op order, then fire any postponed done
    # callbacks — UNCONDITIONALLY, even when there are no active tasks this
    # iteration. Pairs deferred in earlier iterations (their tasks may already
    # be gone) can still have metadata land later; gating this on `active_tasks`
    # would strand them and stall output forever. Deferred metadata-fetch
    # failures go through the same `max_errored_blocks` accounting as inline
    # `on_data_ready` errors. (Inline mode returns nothing here.)
    for (
        failed_op_name,
        fetch_exc,
    ) in metadata_fetcher.emit_ready_and_fire_done_callbacks():
        num_errored_blocks = _handle_data_op_task_exception(
            fetch_exc,
            failed_op_name,
            max_errored_blocks,
            num_errored_blocks,
        )

    # Pull any operator outputs into the streaming op state.
    for op, op_state in topology.items():
        while op.has_next():
            op_state.add_output(op.get_next())

        if not op.has_completed():
            # Only pull outputs into this op if it's not completed
            # Limit, for example, can be completed early, so we
            # don't want to pull anymore inputs into it.
            while op_state.has_pending_input_bundles():
                op_state.move_input_into_op()

        # TODO(Prasad): Please remove this line, it's only here
        # because the default actor pool autoscaler won't scale up
        # unless this is True, so we can't compare against a baseline.
        op_state._scheduling_status.under_resource_limits = True

    return num_errored_blocks


def detect_if_idle(
    output_operator: PhysicalOperator, resource_bank: ResourceBankBase
) -> PhysicalOperator | None:
    """Detect whether output backpressure is too aggressive and, if so, bump a
    single operator's output limit.

    Data flows producer -> consumer, so we DFS upstream from the output operator.

    Only an ``ExperimentalAPMO`` can be output backpressured, so it's the only
    operator we ever upgrade. We upgrade when it's backpressured AND either:

    (a) a downstream consumer is starved and there are no downstream tasks running
        to relieve it (upgrading is the only way to make progress), or
    (b) it doesn't even have budget for its prebuffered outputs -- a low
        object-store stall that waiting can't fix.

    At most one operator is upgraded per call.

    Currently the only operators that require input are the ExperimentalAMPO and OutputSplitter
    both can hold input until a rows requirement is satisfied.
    """
    # Each stack entry is (op, downstream_rows_needed, downstream_has_active_tasks):
    # downstream_rows_needed: how many output rows op must supply for the data
    #   path below it to make progress (e.g. an equal ``OutputSplitter`` needs
    #   ``num_output_splits`` rows to dispatch a balanced split). A producer whose
    #   pulled output rows fall short of this is starving its consumer.
    # downstream_has_active_tasks: a downstream task-launcher has tasks in
    #   flight that will consume refs and, in doing so, free the producer's budget.
    stack: List[Tuple[PhysicalOperator, int, bool]] = [(output_operator, 0, False)]
    while stack:
        curr_op, downstream_rows_needed, downstream_has_active_tasks = stack.pop()
        if curr_op.has_completed():
            continue

        # 1) Only an ExperimentalAPMO can be output backpressured / upgraded.
        if isinstance(curr_op, ExperimentalAPMO):
            curr_producer_in_obp = curr_op.is_output_backpressured()
            curr_op.notify_in_task_output_backpressure(
                in_backpressure=curr_producer_in_obp,
                policy_name="Output",
            )
            # This producer's consumer is starved when the producer hasn't pulled
            # as many output rows as the consumer needs to make progress. We read
            # the producer's OWN output store: it's the operator that holds the
            # refs the consumer is waiting on.
            pulled_output_rows = resource_bank.live_object_store(
                op=curr_op
            ).num_pulled_output_rows
            downstream_needs_input = pulled_output_rows < downstream_rows_needed
            if curr_producer_in_obp and (
                (downstream_needs_input and not downstream_has_active_tasks)
                or not curr_op.has_enough_budget_for_prebuffered_outputs()
            ):
                # Only spend the one-shot upgrade on a call that took effect;
                # upgrade_output_limit() no-ops on debounce / no pool nodes.
                if curr_op.upgrade_output_limit():
                    return curr_op

        # 2) Compute the state to hand this op's producers.
        if isinstance(curr_op, ExperimentalAPMO):
            # A task-launcher drains its producer, so its producer only needs to
            # feed THIS operator enough rows to launch a task -- downstream row
            # requirements don't reach past it, and its active tasks are what
            # relieve its producer's backpressure.
            curr_has_active_tasks = curr_op.get_actor_info().active > 0
            curr_rows_needed = num_rows_required(curr_op)
        else:
            # A pass-through launches no tasks and routes refs unchanged, so it
            # forwards its downstream's requirement (and active-task signal) to
            # its own producer -- taking the sum so the requirement survives a
            # chain of pass-throughs (e.g. APMO -> limit -> OutputSplitter).
            curr_has_active_tasks = downstream_has_active_tasks
            curr_rows_needed = num_rows_required(curr_op) + downstream_rows_needed

        # 3) Descend into producers (upstream).
        for producer in curr_op.input_dependencies:
            stack.append((producer, curr_rows_needed, curr_has_active_tasks))

    return None


def num_rows_required(curr_op: PhysicalOperator) -> int:
    """Returns the # of rows required to form a ready bundle (by ready, I mean
    the operator can now process the input). Each queueing operator knows its
    own requirement (a map's bundle target, an equal split's balance, ...);
    everything else needs nothing."""
    if isinstance(curr_op, InternalQueueOperatorMixin):
        return curr_op.min_num_rows_needed_to_make_progress()
    return 0
