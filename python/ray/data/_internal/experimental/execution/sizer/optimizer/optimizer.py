import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Set

from ray.data._internal.execution.execution_flags import (
    SIZER_SHORTFALL_TICKS,
    SIZER_SIGNAL_WINDOW_SAMPLES,
)
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
    ExperimentalAPMO,
)
from ray.data._internal.experimental.execution.sizer.optimizer.capacity_transfer import (  # noqa: E501
    CapacityTransfer,
)
from ray.data._internal.experimental.execution.sizer.optimizer.context import (
    OpShortfall,
    OptimizationContext,
)
from ray.data._internal.experimental.execution.sizer.optimizer.metrics import (
    OptimizerMetrics,
)
from ray.data._internal.experimental.execution.sizer.optimizer.rule import (
    OptimizationRule,
)
from ray.data._internal.experimental.execution.sizer.optimizer.silent_bottleneck import (  # noqa: E501
    SilentBottleneckGrant,
)
from ray.data._internal.experimental.execution.sizer.optimizer.windowed import (
    BoolStreak,
    WindowedSignal,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.streaming_executor_state import Topology

logger = logging.getLogger(__name__)

# Signals are sampled at most this often, however frequently observe() runs.
_OBSERVE_INTERVAL_S = 0.5


class PipelineOptimizer:
    """Cross-operator sizing corrections, applied as registered rules.

    The sizer's own passes stay strictly per-operator; anything that must
    look at the whole topology (moving capacity between operators, growing
    an op that cannot ask for itself) lives here. Each rule owns its own
    state and guards; the optimizer builds a shared per-tick context,
    advances every rule's in-flight work, and collects proposals.
    """

    def __init__(
        self,
        dataset_id: str,
        resource_reporter,
        placement,
        clock: Callable[[], float],
        is_self_placed: Callable[["PhysicalOperator"], bool],
        outqueue_full: Callable[["PhysicalOperator"], bool],
    ):
        self._resource_reporter = resource_reporter
        self._clock = clock
        self._is_self_placed = is_self_placed
        self._outqueue_full = outqueue_full
        self._metrics = OptimizerMetrics(dataset_id)
        self._last_observed_t = float("-inf")
        self._busy_signal: Dict["PhysicalOperator", WindowedSignal] = {}
        self._productive_signal: Dict["PhysicalOperator", WindowedSignal] = {}
        self._out_full_streak: Dict["PhysicalOperator", BoolStreak] = {}
        self._out_not_full_streak: Dict["PhysicalOperator", BoolStreak] = {}
        self._shortfall_streak: Dict["PhysicalOperator", BoolStreak] = {}
        # Unplaced silent-bottleneck wants, merged into the next sizing pass
        # and consumed there (see record_sizing_results).
        self._silent_shortfalls: Dict["PhysicalOperator", OpShortfall] = {}
        self._shortfalls: Dict["PhysicalOperator", OpShortfall] = {}
        self._rules: List[OptimizationRule] = []
        self.register(CapacityTransfer(self._metrics, placement, clock))
        self.register(SilentBottleneckGrant(self._metrics))

    def register(self, rule: OptimizationRule) -> None:
        self._rules.append(rule)

    # --- signals ----------------------------------------------------------

    def observe(self, topology: "Topology") -> None:
        """Sample per-op signals. Cheap to call every scheduling-loop
        iteration; records at most every ``_OBSERVE_INTERVAL_S``."""
        now = self._clock()
        if now - self._last_observed_t < _OBSERVE_INTERVAL_S:
            return
        self._last_observed_t = now
        for op in self._managed_ops(topology):
            pool = op.get_autoscaling_actor_pools()[0]
            in_flight = pool.num_tasks_in_flight()
            # The full flag is actor-denominated (idle actors count as
            # unblocked capacity); the productive signal must stay in TASK
            # units to match in_flight, so it takes blocked tasks from the
            # task-denominated helper directly.
            out_full, _blocked_actors, _running = self._outqueue_full(op)
            blocked_tasks, _active = op.output_backpressured_fraction()
            self._signal(self._busy_signal, op).record(in_flight)
            self._signal(self._productive_signal, op).record(
                max(0, in_flight - blocked_tasks)
            )
            streak = self._out_full_streak.get(op)
            if streak is None:
                streak = self._out_full_streak[op] = BoolStreak(
                    SIZER_SIGNAL_WINDOW_SAMPLES
                )
            streak.record(out_full)
            not_full_streak = self._out_not_full_streak.get(op)
            if not_full_streak is None:
                not_full_streak = self._out_not_full_streak[op] = BoolStreak(
                    SIZER_SIGNAL_WINDOW_SAMPLES
                )
            not_full_streak.record(not out_full)

    def record_sizing_results(
        self,
        evaluated_ops: List["PhysicalOperator"],
        starved_ops: Set["PhysicalOperator"],
        shortfalls: Dict["PhysicalOperator", OpShortfall],
        neutral_ops: Set["PhysicalOperator"] = frozenset(),
    ) -> None:
        """Called by the sizer at the end of each sizing pass: which ops
        asked and were granted nothing, and each op's unplaced ask."""
        silent = self._silent_shortfalls
        self._silent_shortfalls = {}
        for op in evaluated_ops:
            streak = self._shortfall_streak.get(op)
            if streak is None:
                streak = self._shortfall_streak[op] = BoolStreak(SIZER_SHORTFALL_TICKS)
            if op in starved_ops or op in silent:
                streak.record(True)
            elif op in neutral_ops:
                # Held for output backpressure while demand remains: no
                # evidence either way, so the streak neither grows nor
                # resets. The streak can only be BUILT from real refused
                # asks; an op oscillating between asking and being braked
                # keeps its progress instead of starting over each time.
                continue
            else:
                streak.record(False)
        # Sizer-recorded shortfalls win; silent ones fill the gaps.
        self._shortfalls = {**silent, **shortfalls}

    # --- optimization -----------------------------------------------------

    def _record_silent_shortfall(self, op, shortfall: OpShortfall) -> None:
        """Store an unplaced silent-bottleneck want for the next sizing pass."""
        self._silent_shortfalls[op] = shortfall

    def can_op_scale(self, op: "PhysicalOperator") -> bool:
        """Whether any rule needs ``op`` held (consulted by the sizer
        before upscaling)."""
        return all(rule.can_op_scale(op) for rule in self._rules)

    def optimize_pipeline(
        self,
        topology: "Topology",
        *,
        warmup: bool,
        eligible_nodes: Callable[["PhysicalOperator"], List[str]],
        pool_cap: Callable[["PhysicalOperator"], int],
        growth_step: Callable[["PhysicalOperator"], int],
    ) -> List:
        ctx = self._build_context(
            topology,
            warmup=warmup,
            eligible_nodes=eligible_nodes,
            pool_cap=pool_cap,
            growth_step=growth_step,
        )
        self._last_ctx = ctx
        for rule in self._rules:
            rule.advance(ctx)
        requests = []
        # Rules run one at a time, and ctx.can_op_scale is evaluated live:
        # each rule sees the ops that earlier rules (and its own in-flight
        # work) are holding, so only one rule acts on an op at a time.
        for rule in self._rules:
            requests.extend(rule.apply(ctx))
        return requests

    def after_apply(self) -> None:
        """Called once this tick's proposals have been executed."""
        for rule in self._rules:
            rule.after_apply(self._last_ctx)

    # --- internals ---------------------------------------------------------

    def _managed_ops(self, topology: "Topology") -> List["PhysicalOperator"]:
        return [
            op
            for op in topology
            if isinstance(op, ExperimentalAPMO)
            and op.get_autoscaling_actor_pools()
            and not self._is_self_placed(op)
        ]

    def _signal(self, signals, op) -> WindowedSignal:
        signal = signals.get(op)
        if signal is None:
            signal = signals[op] = WindowedSignal(SIZER_SIGNAL_WINDOW_SAMPLES)
        return signal

    def _build_context(
        self, topology, *, warmup, eligible_nodes, pool_cap, growth_step
    ) -> OptimizationContext:
        alloc = self._resource_reporter.get_reserved_resources_by_node()
        free = self._resource_reporter.free_by_node(alloc)
        return OptimizationContext(
            topology=topology,
            warmup=warmup,
            alloc=alloc,
            free=free,
            managed_ops=self._managed_ops(topology),
            shortfall_streak=self._shortfall_streak,
            record_silent_shortfall=self._record_silent_shortfall,
            shortfalls=self._shortfalls,
            busy_signal=self._busy_signal,
            productive_signal=self._productive_signal,
            out_full_streak=self._out_full_streak,
            out_not_full_streak=self._out_not_full_streak,
            eligible_nodes=eligible_nodes,
            pool_cap=pool_cap,
            growth_step=growth_step,
            seconds_since_allocation_change=(
                self._resource_reporter.seconds_since_allocation_change()
            ),
            can_op_scale=self.can_op_scale,
            now=self._clock(),
        )
