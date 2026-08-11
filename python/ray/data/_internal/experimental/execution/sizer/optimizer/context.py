from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.interfaces.common import NodeIdStr
from ray.data._internal.experimental.execution.sizer.optimizer.windowed import (
    BoolStreak,
    WindowedSignal,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.streaming_executor_state import Topology


@dataclass
class OpShortfall:
    """One operator's unplaced ask from the last sizing pass."""

    resources: ExecutionResources
    num_actors: int
    # True when this is a bridged silent-bottleneck want rather than an ask
    # the op actually placed and had clamped. Transfers serve real asks
    # first when donors are scarce.
    silent: bool = False


@dataclass
class OptimizationContext:
    """Per-tick inputs shared by every rule; read-only for rules."""

    topology: "Topology"
    warmup: bool
    alloc: Dict[NodeIdStr, ExecutionResources]
    free: Dict[NodeIdStr, ExecutionResources]
    # Actor-pool ops the optimizer may act on (has pools, not core-placed).
    managed_ops: List["PhysicalOperator"]
    # True for a full SIZER_SHORTFALL_TICKS run of sizing ticks in which the
    # op asked to grow and was granted nothing.
    shortfall_streak: Dict["PhysicalOperator", BoolStreak]
    # Last sizing pass's unplaced ask per op.
    shortfalls: Dict["PhysicalOperator", OpShortfall]
    # In-flight tasks per op, sampled by observe().
    busy_signal: Dict["PhysicalOperator", WindowedSignal]
    # In-flight minus output-blocked tasks per op.
    productive_signal: Dict["PhysicalOperator", WindowedSignal]
    # Whether the op's output has read full at every sample of the window.
    out_full_streak: Dict["PhysicalOperator", BoolStreak]
    # Whether the op's output has read not-full at every sample of the window.
    out_not_full_streak: Dict["PhysicalOperator", BoolStreak]
    # Constraint-match node set for an op; None = unconstrained (rules
    # intersect with ``free``).
    eligible_nodes: Callable[["PhysicalOperator"], object]
    # Hard pool ceiling (max_size, plus the sink cap for write ops).
    pool_cap: Callable[["PhysicalOperator"], int]
    # Rate-limited growth step for one tick.
    growth_step: Callable[["PhysicalOperator"], int]
    seconds_since_allocation_change: float
    now: float
    # Whether any rule needs an op held (e.g. a recent donor). Rules must
    # consult this before proposing growth; the optimizer supplies it.
    can_op_scale: Callable[["PhysicalOperator"], bool] = lambda _op: True
    # Surface an unplaced silent-bottleneck want as a shortfall, routed through
    # the sizer's per-pass recording so streak semantics hold (a direct streak
    # record would interleave with the sizer's not-starved record).
    record_silent_shortfall: Callable[
        ["PhysicalOperator", OpShortfall], None
    ] = lambda _op, _shortfall: None
