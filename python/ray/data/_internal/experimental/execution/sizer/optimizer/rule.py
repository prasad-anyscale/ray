import abc
from typing import TYPE_CHECKING, List

from ray.data._internal.experimental.execution.sizer.optimizer.context import (
    OptimizationContext,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.experimental.execution.sizer.operator_sizer import (
        PlacedSizingRequest,
    )


class OptimizationRule(abc.ABC):
    """One cross-operator optimization.

    A rule owns its in-flight state, guards, and metadata. The optimizer
    builds the shared context, advances every rule's in-flight work each
    tick, and applies the requests rules propose.
    """

    name: str

    def advance(self, ctx: OptimizationContext) -> None:
        """Progress in-flight work (landings, timeouts). Called every tick,
        whether or not the rule proposes anything."""

    @abc.abstractmethod
    def apply(self, ctx: OptimizationContext) -> List["PlacedSizingRequest"]:
        """Propose fully-placed requests for this tick."""

    def after_apply(self, ctx: OptimizationContext) -> None:
        """Called after this tick's proposals were executed; finalize any
        state that needs post-apply pool observation."""

    def can_op_scale(self, op: "PhysicalOperator") -> bool:
        """False while this rule needs ``op`` held (e.g. a donor whose
        victims are still draining). The optimizer ANDs this across rules;
        the sizer asks the optimizer, never a rule directly."""
        return True
