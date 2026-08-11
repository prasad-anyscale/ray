import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Hashable, List, Optional

logger = logging.getLogger(__name__)


class OrderingPolicy(str, Enum):
    """The ordering policy the sizer supports (``RAY_DATA_SIZER_ORDERING_POLICY``)."""

    REVERSE_TOPOLOGICAL = "reverse_topological"
    SCARCITY_FIRST = "scarcity_first"

    @classmethod
    def from_env(cls, value: str) -> "OrderingPolicy":
        try:
            return cls(value)
        except ValueError:
            logger.warning(
                "Unknown ordering policy %r; using %s. Known: %s",
                value,
                cls.REVERSE_TOPOLOGICAL.value,
                [p.value for p in cls],
            )
            return cls.REVERSE_TOPOLOGICAL


@dataclass(frozen=True)
class OpOrderingSignals:
    """Per-operator inputs a policy sorts on."""

    # Position in source-first topology order; larger = closer to the sink.
    topo_index: int
    # PlacementConstraint.strictness(): constrained(1) < unconstrained(2).
    strictness: int
    has_placement_group: bool
    is_gpu: bool
    is_fixed_concurrency: bool
    # Memory request is large relative to the actor's CPUs (memory-bound).
    is_large_memory: bool
    # How many nodes the op's constraint currently matches (its eligible set
    # size, from the tick snapshot). None = unknown or unconstrained. Within a
    # tier, fewer eligible nodes = more specific = processed first: an op
    # pinned to one node has zero placement freedom, so granting it first can
    # never hurt a pool-selector op that has alternatives, while the reverse
    # can wedge the pinned op.
    num_eligible_nodes: Optional[int] = None

    @classmethod
    def from_operator(
        cls,
        op: "Any",
        *,
        topo_index: int,
        constraint: "Optional[Any]",
        constraint_aware: bool,
        memory_per_cpu_bytes: float,
        num_eligible_nodes: Optional[int] = None,
    ) -> "OpOrderingSignals":
        """Derive the sort features from an operator + its parsed constraint.

        Lives here (not the sizer) so all knowledge of *what the tiers mean* is
        in one place; the caller only supplies the raw op, its topo position,
        and the large-memory threshold (kept a parameter so this module needs no
        sizer constants)."""
        pool = op.get_autoscaling_actor_pools()[0]
        per = pool.per_actor_resource_usage()
        return cls(
            topo_index=topo_index,
            strictness=constraint.strictness() if constraint else 2,
            has_placement_group=constraint_aware
            and bool(constraint and constraint.has_placement_group),
            is_gpu=per.gpu > 0,
            is_fixed_concurrency=pool.min_size() == pool.max_size(),
            is_large_memory=per.memory > per.cpu * memory_per_cpu_bytes,
            num_eligible_nodes=num_eligible_nodes,
        )


class OperatorOrderingPolicy(abc.ABC):
    """Decides the order the sizer grants/places operators in."""

    @abc.abstractmethod
    def order(self, features: Dict[Hashable, OpOrderingSignals]) -> List[Hashable]:
        """Return the operator keys in processing order (first processed first)."""
        ...


class ReverseTopologicalOrdering(OperatorOrderingPolicy):
    """Sink-most operator first."""

    def order(self, features: Dict[Hashable, OpOrderingSignals]) -> List[Hashable]:
        return sorted(features, key=lambda op: -features[op].topo_index)


class ScarcityFirstOrdering(OperatorOrderingPolicy):
    """Scarcest / most-constrained operators first, sink-most within a tier, so an
    unconstrained (or abundant-resource) op can't camp on the scarce nodes a
    constrained op needs. Tiers (lowest first): PG, label/pin, GPU, fixed
    concurrency, large-memory, everything else. An op takes its lowest tier.

    Within a tier, more SPECIFIC constraints go first (ascending eligible-node
    count): a node-pinned op precedes a pool-selector op whose eligible set it
    overlaps, so the pool op can't consume the pinned op's only node. Unknown
    eligibility (None) sorts last within the tier; sink-most breaks the
    remaining ties."""

    @staticmethod
    def _tier(f: OpOrderingSignals) -> int:
        if f.has_placement_group:
            return 0
        if f.strictness < 2:
            return 1
        if f.is_gpu:
            return 2
        if f.is_fixed_concurrency:
            return 3
        if f.is_large_memory:
            return 4
        return 5

    def order(self, features: Dict[Hashable, OpOrderingSignals]) -> List[Hashable]:
        def key(op: Hashable):
            f = features[op]
            specificity = (
                f.num_eligible_nodes
                if f.num_eligible_nodes is not None
                else float("inf")
            )
            return (self._tier(f), specificity, -f.topo_index)

        return sorted(features, key=key)


_POLICIES = {
    OrderingPolicy.REVERSE_TOPOLOGICAL: ReverseTopologicalOrdering,
    OrderingPolicy.SCARCITY_FIRST: ScarcityFirstOrdering,
}


def make_ordering_policy(policy: OrderingPolicy) -> OperatorOrderingPolicy:
    return _POLICIES[policy]()
