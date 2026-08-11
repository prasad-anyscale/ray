import abc
from dataclasses import dataclass, field
from typing import AbstractSet, Dict, List, Optional, Tuple

from ray.data._internal.execution.interfaces import ExecutionResources, NodeIdStr
from ray.data._internal.execution.resource_bank import LogicalActorId

# Fallback bytes-per-actor used to convert a node's queued input into a
# "reasonable" collocation fill when the operator has no input-bytes cap
# (``max_input_bytes_per_actor is None``). Only affects how aggressively a
# queue-heavy node is collocated before the remainder spreads.
_COLLOCATE_BYTES_PER_ACTOR = 384 * 1024 * 1024  # 384 MiB


def max_placeable_actors(
    free: ExecutionResources, per_actor: ExecutionResources
) -> float:
    """How many actors of shape ``per_actor`` fit into ``free``.

    Dimensions the actor doesn't request don't constrain it; an actor with no
    resource requests at all fits anywhere (``inf``).
    """
    count = float("inf")
    for dim in ("cpu", "gpu", "memory"):
        per_actor_dim = getattr(per_actor, dim)
        if per_actor_dim > 0:
            count = min(count, (getattr(free, dim) or 0) // per_actor_dim)
    return count


@dataclass(frozen=True)
class OpPlacementView:
    """The placement strategies mainly operate by looking at a snapshot of the operator being up/downscaled, its upstream and downstream opertator actor placement view"""

    op_id: str
    # True for source operators
    is_source: bool
    per_actor_usage: ExecutionResources
    # Soft cap on input bytes one actor holds in flight
    # (``max_input_bytes_per_actor``); None when uncapped. Used to convert a
    # node's queued input into a reasonable collocation fill.
    input_bytes_per_actor: Optional[int]
    # Alive non-terminating actors + targeted pendings, per node.
    actors_by_node: Dict[NodeIdStr, int]
    # bytes of this op's queued input per node.
    input_queue_bytes_by_node: Dict[NodeIdStr, int]
    # alive actors of upstream producer /
    # downstream consumer actor-pool operators, per node.
    upstream_actors_by_node: Dict[NodeIdStr, int]
    downstream_actors_by_node: Dict[NodeIdStr, int]
    # All non-terminating running actors per node, sorted least-busy first
    # (tasks in flight asc, then last task submission ts). Busy actors ARE
    # eligible downscale victims — they just take longer to drain.
    actor_ids_by_node: Dict[NodeIdStr, List[LogicalActorId]]
    # Non-terminating actors with zero tasks in flight, per node. Evicting
    # these (like cancelling pendings) frees capacity ~immediately.
    idle_actors_by_node: Dict[NodeIdStr, int]
    # Pending (not yet ready) actors per target node — the cheapest victims:
    # cancelling one frees its claimed capacity immediately.
    pending_ids_by_node: Dict[NodeIdStr, List[LogicalActorId]]
    # Constraint-match node set (label_selector / node pin): the nodes this op is
    # ALLOWED on, BEFORE the has-capacity filter. None = unconstrained (all
    # nodes). The has-capacity half of eligibility is still enforced by each
    # strategy's ``_fits`` check, so ``candidates(free)`` = match ∩ free-keys and
    # ``_fits`` narrows that to nodes with room.
    match_nodes: Optional[AbstractSet[NodeIdStr]] = None
    # How many operators can be scheduled on a given node (lower = scarcer). Used to break ties when multiple ops want the same node. Only populated on the constraint-aware path.
    overlapping_op_count_by_node: Dict[NodeIdStr, int] = field(default_factory=dict)
    # Per-actor drain cost aligned with ``actor_ids_by_node``: (tasks in
    # flight, unconsumed output blocks). Both must reach zero before a
    # draining victim exits, so lower = frees capacity sooner.
    actor_drain_cost_by_node: Dict[NodeIdStr, List[Tuple[int, int]]] = field(
        default_factory=dict
    )
    # This op's input/output share of its total observed byte flow (each in
    # [0, 1], summing to 1). 0.5/0.5 = no data yet or weighting off; the rank
    # multiplies each side's actor counts by 2 x weight so the neutral value
    # preserves the unweighted rank exactly.
    upstream_edge_weight: float = 0.5
    downstream_edge_weight: float = 0.5

    def candidates(self, free: Dict[NodeIdStr, ExecutionResources]) -> List[NodeIdStr]:
        """Free-map nodes this op is allowed on (constraint-match ∩ free keys).

        Capacity is NOT checked here -- callers still gate on ``_fits``. When the
        op is unconstrained (``match_nodes is None``) every free node is a
        candidate, matching today's ``for node_id in free`` behavior exactly.
        """
        if self.match_nodes is None:
            return list(free)
        return [node_id for node_id in free if node_id in self.match_nodes]


class ActorPlacementStrategy(abc.ABC):
    """Decides node assignment / victim selection for one operator's delta."""

    @abc.abstractmethod
    def place_upscale(
        self,
        view: OpPlacementView,
        free: Dict[NodeIdStr, ExecutionResources],
        count: int,
    ) -> List[NodeIdStr]:
        """Assign ``count`` new actors to nodes.

        Args:
            view: The operator's placement snapshot.
            free: Free capacity per allocated node. MUTATED in place: chosen
                capacity is deducted, so successive calls for different
                operators in the same tick never double-book.
            count: Number of actors to place.

        Returns:
            One node id per placed actor (node id's can repeat if multiple actors are placed on the same node). May be shorter than ``count`` on a
            capacity shortfall; the next sizing tick reconciles.
        """
        ...

    @abc.abstractmethod
    def pick_downscale_victims(
        self,
        view: OpPlacementView,
        count: int,
        favored_nodes: AbstractSet[NodeIdStr],
    ) -> List[LogicalActorId]:
        """Pick ``count`` exact victim actors for a downscale.

        Args:
            view: The operator's placement snapshot.
            count: Number of victims wanted.
            favored_nodes: Nodes that capacity-starved upscaling operators
                want (the design doc's "set 1") — victims there are preferred
                so the freed capacity benefits the upscaler.

        Returns:
            Victim logical actor ids (pending or running). May be shorter
            than ``count``.
        """
        ...

    @abc.abstractmethod
    def compute_favored_nodes(
        self,
        starved_views: List[Tuple[OpPlacementView, int]],
        free: Dict[NodeIdStr, ExecutionResources],
        all_nodes: AbstractSet[NodeIdStr],
        fast_free_nodes: AbstractSet[NodeIdStr],
    ) -> AbstractSet[NodeIdStr]:
        """Nodes that capacity-starved upscaling operators want freed.

        Fed into ``pick_downscale_victims`` as ``favored_nodes`` so downscale
        victims are taken where the freed capacity benefits the upscalers.

        Args:
            starved_views: (view, unmet delta) per upscaling op, in priority
                order (sink-most operator first).
            free: Free capacity per allocated node (not mutated).
            all_nodes: Every node in the allocation.
            fast_free_nodes: Nodes where a downscaling op has pending/idle
                victims — eviction there is ~instant, so "capacity free now"
                ranking applies; elsewhere the capacity arrives only after a
                slow drain of busy actors.

        Returns:
            The set of nodes to favor when picking downscale victims.
        """
        ...
