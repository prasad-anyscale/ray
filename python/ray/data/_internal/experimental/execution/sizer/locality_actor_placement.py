import heapq
import logging
import math
from collections import Counter
from typing import AbstractSet, Dict, List, Tuple

from ray.data._internal.execution.interfaces import ExecutionResources, NodeIdStr
from ray.data._internal.execution.resource_bank import LogicalActorId
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    _COLLOCATE_BYTES_PER_ACTOR,
    ActorPlacementStrategy,
    OpPlacementView,
    max_placeable_actors,
)

logger = logging.getLogger(__name__)


class LocalityBasedActorPlacement(ActorPlacementStrategy):
    """scaled up actors to favour locality when possible then falls back to a proportional spread across nodes."""

    def __init__(self):
        # Last upscale's phase breakdown per op_id: (collocate_count,
        # spread_count). Read by the sizer to log WHY each placement landed.
        self.last_upscale_breakdown: Dict[str, Tuple[int, int]] = {}

    @staticmethod
    def _fits(
        view: OpPlacementView,
        free: Dict[NodeIdStr, ExecutionResources],
        node_id: NodeIdStr,
    ) -> bool:
        return max_placeable_actors(free[node_id], view.per_actor_usage) >= 1

    @staticmethod
    def _locality_rank(
        view: OpPlacementView,
        node_id: NodeIdStr,
        extra_actors: int = 0,
        include_queue: bool = True,
    ) -> Tuple:
        """rank nodes based on locality (ascending = better).

        rank order:
            1: structural imbalance: max(upstream, downstream) - here
            2: queue pressure per actor: decays as actors land on the node
            3: upstream actors (more upstream = worse)
            4: node_id (tie-breaker, deterministic)

        ``include_queue=False`` ranks nodes whose capacity arrives only after
        a slow drain: the queue snapshot will be stale by then, so only the
        durable structural signal counts.
        """
        here = view.actors_by_node.get(node_id, 0) + extra_actors
        queue = view.input_queue_bytes_by_node.get(node_id, 0) if include_queue else 0
        return (
            # Structural imbalance: max(up, down) - here, matching
            # _locality_demand (one actor bridges a producer AND consumer).
            -(
                max(
                    2
                    * view.upstream_edge_weight
                    * view.upstream_actors_by_node.get(node_id, 0),
                    2
                    * view.downstream_edge_weight
                    * view.downstream_actors_by_node.get(node_id, 0),
                )
                - here
            ),
            # Queue pressure per actor: decays as actors land on the node.
            -queue / (here + 1),
            -view.upstream_actors_by_node.get(node_id, 0),
            node_id,
        )

    def _locality_demand(self, view: OpPlacementView, node_id: NodeIdStr) -> int:
        """Additional actors the node's locality signal warrants.

        This number does not take the capcity into account (will be accounted for during the actual placement).
        The number of actors to place on a node is the minimum of this and the capacity of the node.
        num_actors_on_node = max(upstream_actors, downstream_actors) - current_actors + buffer, where:
        buffer = 0 if there are idle or pending actors on the node
        else ceil(queued_input_bytes / bytes_per_actor)

        """
        up = (
            2 * view.upstream_edge_weight * view.upstream_actors_by_node.get(node_id, 0)
        )
        down = (
            2
            * view.downstream_edge_weight
            * view.downstream_actors_by_node.get(node_id, 0)
        )
        here = view.actors_by_node.get(node_id, 0)
        buffer = 0
        if view.idle_actors_by_node.get(
            node_id, 0
        ) == 0 and not view.pending_ids_by_node.get(node_id):
            bytes_per_actor = view.input_bytes_per_actor
            if not bytes_per_actor or bytes_per_actor == float("inf"):
                bytes_per_actor = _COLLOCATE_BYTES_PER_ACTOR
            queue_bytes = view.input_queue_bytes_by_node.get(node_id, 0)
            buffer = -(-queue_bytes // bytes_per_actor)  # integer ceil
        return max(0, math.ceil(max(up, down)) - here + buffer)

    def place_upscale(
        self,
        view: OpPlacementView,
        free: Dict[NodeIdStr, ExecutionResources],
        count: int,
    ) -> List[NodeIdStr]:
        placed: List[NodeIdStr] = []
        # Actors placed on each node during this call (Counter: missing nodes
        # read as 0). Feeds the locality rank, which is recomputed every time
        # an actor is placed.
        placed_per_node: Counter = Counter()

        def locality_rank(node_id: NodeIdStr) -> Tuple:
            return self._locality_rank(
                view, node_id, extra_actors=placed_per_node[node_id]
            )

        # Phase 1 — locality-based COLLOCATE: heapified based on the locality_rank.
        # identify nodes with unmet locality demand and place actors on them
        heap = [
            (locality_rank(node_id), node_id)
            for node_id in view.candidates(free)
            if self._locality_demand(view, node_id)
            > 0  # there is demand to improve locality on this node
            and self._fits(view, free, node_id)  # more actors can fit on this node
        ]
        heapq.heapify(heap)
        while len(placed) < count and heap:
            _, node_id = heapq.heappop(heap)
            placed.append(node_id)
            placed_per_node[node_id] += 1
            free[node_id] = free[node_id].subtract(view.per_actor_usage)
            if placed_per_node[node_id] < self._locality_demand(
                view, node_id
            ) and self._fits(view, free, node_id):
                # locality demand not met and there is still capacity on this node, so push it back to the heap
                heapq.heappush(heap, (locality_rank(node_id), node_id))
        collocate_count = len(placed)

        # Phase 2 — SPREAD: proportional split by free actor slots. The remaining unplaced actors are spread across the nodes based on their free capacity.
        remainder = count - len(placed)
        if remainder > 0:
            self._proportional_spread(view, free, remainder, placed, placed_per_node)

        self.last_upscale_breakdown[view.op_id] = (
            collocate_count,
            len(placed) - collocate_count,
        )
        logger.debug(
            "place_upscale[%s]: source=%s requested=%d placed=%d "
            "(collocate=%d spread=%d) nodes=%s",
            view.op_id,
            view.is_source,
            count,
            len(placed),
            collocate_count,
            len(placed) - collocate_count,
            dict(Counter(placed)),
        )
        return placed

    def _proportional_spread(
        self,
        view: OpPlacementView,
        free: Dict[NodeIdStr, ExecutionResources],
        num_actors_to_place: int,
        placed: List[NodeIdStr],
        placed_per_node: Counter,
    ) -> None:
        """Spread the actors across free nodes proportional to their free capacity.

        each node gets num_actors_to_place * (node_free_slots / total_free_slots) actors,
        and the leftovers from rounding down go to the nodes with the largest fractional
        remainders. so a node with 2x the free slots gets ~2x the actors and all nodes
        end up proportionally utilized.
        """

        def actors_per_node(node_id: NodeIdStr) -> int:
            cap = max_placeable_actors(free[node_id], view.per_actor_usage)
            # when actor has no resource specifications, each node can house num_actors_to_place.
            return num_actors_to_place if math.isinf(cap) else int(cap)

        max_actors_per_node = {
            node_id: actors_per_node(node_id)
            for node_id in view.candidates(free)
            if self._fits(view, free, node_id)
        }
        total_actor_slots = sum(max_actors_per_node.values())
        if total_actor_slots == 0:
            # no node has room for even one actor, so we can't place any more actors
            # (also guards the divmod below against a zero divisor)
            return
        num_actors_to_place = min(
            num_actors_to_place, total_actor_slots
        )  # can't place more than total capacity
        per_node_actor_share: Dict[
            NodeIdStr, int
        ] = {}  # how many actors to place on each node to satisfy num_actors_to_place
        frac: Dict[NodeIdStr, int] = {}
        for node_id, w in max_actors_per_node.items():
            per_node_actor_share[node_id], frac[node_id] = divmod(
                num_actors_to_place * w, total_actor_slots
            )
        leftover = num_actors_to_place - sum(per_node_actor_share.values())
        # Leftovers go to the largest fractional remainders; ties broken toward
        # scarcer-node-avoidance first (fewer constrained ops can use the node),
        # then toward nodes with fewest of our actors.
        for node_id in sorted(
            max_actors_per_node,
            key=lambda node_id: (
                -frac[node_id],
                view.overlapping_op_count_by_node.get(node_id, 0),
                view.actors_by_node.get(node_id, 0) + placed_per_node[node_id],
                node_id,
            ),
        )[:leftover]:
            per_node_actor_share[node_id] += 1
        for node_id, k in per_node_actor_share.items():
            if k > 0:
                placed.extend([node_id] * k)
                # deduct all k actors' usage in one go instead of k subtractions
                free[node_id] = free[node_id].subtract(view.per_actor_usage.scale(k))
                placed_per_node[node_id] += k

    def pick_downscale_victims(
        self,
        view: OpPlacementView,
        count: int,
        favored_nodes: AbstractSet[NodeIdStr],
    ) -> List[LogicalActorId]:
        all_pending = [aid for ids in view.pending_ids_by_node.values() for aid in ids]
        all_running = [aid for ids in view.actor_ids_by_node.values() for aid in ids]
        # Full drain: everyone goes; pendings first (cancelled without a drain).
        if count >= len(all_pending) + len(all_running):
            return all_pending + all_running

        victims: List[LogicalActorId] = []

        def take(actor_ids: List[LogicalActorId]) -> None:
            for actor_id in actor_ids:
                if len(victims) >= count:
                    return
                victims.append(actor_id)

        def cheap(node_id: NodeIdStr) -> List[LogicalActorId]:
            """Pendings + idle actors: evicting frees capacity ~immediately."""
            pendings = view.pending_ids_by_node.get(node_id, [])
            num_idle = view.idle_actors_by_node.get(node_id, 0)
            return [*pendings, *view.actor_ids_by_node.get(node_id, [])[:num_idle]]

        def busy(node_id: NodeIdStr) -> List[LogicalActorId]:
            """Everyone else, least-busy first: frees capacity after a drain."""
            num_idle = view.idle_actors_by_node.get(node_id, 0)
            return view.actor_ids_by_node.get(node_id, [])[num_idle:]

        nodes = sorted(
            set(view.actor_ids_by_node) | set(view.pending_ids_by_node),
            key=lambda node_id: (
                0 if node_id in favored_nodes else 1,  # sort favored nodes first
                view.input_queue_bytes_by_node.get(
                    node_id, 0
                ),  # tie break by queue pressure (less is better)
                node_id,
            ),
        )
        # Cost-class-major: every cheap victim across all nodes before any
        # busy actor. Killing a busy actor wastes its in-flight work AND
        # delivers the capacity late. Favored nodes are picked first within each class due to the sorting key.
        for node_id in nodes:
            take(cheap(node_id))
        for node_id in nodes:
            take(busy(node_id))

        logger.debug(
            "pick_downscale_victims[%s]: wanted=%d picked=%d favored=%s victims=%s",
            view.op_id,
            count,
            len(victims),
            sorted(favored_nodes),
            victims,
        )
        return victims

    def compute_favored_nodes(
        self,
        starved_views: List[Tuple[OpPlacementView, int]],
        free: Dict[NodeIdStr, ExecutionResources],
        all_nodes: AbstractSet[NodeIdStr],
        fast_free_nodes: AbstractSet[NodeIdStr],
    ) -> AbstractSet[NodeIdStr]:
        """Each starved upscaler nominates its favoured nodes on both
        reclaim timescales, ranked by the common locality_rank().

        Fast-free nodes (a downscaler has pending/idle victims there) free
        capacity ~immediately, so the queue snapshot is still valid and counts.
        Slow-free nodes only free after a busy-actor drain, by which time the
        queue snapshot is stale — rank those on the structural signal alone.
        """
        favored: set = set()
        for view, unmet in starved_views:
            if unmet <= 0:
                continue
            # Only nodes this op is allowed on: freeing a node it can't use is
            # wasted eviction (match_nodes is None => unconstrained => all nodes).
            eligible = (
                all_nodes
                if view.match_nodes is None
                else [n for n in all_nodes if n in view.match_nodes]
            )
            fast = sorted(
                (n for n in eligible if n in fast_free_nodes),
                key=lambda n: self._locality_rank(view, n),
            )
            slow = sorted(
                (n for n in eligible if n not in fast_free_nodes),
                key=lambda n: self._locality_rank(view, n, include_queue=False),
            )
            favored.update(fast[:unmet])
            favored.update(slow[:unmet])
        return favored
