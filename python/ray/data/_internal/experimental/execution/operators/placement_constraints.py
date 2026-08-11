import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, AbstractSet, Dict, Optional

from ray.data._internal.execution.interfaces import ExecutionResources, NodeIdStr
from ray.data._internal.experimental.execution.actor_only_resource_reporter import (
    ClusterView,
)
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    max_placeable_actors,
)

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = logging.getLogger(__name__)

# Resource dimensions an actor can constrain a node on. A node that has none of
# a dimension the actor requests can never host it, regardless of labels.
_CONSTRAINED_DIMS = ("cpu", "gpu", "memory")


@dataclass(frozen=True)
class PlacementConstraint:
    """One operator's scheduling constraints, parsed once at bootstrap.

    A node pin (``ray.io/node-id`` / ``NodeAffinity``) is just a
    ``label_selector`` on the node-id label -- there is no separate pin concept.
    ``matching_nodes`` returns the constraint-match set; capacity is layered on by
    the caller.
    """

    # The op's ``label_selector`` (a node pin appears here as
    # ``{"ray.io/node-id": <node>}``). None if the op selects no labels.
    label_selector: Optional[Dict[str, str]] = None
    # Placement group + optional bundle index. PG ops are sized against bundle
    # capacity but placed by Ray core, so they impose no label filter.
    placement_group: Optional["PlacementGroup"] = None
    placement_group_bundle_index: Optional[int] = None

    @property
    def has_placement_group(self) -> bool:
        return self.placement_group is not None

    @property
    def has_label_selector(self) -> bool:
        return self.label_selector is not None

    def strictness(self) -> int:
        """Ordering rank ("most-constrained first"):
        1 -> has label_selector
        2 -> all other cases
        """
        return 1 if self.label_selector is not None else 2

    def pg_capacity(self, per_actor: ExecutionResources, view: ClusterView) -> int:
        """Max actors of shape ``per_actor`` that fit in this op's PG bundles.

        Per-bundle, not summed: a 4-GPU actor fits zero 2-GPU bundles even if the
        PG totals 8 GPU. Reads the bundle shapes cached on ``view`` (fetched once
        by the resource reporter -- shapes are fixed at PG creation). Returns 0
        when the op has no PG, or when its PG isn't resolved in ``view`` yet
        (bundles absent), so the pool is held at 0 until the PG is available.
        """
        if self.placement_group is None:
            return 0
        bundles = view.pg_bundles.get(self.placement_group.id.hex())
        if not bundles:
            return 0
        idx = self.placement_group_bundle_index
        if idx is not None and 0 <= idx < len(bundles):
            shapes = [bundles[idx]]
        else:
            shapes = bundles
        cap = 0
        for shape in shapes:
            count = max_placeable_actors(shape, per_actor)
            if not math.isinf(count):
                cap += int(count)
        return cap

    @staticmethod
    def _matches_selector(
        node_labels: Dict[str, str], label_selector: Dict[str, str]
    ) -> bool:
        """Delegates to Ray core's matcher so the grammar is identical by construction."""
        import ray

        return ray._raylet.node_labels_match_selector(node_labels, label_selector)

    def matching_nodes(
        self,
        view: ClusterView,
        per_actor: Optional[ExecutionResources] = None,
    ) -> AbstractSet[NodeIdStr]:
        """Allocated nodes the op is allowed on.

        No selector -> every allocated node. Otherwise the nodes whose labels
        satisfy the selector. On an unparseable selector, degrade to "eligible
        everywhere" with a WARN -- never an empty set.

        When ``per_actor`` is given, nodes that structurally lack a resource the
        actor requests (e.g. a GPU actor on a CPU-only node) are also excluded:
        their allocation carries none of that dimension, so no actor can ever be
        placed there no matter how much capacity frees up. This keeps the
        constraint-match set honest for the unsatisfiable-constraint fail-fast
        and the per-node scarcity accounting.
        """
        all_nodes = frozenset(view.alloc_by_node)
        if self.label_selector is None:
            return self._filter_by_resource(all_nodes, view, per_actor)
        try:
            matched = frozenset(
                node_id
                for node_id in all_nodes
                if self._matches_selector(
                    view.labels_by_node.get(node_id, {}), self.label_selector
                )
            )
        except Exception:  # noqa: BLE001 -- degrade, never crash the sizer
            logger.warning(
                "could not evaluate label_selector %s; treating the "
                "operator as unconstrained. Node labels: %s",
                self.label_selector,
                {n: dict(lbl) for n, lbl in view.labels_by_node.items()},
            )
            return self._filter_by_resource(all_nodes, view, per_actor)
        return self._filter_by_resource(matched, view, per_actor)

    @staticmethod
    def _filter_by_resource(
        nodes: AbstractSet[NodeIdStr],
        view: ClusterView,
        per_actor: Optional[ExecutionResources],
    ) -> AbstractSet[NodeIdStr]:
        """Drop nodes whose allocation has zero of a resource the actor requests.

        This is the structural (does-the-node-have-any-GPU) filter, distinct from
        the dynamic has-free-capacity check the caller layers on top. A node with
        the resource but currently full stays eligible.
        """
        if per_actor is None:
            return nodes
        return frozenset(
            node_id
            for node_id in nodes
            if all(
                getattr(view.alloc_by_node[node_id], dim) > 0
                for dim in _CONSTRAINED_DIMS
                if getattr(per_actor, dim) > 0
            )
        )
