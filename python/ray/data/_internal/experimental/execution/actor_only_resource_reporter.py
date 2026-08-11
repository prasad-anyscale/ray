import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from ray.data._internal.cached_ray_internals import get_alive_node_labels
from ray.data._internal.cluster_autoscaler.base_autoscaling_coordinator import (
    AutoscalingCoordinator,
)
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.interfaces.common import NodeIdStr
from ray.data._internal.execution.resource_bank import (
    ResourceBankBase,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.streaming_executor_state import Topology
    from ray.util.placement_group import PlacementGroup

logger = logging.getLogger(__name__)

# Placement-group ids are opaque strings (``PlacementGroup.id.hex()``) used as
# dict keys.
PlacementGroupIdStr = str


@dataclass(frozen=True)
class ClusterView:
    """Per-tick view of the cluster the sizer plans against, built once at the
    top of a tick and shared by both phases."""

    # This dataset's coordinator allocation, per node.
    alloc_by_node: Dict[NodeIdStr, ExecutionResources]
    # Node labels (``ray.nodes()[...]["Labels"]``) per alive node; resolves
    # label_selector eligibility. Empty when constraint-aware placement is off.
    labels_by_node: Dict[NodeIdStr, Dict[str, str]] = field(default_factory=dict)
    # Bundle shapes per placement group the topology references (fetched once).
    # Only populated on the PG-aware path; no node mapping (Ray core places PGs).
    pg_bundles: Dict[PlacementGroupIdStr, List[ExecutionResources]] = field(
        default_factory=dict
    )


class ActorOnlyResourceReporter:
    """Reports global resource usage/limits for the actor-only backend.

      * CPU/GPU/memory usage from each operator's ``running_logical_usage``,
      * object store memory usage from the ``ResourceBank`` (live bytes),
      * limits from this dataset's allocation on the ``AutoscalingCoordinator``.

    Per dataset execution: "global" means across this dataset's operators, not
    cluster-wide. Cross-dataset arbitration lives in the shared coordinator.
    """

    def __init__(
        self,
        topology: "Topology",
        autoscaling_coordinator: AutoscalingCoordinator,
        resource_bank: ResourceBankBase,
    ):
        self._topology = topology
        self._autoscaling_coordinator = autoscaling_coordinator
        self._resource_bank = resource_bank
        # Placement-group bundle shapes, keyed by ``PlacementGroup.id.hex()``.
        # Shapes are fixed at PG creation, so each PG is fetched at most once via
        # ``prefetch_pg_bundles`` and served from here afterwards. A PG that
        # isn't resolvable yet is simply absent (retried on the next prefetch).
        self._pg_bundles_by_id: Dict[str, List[ExecutionResources]] = {}
        # Allocation-change tracking: signature of the last-seen per-node
        # allocation and when it last changed (see
        # ``seconds_since_allocation_change``).
        self._alloc_signature = None
        self._alloc_changed_at = float("-inf")

    def get_global_usage(self) -> ExecutionResources:
        """Resources used by running work across all operators.

        CPU/GPU/memory come from each operator's ``running_logical_usage``;
        object store memory comes from the live bytes tracked by the
        """
        usage = ExecutionResources.zero()
        for op in self._topology:
            live = self._resource_bank.live_object_store(op=op)
            usage = usage.add(
                ExecutionResources(object_store_memory=live.total_bytes())
            )
            usage = usage.add(op.running_logical_usage())
        return usage

    def get_global_pending_usage(self) -> ExecutionResources:
        """Resources for actors requested but not yet running, across operators."""
        pending = ExecutionResources.zero()
        for op in self._topology:
            pending = pending.add(op.pending_logical_usage())
        return pending

    def get_global_limits(self) -> ExecutionResources:
        """This dataset's allocated cluster capacity, from the coordinator."""
        limits = ExecutionResources.zero()
        for bundle in self._autoscaling_coordinator.get_reserved_resources():
            limits = limits.add(ExecutionResources.from_resource_dict(bundle))
        return limits

    def get_reserved_resources_by_node(self) -> Dict[NodeIdStr, ExecutionResources]:
        """This dataset's coordinator allocation, keyed by node id.

        The per-node view the sizer uses to place actors; summing it gives the
        same total as ``get_global_limits``.
        """
        alloc = {
            node_id: ExecutionResources.from_resource_dict(resources)
            for node_id, resources in (
                self._autoscaling_coordinator.get_reserved_resources_by_node().items()
            )
        }
        signature = tuple(sorted((n, r.cpu, r.gpu, r.memory) for n, r in alloc.items()))
        if signature != self._alloc_signature:
            self._alloc_signature = signature
            self._alloc_changed_at = time.monotonic()
        return alloc

    def seconds_since_allocation_change(self) -> float:
        """How long the per-node allocation has been unchanged, as of the
        last ``get_reserved_resources_by_node`` call."""
        return time.monotonic() - self._alloc_changed_at

    def committed_by_node(self) -> Dict[NodeIdStr, ExecutionResources]:
        """Resources this dataset's actor pools occupy or have promised, per node."""
        # Lazy import: actor_pool_map_operator -> placement_constraints ->
        # ClusterView (defined here), so a top-level import would cycle.
        from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
            ExperimentalAPMO,
        )

        zero = ExecutionResources.zero()
        committed: Dict[NodeIdStr, ExecutionResources] = {}
        for op in self._topology:
            if not isinstance(op, ExperimentalAPMO):
                continue
            if op.placement_constraint().has_placement_group:
                continue
            for node_id, usage in op.committed_usage_by_node().items():
                committed[node_id] = committed.get(node_id, zero).add(usage)
        return committed

    def free_by_node(
        self, alloc: Dict[NodeIdStr, ExecutionResources]
    ) -> Dict[NodeIdStr, ExecutionResources]:
        """``alloc - committed``, clamped >= 0, over the allocated nodes."""
        zero = ExecutionResources.zero()
        committed = self.committed_by_node()
        return {
            node_id: alloc[node_id].subtract(committed.get(node_id, zero)).max(zero)
            for node_id in alloc
        }

    def prefetch_pg_bundles(self, placement_groups: List["PlacementGroup"]) -> None:
        """Fetch and cache the bundle shapes of the given placement groups.

        Bundle shapes are fixed at PG creation, so each PG is fetched at most
        once and served from the cache thereafter. A PG that can't be resolved
        yet (table not available) is left uncached and retried on the next call,
        so an early bootstrap fetch doesn't pin it to "no bundles" forever.
        """
        from ray.util.placement_group import placement_group_table

        for pg in placement_groups:
            pg_id = pg.id.hex()
            if pg_id in self._pg_bundles_by_id:
                continue
            try:
                table = placement_group_table(pg)
            except Exception:  # noqa: BLE001 -- not resolvable yet; retry later
                logger.exception(
                    "ActorOnlyResourceReporter: could not read "
                    "placement_group_table for PG %s;",
                    pg_id,
                )
                raise
            bundles = table.get("bundles", {}) or {}
            # ``bundles`` is keyed by bundle index (0..n-1); materialize an
            # index-ordered list of ExecutionResources so consumers can index by
            # ``placement_group_bundle_index``.
            self._pg_bundles_by_id[pg_id] = [
                ExecutionResources.from_resource_dict(bundles[idx])
                for idx in sorted(bundles)
            ]

    def get_cluster_view(
        self, *, include_labels: bool = False, include_pg_bundles: bool = False
    ) -> ClusterView:
        """Snapshot of the cluster (visible to the dataset) state at this moment."""
        labels_by_node = get_alive_node_labels() if include_labels else {}
        pg_bundles = dict(self._pg_bundles_by_id) if include_pg_bundles else {}
        return ClusterView(
            alloc_by_node=self.get_reserved_resources_by_node(),
            labels_by_node=labels_by_node,
            pg_bundles=pg_bundles,
        )
