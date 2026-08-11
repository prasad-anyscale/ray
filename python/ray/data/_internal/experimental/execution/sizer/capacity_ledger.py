import abc
import math
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Callable, Dict, List

from ray.data._internal.execution.interfaces import ExecutionResources, NodeIdStr
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    max_placeable_actors,
)

# Placeable count for an actor that requests no cpu/gpu/memory (fits anywhere).
_UNBOUNDED_PLACEABLE = 1 << 30


@dataclass
class TickSnapshot:
    """The per-tick view both sizer phases share in the constraint-aware path."""

    # alloc - committed(all non-PG pools), clamped >= 0. The shared free map.
    free_by_node: Dict[NodeIdStr, ExecutionResources]
    # The raw allocation (favored-nodes' all-nodes set + logging).
    alloc_by_node: Dict[NodeIdStr, ExecutionResources]
    # The nodes each op's LABEL constraint allows (all allocated nodes when
    # unconstrained). This is the static, label-match half of eligibility -- it
    # does NOT encode free capacity and never changes within a tick. The dynamic
    # has-capacity half is read live off ``free_by_node``; a node freed by
    # ``revoke`` mid-tick becomes usable without touching this set.
    eligible_nodes_per_op: Dict[Any, AbstractSet[NodeIdStr]]
    per_actor_resources_by_op: Dict[Any, ExecutionResources]
    # How many constrained ops can use each node (lower = scarcer): grant order
    # + spread tie-break.
    overlapping_op_count_by_node: Dict[NodeIdStr, int] = field(default_factory=dict)
    # Remaining PG bundle capacity (actors) per PG op = cap - our live+pending in
    # the PG. These ops size against the PG, not the free map.
    pg_remaining_by_op: Dict[Any, int] = field(default_factory=dict)


class CapacityLedger(abc.ABC):
    """How ``scale_how_many`` bounds and debits per-op upscale capacity."""

    @abc.abstractmethod
    def max_placeable(self, op: Any, pool: Any) -> int:
        """Max additional actors of ``op`` that currently fit (reclaimable
        draining actors included -- reviving them needs no new capacity)."""
        ...

    @abc.abstractmethod
    def grant(self, op: Any, pool: Any, count: int) -> None:
        """Debit a grant of ``count`` actors so later ops see reduced capacity.

        ``count`` is the full delta; the ledger figures out how much of it is
        genuinely-new capacity (reclaimable draining actors, revived for free,
        are not debited)."""
        ...

    @abc.abstractmethod
    def revoke(self, op: Any, pool: Any, count: int) -> None:
        """Credit back capacity for ``count`` actors downscaled this tick so
        later grants this tick can reuse it."""
        ...


class GlobalCapacityLedger(CapacityLedger):
    """Maintains a global capacity aggregate which is then used to bound per-operator grants. Does not support label based/ placement groups. suffers from fragmentation, but is simpler and faster than the constraint-aware ledger."""

    def __init__(
        self,
        # Cluster-wide "how many more actors of this pool fit given resources
        # already granted this tick" (the sizer's ``_max_placeable_actors``).
        # Named ``_fn`` to avoid shadowing the module-level
        # ``max_placeable_actors`` free function, which is the per-node fit.
        max_placeable_fn: Callable[[Any, ExecutionResources], int],
        pool_resources: Callable[[Any, int], ExecutionResources],
    ):
        self._max_placeable_fn = max_placeable_fn
        self._pool_resources = pool_resources
        self._granted = ExecutionResources.zero()

    def max_placeable(self, op: Any, pool: Any) -> int:
        return (
            self._max_placeable_fn(pool, self._granted)
            + op.num_reclaimable_terminating_actors()
        )

    def grant(self, op: Any, pool: Any, count: int) -> None:
        # Reclaimable draining actors revive from their held slot -- only the
        # genuinely-new actors beyond them consume cluster capacity.
        to_debit = max(0, count - op.num_reclaimable_terminating_actors())
        self._granted = self._granted.add(self._pool_resources(pool, to_debit))

    def revoke(self, op: Any, pool: Any, count: int) -> None:
        # Credit the freed capacity back by un-granting it (``_granted`` may go
        # negative, which raises max_placeable above the current baseline). The
        # actors aren't dead until scale() applies, so this is optimistic and
        # reconciles next tick; Ray core is the safety net.
        self._granted = self._granted.subtract(self._pool_resources(pool, count))


class ConstraintAwareCapacityLedger(CapacityLedger):
    """Per-node, fragmentation-resilient, constraint-aware accounting."""

    def __init__(self, snapshot: TickSnapshot):
        self._snapshot = snapshot
        self._free: Dict[NodeIdStr, ExecutionResources] = {
            node_id: res.copy() for node_id, res in snapshot.free_by_node.items()
        }
        # PG ops size against the PG's remaining bundle capacity, not the free
        # map (their actors live in the PG reservation, outside our allocation).
        self._pg_remaining: Dict[Any, int] = dict(snapshot.pg_remaining_by_op)

    def _eligible_nodes(self, op: Any) -> List[NodeIdStr]:
        """The op's label-match nodes that currently have room, scarcity-first
        (low pressure first) so consuming them leaves contended nodes for ops
        that need them."""
        match = self._snapshot.eligible_nodes_per_op[op]
        pressure = self._snapshot.overlapping_op_count_by_node
        return sorted(
            (node_id for node_id in self._free if node_id in match),
            key=lambda node_id: (pressure.get(node_id, 0), node_id),
        )

    def max_placeable(self, op: Any, pool: Any) -> int:
        if op in self._pg_remaining:
            return self._pg_remaining[op]
        per = self._snapshot.per_actor_resources_by_op[op]
        total = 0.0
        for node_id in self._eligible_nodes(op):
            count = max_placeable_actors(self._free[node_id], per)
            if math.isinf(count):
                return _UNBOUNDED_PLACEABLE
            total += count
        # Draining actors this op can revive hold their (committed, i.e.
        # not-free) slot, so they place without consuming free capacity -- add
        # them on top of what the free map fits.
        return int(total) + op.num_reclaimable_terminating_actors()

    def grant(self, op: Any, pool: Any, count: int) -> None:
        if op in self._pg_remaining:
            # PG op: debit the PG's bundle counter, not the free map (Ray core
            # places the actor into a bundle -- see scale_where delegation).
            self._pg_remaining[op] -= min(count, self._pg_remaining[op])
            return
        # Reclaimable draining actors are revived from their held slot, so only
        # the genuinely-new actors beyond them consume free capacity.
        to_debit = max(0, count - op.num_reclaimable_terminating_actors())
        per = self._snapshot.per_actor_resources_by_op[op]
        placed = 0
        for node_id in self._eligible_nodes(op):
            while (
                placed < to_debit
                and max_placeable_actors(self._free[node_id], per) >= 1
            ):
                self._free[node_id] = self._free[node_id].subtract(per)
                placed += 1
            if placed >= to_debit:
                break

    def revoke(self, op: Any, pool: Any, count: int) -> None:
        if op in self._pg_remaining:
            self._pg_remaining[op] += count
            return
        # Credit freed capacity back onto the nodes where the op runs (densest
        # first -- the downscale sheds from the biggest concentration), so other
        # constraint-matching ops can grant onto them this same tick.
        actors_by_node = op.build_placement_view(need_victim_order=False).actors_by_node
        per = self._snapshot.per_actor_resources_by_op[op]
        remaining = count
        for node_id in sorted(actors_by_node, key=lambda n: (-actors_by_node[n], n)):
            if remaining <= 0:
                break
            if node_id not in self._free:
                continue
            give = min(remaining, actors_by_node[node_id])
            self._free[node_id] = self._free[node_id].add(per.scale(give))
            remaining -= give
