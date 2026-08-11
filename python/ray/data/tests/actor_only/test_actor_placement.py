"""Unit tests for ``LocalityBasedActorPlacement`` (pure logic, no Ray cluster).

Covers max-not-sum collocation demand with the idle-gated queue buffer, live
re-ranked collocation, the closed-form proportional spread, and cost-class-major
downscale victim ordering.
"""
from collections import Counter

import pytest

from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    OpPlacementView,
)
from ray.data._internal.experimental.execution.sizer.locality_actor_placement import (
    LocalityBasedActorPlacement,
)


def _view(**overrides) -> OpPlacementView:
    base = dict(
        op_id="op",
        is_source=False,
        per_actor_usage=ExecutionResources(cpu=1),
        input_bytes_per_actor=None,
        actors_by_node={},
        input_queue_bytes_by_node={},
        upstream_actors_by_node={},
        downstream_actors_by_node={},
        actor_ids_by_node={},
        idle_actors_by_node={},
        pending_ids_by_node={},
    )
    base.update(overrides)
    return OpPlacementView(**base)


def _free(**by_node) -> dict:
    return {n: ExecutionResources(cpu=cpu) for n, cpu in by_node.items()}


def _victim_view() -> OpPlacementView:
    # Node A: 1 pending + [idle, busy]; B: [idle, busy] (queue 10);
    # C: [idle] (queue 100). idle = the 0-task prefix of actor_ids_by_node.
    return _view(
        pending_ids_by_node={"A": ["pA"]},
        actor_ids_by_node={"A": ["iA", "bA"], "B": ["iB", "bB"], "C": ["iC"]},
        idle_actors_by_node={"A": 1, "B": 1, "C": 1},
        input_queue_bytes_by_node={"B": 10, "C": 100},
    )


def test_locality_demand_is_max_not_sum():
    """1 producer + 1 consumer on a node need 1 bridging actor, not 2."""
    p = LocalityBasedActorPlacement()
    view = _view(
        upstream_actors_by_node={"A": 1},
        downstream_actors_by_node={"A": 1},
    )
    placed = Counter(p.place_upscale(view, _free(A=10, B=10), 3))
    assert p.last_upscale_breakdown["op"] == (1, 2)
    assert placed["A"] >= 1


def test_locality_queue_buffer_gated_on_idle():
    """Queued input adds collocation demand only when no local actor is idle."""
    # No idle actor -> the 300-byte queue converts to 3 actors of demand.
    p = LocalityBasedActorPlacement()
    view = _view(input_queue_bytes_by_node={"A": 300}, input_bytes_per_actor=100)
    p.place_upscale(view, _free(A=10, B=10), 8)
    assert p.last_upscale_breakdown["op"][0] == 3

    # Same queue, but an idle actor on the node -> queue buffer suppressed.
    view = _view(
        input_queue_bytes_by_node={"A": 300},
        input_bytes_per_actor=100,
        actors_by_node={"A": 1},
        actor_ids_by_node={"A": ["a1"]},
        idle_actors_by_node={"A": 1},
    )
    p.place_upscale(view, _free(A=10, B=10), 8)
    assert p.last_upscale_breakdown["op"][0] == 0

    # A pending actor gates the buffer like an idle one: it will absorb the
    # queue as soon as it lands, so the queue justifies no additional actors.
    view = _view(
        input_queue_bytes_by_node={"A": 300},
        input_bytes_per_actor=100,
        actors_by_node={"A": 1},
        pending_ids_by_node={"A": ["p1"]},
    )
    p.place_upscale(view, _free(A=10, B=10), 8)
    assert p.last_upscale_breakdown["op"][0] == 0


def test_locality_live_rerank_interleaves():
    """Placing on a node decays its rank, so a big-queue node doesn't take the
    whole upscale: queue pressure per actor equalizes across nodes."""
    p = LocalityBasedActorPlacement()
    view = _view(
        upstream_actors_by_node={"A": 3, "B": 1},
        downstream_actors_by_node={"B": 1},
        actors_by_node={"A": 1},
        input_queue_bytes_by_node={"A": 400, "B": 100},
        input_bytes_per_actor=200,
    )
    # Walk (imbalance first, then queue pressure): A imb 2 -> imb tie at 1,
    # A wins on pressure 400/3 > 100 -> A exhausts imbalance, B's 1 remains
    # -> B -> imb tie at 0, A wins on pressure 100 > 50.
    # Demands: A = max(3,0)-1+2 = 4, B = max(1,1)-0+1 = 2.
    assert p.place_upscale(view, _free(A=10, B=10), 4) == ["A", "A", "B", "A"]
    assert p.last_upscale_breakdown["op"] == (4, 0)


def test_locality_imbalance_outranks_queue_pressure():
    """Unmatched producers (a durable rate-mismatch signal) beat a big queue
    (a noisy snapshot): balance the node first, then work on the buffer."""
    p = LocalityBasedActorPlacement()
    view = _view(
        upstream_actors_by_node={"A": 2, "B": 1},
        input_queue_bytes_by_node={"B": 1000},
        input_bytes_per_actor=100,
    )
    # Queue-first ranking would send both to B. Imbalance-first sends the
    # first to A (imb 2 > 1); at the imbalance tie, B's queue wins the second.
    assert p.place_upscale(view, _free(A=10, B=10), 2) == ["A", "B"]


def test_locality_spread_proportional_to_capacity():
    """Spread is proportional to free slots (not water-filling): a node with
    8x the capacity gets 8x the actors."""
    p = LocalityBasedActorPlacement()
    placed = Counter(p.place_upscale(_view(), _free(A=64, B=8), 9))
    assert placed == Counter({"A": 8, "B": 1})


def test_locality_spread_leftovers_by_largest_remainder():
    p = LocalityBasedActorPlacement()
    # Quotas are 4 * 2/6 = 1.33 each: floors give 1 apiece, the leftover goes
    # to the remainder tiebreak (fewest of ours, then node id) -> A.
    placed = Counter(p.place_upscale(_view(), _free(A=2, B=2, C=2), 4))
    assert placed == Counter({"A": 2, "B": 1, "C": 1})
    assert sum(placed.values()) == 4


def test_locality_spread_shortfall_returns_short():
    p = LocalityBasedActorPlacement()
    placed = p.place_upscale(_view(), _free(A=1, B=1), 10)
    assert Counter(placed) == Counter({"A": 1, "B": 1})


def test_locality_spread_all_nodes_full_places_nothing():
    """Every node at zero capacity: place nothing (and don't divide by zero)."""
    p = LocalityBasedActorPlacement()
    assert p.place_upscale(_view(), _free(A=0, B=0), 3) == []


def test_locality_spread_even_when_actor_requests_no_resources():
    """max_placeable_actors is inf for a resource-less actor: split evenly, and the
    total-capacity clamp must NOT bound the placement to the node count."""
    p = LocalityBasedActorPlacement()
    view = _view(per_actor_usage=ExecutionResources())
    placed = Counter(p.place_upscale(view, _free(A=4, B=4, C=4), 6))
    assert placed == Counter({"A": 2, "B": 2, "C": 2})
    # More actors than nodes: all 10 land (unbounded capacity), 5 apiece.
    placed = Counter(p.place_upscale(view, _free(A=1, B=1), 10))
    assert placed == Counter({"A": 5, "B": 5})


def test_locality_victims_all_cheap_before_any_busy():
    """Cost-class-major: pending/idle victims on EVERY node are taken before a
    favored node's busy actor (node-major would take bA third)."""
    p = LocalityBasedActorPlacement()
    assert p.pick_downscale_victims(_victim_view(), 4, {"A"}) == [
        "pA",
        "iA",
        "iB",
        "iC",
    ]
    # Busy actors follow, favored node first.
    assert p.pick_downscale_victims(_victim_view(), 5, {"A"}) == [
        "pA",
        "iA",
        "iB",
        "iC",
        "bA",
    ]


def test_locality_victims_favored_first_within_class():
    p = LocalityBasedActorPlacement()
    assert p.pick_downscale_victims(_victim_view(), 2, {"A"}) == ["pA", "iA"]
    # No favored nodes: cheap class walks nodes by queue size (A=0, B=10, C=100).
    assert p.pick_downscale_victims(_victim_view(), 3, set()) == ["pA", "iA", "iB"]


def test_locality_victims_full_drain_returns_all_pendings_first():
    p = LocalityBasedActorPlacement()
    victims = p.pick_downscale_victims(_victim_view(), 99, {"A"})
    assert victims[0] == "pA"
    assert sorted(victims) == ["bA", "bB", "iA", "iB", "iC", "pA"]


def test_locality_favored_nodes_ignore_queue_on_slow_path():
    """Each starved op nominates top-unmet nodes per reclaim timescale; the
    slow (drain-to-free) tier ranks on structure only — C's big queue would
    win the imbalance tie if it counted, but the snapshot is stale by drain
    time, so the node-id tiebreak picks A."""
    p = LocalityBasedActorPlacement()
    view = _view(
        upstream_actors_by_node={"A": 2, "C": 2},
        input_queue_bytes_by_node={"C": 1000},
    )
    favored = p.compute_favored_nodes(
        [(view, 1)], free={}, all_nodes={"A", "B", "C"}, fast_free_nodes={"B"}
    )
    assert favored == {"A", "B"}
    # No starved ops -> no favored nodes.
    assert p.compute_favored_nodes([(view, 0)], {}, {"A", "B"}, {"B"}) == set()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
