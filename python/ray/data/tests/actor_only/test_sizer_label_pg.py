# ABOUTME: Unit tests for label/PG-aware sizing: constraints, ledger, ordering, sizer.
# ABOUTME: Pure-Python doubles (reuses test_operator_sizer fakes); no cluster needed.

import pytest
from test_operator_sizer import _FakeAPMO, _FakeCoordinator, _FakePool, _topology

from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.experimental.execution.actor_only_resource_reporter import (
    ActorOnlyResourceReporter,
    ClusterView,
)
from ray.data._internal.experimental.execution.operators.placement_constraints import (
    PlacementConstraint,
)
from ray.data._internal.experimental.execution.sizer.capacity_ledger import (
    ConstraintAwareCapacityLedger,
    TickSnapshot,
)
from ray.data._internal.experimental.execution.sizer.operator_ordering import (
    OpOrderingSignals,
    OrderingPolicy,
    ReverseTopologicalOrdering,
    ScarcityFirstOrdering,
    make_ordering_policy,
)
from ray.data._internal.experimental.execution.sizer.operator_sizer import OperatorSizer

_NODE_ID_KEY = "ray.io/node-id"

# --------------------------------------------------------------------------- #
# PlacementConstraint.matching_nodes — Ray-core selector grammar.
# --------------------------------------------------------------------------- #


def _view():
    alloc = {
        "n1": ExecutionResources(cpu=16),
        "g1": ExecutionResources(gpu=4),
        "g2": ExecutionResources(gpu=4),
        "g3": ExecutionResources(gpu=4),
    }
    labels = {
        "n1": {_NODE_ID_KEY: "n1"},
        "g1": {_NODE_ID_KEY: "g1", "accelerator-type": "A100"},
        "g2": {_NODE_ID_KEY: "g2", "accelerator-type": "A100"},
        "g3": {_NODE_ID_KEY: "g3", "accelerator-type": "L4"},
    }
    return ClusterView(alloc_by_node=alloc, labels_by_node=labels)


@pytest.mark.parametrize(
    "selector,expected",
    [
        ({"accelerator-type": "A100"}, {"g1", "g2"}),  # exact
        ({"accelerator-type": "!A100"}, {"n1", "g3"}),  # negation
        # NOTE: Ray core's matcher is whitespace-sensitive inside in(...); we
        # delegate to it verbatim, so values must not have leading spaces.
        ({"accelerator-type": "in(A100,L4)"}, {"g1", "g2", "g3"}),  # in-set
        ({"accelerator-type": "!in(A100)"}, {"n1", "g3"}),  # not-in-set
        ({"accelerator-type": "H100"}, set()),  # matches nothing
    ],
)
def test_selector_matching_grammar(selector, expected):
    c = PlacementConstraint(label_selector=selector)
    assert set(c.matching_nodes(_view())) == expected


def test_unconstrained_matches_all_nodes():
    assert set(PlacementConstraint().matching_nodes(_view())) == {
        "n1",
        "g1",
        "g2",
        "g3",
    }


def test_pin_is_a_node_id_selector():
    # A node pin is just a selector on the node-id label -> the single node.
    c = PlacementConstraint(label_selector={_NODE_ID_KEY: "n1"})
    assert set(c.matching_nodes(_view())) == {"n1"}


def test_strictness_selector_then_unconstrained():
    assert PlacementConstraint(label_selector={"a": "b"}).strictness() == 1
    assert PlacementConstraint().strictness() == 2


def test_unparseable_selector_degrades_to_all_nodes(caplog):
    # A value Ray core's matcher rejects -> degrade to "eligible everywhere"
    # with a warning, never an empty set.
    c = PlacementConstraint(label_selector={"accelerator-type": "in()"})
    assert set(c.matching_nodes(_view())) == {"n1", "g1", "g2", "g3"}


def test_matching_nodes_filters_nodes_lacking_requested_resource():
    # In _view(), n1 is CPU-only and g1/g2/g3 are GPU-only. An op is eligible
    # only on nodes whose allocation carries every resource it requests.
    view = _view()
    gpu_actor = ExecutionResources(gpu=1)
    cpu_actor = ExecutionResources(cpu=1)

    # Unconstrained (no selector) but GPU-requesting -> only the GPU nodes.
    assert set(PlacementConstraint().matching_nodes(view, gpu_actor)) == {
        "g1",
        "g2",
        "g3",
    }
    # CPU-requesting -> only the CPU node.
    assert set(PlacementConstraint().matching_nodes(view, cpu_actor)) == {"n1"}
    # per_actor=None keeps today's behavior: every allocated node.
    assert set(PlacementConstraint().matching_nodes(view)) == {"n1", "g1", "g2", "g3"}


def test_matching_nodes_intersects_label_and_resource():
    # A GPU op pinned to A100 gets label-match {g1, g2} (g3 is L4); the resource
    # filter is a no-op here since both A100 nodes have GPUs.
    view = _view()
    c = PlacementConstraint(label_selector={"accelerator-type": "A100"})
    assert set(c.matching_nodes(view, ExecutionResources(gpu=1))) == {"g1", "g2"}


# --------------------------------------------------------------------------- #
# ConstraintAwareCapacityLedger — per-node, fragmentation-resilient, overlapping sets, revoke, PG.
# --------------------------------------------------------------------------- #


def _snapshot(
    free,
    eligible_nodes_per_op,
    per_actor_resources_by_op,
    pressure=None,
    pg_remaining=None,
):
    # None match = unconstrained = all nodes (matching_nodes never returns None).
    eligible_nodes_per_op = {
        op: (frozenset(free) if match is None else match)
        for op, match in eligible_nodes_per_op.items()
    }
    return TickSnapshot(
        free_by_node=free,
        alloc_by_node=dict(free),
        eligible_nodes_per_op=eligible_nodes_per_op,
        per_actor_resources_by_op=per_actor_resources_by_op,
        overlapping_op_count_by_node=pressure or {},
        pg_remaining_by_op=pg_remaining or {},
    )


class _LedgerOp:
    """Duck-typed op the ledger reads reclaimable/actor-location off of.

    The ledger now owns the reclaimable-draining-actor accounting, so it pulls
    ``num_reclaimable_terminating_actors`` and (for revoke) ``build_placement_view``
    straight off the op instead of the sizer threading them in."""

    def __init__(self, reclaimable=0, actors_by_node=None):
        self._reclaimable = reclaimable
        self._actors_by_node = dict(actors_by_node or {})

    def num_reclaimable_terminating_actors(self):
        return self._reclaimable

    def build_placement_view(self, need_victim_order=False):
        return type("V", (), {"actors_by_node": dict(self._actors_by_node)})()


def test_ledger_max_placeable_per_node_fragmentation():
    # 2 nodes x 3 CPU, actor needs 2 CPU: per-node floor (1+1=2), NOT global
    # floor(6/2)=3. This is the fragmentation-resilience the aggregate lacked.
    op = _LedgerOp()
    free = {"a": ExecutionResources(cpu=3), "b": ExecutionResources(cpu=3)}
    snap = _snapshot(free, {op: None}, {op: ExecutionResources(cpu=2)})
    assert ConstraintAwareCapacityLedger(snap).max_placeable(op, None) == 2


def test_ledger_overlapping_sets_contend():
    # op1 eligible on all; op2 only on the shared GPU nodes. Granting op2 first
    # consumes the shared nodes, so op1 then sees less.
    op1, op2 = _LedgerOp(), _LedgerOp()
    free = {
        "n1": ExecutionResources(cpu=8),
        "g1": ExecutionResources(gpu=4),
        "g2": ExecutionResources(gpu=4),
    }
    match = {op1: None, op2: frozenset({"g1", "g2"})}
    per = {op1: ExecutionResources(gpu=1), op2: ExecutionResources(gpu=1)}
    ledger = ConstraintAwareCapacityLedger(_snapshot(free, match, per))
    assert ledger.max_placeable(op2, None) == 8  # 2 GPU nodes x 4
    ledger.grant(op2, None, 8)
    assert ledger.max_placeable(op2, None) == 0
    # op1 (GPU actor) can no longer use the drained GPU nodes.
    assert ledger.max_placeable(op1, None) == 0


def test_ledger_revoke_returns_capacity():
    op = _LedgerOp(actors_by_node={"g1": 4})
    free = {"g1": ExecutionResources(gpu=4)}
    snap = _snapshot(free, {op: None}, {op: ExecutionResources(gpu=1)})
    ledger = ConstraintAwareCapacityLedger(snap)
    ledger.grant(op, None, 4)
    assert ledger.max_placeable(op, None) == 0
    # revoke reads the op's per-node actor locations itself (build_placement_view).
    ledger.revoke(op, None, 2)
    assert ledger.max_placeable(op, None) == 2


def test_ledger_reclaimable_placed_without_consuming_free():
    # A node with room for 4 actors + 3 draining actors this op can revive.
    # max_placeable = 4 (free) + 3 (reclaim) = 7. Granting the 3 reclaimable
    # revives them from their held slots and consumes ZERO free, so a peer op
    # contending for the same node still sees the full 4.
    op = _LedgerOp(reclaimable=3)
    peer = _LedgerOp()
    free = {"n1": ExecutionResources(cpu=4)}
    per = {op: ExecutionResources(cpu=1), peer: ExecutionResources(cpu=1)}
    ledger = ConstraintAwareCapacityLedger(_snapshot(free, {op: None, peer: None}, per))
    assert ledger.max_placeable(op, None) == 7  # 4 free + 3 reclaim
    ledger.grant(op, None, 3)  # all 3 revived from held slots, no free debited
    assert ledger.max_placeable(peer, None) == 4  # node untouched
    # Granting 2 more now spills past the reclaimable pool -> 2 debit free.
    ledger.grant(op, None, 5)
    assert ledger.max_placeable(peer, None) == 2  # 4 - 2 debited


def test_ledger_pg_cap_path():
    # PG op sizes against pg_remaining, not the free map.
    op = _LedgerOp()
    snap = _snapshot(
        {"n1": ExecutionResources(cpu=8)},
        {op: None},
        {op: ExecutionResources(gpu=1)},
        pg_remaining={op: 3},
    )
    ledger = ConstraintAwareCapacityLedger(snap)
    assert ledger.max_placeable(op, None) == 3
    ledger.grant(op, None, 5)  # capped at remaining
    assert ledger.max_placeable(op, None) == 0
    ledger.revoke(op, None, 1)
    assert ledger.max_placeable(op, None) == 1


def test_global_ledger_revoke_reuses_downscaled_capacity_same_tick():
    from ray.data._internal.experimental.execution.sizer.capacity_ledger import (
        GlobalCapacityLedger,
    )

    a, b = _LedgerOp(), _LedgerOp()

    # Baseline capacity of 5 actors; each grant/revoke moves it by whole actors.
    def max_placeable_actors(pool, granted):
        return int(5 - granted.cpu)

    def pool_resources(pool, n):
        return ExecutionResources(cpu=float(n))

    ledger = GlobalCapacityLedger(max_placeable_actors, pool_resources)
    ledger.grant(a, None, 5)
    assert ledger.max_placeable(a, None) == 0
    # A same-tick downscale of 2 returns capacity for later ops this tick.
    ledger.revoke(b, None, 2)
    assert ledger.max_placeable(b, None) == 2


def test_global_ledger_reclaimable_added_and_not_debited():
    # The old path now mirrors the new one: an op's reclaimable draining actors
    # add to max_placeable and revive without debiting cluster capacity.
    from ray.data._internal.experimental.execution.sizer.capacity_ledger import (
        GlobalCapacityLedger,
    )

    op = _LedgerOp(reclaimable=3)
    peer = _LedgerOp()

    def max_placeable_actors(pool, granted):
        return int(4 - granted.cpu)  # baseline room for 4 new actors

    def pool_resources(pool, n):
        return ExecutionResources(cpu=float(n))

    ledger = GlobalCapacityLedger(max_placeable_actors, pool_resources)
    assert ledger.max_placeable(op, None) == 7  # 4 baseline + 3 reclaim
    ledger.grant(op, None, 3)  # all 3 revived -> nothing debited
    assert ledger.max_placeable(peer, None) == 4  # baseline untouched


def _reporter_over(*ops):
    return ActorOnlyResourceReporter(
        topology=list(ops), autoscaling_coordinator=None, resource_bank=None
    )


def test_reporter_free_and_committed_by_node():
    a = _FakeAPMO("a", _FakePool(min_size=1, max_size=10, cpu=1))
    b = _FakeAPMO("b", _FakePool(min_size=1, max_size=10, cpu=1))
    pg = _FakeAPMO("pg", _FakePool(min_size=1, max_size=10, gpu=1))
    a._placement_constraint = PlacementConstraint()
    b._placement_constraint = PlacementConstraint()
    pg._placement_constraint = PlacementConstraint(placement_group=object())
    a.committed_usage_by_node = lambda: {
        "n1": ExecutionResources(cpu=2),
        "n2": ExecutionResources(cpu=3),
    }
    b.committed_usage_by_node = lambda: {"n1": ExecutionResources(cpu=1)}
    # PG op usage must be excluded (its actors live in the PG reservation).
    pg.committed_usage_by_node = lambda: {"n1": ExecutionResources(gpu=1)}
    reporter = _reporter_over(a, b, pg)

    committed = reporter.committed_by_node()
    assert committed["n1"].cpu == 3  # a(2) + b(1)
    assert committed["n2"].cpu == 3
    assert committed["n1"].gpu == 0  # PG op's usage excluded

    alloc = {"n1": ExecutionResources(cpu=4), "n2": ExecutionResources(cpu=4)}
    free = reporter.free_by_node(alloc)
    assert free["n1"].cpu == 1 and free["n2"].cpu == 1

    # Over-commit clamps free at 0, never negative.
    tight = {"n1": ExecutionResources(cpu=1), "n2": ExecutionResources(cpu=4)}
    assert reporter.free_by_node(tight)["n1"].cpu == 0


# --------------------------------------------------------------------------- #
# Ordering policies.
# --------------------------------------------------------------------------- #


def _feat(i, strict=2, pg=False, gpu=False, fixed=False, mem=False, eligible=None):
    return OpOrderingSignals(
        topo_index=i,
        strictness=strict,
        has_placement_group=pg,
        is_gpu=gpu,
        is_fixed_concurrency=fixed,
        is_large_memory=mem,
        num_eligible_nodes=eligible,
    )


def test_reverse_topological_is_sink_most_first():
    feats = {"read": _feat(0), "embed": _feat(1), "write": _feat(2)}
    assert ReverseTopologicalOrdering().order(feats) == ["write", "embed", "read"]


def test_scarcity_first_tiers():
    feats = {
        "read": _feat(0),  # tier 5 (unconstrained cpu)
        "embed": _feat(1, strict=1, gpu=True),  # tier 1 (selector)
        "write": _feat(2),  # tier 5, sink-most
        "pg": _feat(3, pg=True),  # tier 0
        "gpu": _feat(4, gpu=True),  # tier 2
    }
    # pg(0) < embed(1) < gpu(2) < write(5, topo2) < read(5, topo0)
    assert ScarcityFirstOrdering().order(feats) == [
        "pg",
        "embed",
        "gpu",
        "write",
        "read",
    ]


def test_scarcity_first_sink_most_within_tier():
    feats = {"a": _feat(0, gpu=True), "b": _feat(1, gpu=True)}
    assert ScarcityFirstOrdering().order(feats) == ["b", "a"]


def test_scarcity_first_specificity_within_label_tier():
    # Within the label tier, more specific (fewer eligible nodes) goes first,
    # overriding sink-most: a node pin (1 eligible) precedes a 2-node pool
    # selector precedes a 4-node in(...) selector, regardless of topo order.
    feats = {
        "in_ab": _feat(3, strict=1, eligible=4),
        "pin": _feat(0, strict=1, eligible=1),
        "pool": _feat(2, strict=1, eligible=2),
    }
    assert ScarcityFirstOrdering().order(feats) == ["pin", "pool", "in_ab"]


def test_scarcity_first_unknown_specificity_sorts_last_in_tier():
    # None (unknown eligibility, e.g. outside a constraint-aware tick) sorts
    # after known counts; sink-most breaks the tie among unknowns.
    feats = {
        "known": _feat(0, strict=1, eligible=3),
        "unknown_a": _feat(1, strict=1),
        "unknown_b": _feat(2, strict=1),
    }
    assert ScarcityFirstOrdering().order(feats) == [
        "known",
        "unknown_b",
        "unknown_a",
    ]


def test_scarcity_first_specificity_does_not_cross_tiers():
    # A very specific label op still sorts after any PG op, and a broad label
    # op still sorts before any GPU/unconstrained op: specificity only breaks
    # ties WITHIN a tier.
    feats = {
        "pg": _feat(0, pg=True, eligible=100),
        "pin": _feat(1, strict=1, eligible=1),
        "gpu": _feat(2, gpu=True, eligible=1),
    }
    assert ScarcityFirstOrdering().order(feats) == ["pg", "pin", "gpu"]


def test_ordering_policy_from_env_and_factory():
    assert OrderingPolicy.from_env("scarcity_first") is OrderingPolicy.SCARCITY_FIRST
    # Unknown env value degrades to reverse-topological.
    assert OrderingPolicy.from_env("nope") is OrderingPolicy.REVERSE_TOPOLOGICAL
    assert isinstance(
        make_ordering_policy(OrderingPolicy.REVERSE_TOPOLOGICAL),
        ReverseTopologicalOrdering,
    )
    assert isinstance(
        make_ordering_policy(OrderingPolicy.SCARCITY_FIRST), ScarcityFirstOrdering
    )


def test_ordered_apmo_ops_uses_snapshot_eligibility_for_specificity(monkeypatch):
    """End-to-end wiring: with a constraint-aware tick snapshot present, a
    node-pinned op (1 eligible node) is ordered before an overlapping pool
    selector (2 eligible nodes) even though sink-most alone would reverse
    them. Without a snapshot (bootstrap / initial sizing), ordering falls
    back to sink-most within the tier."""
    alloc = {
        "a1": ExecutionResources(cpu=8),
        "a2": ExecutionResources(cpu=8),
    }
    labels = {
        "a1": {_NODE_ID_KEY: "a1", "ray-pool": "pool_a"},
        "a2": {_NODE_ID_KEY: "a2", "ray-pool": "pool_a"},
    }
    view = ClusterView(alloc_by_node=alloc, labels_by_node=labels)

    # pinned is UPSTREAM of pool: sink-most within the tier would put pool
    # first; specificity must override that.
    pinned = _FakeAPMO("pinned", _FakePool(min_size=1, max_size=4, cpu=1))
    pool = _FakeAPMO("pool", _FakePool(min_size=1, max_size=4, cpu=1), inputs=[pinned])
    pinned._ray_remote_args = {"label_selector": {_NODE_ID_KEY: "a1"}}
    pool._ray_remote_args = {"label_selector": {"ray-pool": "pool_a"}}
    _stub_pool_accounting(pinned, pool)

    topo = _topology(pinned, pool)
    sizer = _label_aware_sizer(view, monkeypatch)
    sizer.bootstrap(topology=topo)

    # No snapshot yet -> sink-most within the tier: pool (downstream) first.
    sizer._tick_snapshot = None
    assert sizer._ordered_apmo_ops(topo) == [pool, pinned]

    # With the tick snapshot, the pin (1 eligible node) outranks the pool
    # selector (2 eligible nodes).
    sizer._tick_snapshot = sizer._build_tick_snapshot(topo)
    assert sizer._ordered_apmo_ops(topo) == [pinned, pool]


# --------------------------------------------------------------------------- #
# Sizer label-aware integration: constrained op capped + shaped unmet demand.
# --------------------------------------------------------------------------- #


def _label_aware_sizer(cluster_view, monkeypatch):
    monkeypatch.setattr(
        "ray.data._internal.experimental.execution.sizer.operator_sizer."
        "get_draining_nodes",
        lambda: {},
    )
    sizer = OperatorSizer(dataset_id="ds")
    sizer._resource_reporter = ActorOnlyResourceReporter(
        topology={},
        autoscaling_coordinator=_FakeCoordinator(cpu=0, gpu=0),
        resource_bank=None,
    )
    sizer._resource_reporter.get_cluster_view = lambda **k: cluster_view
    sizer._resource_reporter.get_reserved_resources_by_node = lambda: dict(
        cluster_view.alloc_by_node
    )
    # PG bundle shapes come pre-populated on ``cluster_view``; skip the real
    # ``placement_group_table`` fetch so the tests stay hermetic.
    sizer._resource_reporter.prefetch_pg_bundles = lambda pgs: None
    sizer._constraint_aware = True
    sizer._ordering = make_ordering_policy(OrderingPolicy.SCARCITY_FIRST)
    return sizer


def _stub_pool_accounting(*ops):
    for op in ops:
        op.committed_usage_by_node = lambda: {}
        op.num_reclaimable_terminating_actors = lambda: 0
        op.build_placement_view = lambda need_victim_order=False: type(
            "V", (), {"actors_by_node": {}}
        )()


def test_label_aware_caps_constrained_op_and_shapes_demand(monkeypatch):
    alloc = {
        "n1": ExecutionResources(cpu=16),
        "g1": ExecutionResources(gpu=4),
        "g2": ExecutionResources(gpu=4),
        "g3": ExecutionResources(gpu=4),  # L4 -- must NOT be used by the A100 op
    }
    labels = {
        "n1": {},
        "g1": {"accelerator-type": "A100"},
        "g2": {"accelerator-type": "A100"},
        "g3": {"accelerator-type": "L4"},
    }
    view = ClusterView(alloc_by_node=alloc, labels_by_node=labels)

    read = _FakeAPMO("read", _FakePool(min_size=1, max_size=500, cpu=1, running=1))
    embed = _FakeAPMO(
        "embed",
        _FakePool(min_size=1, max_size=500, gpu=1, running=20, in_flight=50),
        inputs=[read],
    )
    read._ray_remote_args = {}
    embed._ray_remote_args = {"label_selector": {"accelerator-type": "A100"}}
    _stub_pool_accounting(read, embed)

    topo = _topology(read, embed, enqueued=10_000)
    sizer = _label_aware_sizer(view, monkeypatch)
    sizer.bootstrap(topology=topo)
    sizer._warmup = False
    read._pool.in_flight = 50

    # embed matches only the 2 A100 nodes -> 8 GPU slots; L4 (g3) excluded.
    # read requests CPU, so it's eligible only where CPU is allocated (n1); the
    # GPU-only nodes (g1/g2/g3 have zero CPU) are filtered out by the
    # constrained-resource check in matching_nodes.
    snap = sizer._build_tick_snapshot(topo)
    assert set(snap.eligible_nodes_per_op[embed]) == {"g1", "g2"}
    assert set(snap.eligible_nodes_per_op[read]) == {"n1"}

    req = sizer.scale_how_many(topology=topo)
    # embed wants ceil(0.5*20)=10 but only 8 A100 GPUs are free -> capped at 8.
    assert req.request[embed].delta == 8
    # The 2 unplaced actors are shaped demand tagged with the A100 selector.
    bundles = sizer.get_unmet_demand_bundles()
    embed_bundles = [b for b in bundles if b[2] == {"accelerator-type": "A100"}]
    assert embed_bundles and sum(b[1] for b in embed_bundles) == 2


def test_below_min_constrained_op_granted_full_min(monkeypatch):
    """A label-constrained below-min op must be granted its FULL min from its
    eligible nodes. Regression test for the (since removed) one-slot-per-
    unstarted-peer reservation being carved out of a constrained op's private
    (scarce) capacity: a node-pinned pool was held at 1 of min 4 while its
    node's remaining CPUs were granted to an unconstrained peer (release
    flake, build 7786 node case)."""
    alloc = {
        "picked": ExecutionResources(cpu=8),
        "n2": ExecutionResources(cpu=8),
        "n3": ExecutionResources(cpu=8),
    }
    labels = {"picked": {"ray.io/node-id": "picked"}, "n2": {}, "n3": {}}
    view = ClusterView(alloc_by_node=alloc, labels_by_node=labels)

    # Pinned op: min 4 x 2 CPUs = exactly the picked node. Three unstarted
    # unconstrained peers would previously reserve 3 of its 4 slots.
    peer1 = _FakeAPMO("peer1", _FakePool(min_size=1, max_size=10, cpu=1))
    pinned = _FakeAPMO(
        "pinned", _FakePool(min_size=4, max_size=4, cpu=2), inputs=[peer1]
    )
    peer2 = _FakeAPMO(
        "peer2", _FakePool(min_size=1, max_size=10, cpu=1), inputs=[pinned]
    )
    peer3 = _FakeAPMO(
        "peer3", _FakePool(min_size=1, max_size=10, cpu=1), inputs=[peer2]
    )
    pinned._ray_remote_args = {"label_selector": {"ray.io/node-id": "picked"}}
    for peer in (peer1, peer2, peer3):
        peer._ray_remote_args = {}
    _stub_pool_accounting(pinned, peer1, peer2, peer3)

    topo = _topology(peer1, pinned, peer2, peer3)
    sizer = _label_aware_sizer(view, monkeypatch)
    sizer.bootstrap(topology=topo)

    req = sizer.scale_how_many(topology=topo)
    # Full min for the constrained op, not min minus peer reservations.
    assert req.request[pinned].delta == 4
    # Unconstrained peers still get their first actors from the shared pool.
    for peer in (peer1, peer2, peer3):
        assert req.request[peer].delta >= 1


# --------------------------------------------------------------------------- #
# Placement groups (Phase 3).
# --------------------------------------------------------------------------- #


def test_pg_actors_excluded_from_committed_usage():
    # A PG actor consumes the PG reservation, not our allocation; its node must
    # NOT be subtracted from the free map (else other ops there are starved).
    pg_op = _FakeAPMO("pg", _FakePool(min_size=1, max_size=10, gpu=1, running=2))
    normal = _FakeAPMO("normal", _FakePool(min_size=1, max_size=10, cpu=1, running=2))
    # Constraints are cached on the op itself; pre-seed the cache.
    pg_op._placement_constraint = PlacementConstraint(placement_group=object())
    normal._placement_constraint = PlacementConstraint()
    pg_op.committed_usage_by_node = lambda: {"g1": ExecutionResources(gpu=2)}
    normal.committed_usage_by_node = lambda: {"n1": ExecutionResources(cpu=2)}

    committed = _reporter_over(pg_op, normal).committed_by_node()
    assert "g1" not in committed  # PG op's node excluded
    assert committed["n1"].cpu == 2  # normal op still counted


def test_gpu_nodes_are_scarce_in_pressure(monkeypatch):
    # A GPU op with NO label_selector still marks GPU nodes as scarce, so a plain
    # CPU op deprioritizes them; a CPU-only node stays unpressured.
    view = ClusterView(
        alloc_by_node={
            "n1": ExecutionResources(cpu=16),
            "g1": ExecutionResources(cpu=8, gpu=4),
        },
        labels_by_node={"n1": {}, "g1": {}},
    )
    cpu_op = _FakeAPMO("read", _FakePool(min_size=1, max_size=10, cpu=1, running=1))
    gpu_op = _FakeAPMO(
        "embed",
        _FakePool(min_size=1, max_size=10, gpu=1, running=1),
        inputs=[cpu_op],
    )
    cpu_op._ray_remote_args = {}
    gpu_op._ray_remote_args = {}
    _stub_pool_accounting(cpu_op, gpu_op)
    topo = _topology(cpu_op, gpu_op)
    sizer = _label_aware_sizer(view, monkeypatch)
    sizer.bootstrap(topology=topo)

    snap = sizer._build_tick_snapshot(topo)
    # g1 (has GPU) gets pressure from the GPU op; the CPU op adds none anywhere.
    assert snap.overlapping_op_count_by_node.get("g1", 0) == 1
    assert snap.overlapping_op_count_by_node.get("n1", 0) == 0


class _FakePG:
    """Minimal placement-group stand-in exposing ``.id.hex()`` for tests."""

    def __init__(self, id_hex):
        self.id = type("Id", (), {"hex": lambda _self, h=id_hex: h})()


def test_pg_capacity_is_per_bundle():
    pg = _FakePG("pg1")
    view = ClusterView(
        alloc_by_node={},
        pg_bundles={"pg1": [ExecutionResources(gpu=2), ExecutionResources(gpu=2)]},
    )
    # 1-GPU actor: each 2-GPU bundle holds 2 -> 4 across both bundles.
    assert (
        PlacementConstraint(placement_group=pg).pg_capacity(
            ExecutionResources(gpu=1), view
        )
        == 4
    )
    # 4-GPU actor: fits zero 2-GPU bundles even though the PG totals 4 GPU.
    assert (
        PlacementConstraint(placement_group=pg).pg_capacity(
            ExecutionResources(gpu=4), view
        )
        == 0
    )
    # bundle_index restricts capacity to a single bundle.
    assert (
        PlacementConstraint(
            placement_group=pg, placement_group_bundle_index=0
        ).pg_capacity(ExecutionResources(gpu=1), view)
        == 2
    )
    # Unresolved PG (bundles absent) and the no-PG case both -> 0.
    assert (
        PlacementConstraint(placement_group=_FakePG("missing")).pg_capacity(
            ExecutionResources(gpu=1), view
        )
        == 0
    )
    assert PlacementConstraint().pg_capacity(ExecutionResources(gpu=1), view) == 0


def test_bootstrap_ok_when_pg_bundles_meet_min(monkeypatch):
    pg = _FakePG("pg1")
    view = ClusterView(
        alloc_by_node={"g1": ExecutionResources(gpu=4)},
        pg_bundles={"pg1": [ExecutionResources(gpu=4)]},
    )
    op = _FakeAPMO("embed", _FakePool(min_size=2, max_size=10, gpu=1, running=0))
    op._placement_constraint = PlacementConstraint(placement_group=pg)
    _stub_pool_accounting(op)
    sizer = _label_aware_sizer(view, monkeypatch)
    sizer.bootstrap(topology=_topology(op))  # cap=4 >= min=2 -> no raise
    assert sizer._pg_cap_by_op[op] == 4


def test_bootstrap_caps_ray_remote_args_fn_op_at_concurrency(monkeypatch):
    # A ray_remote_args_fn op (vLLM "ray" backend) assigns a fresh PG per actor
    # that the sizer can't inspect here, so it's capped at its user concurrency
    # (max_size) and routed through the ledger's PG path -- no free-map debit.
    view = ClusterView(alloc_by_node={"g1": ExecutionResources(gpu=4)})
    op = _FakeAPMO("vLLMEngineStageUDF", _FakePool(min_size=1, max_size=1, gpu=0))
    op._user_set_ray_remote_args_fn = True
    _stub_pool_accounting(op)
    sizer = _label_aware_sizer(view, monkeypatch)
    sizer.bootstrap(topology=_topology(op))
    # Capped at concurrency (max_size), not a bundle count, and self-placed.
    assert sizer._pg_cap_by_op[op] == 1
    assert sizer._is_self_placed(op) is True


def test_validate_allows_pg_under_flag(monkeypatch):
    import ray.data._internal.experimental.execution.operators.actor_pool_map_operator as apmo  # noqa: E501

    monkeypatch.setattr(apmo, "SIZER_CONSTRAINT_AWARE_PLACEMENT", True)
    op = _FakeAPMO("ok", _FakePool(min_size=1, max_size=10, gpu=1))
    op._user_ray_remote_args = {"placement_group": object()}
    op.validate_ray_remote_args()  # does not raise under the flag

    # placement_group_capture_child_tasks is always rejected.
    op._user_ray_remote_args = {"placement_group_capture_child_tasks": True}
    with pytest.raises(NotImplementedError):
        op.validate_ray_remote_args()


def test_validate_rejects_pg_without_flag(monkeypatch):
    import ray.data._internal.experimental.execution.operators.actor_pool_map_operator as apmo  # noqa: E501

    # Constraint-aware placement is the flag default now; pin it off to exercise
    # the pre-flag rejection path.
    monkeypatch.setattr(apmo, "SIZER_CONSTRAINT_AWARE_PLACEMENT", False)
    op = _FakeAPMO("bad", _FakePool(min_size=1, max_size=10, gpu=1))
    op._user_ray_remote_args = {"placement_group": object()}
    with pytest.raises(NotImplementedError):
        op.validate_ray_remote_args()


def test_reclaim_not_over_reserved_for_later_ops(monkeypatch):
    # A reclaim-heavy op processed first must not over-debit the per-node ledger:
    # only genuinely-new actors consume free capacity, so a later op contending
    # for the same node still gets its full grant.
    view = ClusterView(
        alloc_by_node={"n1": ExecutionResources(cpu=10)},
        labels_by_node={"n1": {}},
    )
    up = _FakeAPMO(
        "up", _FakePool(min_size=1, max_size=500, cpu=1, running=16, in_flight=1)
    )
    sink = _FakeAPMO(
        "sink",
        _FakePool(min_size=1, max_size=500, cpu=1, running=10, in_flight=1),
        inputs=[up],
    )
    up._ray_remote_args = {}
    sink._ray_remote_args = {}
    _stub_pool_accounting(up, sink)
    # sink (processed first, sink-most) has 3 reclaimable draining actors, revived
    # without consuming new free capacity.
    sink.num_reclaimable_terminating_actors = lambda: 3

    topo = _topology(up, sink, enqueued=100_000)
    sizer = _label_aware_sizer(view, monkeypatch)
    sizer.bootstrap(topology=topo)
    sizer._warmup = False

    req = sizer.scale_how_many(topology=topo)
    # sink: want=ceil(0.5*10)=5, of which 3 are reclaimed -> only 2 consume free,
    #   leaving 8 CPU. up: want=ceil(0.5*16)=8, and it gets all 8 (not 5, which is
    #   what over-reserving the full sink delta would have left).
    assert req.request[sink].delta == 5
    assert req.request[up].delta == 8


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
