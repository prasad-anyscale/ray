import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Deque,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from ray.data._internal.execution.execution_flags import (
    SIZER_CLUSTER_SETTLE_S,
    SIZER_SIGNAL_WINDOW_SAMPLES,
    SIZER_TRANSFER_TIMEOUT_S,
)
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.interfaces.common import NodeIdStr
from ray.data._internal.execution.resource_bank import LogicalActorId
from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
    OpPlacementView,
    max_placeable_actors,
)
from ray.data._internal.experimental.execution.sizer.optimizer.context import (
    OptimizationContext,
)
from ray.data._internal.experimental.execution.sizer.optimizer.metrics import (
    OptimizerMetrics,
)
from ray.data._internal.experimental.execution.sizer.optimizer.rule import (
    OptimizationRule,
)
from ray.data._internal.experimental.execution.sizer.optimizer.windowed import (
    SAMPLE_INTERVAL_S,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator

logger = logging.getLogger(__name__)

# A donor is held from growing until its signals reflect the smaller pool --
# one full signal window past the transfer closing. Without this, another
# rule (or normal sizing) regrows the actors this rule just moved.
DONOR_HOLD_S = SIZER_SIGNAL_WINDOW_SAMPLES * SAMPLE_INTERVAL_S

# A transfer plan: per-node portions, each all-or-nothing per actor.
# (node, victims per donor op on that node, recipient actors placed there)
NodeAllocation = Tuple[NodeIdStr, Dict["PhysicalOperator", int], int]


@dataclass
class TransferInFlight:
    """One recipient's transfer, from claim until every claimed pending
    lands (or the watchdog cancels it)."""

    recipient: "PhysicalOperator"
    donors: FrozenSet["PhysicalOperator"]
    nodes: Tuple[NodeIdStr, ...]
    started_t: float
    # Pending ids that existed BEFORE the claim; the post-apply diff against
    # these is the claim, so organic pendings never confuse landing or get
    # evicted by the watchdog.
    pre_pending: FrozenSet[LogicalActorId]
    claimed_ids: Tuple[LogicalActorId, ...] = ()
    num_actors: int = 0


@dataclass
class DonorCapacity:
    """How much an operator can give up without hurting its throughput:
    actors above the windowed peak of PRODUCTIVE work (in-flight minus
    output-blocked). Blocked actors aren't producing, so they aren't
    protected; whether the op is asking to grow is deliberately ignored --
    a backpressured feeder's backlog demand is phantom exactly when its
    consumer is the bottleneck."""

    op: "PhysicalOperator"
    view: Optional[OpPlacementView]
    per_actor: ExecutionResources
    removable: int


def assess_donor(op, pool, ctx: OptimizationContext) -> DonorCapacity:
    cur = pool.serving_size()
    conc = max(1, pool.max_actor_concurrency())
    signal = ctx.productive_signal.get(op)
    peak = signal.peak() if signal is not None else None
    if peak is None:
        protected = cur
    else:
        protected = max(pool.min_size(), math.ceil(peak / conc))
    return DonorCapacity(
        op=op,
        view=None,
        per_actor=pool.per_actor_resource_usage(),
        removable=cur - protected,
    )


def plan_transfer(
    recipient,
    shape: ExecutionResources,
    want: int,
    nodes: List[NodeIdStr],
    free: Dict[NodeIdStr, ExecutionResources],
    donors: List[DonorCapacity],
    node_rank: Dict[NodeIdStr, Tuple],
) -> Tuple[Optional[List[NodeAllocation]], Optional[str]]:
    """Plan up to ``want`` recipient actors across nodes, funded by free
    capacity plus victims. Victims are ranked per node across donors,
    cheapest drain first: pending, then idle, then busy by fewest
    unconsumed outputs / least in-flight. Per-donor caps are global across
    the plan. Returns ``(plan, refusal_reason)``.
    """
    if any(max_placeable_actors(free[n], shape) >= 1 for n in nodes):
        # Normal sizing can already grant; the shortfall was a race.
        return None, "capacity_already_free"

    dims = [d for d in ("cpu", "gpu", "memory") if (getattr(shape, d) or 0) > 0]
    remaining_cap = {d.op: d.removable for d in donors}

    def node_portion(
        node: NodeIdStr, want_here: int
    ) -> Tuple[int, List[Tuple["PhysicalOperator", LogicalActorId]], Dict[str, float]]:
        """Best (k, victim picks, gap-at-k1) for one node with current caps."""
        cands = []
        for donor in donors:
            if remaining_cap[donor.op] <= 0:
                continue
            scarce = 1 if (donor.per_actor.gpu or 0) > 0 and not (shape.gpu or 0) else 0
            for pid in donor.view.pending_ids_by_node.get(node, []):
                cands.append((scarce, 0, 0, 0, donor.op, donor.per_actor, pid))
            ids = donor.view.actor_ids_by_node.get(node, [])
            costs = donor.view.actor_drain_cost_by_node.get(node, [])
            for aid, (in_flight, unconsumed) in zip(ids, costs):
                rank = 1 if in_flight == 0 else 2
                cands.append(
                    (
                        scarce,
                        rank,
                        unconsumed,
                        in_flight,
                        donor.op,
                        donor.per_actor,
                        aid,
                    )
                )
        cands.sort(key=lambda c: c[:4])

        def placeable_k(freed: Dict[str, float]) -> int:
            k = want_here
            for d in dims:
                have = (getattr(free[node], d) or 0) + freed[d]
                k = min(k, int(have // getattr(shape, d)))
            return k

        freed = {d: 0.0 for d in dims}
        taken: Dict["PhysicalOperator", int] = {}
        picks: List[Tuple["PhysicalOperator", LogicalActorId]] = []
        k_at: List[int] = []
        for _, _, _, _, op, per, aid in cands:
            if remaining_cap[op] - taken.get(op, 0) <= 0:
                continue
            picks.append((op, aid))
            taken[op] = taken.get(op, 0) + 1
            for d in dims:
                freed[d] += getattr(per, d) or 0
            k_at.append(placeable_k(freed))
            if k_at[-1] >= want_here:
                break
        k_node = max(k_at, default=0)
        if k_node >= 1:
            n_min = k_at.index(k_node) + 1
            return k_node, picks[:n_min], {}
        gap = {}
        for d in dims:
            need = getattr(shape, d) - (getattr(free[node], d) or 0) - freed[d]
            if need > 0:
                gap[d] = need
        return 0, [], gap

    # Rank nodes by their standalone best k, then the recipient's locality.
    standalone = [(node, node_portion(node, want)) for node in nodes]
    standalone.sort(key=lambda e: (-e[1][0], node_rank.get(e[0], ())))

    plan: List[NodeAllocation] = []
    best_gap: Optional[Tuple[NodeIdStr, Dict[str, float]]] = None
    left = want
    for node, (k_probe, _, gap) in standalone:
        if left <= 0:
            break
        if k_probe <= 0:
            if gap and (
                best_gap is None or sum(gap.values()) < sum(best_gap[1].values())
            ):
                best_gap = (node, gap)
            continue
        # Recompute with the caps remaining after earlier portions.
        k_node, picks, _ = node_portion(node, left)
        if k_node <= 0:
            continue
        victims: Dict["PhysicalOperator", int] = {}
        for op, _aid in picks:
            victims[op] = victims.get(op, 0) + 1
            remaining_cap[op] -= 1
        plan.append((node, victims, k_node))
        left -= k_node

    if plan:
        return plan, None
    if best_gap is not None:
        logger.info(
            "PipelineOptimizer transfer refused: %s needs %s; best node %s "
            "still short by %s after all eligible victims.",
            recipient.name,
            {d: getattr(shape, d) for d in ("cpu", "gpu", "memory")},
            best_gap[0][:8],
            {d: round(v, 2) for d, v in best_gap[1].items()},
        )
    return None, "victims_insufficient"


def _consume_victims_from_view(
    view: OpPlacementView, node: NodeIdStr, count: int
) -> None:
    """Remove planned victims from a donor placement view."""
    remaining = count
    removed = 0

    pending = view.pending_ids_by_node.get(node)
    if pending:
        take = min(remaining, len(pending))
        del pending[:take]
        remaining -= take
        removed += take
        if not pending:
            view.pending_ids_by_node.pop(node, None)

    actor_ids = view.actor_ids_by_node.get(node)
    if remaining > 0 and actor_ids:
        costs = view.actor_drain_cost_by_node.get(node)
        take = min(remaining, len(actor_ids))
        removed_costs = costs[:take] if costs is not None else []

        del actor_ids[:take]
        if costs is not None:
            del costs[:take]
        removed += take

        if not actor_ids:
            view.actor_ids_by_node.pop(node, None)
        if costs is not None and not costs:
            view.actor_drain_cost_by_node.pop(node, None)

        idle_removed = (
            sum(1 for in_flight, _ in removed_costs if in_flight == 0)
            if removed_costs
            else min(view.idle_actors_by_node.get(node, 0), take)
        )
        idle = view.idle_actors_by_node.get(node, 0) - idle_removed
        if idle > 0:
            view.idle_actors_by_node[node] = idle
        else:
            view.idle_actors_by_node.pop(node, None)

    if removed:
        actors = view.actors_by_node.get(node, 0) - removed
        if actors > 0:
            view.actors_by_node[node] = actors
        else:
            view.actors_by_node.pop(node, None)


def _consume_plan_from_donors(
    plan: List[NodeAllocation], donors_by_op: Dict["PhysicalOperator", DonorCapacity]
) -> None:
    for node, victims, _ in plan:
        for op, count in victims.items():
            donor = donors_by_op[op]
            donor.removable = max(0, donor.removable - count)
            if donor.view is not None:
                _consume_victims_from_view(donor.view, node, count)


class CapacityTransfer(OptimizationRule):
    """Move capacity from operators with provable surplus to operators that
    keep asking and getting nothing. Claim -> evict -> land, per node: the
    recipient's targeted pendings are created in the same request as the
    victims' eviction, so committed accounting reserves the space."""

    name = "capacity_transfer"

    def __init__(self, metrics: OptimizerMetrics, placement, clock):
        self._metrics = metrics
        self._placement = placement
        self._clock = clock
        self._transfers: Dict["PhysicalOperator", TransferInFlight] = {}
        # Donor -> time until which it must not grow (see DONOR_HOLD_S).
        self._donor_hold_until: Dict["PhysicalOperator", float] = {}
        # Recent (recipient, donors) pairs; a transfer that inverts one of
        # these is thrashing the same capacity back and forth.
        self._recent: Deque[Tuple["PhysicalOperator", FrozenSet]] = deque(maxlen=8)
        # Edge-triggered: log "no donor surplus" once per episode, not per tick.
        self._logged_no_donors = False

    # --- lifecycle -------------------------------------------------------

    def advance(self, ctx: OptimizationContext) -> None:
        self.advance_inflight_transfers(ctx)

    def advance_inflight_transfers(self, ctx: OptimizationContext) -> None:
        from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
            ExperimentalActorPoolScalingRequest,
        )

        for recipient, transfer in list(self._transfers.items()):
            pools = recipient.get_autoscaling_actor_pools()
            pool = pools[0] if pools else None
            pending: Set[LogicalActorId] = set()
            if pool is not None:
                for ids in pool.pending_ids_by_target_node().values():
                    pending.update(ids)
            still_pending = [a for a in transfer.claimed_ids if a in pending]
            if pool is None or not still_pending:
                logger.info(
                    "PipelineOptimizer: transfer to %s closed (%.0fs after claim)",
                    recipient.name,
                    ctx.now - transfer.started_t,
                )
                self._hold_donors(transfer, ctx.now)
                del self._transfers[recipient]
                self._metrics.record_action(self.name, recipient.name, "landed")
                continue
            if ctx.now - transfer.started_t > SIZER_TRANSFER_TIMEOUT_S:
                pool.scale(
                    ExperimentalActorPoolScalingRequest(
                        delta=-len(still_pending),
                        force=True,
                        reason="optimizer: transfer timed out",
                        actor_ids_to_downscale=tuple(still_pending),
                    )
                )
                logger.warning(
                    "PipelineOptimizer: transfer to %s timed out after %.0fs; "
                    "claim cancelled.",
                    recipient.name,
                    SIZER_TRANSFER_TIMEOUT_S,
                )
                self._hold_donors(transfer, ctx.now)
                del self._transfers[recipient]
                self._metrics.record_action(self.name, recipient.name, "timeout")

    # --- proposal --------------------------------------------------------

    def apply(self, ctx: OptimizationContext) -> List:
        from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
            ExperimentalActorPoolScalingRequest,
        )
        from ray.data._internal.experimental.execution.sizer.operator_sizer import (
            PlacedSizingRequest,
        )

        if ctx.warmup:
            self._metrics.record_refusal(self.name, "warmup")
            return []
        # Transfers correct a settled allocation; while the cluster is still
        # resizing, growth is the autoscaler's move.
        if ctx.seconds_since_allocation_change < SIZER_CLUSTER_SETTLE_S:
            self._metrics.record_refusal(self.name, "cluster_settling")
            return []
        managed = set(ctx.managed_ops)
        recipients = []
        for op, streak in ctx.shortfall_streak.items():
            if op not in managed or not streak.is_streak() or not ctx.can_op_scale(op):
                continue
            if any(c in ctx.shortfalls for c in op.output_dependencies):
                # The op's own consumer is starved, so this backlog demand
                # is phantom: growing the op would pile output onto a
                # consumer that cannot keep up. Serve the consumer first --
                # this op is donor material, not a recipient.
                continue
            pools = op.get_autoscaling_actor_pools()
            if not pools or pools[0].num_pending_actors() > 0:
                # Sizing already in flight; judge again once it lands.
                continue
            recipients.append(op)
        if not recipients:
            return []
        # Real asks first: an op that asked to grow and was clamped is
        # evidence; a bridged silent want is an inference. When donors are
        # scarce the real asks get the budget.
        recipients.sort(
            key=lambda op: (
                ctx.shortfalls[op].silent if op in ctx.shortfalls else False
            )
        )

        donors: List[DonorCapacity] = []
        for op in ctx.managed_ops:
            # A held op's signals are stale (they predate whatever is
            # holding it), so its removable count cannot be trusted.
            # TODO: remember how much of a held donor's surplus is already
            # spoken for and let it donate the remainder instead of
            # skipping it outright.
            if op in recipients or not ctx.can_op_scale(op):
                continue
            pools = op.get_autoscaling_actor_pools()
            capacity = assess_donor(op, pools[0], ctx)
            if capacity.removable <= 0:
                continue
            capacity.view = op.build_placement_view(need_victim_order=True)
            donors.append(capacity)
        donors_by_op = {donor.op: donor for donor in donors}
        if not donors:
            # Every managed op is starved, held, or has no provable surplus:
            # there is nothing to move. Normal while demand exceeds the
            # cluster (e.g. during an autoscaling ramp) -- adding capacity is
            # the autoscaler's job, not this rule's.
            self._metrics.record_refusal(self.name, "no_donor_surplus")
            if not self._logged_no_donors:
                logger.info(
                    "PipelineOptimizer: %d starved op(s) but no donor has "
                    "surplus; nothing to move.",
                    len(recipients),
                )
                self._logged_no_donors = True
            return []
        self._logged_no_donors = False

        requests: List[PlacedSizingRequest] = []
        for recipient in recipients:
            allowed = ctx.eligible_nodes(recipient)
            nodes = [n for n in ctx.free if allowed is None or n in allowed]
            if not nodes:
                continue
            pool = recipient.get_autoscaling_actor_pools()[0]
            shape = pool.per_actor_resource_usage()
            shortfall = ctx.shortfalls.get(recipient)
            want = max(1, shortfall.num_actors if shortfall else 1)
            rview = recipient.build_placement_view(need_victim_order=False)
            node_rank = {n: self._placement._locality_rank(rview, n) for n in nodes}
            plan, refusal = plan_transfer(
                recipient,
                shape,
                want,
                nodes,
                ctx.free,
                [d for d in donors if d.op is not recipient],
                node_rank,
            )
            if plan is None:
                self._metrics.record_refusal(self.name, refusal)
                continue
            donor_ops = frozenset(op for _, victims, _ in plan for op in victims)
            if self._is_thrashing(recipient, donor_ops):
                self._metrics.record_refusal(self.name, "thrashing")
                continue
            _consume_plan_from_donors(plan, donors_by_op)
            pre_pending: Set[LogicalActorId] = set()
            for ids in pool.pending_ids_by_target_node().values():
                pre_pending.update(ids)
            total_k = 0
            for node, victims, k_node in plan:
                req = {
                    recipient: ExperimentalActorPoolScalingRequest(
                        delta=k_node,
                        reason="optimizer: capacity transfer in",
                        target_nodes_to_scale_actors_on=(node,) * k_node,
                    )
                }
                for donor, count in victims.items():
                    req[donor] = ExperimentalActorPoolScalingRequest(
                        delta=-count,
                        force=True,
                        reason=f"optimizer: donate to {recipient.name}",
                        victim_node=node,
                    )
                requests.append(PlacedSizingRequest(request=req))
                total_k += k_node
            self._transfers[recipient] = TransferInFlight(
                recipient=recipient,
                donors=donor_ops,
                nodes=tuple(node for node, _, _ in plan),
                started_t=ctx.now,
                pre_pending=frozenset(pre_pending),
                num_actors=total_k,
            )
            self._recent.append((recipient, donor_ops))
            self._metrics.record_action(self.name, recipient.name, "fired")
            for node, victims, _ in plan:
                for donor, count in victims.items():
                    self._metrics.record_corrective_transfer(
                        donor.name, recipient.name, count
                    )
            logger.info(
                "PipelineOptimizer: transfer +%d %s across %d node(s) from %s",
                total_k,
                recipient.name,
                len(plan),
                "/".join(sorted(d.name for d in donor_ops)),
            )
        return requests

    def after_apply(self, ctx: OptimizationContext) -> None:
        """Record the claimed pending ids: everything pending now that
        wasn't pending before the claim."""
        for transfer in self._transfers.values():
            if transfer.claimed_ids:
                continue
            pools = transfer.recipient.get_autoscaling_actor_pools()
            if not pools:
                continue
            pending: Set[LogicalActorId] = set()
            for ids in pools[0].pending_ids_by_target_node().values():
                pending.update(ids)
            transfer.claimed_ids = tuple(pending - transfer.pre_pending)

    # --- guards ----------------------------------------------------------

    def _hold_donors(self, transfer: TransferInFlight, now: float) -> None:
        for donor in transfer.donors:
            self._donor_hold_until[donor] = now + DONOR_HOLD_S

    def can_op_scale(self, op: "PhysicalOperator") -> bool:
        # One action at a time per op: held while it is the recipient or a
        # donor of an in-flight transfer, and as a donor until its signals
        # reflect the smaller pool -- otherwise the donation boomerangs.
        if op in self._transfers:
            return False
        if any(op in t.donors for t in self._transfers.values()):
            return False
        return self._clock() >= self._donor_hold_until.get(op, float("-inf"))

    def _is_thrashing(self, recipient: "PhysicalOperator", donors: FrozenSet) -> bool:
        return any(
            recipient in past_donors and past_recipient in donors
            for past_recipient, past_donors in self._recent
        )
