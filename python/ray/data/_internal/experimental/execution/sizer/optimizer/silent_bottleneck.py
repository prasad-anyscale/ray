import logging
from typing import TYPE_CHECKING, List

from ray.data._internal.experimental.execution.sizer.actor_placement_strategy import (
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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SilentBottleneckGrant(OptimizationRule):
    """Grow a bottleneck that cannot ask for itself.

    A fully-busy op braked by per-actor output flow control reads out_full,
    so its local grow arm holds -- it never asks, never records a
    shortfall, and normal sizing never sees it. This rule detects it from
    the pipeline view and grants growth from FREE capacity only (no
    victims, no claim to watch), so it may act even while the cluster is
    still resizing.
    """

    name = "silent_bottleneck"

    def __init__(self, metrics: OptimizerMetrics):
        self._metrics = metrics

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
        requests: List[PlacedSizingRequest] = []
        for op in ctx.managed_ops:
            if not ctx.can_op_scale(op):
                # Another rule needs this op held (e.g. it just donated).
                continue
            pool = op.get_autoscaling_actor_pools()[0]
            if pool.num_pending_actors() > 0:
                continue
            want = self._silent_want(op, pool, ctx)
            if want <= 0:
                continue
            shape = pool.per_actor_resource_usage()
            # Fill from free capacity across eligible nodes.
            allowed = ctx.eligible_nodes(op)
            targets = []
            for node in (n for n in ctx.free if allowed is None or n in allowed):
                fit = int(min(max_placeable_actors(ctx.free[node], shape), want))
                if fit > 0:
                    targets.extend([node] * fit)
                    want -= fit
                if want <= 0:
                    break
            if want > 0:
                # Free capacity couldn't cover the want: surface the remainder
                # as a shortfall so CapacityTransfer can qualify this op as a
                # recipient and tap donors -- a silent op never asks on its
                # own, which otherwise leaves it invisible to transfers.
                from ray.data._internal.experimental.execution.sizer.optimizer.context import (  # noqa: E501
                    OpShortfall,
                )

                ctx.record_silent_shortfall(
                    op,
                    OpShortfall(
                        resources=shape.scale(want), num_actors=want, silent=True
                    ),
                )
            if not targets:
                self._metrics.record_refusal(self.name, "no_free_capacity")
                continue
            requests.append(
                PlacedSizingRequest(
                    request={
                        op: ExperimentalActorPoolScalingRequest(
                            delta=len(targets),
                            reason="optimizer: silent bottleneck grant",
                            target_nodes_to_scale_actors_on=tuple(targets),
                        )
                    }
                )
            )
            self._metrics.record_action(self.name, op.name, "granted")
            logger.info(
                "PipelineOptimizer: silent bottleneck grant +%d %s",
                len(targets),
                op.name,
            )
        return requests

    def _silent_want(self, op, pool, ctx: OptimizationContext) -> int:
        """Rate-limited growth step, or 0. Deliberately tight -- every
        condition holds for the full signal window: busy at every sample,
        output full at every sample, and every direct consumer demonstrably
        absorbing (shallow queue, no shortfall, not mid-resize, and -- for
        non-terminal consumers -- own output unblocked). Deliberately NO
        consumer-saturation check: a consumer that reads saturated while
        its queue stays shallow is draining a catch-up backlog, and
        refusing the grant then closes the op's out_full window for good
        (measured as a 4% read-fixed regression)."""
        cur = pool.serving_size()
        cap = ctx.pool_cap(op)
        if cur == 0 or cur >= cap:
            return 0
        streak = ctx.out_full_streak.get(op)
        if streak is None or not streak.is_streak():
            return 0
        busy = ctx.busy_signal.get(op)
        min_busy = busy.minimum() if busy is not None else None
        # Busy = every actor holds at least one task at every window sample.
        # Do NOT require every task SLOT full (cur x max_actor_concurrency):
        # the output flow control this rule rescues from makes dispatch
        # breathe, so a braked op can never sustain full slots.
        if min_busy is None or min_busy < cur:
            return 0
        for consumer in op.output_dependencies:
            state = ctx.topology.get(consumer)
            pools = consumer.get_autoscaling_actor_pools()
            if state is None or not pools:
                return 0
            if pools[0].num_pending_actors() > 0:
                # The consumer is mid-resize; how much it absorbs is about
                # to change, so any verdict from its current signals is
                # stale. Judge again once its pendings land.
                return 0
            if consumer in ctx.shortfalls:
                # The consumer is itself starved: this op's backpressure is
                # real, and growing it would only pile up memory.
                return 0
            if consumer.output_dependencies:
                # Only meaningful for non-terminal consumers: a terminal
                # op's output is never blocked, so its streak is vacuous.
                unblocked = ctx.out_not_full_streak.get(consumer)
                if unblocked is None or not unblocked.is_streak():
                    return 0
            if state.total_enqueued_input_blocks() > max(1, pools[0].serving_size()):
                return 0
        return min(ctx.growth_step(op), cap - cur)
