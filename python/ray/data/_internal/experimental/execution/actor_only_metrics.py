from __future__ import annotations

import math
import time
from collections import Counter as cCounter, defaultdict
from enum import Enum
from typing import TYPE_CHECKING, DefaultDict, Dict, Set, Tuple

from ray.data._internal.execution.interfaces.common import NodeIdStr
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (
    ActorStatus,
    ExperimentalAPMO,
)
from ray.util.metrics import Counter, Gauge

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_bank import ResourceBankBase
    from ray.data._internal.execution.streaming_executor_state import Topology


class ObjStoreBreakdown(str, Enum):

    INPUTS = "inputs"
    OUTPUTS = "outputs"
    REMOTE = "remote"
    LOCAL = "local"
    FREED = "freed"


class ObjStoreUnit(str, Enum):

    BYTES = "bytes"
    BLOCKS = "blocks"


class Stat(str, Enum):

    MIN = "min"
    P50 = "p50"
    P90 = "p90"
    MAX = "max"
    MEAN = "mean"


class ActorOnlyTags(tuple[str, ...], Enum):

    DATASET_NODE_UNIT_BREAKDOWN = ("dataset", "node", "unit", "breakdown")
    DATASET_NODE_STATUS = ("dataset", "node", "status")
    DATASET_NODE_OPERATOR_STATUS = ("dataset", "node", "operator", "status")
    DATASET_NODE_OPERATOR_UNIT_BREAKDOWN = (
        "dataset",
        "node",
        "operator",
        "unit",
        "breakdown",
    )
    DATASET_OPERATOR_UNIT_STAT = ("dataset", "operator", "unit", "stat")
    DATASET_OPERATOR_KIND = ("dataset", "operator", "kind")


def _limit_stats(values: "list[float]") -> "Dict[Stat, float]":
    """Snapshot min/p50/p90/max/mean across a list of per-actor values.

    The list is already materialized and small (one entry per actor), so the
    stats are computed exactly rather than via an approximate KLL sketch.
    """
    if not values:
        return {stat: 0.0 for stat in Stat}
    ordered = sorted(values)
    n = len(ordered)

    def _pct(q: float) -> float:
        # Nearest-rank percentile.
        rank = min(n - 1, max(0, math.ceil(q * n) - 1))
        return ordered[rank]

    return {
        Stat.MIN: ordered[0],
        Stat.P50: _pct(0.50),
        Stat.P90: _pct(0.90),
        Stat.MAX: ordered[-1],
        Stat.MEAN: sum(ordered) / n,
    }


class ActorOnlyRecorders:
    """Gauges / counters for actor-only ResourceBank telemetry."""

    node_obj_store = Gauge(
        "data_actor_only_node_obj_store",
        description="Object store on a node",
        tag_keys=ActorOnlyTags.DATASET_NODE_UNIT_BREAKDOWN,
    )

    node_num_actors = Gauge(
        "data_actor_only_node_num_actors",
        description="Number of actors on a node by state (see ActorStatus)",
        tag_keys=ActorOnlyTags.DATASET_NODE_STATUS,
    )

    node_op_num_actors = Gauge(
        "data_actor_only_node_operator_num_actors",
        description="Number of actors on a node for an operator by state",
        tag_keys=ActorOnlyTags.DATASET_NODE_OPERATOR_STATUS,
    )

    node_op_input_locality = Gauge(
        "data_actor_only_node_operator_input_locality",
        description=(
            "Cumulative input bytes/blocks submitted to an operator's actors on "
            "a node, by locality. Summing over nodes gives the per-operator "
            "split; summing over operators gives the per-node split."
        ),
        tag_keys=ActorOnlyTags.DATASET_NODE_OPERATOR_UNIT_BREAKDOWN,
    )

    output_limit_upgrades = Counter(
        "data_actor_only_output_limit_upgrades",
        description=(
            "# of Times the idle/deadlock detector upgraded an actor's "
            "output-bytes/blocks limit to prevent deadlock."
        ),
        tag_keys=("dataset", "operator"),
    )

    op_output_limit = Gauge(
        "data_actor_only_operator_output_limit",
        description=(
            "Snapshot distribution of per-actor output limits for an operator "
            "(by stat)."
        ),
        tag_keys=ActorOnlyTags.DATASET_OPERATOR_UNIT_STAT,
    )

    op_output_backpressured_actors = Gauge(
        "data_actor_only_operator_output_backpressured_actors",
        description=(
            "Number of actors whose in-flight output task is output-"
            "backpressured (its pull budget is exhausted) for an operator."
        ),
        tag_keys=("dataset", "operator"),
    )

    op_output_overshoots = Counter(
        "data_actor_only_operator_output_overshoots",
        description=(
            "Cumulative count, per operator, of output-limit overshoots by "
            "kind: 'block_size' (a produced block exceeded 150% of its target "
            "block size) and 'output_bytes' (an actor's outstanding output "
            "exceeded 150% of its output-bytes limit)."
        ),
        tag_keys=ActorOnlyTags.DATASET_OPERATOR_KIND,
    )


class _StaleTrackingGauge:
    """Wraps a Gauge and resets series that stop being emitted.

    A Gauge holds its last value forever, so a series whose label-set stops
    being written between cycles (an actor leaves a node, an operator finishes,
    a node dies) would linger at a stale value in Prometheus. Callers write via
    ``set`` during a cycle and then call ``flush_stale`` exactly once: any
    series written in the previous cycle but not the current one is reset to 0.

    State is per ``ActorOnlyMetrics`` instance (i.e. per dataset), even though
    the underlying Gauge is a module-level singleton shared across datasets.
    """

    def __init__(self, gauge: Gauge):
        self._gauge = gauge
        self._emitted: Set[Tuple[Tuple[str, str], ...]] = set()
        self._current: Set[Tuple[Tuple[str, str], ...]] = set()

    def set(self, value: float, tags: Dict[str, str]) -> None:
        self._gauge.set(value, tags=tags)
        self._current.add(tuple(sorted(tags.items())))

    def flush_stale(self) -> None:
        for key in self._emitted - self._current:
            self._gauge.set(0, tags=dict(key))
        self._emitted = self._current
        self._current = set()


class ActorOnlyMetrics:
    def __init__(self, dataset_id: str, update_interval: float):
        self._dataset_id = dataset_id
        self._last_updated: float = 0.0
        self._update_interval: float = update_interval
        self.recorder = ActorOnlyRecorders()
        # Gauges wrapped for stale-series resetting. The Counter is cumulative
        # and never needs resetting, so it stays on ``recorder``.
        self._node_obj_store = _StaleTrackingGauge(self.recorder.node_obj_store)
        self._node_num_actors = _StaleTrackingGauge(self.recorder.node_num_actors)
        self._node_op_num_actors = _StaleTrackingGauge(self.recorder.node_op_num_actors)
        self._node_op_input_locality = _StaleTrackingGauge(
            self.recorder.node_op_input_locality
        )
        self._op_output_limit = _StaleTrackingGauge(self.recorder.op_output_limit)
        self._op_output_backpressured_actors = _StaleTrackingGauge(
            self.recorder.op_output_backpressured_actors
        )

    def record_output_limit_upgrade(self, op_tag: str) -> None:
        self.recorder.output_limit_upgrades.inc(
            1, tags={"dataset": self._dataset_id, "operator": op_tag}
        )

    def record_block_size_overshoot(self, op_tag: str) -> None:
        """Increment when a produced block exceeds 150% of its target size."""
        self.recorder.op_output_overshoots.inc(
            1,
            tags={
                "dataset": self._dataset_id,
                "operator": op_tag,
                "kind": "block_size",
            },
        )

    def record_output_limit_overshoot(self, op_tag: str) -> None:
        """Increment when an actor's outstanding output exceeds 150% of its
        output-bytes limit."""
        self.recorder.op_output_overshoots.inc(
            1,
            tags={
                "dataset": self._dataset_id,
                "operator": op_tag,
                "kind": "output_bytes",
            },
        )

    def maybe_update(
        self,
        *,
        topology: "Topology",
        resource_bank: "ResourceBankBase",
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if (
            not force
            and self._last_updated > 0
            and (now - self._last_updated) < self._update_interval
        ):
            return
        self._last_updated = now
        self._update_node_object_store(resource_bank=resource_bank)
        self._update_actor_counts(topology=topology)
        self._update_operator_limits(topology=topology)
        self._update_node_operator_locality(
            topology=topology, resource_bank=resource_bank
        )
        self._update_output_backpressure(topology=topology)

    def _update_node_object_store(
        self,
        resource_bank: "ResourceBankBase",
    ) -> None:
        for node_id in resource_bank.node_ids():
            node_obj_store = resource_bank.live_object_store(node_id=node_id)
            base_tags = {"dataset": self._dataset_id, "node": node_id}
            # TODO(Justin): Also, get the total too, but use a grafana query for that
            # Input/Output breakdown of blocks/bytes of object store
            self._node_obj_store.set(
                node_obj_store.output_bytes(),
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BYTES.value,
                    "breakdown": ObjStoreBreakdown.OUTPUTS.value,
                },
            )
            self._node_obj_store.set(
                node_obj_store.output_blocks(),
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BLOCKS.value,
                    "breakdown": ObjStoreBreakdown.OUTPUTS.value,
                },
            )
            self._node_obj_store.set(
                node_obj_store.input_bytes(),
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BYTES.value,
                    "breakdown": ObjStoreBreakdown.INPUTS.value,
                },
            )
            self._node_obj_store.set(
                node_obj_store.input_blocks(),
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BLOCKS.value,
                    "breakdown": ObjStoreBreakdown.INPUTS.value,
                },
            )

            # Freed outputs only. Including inputs double-counts same-node local
            # transfers (output withdrawn, then re-charged as input).
            acc_node_obj_store = resource_bank.cumulative_object_store(node_id=node_id)
            # NOTE: CumulativeObjStore adds exactly one output_bytes sample per
            # output block, so ``output_bytes.num_samples`` doubles as the
            # cumulative output block count.
            #
            # The live store also holds prebuffered outputs (generated but not
            # yet pulled from the generators); the cumulative running sum only
            # counts outputs once they are pulled. Subtracting the prebuffered
            # portion of the live store would therefore let ``freed`` go
            # negative, so exclude it from the subtracted term.
            live_output_bytes = (
                node_obj_store.output_bytes()
                - node_obj_store.num_prebuffered_output_bytes
            )
            live_output_blocks = (
                node_obj_store.output_blocks()
                - node_obj_store.num_prebuffered_output_blocks
            )
            freed_bytes = acc_node_obj_store.output_bytes.sum - live_output_bytes
            freed_blocks = (
                acc_node_obj_store.output_bytes.num_samples - live_output_blocks
            )
            assert (
                freed_bytes >= 0 and freed_blocks >= 0
            ), f"{freed_bytes=}, {freed_blocks=}"
            self._node_obj_store.set(
                freed_bytes,
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BYTES.value,
                    "breakdown": ObjStoreBreakdown.FREED.value,
                },
            )
            self._node_obj_store.set(
                freed_blocks,
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BLOCKS.value,
                    "breakdown": ObjStoreBreakdown.FREED.value,
                },
            )

            # Locality metrics with bytes. NOTE: As a reminder bytes are only
            # remote for task inputs,all outputs must be local to the task itself.
            local_bytes = (
                acc_node_obj_store.input_bytes_local.sum
                + acc_node_obj_store.output_bytes.sum
            )
            remote_bytes = acc_node_obj_store.input_bytes_remote.sum
            self._node_obj_store.set(
                local_bytes,
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BYTES.value,
                    "breakdown": ObjStoreBreakdown.LOCAL.value,
                },
            )
            self._node_obj_store.set(
                remote_bytes,
                tags={
                    **base_tags,
                    "unit": ObjStoreUnit.BYTES.value,
                    "breakdown": ObjStoreBreakdown.REMOTE.value,
                },
            )

        # Reset series for nodes/breakdowns that disappeared this cycle.
        self._node_obj_store.flush_stale()

    def _update_actor_counts(self, topology: "Topology") -> None:
        # Aggregate per-node across all ExperimentalAPMO ops. The per-operator
        # breakdown is intentionally omitted: it is derivable in Grafana by
        # summing the per-node-operator series over nodes.
        node_totals: DefaultDict[NodeIdStr, cCounter[ActorStatus]] = defaultdict(
            cCounter
        )

        for op, op_state in topology.items():
            if not isinstance(op, ExperimentalAPMO):
                continue

            op_tag = op_state.op_tag()

            # Per-op-node breakdown
            for node_id, counts in op.actor_counts_by_node().items():
                for status, value in counts.items():
                    node_totals[node_id][status] += value
                    self._node_op_num_actors.set(
                        value,
                        tags={
                            "dataset": self._dataset_id,
                            "node": node_id,
                            "operator": op_tag,
                            "status": status.value,
                        },
                    )

        # Per-node breakdown
        for node_id, counts in node_totals.items():
            for status, value in counts.items():
                self._node_num_actors.set(
                    value,
                    tags={
                        "dataset": self._dataset_id,
                        "node": node_id,
                        "status": status.value,
                    },
                )

        self._node_op_num_actors.flush_stale()
        self._node_num_actors.flush_stale()

    def _update_operator_limits(self, topology: "Topology") -> None:
        for op, op_state in topology.items():
            if not isinstance(op, ExperimentalAPMO):
                continue
            op_tag = op_state.op_tag()

            # Snapshot the per-actor output limits across the pool, then emit
            # the current distribution (min/p50/p90/max/mean) per unit. Infinite
            # (unbounded) limits are excluded so they don't skew the stats.
            pool = op.actor_pool
            limits_by_unit: "Dict[ObjStoreUnit, list[float]]" = {
                ObjStoreUnit.BYTES: [],
                ObjStoreUnit.BLOCKS: [],
            }
            for actor_id in pool.get_logical_ids():
                bytes_limit = pool.output_bytes_limit(actor_id)
                blocks_limit = pool.output_count_limit(actor_id)
                if math.isfinite(bytes_limit):
                    limits_by_unit[ObjStoreUnit.BYTES].append(bytes_limit)
                if math.isfinite(blocks_limit):
                    limits_by_unit[ObjStoreUnit.BLOCKS].append(blocks_limit)

            for unit, values in limits_by_unit.items():
                if not values:
                    # No live actors, or all unbounded: let flush_stale reset
                    # any prior series.
                    continue
                for stat, value in _limit_stats(values).items():
                    self._op_output_limit.set(
                        value,
                        tags={
                            "dataset": self._dataset_id,
                            "operator": op_tag,
                            "unit": unit.value,
                            "stat": stat.value,
                        },
                    )

        self._op_output_limit.flush_stale()

    def _update_node_operator_locality(
        self,
        topology: "Topology",
        resource_bank: "ResourceBankBase",
    ) -> None:
        """Emit the input-locality split joined on (node, operator).

        Grafana derives the per-operator and per-node views by summing this
        series over the other dimension, so all three groupings stay consistent
        with one another by construction.
        """
        op_tags: Dict[PhysicalOperator, str] = {
            op: op_state.op_tag()
            for op, op_state in topology.items()
            if isinstance(op, ExperimentalAPMO)
        }

        for op, node_id in resource_bank.op_node_ids():
            op_tag = op_tags.get(op)
            if op_tag is None:
                # The operator is no longer in the topology (e.g. it finished and
                # was dropped); let flush_stale reset its series.
                continue
            cumulative = resource_bank.cumulative_object_store(op=op, node_id=node_id)
            local = cumulative.input_bytes_local
            remote = cumulative.input_bytes_remote
            base_tags = {
                "dataset": self._dataset_id,
                "node": node_id,
                "operator": op_tag,
            }
            for unit, breakdown, value in (
                (ObjStoreUnit.BYTES, ObjStoreBreakdown.LOCAL, local.sum),
                (ObjStoreUnit.BYTES, ObjStoreBreakdown.REMOTE, remote.sum),
                (ObjStoreUnit.BLOCKS, ObjStoreBreakdown.LOCAL, local.num_samples),
                (ObjStoreUnit.BLOCKS, ObjStoreBreakdown.REMOTE, remote.num_samples),
            ):
                self._node_op_input_locality.set(
                    value,
                    tags={
                        **base_tags,
                        "unit": unit.value,
                        "breakdown": breakdown.value,
                    },
                )

        self._node_op_input_locality.flush_stale()

    def _update_output_backpressure(self, topology: "Topology") -> None:
        # ``output_backpressured_fraction`` returns (blocked, active) in-flight
        # data-task counts; the actor-only backend runs one data task per actor,
        # so ``blocked`` is the number of actors currently output-backpressured.
        for op, op_state in topology.items():
            if not isinstance(op, ExperimentalAPMO):
                continue
            blocked, _active = op.output_backpressured_actors()
            self._op_output_backpressured_actors.set(
                blocked,
                tags={"dataset": self._dataset_id, "operator": op_state.op_tag()},
            )

        self._op_output_backpressured_actors.flush_stale()
