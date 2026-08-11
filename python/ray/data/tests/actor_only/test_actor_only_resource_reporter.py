# ABOUTME: Unit tests for ActorOnlyResourceReporter and its wiring into the
# ABOUTME: cluster autoscaler v2 (gauge -> scale-up). Fakes only; no cluster.

import sys

import pytest

from ray.data._internal.cluster_autoscaler import (
    default_cluster_autoscaler_v2 as autoscaler_module,
)
from ray.data._internal.cluster_autoscaler.default_cluster_autoscaler_v2 import (
    DefaultClusterAutoscalerV2,
    _NodeResourceSpec,
)
from ray.data._internal.cluster_autoscaler.fake_autoscaling_coordinator import (
    FakeAutoscalingCoordinator,
)
from ray.data._internal.cluster_autoscaler.resource_utilization_gauge import (
    RollingLogicalUtilizationGauge,
)
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.resource_bank import LiveObjStore
from ray.data._internal.experimental.execution.actor_only_resource_reporter import (
    ActorOnlyResourceReporter,
)


class _ReporterOp:
    """Duck-typed operator exposing only what the reporter reads."""

    def __init__(self, running: ExecutionResources, pending: ExecutionResources):
        self._running = running
        self._pending = pending

    def running_logical_usage(self) -> ExecutionResources:
        return self._running

    def pending_logical_usage(self) -> ExecutionResources:
        return self._pending


class _FakeResourceBank:
    """Fake ResourceBank exposing per-op live object-store byte counts.

    ``per_op_bytes`` maps op -> (input, pulled_output, prebuffered_output,
    dangling_output) bytes; missing ops report zero.
    """

    def __init__(self, per_op_bytes=None):
        self._per_op = per_op_bytes or {}

    def live_object_store(self, op=None, node_id=None, actor_id=None):
        in_b, pulled, prebuffered, dangling = self._per_op.get(op, (0, 0, 0, 0))
        return LiveObjStore(
            num_input_bytes=in_b,
            num_pulled_output_bytes=pulled,
            num_prebuffered_output_bytes=prebuffered,
            num_dangling_output_bytes=dangling,
        )


def _topology(*ops):
    # The reporter only iterates topology keys; the OpState value is unused.
    return {op: None for op in ops}


def test_get_global_usage_sums_running_and_object_store():
    op1 = _ReporterOp(
        running=ExecutionResources(cpu=2, gpu=0, memory=100),
        pending=ExecutionResources.zero(),
    )
    op2 = _ReporterOp(
        running=ExecutionResources(cpu=3, gpu=1, memory=200),
        pending=ExecutionResources.zero(),
    )
    # (input, pulled, prebuffered, dangling) bytes per op.
    resource_bank = _FakeResourceBank({op1: (10, 20, 5, 3), op2: (5, 0, 0, 0)})

    reporter = ActorOnlyResourceReporter(
        _topology(op1, op2), FakeAutoscalingCoordinator(), resource_bank
    )
    usage = reporter.get_global_usage()

    assert usage.cpu == 5
    assert usage.gpu == 1
    assert usage.memory == 300
    # object store = (10+20+5+3) + (5+0+0+0)
    assert usage.object_store_memory == 43


def test_get_global_pending_usage_sums_pending():
    op1 = _ReporterOp(
        running=ExecutionResources.zero(),
        pending=ExecutionResources(cpu=4, gpu=0, memory=0),
    )
    op2 = _ReporterOp(
        running=ExecutionResources.zero(),
        pending=ExecutionResources(cpu=1, gpu=2, memory=50),
    )
    reporter = ActorOnlyResourceReporter(
        _topology(op1, op2), FakeAutoscalingCoordinator(), _FakeResourceBank()
    )
    pending = reporter.get_global_pending_usage()

    assert pending.cpu == 5
    assert pending.gpu == 2
    assert pending.memory == 50


def test_get_global_limits_sums_coordinator_allocation():
    coordinator = FakeAutoscalingCoordinator()
    # Seed a fixed allocation (request_remaining=False -> exactly these bundles).
    coordinator.request_resources(
        resources=[{"CPU": 4, "GPU": 1}, {"CPU": 8}],
        expire_after_s=1e9,
        request_remaining=False,
    )

    reporter = ActorOnlyResourceReporter(_topology(), coordinator, _FakeResourceBank())
    limits = reporter.get_global_limits()

    assert limits.cpu == 12
    assert limits.gpu == 1


def test_get_reserved_resources_by_node_converts_per_node():
    coordinator = FakeAutoscalingCoordinator(
        initial_cluster_resources_by_node={
            "node0": {"CPU": 4, "GPU": 1},
            "node1": {"CPU": 8},
        },
    )
    # by_node only returns the configured allocation while an unexpired request
    # exists, so register one first (matches how the autoscaler drives it).
    coordinator.request_resources(
        resources=[{"CPU": 1}], expire_after_s=1e9, request_remaining=False
    )

    reporter = ActorOnlyResourceReporter(_topology(), coordinator, _FakeResourceBank())
    by_node = reporter.get_reserved_resources_by_node()

    assert set(by_node) == {"node0", "node1"}
    assert by_node["node0"].cpu == 4
    assert by_node["node0"].gpu == 1
    assert by_node["node1"].cpu == 8
    # Every value is an ExecutionResources, not a raw dict.
    assert all(isinstance(v, ExecutionResources) for v in by_node.values())


def test_empty_topology_and_no_allocation():
    reporter = ActorOnlyResourceReporter(
        _topology(), FakeAutoscalingCoordinator(), _FakeResourceBank()
    )
    assert reporter.get_global_usage() == ExecutionResources.zero()
    assert reporter.get_global_pending_usage() == ExecutionResources.zero()
    assert reporter.get_global_limits() == ExecutionResources.zero()


@pytest.mark.parametrize("cpu_running,should_scale_up", [(4.0, False), (8.0, True)])
def test_reporter_drives_cluster_autoscaler_scale_up(
    monkeypatch, cpu_running, should_scale_up
):
    """End-to-end wiring: reporter -> gauge -> DefaultClusterAutoscalerV2.

    Limits come from the reporter's coordinator allocation (10 CPU); usage from
    the operator. At 8/10 CPU we cross the 0.75 threshold and the autoscaler
    requests node bundles; at 4/10 it doesn't.
    """
    # is_autoscaling_enabled() hits the GCS (needs a running cluster); it only
    # affects log level, so stub it out for this cluster-free unit test.
    monkeypatch.setattr(autoscaler_module, "is_autoscaling_enabled", lambda: True)

    op = _ReporterOp(
        running=ExecutionResources(cpu=cpu_running),
        pending=ExecutionResources.zero(),
    )
    # Reporter's coordinator supplies the limits (cluster capacity = 10 CPU).
    limits_coordinator = FakeAutoscalingCoordinator()
    limits_coordinator.request_resources(
        resources=[{"CPU": 10}], expire_after_s=1e9, request_remaining=False
    )
    reporter = ActorOnlyResourceReporter(
        _topology(op), limits_coordinator, _FakeResourceBank()
    )
    gauge = RollingLogicalUtilizationGauge(reporter)

    # Separate coordinator receives the autoscaler's scale-up requests.
    request_coordinator = FakeAutoscalingCoordinator()
    node_spec = _NodeResourceSpec.of(cpu=10, gpu=0, mem=0)
    autoscaler = DefaultClusterAutoscalerV2(
        resource_manager=None,
        execution_id="test-actor-only",
        resource_utilization_calculator=gauge,
        cluster_scaling_up_util_threshold=0.75,
        cluster_scaling_up_delta=1,
        min_gap_between_autoscaling_requests_s=0,
        autoscaling_coordinator=request_coordinator,
        get_node_counts=lambda: {node_spec: 1},
    )

    autoscaler.try_trigger_scaling()

    allocated = autoscaler.get_total_resources()
    if should_scale_up:
        # 1 existing node + 1 scale-up delta = 20 CPU requested.
        assert allocated.cpu == 20
    else:
        assert allocated == ExecutionResources.zero()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
