"""Tests for _ExperimentalActorPool heap-based scheduling."""

import unittest
from typing import Any, Dict, Optional, Tuple
from unittest import mock

import numpy as np
import pyarrow as pa
import pytest
from typing_extensions import override

import ray
from ray.actor import ActorHandle
from ray.data._internal.actor_autoscaler import ActorPoolScalingRequest
from ray.data._internal.actor_autoscaler.autoscaling_actor_pool import (
    AutoscalingActorConfig,
)
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.execution_flags import (
    BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO,
)
from ray.data._internal.execution.interfaces import ExecutionResources, NodeIdStr
from ray.data._internal.execution.interfaces.ref_bundle import BlockEntry, RefBundle
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.execution.resource_bank import ResourceBankBase
from ray.data._internal.experimental.execution.operators import (
    actor_pool_map_operator as apmo,
)
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (
    _ACTOR_STATE_ALIVE,
    _ACTOR_STATE_RESTARTING,
    ActorStatus,
    ExperimentalAPMO,
    _ExperimentalActorPool,
)
from ray.data._internal.util import MiB
from ray.data.block import BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)


@ray.remote
class PoolWorker:
    def __init__(self, node_id: str = "node1"):
        self.node_id = node_id

    def get_location(self) -> str:
        return self.node_id

    def __ray_shutdown__(self):
        pass


class _RecordingResourceBank(ResourceBankBase):
    """Minimal ``ResourceBankBase`` for the pool scheduling tests."""

    def __init__(self, node_resources: Optional[ExecutionResources] = None):
        self._node_resources = node_resources or ExecutionResources(
            cpu=4, object_store_memory=1024**3
        )

    @override
    def on_block_consumed(self, input_ref: ObjectRef[Any]) -> None:
        pass

    @override
    def node_capacity(self, node_id: NodeIdStr) -> ExecutionResources:
        return self._node_resources

    @override
    def deregister_actor(self, actor_id: str) -> None:
        pass


def _make_bundle(size_bytes: int, node_id: Optional[str] = None) -> RefBundle:
    if node_id is None:
        node_id = "node1"
    """Create a RefBundle with controlled size_bytes and node_id metadata."""
    block = pa.table({"col": [0] * max(1, size_bytes // 8)})
    block_ref = ray.put(block)
    metadata = BlockMetadata(
        num_rows=block.num_rows,
        size_bytes=size_bytes,
        exec_stats=BlockExecStats(node_id=node_id),
    )
    return RefBundle(
        blocks=(BlockEntry(block_ref, metadata),),
        owns_blocks=False,
        schema=block.schema,
    )


class TestExperimentalActorPool(unittest.TestCase):
    def setup_class(self):
        self._last_created_actor_and_ready_ref: Optional[
            Tuple[ActorHandle, ObjectRef[Any]]
        ] = None
        self._actor_node_id = "node1"
        ray.init(num_cpus=4)

    def teardown_class(self):
        ray.shutdown()

    def _create_actor_fn(
        self,
        labels: Dict[str, Any],
        logical_actor_id: str = "Actor1",
        scheduling_strategy: NodeAffinitySchedulingStrategy | str | None = None,
    ) -> Tuple[ActorHandle, ObjectRef[Any], ExecutionResources]:
        actor = PoolWorker.options(_labels=labels).remote(self._actor_node_id)
        ready_ref = actor.get_location.remote()
        self._last_created_actor_and_ready_ref = actor, ready_ref
        return actor, ready_ref, ExecutionResources(cpu=1)

    def _create_pool(
        self,
        max_tasks_in_flight=4,
        max_input_bytes_per_actor=None,
        min_size=1,
        max_size=4,
        initial_size=1,
        max_num_output_bytes_per_actor: Optional[int] = 10_000_000,
        per_actor_resource_usage=None,
        node_resources=None,
    ) -> _ExperimentalActorPool:
        config = AutoscalingActorConfig(
            min_size=min_size,
            max_size=max_size,
            initial_size=initial_size,
            max_tasks_in_flight_per_actor=max_tasks_in_flight,
            max_actor_concurrency=1,
            per_actor_resource_usage=(
                per_actor_resource_usage or ExecutionResources(cpu=1)
            ),
            max_input_bytes_per_actor=max_input_bytes_per_actor,
            max_num_output_bytes_per_actor=max_num_output_bytes_per_actor,
        )
        pool = _ExperimentalActorPool(
            create_actor_fn=self._create_actor_fn,
            config=config,
        )
        pool.set_resource_bank(_RecordingResourceBank(node_resources))
        return pool

    def _add_pending_actor(
        self, pool: _ExperimentalActorPool, node_id="node1"
    ) -> Tuple[ActorHandle, ObjectRef[Any]]:
        self._actor_node_id = node_id
        num_actors = pool.scale(ActorPoolScalingRequest(delta=1, reason="test"))
        assert num_actors == 1
        assert self._last_created_actor_and_ready_ref is not None
        actor, ready_ref = self._last_created_actor_and_ready_ref
        self._last_created_actor_and_ready_ref = None
        return actor, ready_ref

    def _add_ready_actor(
        self, pool: _ExperimentalActorPool, node_id="node1"
    ) -> ActorHandle:
        actor, ready_ref = self._add_pending_actor(pool, node_id)
        ray.get(ready_ref)
        pool.pending_to_running(ready_ref)
        return actor

    def _output_bytes_limit(
        self,
        *,
        node_resources: ExecutionResources,
        max_num_output_bytes_per_actor: Optional[int],
        per_actor_resource_usage: Optional[ExecutionResources] = None,
    ) -> float:
        """Build a one-actor pool, bring the actor up (which derives its budget
        in ``pending_to_running``), and return its output-bytes limit."""
        pool = self._create_pool(
            max_num_output_bytes_per_actor=max_num_output_bytes_per_actor,
            per_actor_resource_usage=per_actor_resource_usage,
            node_resources=node_resources,
        )
        actor = self._add_ready_actor(pool)
        return pool.output_bytes_limit(pool.get_actor_logical_id(actor))

    # ---- dynamic per-actor output-bytes limit tests ----

    def test_output_bytes_limit_derived_from_object_store_size(self):
        """With no explicit per-actor output limit, the limit is derived from the
        node's (coordinator-allocated) object store: a fraction of it, split by
        the actor's CPU share of the node, rounded to MiB. It scales with the
        object-store size. The fraction is pinned so expectations don't depend on
        the module default."""
        # 4-CPU node, 1-CPU actor, fraction 0.5 -> (object_store * 0.5) / 4 cpu.
        cases = [
            (1 * 1024**3, 128 * MiB),  # 0.5 GiB usable / 4 = 128 MiB
            (2 * 1024**3, 256 * MiB),
            (8 * 1024**3, 1024 * MiB),
        ]
        with mock.patch.object(apmo, "OBJECT_STORE_OUTPUT_FRACTION", 0.5):
            for object_store_bytes, expected in cases:
                with self.subTest(object_store_bytes=object_store_bytes):
                    limit = self._output_bytes_limit(
                        node_resources=ExecutionResources(
                            cpu=4, object_store_memory=object_store_bytes
                        ),
                        max_num_output_bytes_per_actor=None,
                    )
                    self.assertEqual(limit, expected)

    def test_output_bytes_limit_scales_with_per_actor_cpu_share(self):
        """The derived limit is the actor's CPU share of the node's usable object
        store: a 2-CPU actor on a 4-CPU node gets twice a 1-CPU actor's budget."""
        node = ExecutionResources(cpu=4, object_store_memory=8 * 1024**3)
        with mock.patch.object(apmo, "OBJECT_STORE_OUTPUT_FRACTION", 0.5):
            one_cpu = self._output_bytes_limit(
                node_resources=node,
                max_num_output_bytes_per_actor=None,
                per_actor_resource_usage=ExecutionResources(cpu=1),
            )
            two_cpu = self._output_bytes_limit(
                node_resources=node,
                max_num_output_bytes_per_actor=None,
                per_actor_resource_usage=ExecutionResources(cpu=2),
            )
        self.assertEqual(one_cpu, 1024 * MiB)  # 4 GiB usable / 4 cpu * 1
        self.assertEqual(two_cpu, 2048 * MiB)  # * 2

    def test_output_bytes_limit_normalizes_on_cpu_only(self):
        """The derived budget normalizes on CPU alone; memory and GPU are
        ignored. This avoids handing an actor nearly the whole object store when
        a non-CPU dimension is tiny relative to the node (e.g. a small memory
        reservation, or a single-GPU actor on a single-GPU node).

        Node: cpu=8, memory=512 GiB, 1 GPU, object_store=100 GiB (usable 50 GiB
        at fraction 0.5). The budget is the actor's CPU share (1/8) of 50 GiB
        regardless of how large its memory/GPU footprint is.
        """
        node = ExecutionResources(
            cpu=8,
            gpu=1,
            memory=512 * 1024**3,
            object_store_memory=100 * 1024**3,
        )
        # 50 GiB usable / 8 CPUs, floored to MiB (matches the implementation's
        # ``int(x / MiB) * MiB``). Raise the max bound so the clamp doesn't mask
        # the CPU-share derivation this test is exercising (6.25 GiB > 2 GiB).
        expected = int((50 * 1024**3 / 8) / MiB) * MiB
        with (
            mock.patch.object(apmo, "OBJECT_STORE_OUTPUT_FRACTION", 0.5),
            mock.patch.object(apmo, "PER_ACTOR_OBJECT_STORE_MAX_BYTES", 100 * 1024**3),
        ):
            # A tiny memory reservation would, under the old cross-dimension
            # max, have claimed nearly the whole store; here it's ignored.
            for per_actor in (
                ExecutionResources(cpu=1),
                ExecutionResources(cpu=1, memory=4 * 1024**3),
                ExecutionResources(cpu=1, memory=1, gpu=1),
            ):
                with self.subTest(per_actor=per_actor):
                    limit = self._output_bytes_limit(
                        node_resources=node,
                        max_num_output_bytes_per_actor=None,
                        per_actor_resource_usage=per_actor,
                    )
                    self.assertEqual(limit, expected)

    def test_explicit_output_bytes_limit_overrides_derivation(self):
        """An explicit ``max_num_output_bytes_per_actor`` (within the system
        bounds) is used verbatim, so the node object-store size does not affect it
        (derivation is bypassed)."""
        explicit = 100 * MiB
        for object_store_bytes in (1 * 1024**3, 100 * 1024**3):
            with self.subTest(object_store_bytes=object_store_bytes):
                limit = self._output_bytes_limit(
                    node_resources=ExecutionResources(
                        cpu=4, object_store_memory=object_store_bytes
                    ),
                    max_num_output_bytes_per_actor=explicit,
                )
                self.assertEqual(limit, explicit)

    def test_output_bytes_limit_clamped_to_system_max(self):
        """A derived budget above the system ceiling is clamped down to it."""
        # 100 GiB object store, fraction 0.5, 1-CPU/4-CPU share -> ~12.5 GiB
        # derived, well above the system ceiling.
        with mock.patch.object(apmo, "OBJECT_STORE_OUTPUT_FRACTION", 0.5):
            limit = self._output_bytes_limit(
                node_resources=ExecutionResources(
                    cpu=4, object_store_memory=100 * 1024**3
                ),
                max_num_output_bytes_per_actor=None,
            )
        self.assertEqual(limit, apmo.PER_ACTOR_OBJECT_STORE_MAX_BYTES)

    def test_output_bytes_limit_clamped_to_system_min(self):
        """A derived budget below the system floor is clamped up to it (and no
        longer errors on a tiny node)."""
        # 4 MiB object store, fraction 0.5, 1-CPU/4-CPU share -> 0.5 MiB derived
        # (floors to 0), well below the system floor.
        with mock.patch.object(apmo, "OBJECT_STORE_OUTPUT_FRACTION", 0.5):
            limit = self._output_bytes_limit(
                node_resources=ExecutionResources(cpu=4, object_store_memory=4 * MiB),
                max_num_output_bytes_per_actor=None,
            )
        self.assertEqual(limit, apmo.PER_ACTOR_OBJECT_STORE_MIN_BYTES)

    def test_output_bytes_limit_zero_cpu_share_floored_to_min(self):
        """A zero per-actor CPU share (num_cpus=0), or a CPU-less landing node,
        can't yield a CPU-proportional budget -- it falls back to 0 (no divide-
        by-zero, no error) and the MIN floor pins it to the system minimum."""
        with mock.patch.object(apmo, "OBJECT_STORE_OUTPUT_FRACTION", 0.5):
            # num_cpus=0 actor on a CPU-bearing node.
            zero_cpu_actor = self._output_bytes_limit(
                node_resources=ExecutionResources(
                    cpu=4, object_store_memory=8 * 1024**3
                ),
                max_num_output_bytes_per_actor=None,
                per_actor_resource_usage=ExecutionResources(cpu=0),
            )
            # 1-CPU actor landing on a CPU:0 node (e.g. a GPU-only node).
            zero_cpu_node = self._output_bytes_limit(
                node_resources=ExecutionResources(
                    cpu=0, gpu=4, object_store_memory=8 * 1024**3
                ),
                max_num_output_bytes_per_actor=None,
                per_actor_resource_usage=ExecutionResources(cpu=1),
            )
        self.assertEqual(zero_cpu_actor, apmo.PER_ACTOR_OBJECT_STORE_MIN_BYTES)
        self.assertEqual(zero_cpu_node, apmo.PER_ACTOR_OBJECT_STORE_MIN_BYTES)

    def test_explicit_output_bytes_limit_escapes_bounds(self):
        """An explicit ``max_num_output_bytes_per_actor`` is an override used
        verbatim -- it is NOT clamped into the system [min, max] range (only the
        auto-derived budget is). So a user value above the ceiling or below the
        floor is honored as-is."""
        node = ExecutionResources(cpu=4, object_store_memory=8 * 1024**3)
        for explicit in (
            5 * 1024**3,  # above the system ceiling
            100,  # below the system floor
        ):
            with self.subTest(explicit=explicit):
                limit = self._output_bytes_limit(
                    node_resources=node,
                    max_num_output_bytes_per_actor=explicit,
                )
                self.assertEqual(limit, explicit)

    def test_output_bytes_limit_bounds_are_overridable(self):
        """The clamp on the derived budget reads the (env-backed) bound constants,
        so overriding them moves the ceiling/floor the derived budget is pinned
        to."""
        with (
            mock.patch.object(apmo, "OBJECT_STORE_OUTPUT_FRACTION", 0.5),
            mock.patch.object(apmo, "PER_ACTOR_OBJECT_STORE_MAX_BYTES", 256 * MiB),
            mock.patch.object(apmo, "PER_ACTOR_OBJECT_STORE_MIN_BYTES", 64 * MiB),
        ):
            # Derived is 4 GiB usable / 4 cpu = 1 GiB, clamped down to the 256 MiB
            # ceiling.
            self.assertEqual(
                self._output_bytes_limit(
                    node_resources=ExecutionResources(
                        cpu=4, object_store_memory=8 * 1024**3
                    ),
                    max_num_output_bytes_per_actor=None,
                ),
                256 * MiB,
            )
            # Derived is 128 MiB usable / 4 cpu = 32 MiB, clamped up to the 64 MiB
            # floor.
            self.assertEqual(
                self._output_bytes_limit(
                    node_resources=ExecutionResources(
                        cpu=4, object_store_memory=256 * MiB
                    ),
                    max_num_output_bytes_per_actor=None,
                ),
                64 * MiB,
            )

    def _select_and_submit(
        self,
        pool: _ExperimentalActorPool,
        bundle: RefBundle,
        actor_locality_enabled: bool = False,
    ) -> Optional[ActorHandle]:
        actor = pool.select_actors(
            bundle=bundle, actor_locality_enabled=actor_locality_enabled
        )
        if actor is not None:
            pool.on_task_submitted(actor, bundle)
        return actor

    # ---- max_tasks_in_flight tests ----

    def test_max_tasks_in_flight_respected(self):
        pool = self._create_pool(max_tasks_in_flight=2)
        actor = self._add_ready_actor(pool)

        b1 = _make_bundle(100)
        b2 = _make_bundle(100)
        b3 = _make_bundle(100)

        a1 = self._select_and_submit(pool, b1)
        assert a1 == actor
        a2 = self._select_and_submit(pool, b2)
        assert a2 == actor

        assert pool.select_actors(bundle=b3) is None

        pool.on_task_completed(actor, b1)
        a3 = self._select_and_submit(pool, b3)
        assert a3 == actor

    def test_max_tasks_in_flight_spreads_across_actors(self):
        pool = self._create_pool(max_tasks_in_flight=2)
        actor1 = self._add_ready_actor(pool, node_id="node1")
        actor2 = self._add_ready_actor(pool, node_id="node2")

        bundles = [_make_bundle(100) for _ in range(4)]
        assigned = []
        for b in bundles:
            a = self._select_and_submit(pool, b)
            assert a is not None
            assigned.append(a)

        actor1_count = sum(1 for a in assigned if a == actor1)
        actor2_count = sum(1 for a in assigned if a == actor2)
        assert actor1_count == 2
        assert actor2_count == 2

        assert pool.select_actors(bundle=_make_bundle(100)) is None

    # ---- input_bytes tests ----

    def test_input_bytes_capacity_respected(self):
        pool = self._create_pool(max_tasks_in_flight=10, max_input_bytes_per_actor=1000)
        actor = self._add_ready_actor(pool)

        b1 = _make_bundle(600)
        b2 = _make_bundle(500)
        b3 = _make_bundle(100)

        a1 = self._select_and_submit(pool, b1)
        assert a1 == actor
        # 600 <= 1000, still in heaps
        a2 = self._select_and_submit(pool, b2)
        assert a2 == actor
        # 600 + 500 = 1100 > 1000, removed from heaps after sync

        assert pool.select_actors(bundle=b3) is None

        pool.on_task_completed(actor, b1)
        # 1100 - 600 = 500 <= 1000, back in heaps
        a3 = self._select_and_submit(pool, b3)
        assert a3 == actor

    def test_input_bytes_boundary_equal_to_max(self):
        pool = self._create_pool(max_tasks_in_flight=10, max_input_bytes_per_actor=1000)
        actor = self._add_ready_actor(pool)

        # Submit 999 bytes — just under the limit, actor stays available
        b_under = _make_bundle(999)
        a1 = self._select_and_submit(pool, b_under)
        assert a1 == actor

        # Next submit pushes to 999+1=1000 which is NOT < 1000, so
        # after sync the actor is removed from heaps
        b_tiny = _make_bundle(1)
        a2 = self._select_and_submit(pool, b_tiny)
        assert a2 == actor

        assert pool.select_actors(bundle=_make_bundle(1)) is None

        # Complete the 1-byte task → 999 < 1000, actor available again
        pool.on_task_completed(actor, b_tiny)
        a3 = self._select_and_submit(pool, _make_bundle(1))
        assert a3 == actor

    def test_input_bytes_none_means_unlimited(self):
        pool = self._create_pool(max_tasks_in_flight=10, max_input_bytes_per_actor=None)
        actor = self._add_ready_actor(pool)

        for _ in range(10):
            b = _make_bundle(1_000_000)
            a = self._select_and_submit(pool, b)
            assert a == actor

    def test_both_flight_and_bytes_gate_scheduling(self):
        """Actor must have BOTH flight capacity and input bytes capacity."""
        pool = self._create_pool(max_tasks_in_flight=2, max_input_bytes_per_actor=500)
        actor = self._add_ready_actor(pool)

        b1 = _make_bundle(400)
        a1 = self._select_and_submit(pool, b1)
        assert a1 == actor
        # flight=1/2, bytes=400/500 — both under cap

        b2 = _make_bundle(200)
        a2 = self._select_and_submit(pool, b2)
        assert a2 == actor
        # flight=2/2, bytes=600/500 — BOTH caps exceeded

        assert pool.select_actors(bundle=_make_bundle(1)) is None

        pool.on_task_completed(actor, b2)
        # flight=1/2, bytes=400/500 — both under cap again
        a3 = self._select_and_submit(pool, _make_bundle(50))
        assert a3 == actor

    # ---- scheduling priority tests ----

    def test_least_loaded_actor_preferred(self):
        pool = self._create_pool(max_tasks_in_flight=4)
        actor1 = self._add_ready_actor(pool, node_id="node1")
        actor2 = self._add_ready_actor(pool, node_id="node2")

        # Submit four tasks; the pool always hands the next one to the
        # least-loaded actor, so picks must alternate. That the sequence is
        # A, B, A, B is the observable proof of an even 2/2 split -- no need to
        # reach into the pool's private per-actor state to confirm the counts.
        picks = [self._select_and_submit(pool, _make_bundle(100)) for _ in range(4)]

        assert None not in picks
        assert {picks[0], picks[1]} == {actor1, actor2}, picks
        # Rounds alternate: pick 3 returns to pick 1's actor, pick 4 to pick 2's.
        assert picks[2] == picks[0], picks
        assert picks[3] == picks[1], picks
        # Total load is what the pool reports publicly: 4 tasks, 2 per actor.
        assert pool.num_tasks_in_flight() == 4

    def test_completing_task_reprioritizes_actor(self):
        pool = self._create_pool(max_tasks_in_flight=4)
        actor1 = self._add_ready_actor(pool, node_id="node1")
        _actor2 = self._add_ready_actor(pool, node_id="node2")

        bundles = [_make_bundle(100) for _ in range(4)]
        for b in bundles[:4]:
            self._select_and_submit(pool, b)
        # Both actors have 2 tasks each

        # Complete both tasks on actor1
        for b in bundles[:2]:
            pool.on_task_completed(actor1, b)
        # actor1: 0 tasks, actor2: 2 tasks

        # Next 2 picks should both go to actor1 (lower rank)
        b5 = _make_bundle(100)
        a5 = self._select_and_submit(pool, b5)
        b6 = _make_bundle(100)
        a6 = self._select_and_submit(pool, b6)
        assert a5 == actor1
        assert a6 == actor1

    def test_task_completion_while_actor_restarting(self):
        """A node death fails the in-flight task AND marks its actor
        RESTARTING; the task's done-callback still fires. It must drain the
        accounting without re-adding the restarting actor to the scheduling
        heaps, and the actor becomes selectable again once it is ALIVE."""
        pool = self._create_pool(max_tasks_in_flight=2)
        actor = self._add_ready_actor(pool)

        b1 = _make_bundle(100)
        assert self._select_and_submit(pool, b1) == actor

        # GCS reports the actor RESTARTING (its node died).
        with mock.patch.object(
            ActorHandle, "_get_local_state", return_value=_ACTOR_STATE_RESTARTING
        ):
            pool.refresh_actor_state()

        # The failed task's done-callback fires while the actor restarts.
        pool.on_task_completed(actor, b1)

        assert pool._running_actors[actor].num_tasks_in_flight == 0
        # Freed capacity must NOT make the restarting actor schedulable.
        assert pool.select_actors(bundle=_make_bundle(100)) is None

        # Recovery: the actor is ALIVE again -> selectable once more.
        with mock.patch.object(
            ActorHandle, "_get_local_state", return_value=_ACTOR_STATE_ALIVE
        ):
            pool.refresh_actor_state()
        assert self._select_and_submit(pool, _make_bundle(100)) == actor

    def test_evicting_restarting_actor_keeps_counters_consistent(self):
        """Draining a RESTARTING victim (e.g. a sizer eviction during chaos)
        must retire it from the restarting accounting: otherwise the
        restarting count leaks forever and num_alive_actors goes negative."""
        pool = self._create_pool()
        actor = self._add_ready_actor(pool)

        # GCS reports the actor RESTARTING (its node died).
        with mock.patch.object(
            ActorHandle, "_get_local_state", return_value=_ACTOR_STATE_RESTARTING
        ):
            pool.refresh_actor_state()
        assert pool.num_restarting_actors() == 1
        assert pool.pending_logical_usage() == ExecutionResources(cpu=1)

        # The sizer picks it as an eviction victim.
        pool.begin_graceful_termination(actor, reclaimable=False)

        assert pool.num_restarting_actors() == 0
        assert pool.num_alive_actors() == 0
        assert pool.pending_logical_usage() == ExecutionResources.zero()

        # Full release retires it without double-decrementing.
        pool._release_running_actor(actor)
        assert pool.num_restarting_actors() == 0
        assert pool.num_alive_actors() == 0
        assert pool._num_terminating_actors == 0

    def test_draining_actor_stays_terminating_on_gcs_restart_report(self):
        """A draining (TERMINATING) actor whose node dies is reported
        RESTARTING by GCS. It must stay TERMINATING: flipping the status
        leaks the terminating count (num_alive_actors goes negative) and
        re-arms an evicted actor for revival."""
        pool = self._create_pool()
        actor = self._add_ready_actor(pool)
        pool.begin_graceful_termination(actor, reclaimable=False)
        assert pool.num_alive_actors() == 0

        with mock.patch.object(
            ActorHandle, "_get_local_state", return_value=_ACTOR_STATE_RESTARTING
        ):
            pool.refresh_actor_state()

        assert pool._running_actors[actor].status == ActorStatus.TERMINATING
        assert pool.num_restarting_actors() == 0
        assert pool.num_alive_actors() == 0

        # GCS then reports the actor DEAD -> the pool fully retires it.
        with mock.patch.object(ActorHandle, "_get_local_state", return_value=None):
            pool.refresh_actor_state()
        assert actor not in pool._running_actors
        assert pool.num_restarting_actors() == 0
        assert pool._num_terminating_actors == 0

    def test_num_idle_actors_counter_tracks_scan(self):
        """``num_idle_actors`` is served from a maintained counter; it must
        equal a brute-force scan of the idle predicate (non-TERMINATING with
        zero in-flight tasks) after every lifecycle transition."""

        def oracle(pool: _ExperimentalActorPool) -> int:
            return sum(
                1
                for state in pool._running_actors.values()
                if state.status != ActorStatus.TERMINATING
                and state.num_tasks_in_flight == 0
            )

        def check(pool: _ExperimentalActorPool, expected: int, label: str) -> None:
            assert oracle(pool) == expected, f"{label}: oracle {oracle(pool)}"
            assert pool.num_idle_actors() == expected, (
                f"{label}: num_idle_actors {pool.num_idle_actors()}"
            )
            assert pool._num_idle_actors == expected, (
                f"{label}: counter {pool._num_idle_actors}"
            )

        pool = self._create_pool(max_tasks_in_flight=4)
        check(pool, 0, "empty pool")

        # Actor add: joins the pool idle.
        actor1 = self._add_ready_actor(pool, node_id="node1")
        check(pool, 1, "one ready actor")
        actor2 = self._add_ready_actor(pool, node_id="node2")
        check(pool, 2, "two ready actors")

        # Task submit: 0 -> 1 in flight leaves idle; 1 -> 2 must not
        # double-count.
        b1 = _make_bundle(100, node_id="node1")
        pool.on_task_submitted(actor1, b1)
        check(pool, 1, "actor1 first task")
        b2 = _make_bundle(100, node_id="node1")
        pool.on_task_submitted(actor1, b2)
        check(pool, 1, "actor1 second task")

        # Task completion: only the 1 -> 0 transition returns it to idle.
        pool.on_task_completed(actor1, b2)
        check(pool, 1, "actor1 one task left")
        pool.on_task_completed(actor1, b1)
        check(pool, 2, "actor1 drained")

        # RESTARTING with zero in-flight still satisfies the idle predicate
        # (only TERMINATING is excluded), in both directions.
        with mock.patch.object(
            ActorHandle, "_get_local_state", return_value=_ACTOR_STATE_RESTARTING
        ):
            pool.refresh_actor_state()
        check(pool, 2, "both restarting")
        with mock.patch.object(
            ActorHandle, "_get_local_state", return_value=_ACTOR_STATE_ALIVE
        ):
            pool.refresh_actor_state()
        check(pool, 2, "both recovered")

        # Graceful termination of an idle actor, then reclaim revives it.
        pool.begin_graceful_termination(actor1, reclaimable=True)
        check(pool, 1, "actor1 draining")
        assert pool.reclaim_draining_actors(1) == 1
        check(pool, 2, "actor1 reclaimed")

        # Terminating an ACTIVE actor does not change the idle count, and a
        # task completing on a draining actor must not return it to idle.
        b3 = _make_bundle(100, node_id="node2")
        pool.on_task_submitted(actor2, b3)
        check(pool, 1, "actor2 active")
        pool.begin_graceful_termination(actor2, reclaimable=True)
        check(pool, 1, "actor2 draining while active")
        pool.on_task_completed(actor2, b3)
        check(pool, 1, "actor2 drained but terminating")

        # Reclaiming the now-idle draining actor restores it.
        assert pool.reclaim_draining_actors(1) == 1
        check(pool, 2, "actor2 reclaimed")

        # Release of an idle actor decrements; release of an active actor
        # (forced shutdown with tasks in flight) does not.
        b4 = _make_bundle(100, node_id="node1")
        pool.on_task_submitted(actor1, b4)
        check(pool, 1, "actor1 active again")
        pool._release_running_actor(actor1)
        check(pool, 1, "active actor1 released")
        pool._release_running_actor(actor2)
        check(pool, 0, "idle actor2 released")

    def test_active_nodes_tracks_actors_with_tasks(self):
        """``active_nodes`` is served from ``_node_to_active_actor_count`` and
        must match a scan of nodes hosting actors with in-flight tasks after
        every lifecycle transition."""

        def oracle(pool: _ExperimentalActorPool) -> set:
            return {
                state.actor_location
                for state in pool._running_actors.values()
                if state.num_tasks_in_flight > 0
            }

        def check(pool: _ExperimentalActorPool, expected: set, label: str) -> None:
            assert oracle(pool) == expected, f"{label}: oracle {oracle(pool)}"
            assert pool.active_nodes() == expected, (
                f"{label}: active_nodes {pool.active_nodes()}"
            )
            assert set(pool._node_to_active_actor_count) == expected, (
                f"{label}: count keys {set(pool._node_to_active_actor_count)}"
            )
            assert sum(pool._node_to_active_actor_count.values()) == sum(
                1
                for state in pool._running_actors.values()
                if state.num_tasks_in_flight > 0
            ), f"{label}: count total {pool._node_to_active_actor_count}"

        pool = self._create_pool(max_tasks_in_flight=4)
        check(pool, set(), "empty pool")

        actor1 = self._add_ready_actor(pool, node_id="node1")
        actor2 = self._add_ready_actor(pool, node_id="node2")
        check(pool, set(), "idle actors do not activate nodes")
        assert pool.nodes() == {"node1", "node2"}

        # 0 -> 1 in flight activates the node; 1 -> 2 must not double-count.
        b1 = _make_bundle(100, node_id="node1")
        pool.on_task_submitted(actor1, b1)
        check(pool, {"node1"}, "actor1 first task")
        b2 = _make_bundle(100, node_id="node1")
        pool.on_task_submitted(actor1, b2)
        check(pool, {"node1"}, "actor1 second task")
        assert pool._node_to_active_actor_count["node1"] == 1

        # A second actor on another node adds that node.
        b3 = _make_bundle(100, node_id="node2")
        pool.on_task_submitted(actor2, b3)
        check(pool, {"node1", "node2"}, "both nodes active")

        # Completing the last task on a node drops it from active_nodes.
        pool.on_task_completed(actor1, b2)
        check(pool, {"node1", "node2"}, "actor1 still has a task")
        pool.on_task_completed(actor1, b1)
        check(pool, {"node2"}, "node1 idle again")
        pool.on_task_completed(actor2, b3)
        check(pool, set(), "all drained")

        # Two active actors on the same node: releasing one must not drop
        # the node until the other finishes.
        actor3 = self._add_ready_actor(pool, node_id="node1")
        b4 = _make_bundle(100, node_id="node1")
        b5 = _make_bundle(100, node_id="node1")
        pool.on_task_submitted(actor1, b4)
        pool.on_task_submitted(actor3, b5)
        check(pool, {"node1"}, "two active actors on node1")
        assert pool._node_to_active_actor_count["node1"] == 2
        pool._release_running_actor(actor1)
        check(pool, {"node1"}, "one active actor remains on node1")
        assert pool._node_to_active_actor_count["node1"] == 1
        pool._release_running_actor(actor3)
        check(pool, set(), "last active actor released")

        # Terminating with in-flight work still counts as active until the
        # task completes.
        b6 = _make_bundle(100, node_id="node2")
        pool.on_task_submitted(actor2, b6)
        pool.begin_graceful_termination(actor2, reclaimable=True)
        check(pool, {"node2"}, "terminating actor still active")
        pool.on_task_completed(actor2, b6)
        check(pool, set(), "terminating actor drained")

    # ---- locality tests ----

    def test_locality_preferred_over_fallback(self):
        # Locality is derived from the bundle's own block metadata
        # (get_preferred_object_locations_from_metadata reads exec_stats.node_id),
        # so building node1-local bundles is the seam -- no mocking needed.
        pool = self._create_pool(max_tasks_in_flight=4)
        actor1 = self._add_ready_actor(pool, node_id="node1")
        actor2 = self._add_ready_actor(pool, node_id="node2")

        # Submit a task to actor2 first so it has lower rank
        b0 = _make_bundle(100, node_id="node1")
        pool.on_task_submitted(actor2, b0)  # manually push actor2 to rank 1

        b1 = _make_bundle(100, node_id="node1")
        a1 = pool.select_actors(bundle=b1, actor_locality_enabled=True)
        # Should pick actor1 (on preferred node) even though actor2
        # would be fallback choice at equal load
        assert a1 == actor1

    def test_locality_falls_back_when_local_at_capacity(self):
        pool = self._create_pool(max_tasks_in_flight=1)
        actor1 = self._add_ready_actor(pool, node_id="node1")
        actor2 = self._add_ready_actor(pool, node_id="node2")

        b1 = _make_bundle(100, node_id="node1")
        a1 = self._select_and_submit(pool, b1, actor_locality_enabled=True)
        assert a1 == actor1

        # actor1 at capacity (1/1), locality can't find it
        b2 = _make_bundle(100, node_id="node1")
        a2 = self._select_and_submit(pool, b2, actor_locality_enabled=True)
        assert a2 == actor2

    def test_no_locality_info_uses_fallback(self):
        pool = self._create_pool(max_tasks_in_flight=4)
        actor1 = self._add_ready_actor(pool, node_id="node1")

        # A bundle with no exec stats reports no preferred locations, so
        # selection must fall back to the global heap (the only actor).
        b1 = _make_bundle(100, node_id=None)
        a1 = pool.select_actors(bundle=b1, actor_locality_enabled=True)
        assert a1 == actor1

    # ---- end to end ----

    def test_e2e_map_with_actor_pool(self):
        ds = ray.data.range(100)
        result = ds.map(lambda row: row).take_all()
        assert len(result) == 100
        values = sorted(r["id"] for r in result)
        assert values == list(range(100))

    def test_e2e_map_batches_with_actor_pool(self):
        class Identity:
            def __call__(self, batch):
                return batch

        ds = ray.data.range(200)
        result = ds.map_batches(
            Identity,
            compute=ActorPoolStrategy(size=2),
        ).take_all()
        assert len(result) == 200
        values = sorted(r["id"] for r in result)
        assert values == list(range(200))

    def test_oversized_unsplittable_row_warns(self):
        """Rows larger than the per-actor target block size cannot be split, so
        block shaping is imprecise. The actor-only backend should warn when a
        pulled block exceeds 150% of the actor's target block size, and again
        when outstanding output exceeds 150% of the actor's output-bytes limit.

        Setup: output limit 4 MiB -> target block size 1 MiB (25% ratio). Each
        UDF row carries an 8 MiB payload (unsplittable), so both thresholds fire
        as soon as the first block is pulled.
        """
        output_limit = 4 * MiB
        row_nbytes = 8 * MiB

        class EmitLargeRows:
            def __call__(self, batch):
                n = len(batch["id"])
                return {
                    "id": batch["id"],
                    "data": [np.zeros(row_nbytes, dtype=np.uint8) for _ in range(n)],
                }

        with mock.patch.object(apmo.logger, "warning") as warn:
            (
                ray.data.range(2, override_num_blocks=2)
                .map_batches(
                    EmitLargeRows,
                    batch_size=1,
                    compute=ActorPoolStrategy(
                        size=1,
                        max_num_output_bytes_per_actor=output_limit,
                    ),
                )
                .materialize()
            )

        messages = [str(call.args[0]) for call in warn.call_args_list if call.args]
        assert any("more than 150% of its target block size" in m for m in messages), (
            messages
        )
        assert any("more than 150% of its output-bytes limit" in m for m in messages), (
            messages
        )


def _make_actor_op(
    ray_remote_args: Dict[str, Any],
    ray_remote_args_fn=None,
    compute_strategy: Optional[ActorPoolStrategy] = None,
) -> ExperimentalAPMO:
    """Build a real ExperimentalAPMO the way the planner does."""
    ctx = DataContext.get_current()
    input_op = InputDataBuffer(ctx, input_data=[])
    transformer = MapTransformer(
        [BlockMapTransformFn(lambda blocks, _ctx: iter(blocks))]
    )
    return ExperimentalAPMO(
        map_transformer=transformer,
        input_op=input_op,
        data_context=ctx,
        compute_strategy=compute_strategy or ActorPoolStrategy(),
        ray_remote_args=ray_remote_args,
        ray_remote_args_fn=ray_remote_args_fn,
        name="Infer",
    )


def test_system_output_bytes_limit_env_setting():
    """The system default (RAY_DATA_PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT,
    surfaced as PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT) is used as the actor
    output-bytes limit when the ActorPoolStrategy doesn't set one; an explicit
    strategy value wins over it.

    There is no public getter for the computed pool config (same as
    ``_ray_remote_args`` in ``test_pg_op_skips_default_memory_reservation``), so
    the assertions read ``_config`` directly."""
    env_limit = 123 * MiB
    with mock.patch.object(
        apmo, "PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT", env_limit
    ):
        # Strategy leaves it unset -> the system env value is used.
        op = _make_actor_op({"num_cpus": 1})
        assert op.actor_pool._config.max_num_output_bytes_per_actor == env_limit

        # Explicit strategy value takes precedence over the system env value.
        strategy_limit = 7 * MiB
        op_override = _make_actor_op(
            {"num_cpus": 1},
            compute_strategy=ActorPoolStrategy(
                max_num_output_bytes_per_actor=strategy_limit
            ),
        )
        assert (
            op_override.actor_pool._config.max_num_output_bytes_per_actor
            == strategy_limit
        )


def test_scale_to_initial_size_skipped_under_sizer(monkeypatch):
    """Under the sizer, pools must NOT scale themselves to their initial size
    at start(): those actors are created untargeted (Ray core places them)
    before the sizer's first tick, bypassing constraint-aware placement -- a
    busy unconstrained actor landing on a labeled/pinned node is never evicted
    and can wedge a constrained pool below its target for the whole run. The
    sizer's initial_sizing_request owns initial sizing instead."""
    import ray.data._internal.experimental.execution.operators.actor_pool_map_operator as apmo  # noqa: E501

    op = _make_actor_op({"num_cpus": 1})
    calls = []

    class _RecordingPool:
        def initial_size(self):
            return 3

        def scale(self, request):
            calls.append(request)

    op._actor_pool = _RecordingPool()

    monkeypatch.setattr(apmo, "ENABLE_OPERATOR_SIZER", True)
    op._scale_to_initial_size()
    assert calls == []

    monkeypatch.setattr(apmo, "ENABLE_OPERATOR_SIZER", False)
    op._scale_to_initial_size()
    assert len(calls) == 1 and calls[0].delta == 3


def _patch_actor_wait(monkeypatch, op, refs, ready_schedule):
    """Drive ``wait_for_min_actors`` deterministically.

    ``ready_schedule`` is an iterable of per-``ray.wait``-call lists of refs to
    resolve (``[]`` means "nothing became ready this poll"); it is padded with
    ``[]`` forever. A fake clock advances by the timeout each ``ray.wait`` asks
    for, so no test actually sleeps.
    """
    import ray as ray_mod
    import ray.data._internal.execution.operators.actor_pool_map_operator as apmo_base

    class _PendingPool:
        def get_pending_actor_refs(self):
            return list(refs)

    op._actor_pool = _PendingPool()

    clock = {"t": 0.0}
    calls = []
    schedule = list(ready_schedule)

    def _wait(pending, num_returns=1, timeout=None):
        calls.append((list(pending), timeout))
        clock["t"] += timeout
        ready = schedule.pop(0) if schedule else []
        return list(ready), [r for r in pending if r not in ready]

    monkeypatch.setattr(ray_mod, "wait", _wait)
    monkeypatch.setattr(ray_mod, "get", lambda refs, **kw: None)
    monkeypatch.setattr(apmo_base.time, "perf_counter", lambda: clock["t"])
    return calls, clock


@pytest.mark.parametrize(
    "num_actors,expected",
    [(1, 315), (5, 375), (20, 600), (100, 1800), (1000, 1800)],
)
def test_wait_for_min_actors_budget_scales_with_pool_size(num_actors, expected):
    """The default (unset ``wait_for_min_actors_s``) budget grows with the
    number of actors waited on and saturates at the max, instead of using one
    fixed 5-minute deadline for every pool size."""
    from ray.data._internal.execution.operators.actor_pool_map_operator import (
        wait_for_min_actors_budget_s,
    )

    assert wait_for_min_actors_budget_s(num_actors) == expected


def test_wait_for_min_actors_waits_on_pending_refs(monkeypatch):
    """Under the sizer, initial actors are created by the sizer's initial
    sizing pass (not op.start()), which then runs this wait. Semantics:
    explicitly-set wait_for_min_actors_s (> 0) wins as a hard deadline; unset
    uses the size-derived budget when the actor-only backend is enabled, and
    no wait at all otherwise."""
    import ray.data._internal.execution.operators.actor_pool_map_operator as apmo_base

    op = _make_actor_op({"num_cpus": 1})

    # Explicitly configured: returns as soon as both refs resolve.
    calls, _ = _patch_actor_wait(monkeypatch, op, ["ref1", "ref2"], [["ref1", "ref2"]])
    monkeypatch.setattr(op.data_context, "wait_for_min_actors_s", 5)
    op.wait_for_min_actors()
    assert len(calls) == 1

    # Unset + actor-only backend (conftest enables it): still waits, and keeps
    # polling across a poll where nothing became ready.
    calls, _ = _patch_actor_wait(
        monkeypatch, op, ["ref1", "ref2"], [[], ["ref1"], ["ref2"]]
    )
    monkeypatch.setattr(op.data_context, "wait_for_min_actors_s", -1)
    op.wait_for_min_actors()
    assert len(calls) == 3

    # Unset + actor-only backend disabled: no wait (OSS behavior unchanged).
    calls, _ = _patch_actor_wait(monkeypatch, op, ["ref1", "ref2"], [["ref1", "ref2"]])
    monkeypatch.setattr(apmo_base, "actor_only_backend_enabled", lambda: False)
    op.wait_for_min_actors()
    assert calls == []

    # No pending refs: nothing to wait on regardless of configuration.
    monkeypatch.setattr(apmo_base, "actor_only_backend_enabled", lambda: True)
    calls, _ = _patch_actor_wait(monkeypatch, op, [], [])
    op.wait_for_min_actors()
    assert calls == []


def test_wait_for_min_actors_timeout_raises_with_guidance(monkeypatch):
    """Expiry fails the dataset with guidance: tune the DAG's per-operator
    min concurrency or raise RAY_DATA_DEFAULT_WAIT_FOR_MIN_ACTORS_S."""
    import ray as ray_mod

    op = _make_actor_op({"num_cpus": 1})
    _patch_actor_wait(monkeypatch, op, ["ref1"], [])
    monkeypatch.setattr(op.data_context, "wait_for_min_actors_s", 5)
    with pytest.raises(
        ray_mod.exceptions.GetTimeoutError,
        match="RAY_DATA_DEFAULT_WAIT_FOR_MIN_ACTORS_S",
    ):
        op.wait_for_min_actors()


def test_wait_for_min_actors_aborts_on_stalled_startup(monkeypatch):
    """On the default path the wait gives up once startup stalls, well before
    the (size-derived) overall budget expires -- a pool that can't be placed
    fails fast no matter how large its budget is."""
    import ray as ray_mod
    import ray.data._internal.execution.operators.actor_pool_map_operator as apmo_base

    op = _make_actor_op({"num_cpus": 1})
    monkeypatch.setattr(op.data_context, "wait_for_min_actors_s", -1)
    monkeypatch.setattr(apmo_base, "DEFAULT_WAIT_FOR_MIN_ACTORS_STALL_S", 10)
    # 100 actors -> a 1800s budget, but only one ever starts.
    refs = [f"ref{i}" for i in range(100)]
    calls, clock = _patch_actor_wait(monkeypatch, op, refs, [["ref0"]])

    with pytest.raises(ray_mod.exceptions.GetTimeoutError, match="1/100"):
        op.wait_for_min_actors()

    # Failed ~10s after the last actor came up, not after the 1800s budget.
    assert clock["t"] < 30


def test_create_task_context_clamps_block_size_to_configured_target(
    restore_data_context,
):
    """``_create_task_context`` sizes blocks as
    ``min(output_limit * ratio, configured target)``: the actor's output
    budget may shrink the target block size, but must not raise it above the
    configured DataContext / op override.
    """
    configured = 10 * MiB
    restore_data_context.target_max_block_size = configured
    op = _make_actor_op({"num_cpus": 1})
    actor = mock.Mock(name="actor")

    # Large output budget: ratio-derived size would exceed the configured
    # target; keep the configured target.
    large_limit = 128 * MiB
    with (
        mock.patch.object(
            op.actor_pool, "get_actor_logical_id", return_value="actor-1"
        ),
        mock.patch.object(
            op.actor_pool, "output_bytes_limit", return_value=large_limit
        ),
    ):
        ctx = op._create_task_context(actor)
    assert ctx.target_max_block_size_override == configured

    # Small output budget: ratio-derived size is below the configured target;
    # shrink to fit under the actor's output-bytes limit.
    small_limit = 4 * MiB
    expected = int(small_limit * BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO)
    assert expected < configured
    with (
        mock.patch.object(
            op.actor_pool, "get_actor_logical_id", return_value="actor-1"
        ),
        mock.patch.object(
            op.actor_pool, "output_bytes_limit", return_value=small_limit
        ),
    ):
        ctx = op._create_task_context(actor)
    assert ctx.target_max_block_size_override == expected


def test_pg_op_skips_default_memory_reservation():
    """A placement-group op must NOT get the default heap reservation: the actor
    is scheduled into a user-defined bundle, and reserving memory the bundle
    didn't declare makes it unplaceable by Ray core and zeroes the sizer's
    per-bundle max_placeable_actors -> the pool stalls at 0. Mirrors the OSS
    ConfigureMapTaskMemoryRule guard.

    The reserved memory is written into ``_ray_remote_args`` (so Ray Core
    admission-controls on it); there's no public getter for the computed remote
    args, so the assertions read that attribute directly."""
    # Non-PG op: the default heap reservation is applied.
    plain = _make_actor_op({"num_cpus": 1})
    assert plain._ray_remote_args.get("memory")  # nonzero

    # PG op via explicit remote-args key: reservation skipped.
    pg_key = _make_actor_op({"num_cpus": 1, "placement_group": object()})
    assert "memory" not in pg_key._ray_remote_args

    # PG op via a PlacementGroupSchedulingStrategy: also skipped.
    pg_strat = _make_actor_op(
        {
            "num_cpus": 1,
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=object()
            ),
        }
    )
    assert "memory" not in pg_strat._ray_remote_args

    # PG op via a ray_remote_args_fn (e.g. the vLLM "ray" backend returns a
    # PlacementGroupSchedulingStrategy per replica): the PG isn't visible in
    # the static remote args, but the reservation must still be skipped -- else
    # the actor requests memory the bundle didn't declare and Ray core rejects
    # it with "cannot fit into any bundles for the placement group".
    pg_fn = _make_actor_op(
        {"num_gpus": 0},
        ray_remote_args_fn=lambda: {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=object()
            )
        },
    )
    assert "memory" not in pg_fn._ray_remote_args


def test_ray_remote_args_fn_must_only_assign_a_placement_group(monkeypatch):
    """With the sizer on (PG-aware path) a ray_remote_args_fn is allowed ONLY if
    it assigns a placement group and nothing else. A fn that assigns a PG merges
    cleanly (the PG strategy is preserved); a fn that overrides the resource
    footprint or sets a non-PG strategy would break the sizer's accounting and is
    rejected at merge time (the one point the fn's realized output is
    available). Without the sizer the pool doesn't own placement, so any fn
    keeps working unchecked."""
    import ray.data._internal.experimental.execution.operators.actor_pool_map_operator as apmo  # noqa: E501

    monkeypatch.setattr(apmo, "ENABLE_OPERATOR_SIZER", True)
    monkeypatch.setattr(apmo, "SIZER_CONSTRAINT_AWARE_PLACEMENT", True)

    # (a) fn assigns a PG -> merge OK, PG strategy preserved.
    ok = _make_actor_op(
        {"num_gpus": 0},
        ray_remote_args_fn=lambda: {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=object()
            )
        },
    )
    merged = ok._merge_ray_remote_args()
    assert isinstance(merged["scheduling_strategy"], PlacementGroupSchedulingStrategy)

    # (b) fn overrides the resource footprint -> rejected (takeover).
    takeover = _make_actor_op(
        {"num_gpus": 0},
        ray_remote_args_fn=lambda: {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=object()
            ),
            "num_gpus": 4,
        },
    )
    with pytest.raises(NotImplementedError, match="num_gpus"):
        takeover._merge_ray_remote_args()

    # (c) fn assigns a non-PG strategy -> rejected.
    non_pg = _make_actor_op(
        {"num_gpus": 0}, ray_remote_args_fn=lambda: {"scheduling_strategy": "SPREAD"}
    )
    with pytest.raises(NotImplementedError, match="placement group"):
        non_pg._merge_ray_remote_args()

    # (d) sizer OFF -> no enforcement; a benign non-PG fn (e.g. one tuning
    # generator backpressure, as in OSS test_map) keeps working unchecked.
    monkeypatch.setattr(apmo, "ENABLE_OPERATOR_SIZER", False)
    benign = _make_actor_op(
        {"num_cpus": 1},
        ray_remote_args_fn=lambda: {"_generator_backpressure_num_objects": 2},
    )
    merged = benign._merge_ray_remote_args()
    assert merged["_generator_backpressure_num_objects"] == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))


def test_node_active_counter_survives_actor_relocation():
    # Regression: the active-actor counter is keyed by actor_location, which
    # changes when Ray core restarts the actor on another node. Decrementing
    # the completion-time location raised KeyError (and leaked the dispatch
    # node's count). The counter must decrement the node recorded at
    # mark-active time.
    from types import SimpleNamespace

    from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
        _ExperimentalActorPool,
    )

    pool = object.__new__(_ExperimentalActorPool)
    pool._node_to_active_actor_count = {}
    state = SimpleNamespace(actor_location="nodeA", active_count_node=None)

    pool._mark_node_actor_active(state)
    assert pool._node_to_active_actor_count == {"nodeA": 1}

    # Actor restarts elsewhere while its task is still in flight.
    state.actor_location = "nodeB"

    pool._mark_node_actor_inactive(state)  # must not KeyError on nodeB
    assert pool._node_to_active_actor_count == {}
    assert state.active_count_node is None


def test_is_output_backpressured_uses_task_denomination():
    # The deadlock hatch's trigger. At the pipeline tail an op has a couple of
    # output-blocked tasks and a mostly-idle pool (hetero wedge: 2 blocked
    # tasks, 38 running actors). "All ACTIVE TASKS blocked" must read True --
    # the actor-denominated pair (2, 38) would read False and permanently
    # disable the hatch at exactly the state it exists for.
    from types import SimpleNamespace

    from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
        ExperimentalAPMO,
    )

    op = object.__new__(ExperimentalAPMO)
    op.get_actor_info = lambda: SimpleNamespace(running=38, pending=0)
    op._data_tasks = {0: object(), 1: object()}
    op.output_backpressured_fraction = lambda: (2, 2)
    op.output_backpressured_actors = lambda: (2, 38)
    assert op.is_output_backpressured() is True


def test_expired_pending_actor_is_released_and_killed(monkeypatch):
    # A pending actor whose ready-ref never resolves (e.g. an actor wedged in
    # failed restarts, ray#53727) must not hold the pool's pending count -- and
    # with it every sizer arm -- forever. After the expiry bound the pool kills
    # the handle, unwinds the bookkeeping, and frees the pending slot.

    import ray
    from ray.data._internal.execution.interfaces import ExecutionResources
    from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
        _ExperimentalActorPool,
        _PendingActorInfo,
    )

    killed = []
    monkeypatch.setattr(ray, "kill", lambda a: killed.append(a))

    class _Handle:
        pass

    pool = object.__new__(_ExperimentalActorPool)
    actor = _Handle()
    ref = object()
    usage = ExecutionResources(cpu=1, gpu=1, memory=0)
    pool._pending_actors = {ref: actor}
    pool._pending_actor_info = {
        ref: _PendingActorInfo(
            logical_id="la-1", target_node_id="nodeA", created_at=1000.0
        )
    }
    pool._actor_resource_usage = {actor: usage}
    pool._total_usage = usage
    pool._pending_or_restarting_usage = usage
    pool._actor_to_logical_id = {actor: "la-1"}
    removed_committed = []
    pool._remove_committed_usage = lambda node, u: removed_committed.append((node, u))

    # Before the bound: nothing happens.
    assert pool.release_expired_pending_actors(expiry_s=300.0, now=1200.0) == 0
    assert pool.num_pending_actors() == 1

    # Past the bound: released, killed, bookkeeping unwound.
    assert pool.release_expired_pending_actors(expiry_s=300.0, now=1301.0) == 1
    assert pool.num_pending_actors() == 0
    assert killed == [actor]
    assert removed_committed == [("nodeA", usage)]
    assert pool._total_usage.cpu == 0
    assert pool._pending_or_restarting_usage.cpu == 0
    assert actor not in pool._actor_to_logical_id
