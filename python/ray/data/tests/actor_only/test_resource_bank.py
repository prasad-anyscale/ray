from dataclasses import dataclass
from typing import Dict, Tuple

import pytest

from ray.data._internal.execution.resource_bank import (
    LiveObjStore,
    ResourceBank,
)
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (
    ActorStatus,
    ExperimentalAPMO,
)
from ray.data.block import BlockExecStats, BlockMetadata

NODE = "node-A"
NODE_B = "node-B"
ACTOR = "actor-1"
TASK = 0


@dataclass(frozen=True)
class _FakeRef:
    key: str

    def hex(self) -> str:
        return self.key


@dataclass(frozen=True)
class _FakeActorState:
    status: ActorStatus


class _FakeActorPool:
    """Stand-in for an operator's actor pool."""

    def __init__(
        self,
        *,
        max_concurrency: int = 1,
        active_actors_by_node: Dict[str, int] | None = None,
    ):
        self._max_concurrency = max_concurrency
        # Shaped like the real pool's index so ExperimentalAPMO's own
        # ``actor_counts_by_node`` runs against the fake unmodified -- the
        # (op, node) scope's prebuffer estimate reads it.
        self._node_to_actor_states: Dict[str, Dict[str, _FakeActorState]] = {
            node_id: {
                f"{node_id}-active-{i}": _FakeActorState(ActorStatus.ACTIVE)
                for i in range(count)
            }
            for node_id, count in (active_actors_by_node or {}).items()
        }

    def max_actor_concurrency(self) -> int:
        return self._max_concurrency

    def num_active_actors(self) -> int:
        return sum(len(states) for states in self._node_to_actor_states.values())

    def pending_ids_by_target_node(self) -> Dict[str, list]:
        return {}

    def average_bytes_per_output(self) -> float | None:
        # No per-actor output-bytes caps in the fake; ResourceBank falls back
        # to the 128 MiB default when cumulative BPO is also unset.
        return None


class _FakeActorOp(ExperimentalAPMO):
    """Minimal ExperimentalAPMO stand-in for ResourceBank accounting tests."""

    def __init__(
        self,
        *,
        prebuffered_outputs: int = 0,
        max_concurrency: int = 1,
        active_actors_by_node: Dict[str, int] | None = None,
    ):
        self._ray_remote_args = {
            "_actor_generator_backpressure_num_objects": prebuffered_outputs * 2,
        }
        self._ray_actor_task_remote_args = {}
        self._actor_pool = _FakeActorPool(
            max_concurrency=max_concurrency,
            active_actors_by_node=active_actors_by_node,
        )

    @property
    def actor_pool(self) -> _FakeActorPool:
        # Bypass ExperimentalAPMO.actor_pool's _ExperimentalActorPool assert.
        return self._actor_pool


def _make_op(
    prebuffered_outputs: int = 0,
    *,
    num_active_actors: int = 0,
    active_actors_by_node: Dict[str, int] | None = None,
) -> _FakeActorOp:
    """Build a fake actor operator.

    Args:
        prebuffered_outputs: Generator backpressure cap, in outputs per actor.
        num_active_actors: Actors the pool reports as active, all placed on
            ``NODE``. Set it to the number of actors the test has left running
            tasks (only relevant when ``prebuffered_outputs > 0``).
        active_actors_by_node: Per-node active actor counts, for tests that
            need the op spread across nodes. Overrides ``num_active_actors``.

    Returns:
        The fake operator.
    """
    if active_actors_by_node is None:
        active_actors_by_node = {NODE: num_active_actors} if num_active_actors else {}
    return _FakeActorOp(
        prebuffered_outputs=prebuffered_outputs,
        active_actors_by_node=active_actors_by_node,
    )


def _bm(
    task_idx: int = 0,
    size_bytes: int = 100,
    num_rows: int = 10,
    node: str = NODE,
) -> BlockMetadata:
    return BlockMetadata(
        num_rows=num_rows,
        size_bytes=size_bytes,
        exec_stats=BlockExecStats(node_id=node, task_idx=task_idx),
        input_files=None,
    )


def _new_output(
    bank: ResourceBank,
    op: ExperimentalAPMO,
    ref_key: str,
    *,
    task_id: int | None = None,
    node: str = NODE,
    size_bytes: int = 100,
) -> _FakeRef:
    ref = _FakeRef(ref_key)
    bank.on_new_output(
        op=op,
        ref=ref,
        bm=_bm(task_idx=task_id or 0, node=node, size_bytes=size_bytes),
        task_id=task_id,
    )
    return ref


def _consume(bank: ResourceBank, *refs: _FakeRef) -> None:
    for ref in refs:
        bank.on_block_consumed(input_ref=ref)
    bank.drain_consumed_blocks()


def _total_blocks_bytes(usage: LiveObjStore) -> Tuple[int, int]:
    blocks = (
        usage.num_pulled_output_blocks
        + usage.num_prebuffered_output_blocks
        + usage.num_input_blocks
        + usage.num_dangling_output_blocks
    )
    bytes_ = (
        usage.num_pulled_output_bytes
        + usage.num_prebuffered_output_bytes
        + usage.num_input_bytes
        + usage.num_dangling_output_bytes
    )
    return blocks, bytes_


def _assert_empty(usage: LiveObjStore) -> None:
    assert usage.num_pulled_output_blocks == 0
    assert usage.num_pulled_output_bytes == 0
    assert usage.num_prebuffered_output_blocks == 0
    assert usage.num_prebuffered_output_bytes == 0
    assert usage.num_input_blocks == 0
    assert usage.num_input_bytes == 0
    assert usage.num_dangling_output_blocks == 0
    assert usage.num_dangling_output_bytes == 0


def test_source_output_created_consumed_and_duplicate_consumption_is_idempotent():
    bank = ResourceBank()
    source_op = _make_op()

    ref = _new_output(bank, source_op, "source-0")

    assert _total_blocks_bytes(bank.live_object_store(op=source_op)) == (1, 100)
    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE)) == (1, 100)
    assert _total_blocks_bytes(bank.live_object_store()) == (1, 100)

    _consume(bank, ref)

    _assert_empty(bank.live_object_store(op=source_op))
    _assert_empty(bank.live_object_store(node_id=NODE))
    _assert_empty(bank.live_object_store())

    # A second consumer report for the same ref should be ignored, not underflow.
    _consume(bank, ref)
    _assert_empty(bank.live_object_store())


@pytest.mark.parametrize("num_outputs", [2, 4, 8])
def test_actor_multi_output_task_drains_after_completion(num_outputs: int):
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK,
        input_bms=[_bm(task_idx=99, size_bytes=50)],
    )
    refs = [
        _new_output(bank, actor_op, f"actor-output-{i}", task_id=TASK)
        for i in range(num_outputs)
    ]

    actor_usage = bank.live_object_store(actor_id=ACTOR)
    assert actor_usage.num_input_blocks == 1
    assert actor_usage.num_input_bytes == 50
    assert actor_usage.num_pulled_output_blocks == num_outputs
    assert actor_usage.num_pulled_output_bytes == num_outputs * 100

    bank.on_task_completed(op=actor_op, task_id=TASK)
    assert bank.live_object_store(actor_id=ACTOR).num_input_blocks == 0

    for remaining, ref in reversed(list(enumerate(refs))):
        _consume(bank, ref)
        assert (
            bank.live_object_store(actor_id=ACTOR).num_pulled_output_blocks == remaining
        )

    _assert_empty(bank.live_object_store(actor_id=ACTOR))
    _assert_empty(bank.live_object_store(op=actor_op))
    _assert_empty(bank.live_object_store(node_id=NODE))
    _assert_empty(bank.live_object_store())


def test_cross_node_input_counts_secondary_copy_until_task_completion():
    bank = ResourceBank()
    source_op = _make_op()
    actor_op = _make_op()

    source_ref = _new_output(
        bank, source_op, "source-remote", node=NODE, size_bytes=100
    )
    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE_B)
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK,
        input_bms=[_bm(task_idx=0, node=NODE, size_bytes=100)],
    )

    assert _total_blocks_bytes(bank.live_object_store(op=source_op)) == (1, 100)
    assert bank.live_object_store(op=actor_op).num_input_blocks == 1
    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE)) == (1, 100)
    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE_B)) == (1, 100)
    assert _total_blocks_bytes(bank.live_object_store()) == (2, 200)

    cumulative = bank.cumulative_object_store(op=actor_op)
    assert cumulative.input_bytes_remote.num_samples == 1
    assert cumulative.input_bytes_remote.mean == 100
    assert cumulative.input_bytes_local.num_samples == 0

    bank.on_task_completed(op=actor_op, task_id=TASK)
    assert _total_blocks_bytes(bank.live_object_store()) == (1, 100)
    _consume(bank, source_ref)
    _assert_empty(bank.live_object_store())


def test_local_input_transfers_primary_copy_without_double_count_after_drain():
    bank = ResourceBank()
    source_op = _make_op()
    actor_op = _make_op()

    source_ref = _new_output(bank, source_op, "source-local", node=NODE)
    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)

    # This mirrors _ExperimentalActorPool.on_task_submitted: local input refs
    # are withdrawn from the producer when ownership moves to the actor.
    bank.on_block_consumed(input_ref=source_ref)
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK,
        input_bms=[_bm(task_idx=0, node=NODE)],
    )
    bank.drain_consumed_blocks()

    assert _total_blocks_bytes(bank.live_object_store(op=source_op)) == (0, 0)
    assert bank.live_object_store(op=actor_op).num_input_blocks == 1
    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE)) == (1, 100)
    assert _total_blocks_bytes(bank.live_object_store()) == (1, 100)

    cumulative = bank.cumulative_object_store(op=actor_op)
    assert cumulative.input_bytes_local.num_samples == 1
    assert cumulative.input_bytes_local.mean == 100
    assert cumulative.input_bytes_remote.num_samples == 0

    bank.on_task_completed(op=actor_op, task_id=TASK)
    _assert_empty(bank.live_object_store())


def test_prebuffered_pulled_and_actorless_usage_are_correct_by_scope():
    bank = ResourceBank()
    # One actor stays busy (a single in-flight task), so its pool reports one
    # active actor for the op-scope prebuffer estimate.
    actor_op = _make_op(prebuffered_outputs=2, num_active_actors=1)
    source_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(actor_id=ACTOR, task_id=TASK, input_bms=[])
    _new_output(bank, actor_op, "actor-0", task_id=TASK, node=NODE)
    _new_output(bank, actor_op, "actor-1", task_id=TASK, node=NODE)
    _new_output(bank, source_op, "source-a", node=NODE, size_bytes=50)
    _new_output(bank, source_op, "source-b", node=NODE_B, size_bytes=70)

    actor_usage = bank.live_object_store(actor_id=ACTOR)
    assert actor_usage.num_pulled_output_blocks == 2
    assert actor_usage.num_pulled_output_bytes == 200
    assert actor_usage.num_prebuffered_output_blocks == 2
    assert actor_usage.num_prebuffered_output_bytes == 200

    assert _total_blocks_bytes(bank.live_object_store(op=actor_op)) == (4, 400)
    assert _total_blocks_bytes(bank.live_object_store(op=source_op)) == (2, 120)
    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE)) == (5, 450)
    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE_B)) == (1, 70)
    assert _total_blocks_bytes(bank.live_object_store(op=actor_op, node_id=NODE)) == (
        4,
        400,
    )
    assert _total_blocks_bytes(bank.live_object_store(op=source_op, node_id=NODE)) == (
        1,
        50,
    )
    assert _total_blocks_bytes(
        bank.live_object_store(op=source_op, node_id=NODE_B)
    ) == (1, 70)
    assert _total_blocks_bytes(bank.live_object_store()) == (6, 520)


def test_downscaled_actor_reports_dangling_until_outputs_are_consumed():
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(actor_id=ACTOR, task_id=TASK, input_bms=[])
    refs = [
        _new_output(bank, actor_op, f"downscaled-{i}", task_id=TASK) for i in range(2)
    ]
    bank.on_task_completed(op=actor_op, task_id=TASK)

    bank.deregister_actor(actor_id=ACTOR)

    actor_usage = bank.live_object_store(actor_id=ACTOR)
    assert actor_usage.num_pulled_output_blocks == 0
    assert actor_usage.num_dangling_output_blocks == 2
    assert actor_usage.num_dangling_output_bytes == 200
    assert _total_blocks_bytes(bank.live_object_store(op=actor_op)) == (2, 200)

    _consume(bank, *refs)

    _assert_empty(bank.live_object_store(actor_id=ACTOR))
    _assert_empty(bank.live_object_store(op=actor_op))
    _assert_empty(bank.live_object_store(node_id=NODE))


def test_actor_can_downscale_after_outputs_are_consumed():
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(actor_id=ACTOR, task_id=TASK, input_bms=[])
    ref = _new_output(bank, actor_op, "completed-before-downscale", task_id=TASK)
    bank.on_task_completed(op=actor_op, task_id=TASK)
    _consume(bank, ref)

    bank.deregister_actor(actor_id=ACTOR)

    _assert_empty(bank.live_object_store(actor_id=ACTOR))
    _assert_empty(bank.live_object_store(op=actor_op))
    _assert_empty(bank.live_object_store(node_id=NODE))


def test_finalize_dataset_clears_all_outputs_inputs_and_refs():
    bank = ResourceBank()
    op1 = _make_op()
    op2 = _make_op()

    bank.maybe_register_actor(op=op1, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK,
        input_bms=[_bm(task_idx=0, node=NODE, size_bytes=50)],
    )
    _new_output(bank, op1, "op1-output", task_id=TASK, node=NODE, size_bytes=100)
    op2_ref = _new_output(bank, op2, "op2-output", node=NODE, size_bytes=70)

    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE)) == (3, 220)

    bank.finalize_dataset(ops={op1})

    _assert_empty(bank.live_object_store(op=op1))
    _assert_empty(bank.live_object_store(op=op2))
    _assert_empty(bank.live_object_store(actor_id=ACTOR))
    _assert_empty(bank.live_object_store(node_id=NODE))
    _assert_empty(bank.live_object_store())

    # Late duplicate consumption reports after finalization should be ignored.
    _consume(bank, op2_ref)
    _assert_empty(bank.live_object_store())


def test_resource_bank_instances_are_isolated():
    bank1 = ResourceBank()
    bank2 = ResourceBank()
    op1 = _make_op()
    op2 = _make_op()

    _new_output(bank1, op1, "bank1-output", node=NODE, size_bytes=100)
    _new_output(bank2, op2, "bank2-output", node=NODE_B, size_bytes=70)

    assert _total_blocks_bytes(bank1.live_object_store()) == (1, 100)
    assert _total_blocks_bytes(bank2.live_object_store()) == (1, 70)

    bank1.finalize_dataset(ops={op1})

    _assert_empty(bank1.live_object_store())
    assert _total_blocks_bytes(bank2.live_object_store()) == (1, 70)


def test_global_cumulative_object_store_merges_operator_trackers():
    bank = ResourceBank()
    op1 = _make_op()
    op2 = _make_op()

    _new_output(bank, op1, "op1-output", size_bytes=100)
    _new_output(bank, op2, "op2-output", size_bytes=300)

    cumulative = bank.cumulative_object_store()
    assert cumulative.output_bytes.num_samples == 2
    assert cumulative.output_bytes.mean == 200
    assert cumulative.output_bytes.min == 100
    assert cumulative.output_bytes.max == 300


def test_late_task_completion_and_actor_lookup_after_finalize_are_noops():
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(actor_id=ACTOR, task_id=TASK, input_bms=[])
    bank.finalize_dataset(ops={actor_op})

    assert bank.get_actor_id_from_task_id(op=actor_op, task_id=TASK) is None
    bank.on_task_completed(op=actor_op, task_id=TASK)

    _assert_empty(bank.live_object_store(op=actor_op))
    _assert_empty(bank.live_object_store(node_id=NODE))
    _assert_empty(bank.live_object_store())


def test_actor_relocation_migrates_node_bookkeeping():
    """When an actor relocates (only possible on node death), its accounting
    must follow it to the new node. Before the per-node migration fix, a task
    submitted after relocation raised KeyError because the actor was never
    registered on the new node's registry, and its usage was still pinned to
    the dead node. This asserts the observable outcome via the public API."""
    bank = ResourceBank()
    actor_op = _make_op()

    # 1) First sighting on NODE, with an in-flight task and a live output.
    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK,
        input_bms=[_bm(task_idx=0, node=NODE, size_bytes=50)],
    )
    _new_output(bank, actor_op, "pre-reloc-output", task_id=TASK, node=NODE)
    assert bank.live_object_store(actor_id=ACTOR).num_input_blocks == 1

    # 2) Node died -> the actor is re-sighted on NODE_B. Core replays the
    # in-flight task elsewhere, so the per-actor live store resets to empty.
    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE_B)
    _assert_empty(bank.live_object_store(actor_id=ACTOR))

    # 3) Regression: a task submitted on the relocated actor must not raise, and
    # its usage must land on NODE_B (the new node's registry now knows it).
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK + 1,
        input_bms=[_bm(task_idx=0, node=NODE_B, size_bytes=100)],
    )
    _new_output(bank, actor_op, "post-reloc-output", task_id=TASK + 1, node=NODE_B)

    assert bank.live_object_store(actor_id=ACTOR).num_input_blocks == 1
    assert _total_blocks_bytes(bank.live_object_store(node_id=NODE_B)) == (2, 200)


def test_straggler_task_completion_after_relocation_is_a_noop():
    """A task submitted before relocation can complete after the actor was
    re-sighted on a new node. Relocation assumes 0 tasks are running (core
    retries them), so the straggler's completion must not assert on the old
    node's registry (where the actor entry was dropped) nor drive the op
    registry's reset in-flight count negative -- while the task keeps its
    actor binding (task_pull_request budgets retried outputs through it)."""
    bank = ResourceBank()
    actor_op = _make_op()

    # 1) First sighting on NODE, with an in-flight task.
    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK,
        input_bms=[_bm(task_idx=0, node=NODE, size_bytes=50)],
    )

    # 2) NODE died -> actor re-sighted on NODE_B. The task stays bound to the
    # actor: core retries it on the restarted actor and its outputs must keep
    # resolving to a live per-actor budget.
    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE_B)
    assert bank.get_actor_id_from_task_id(op=actor_op, task_id=TASK) == ACTOR

    # 3) The pre-relocation task's done-callback fires. Must not raise, and
    # must not drive the relocated actor's in-flight count negative.
    bank.on_task_completed(op=actor_op, task_id=TASK)

    op_stats = bank.stats.per_operator[actor_op]
    actor_info = op_stats.actors.get_actor_info(ACTOR)
    assert actor_info is not None
    assert actor_info.num_tasks_in_flight == 0

    # 4) The relocated actor keeps working normally.
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK + 1,
        input_bms=[_bm(task_idx=0, node=NODE_B, size_bytes=100)],
    )
    bank.on_task_completed(op=actor_op, task_id=TASK + 1)
    assert actor_info.num_tasks_in_flight == 0


def test_retried_task_output_before_actor_reregisters_on_new_node():
    """A restarted actor re-runs its in-flight task on a NEW node and can
    produce outputs there before any fresh submission re-registers it with the
    bank (relocation fires on submit). The output lands on the new node's
    registry, which doesn't know the actor yet -- that must not assert."""
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(
        actor_id=ACTOR,
        task_id=TASK,
        input_bms=[_bm(task_idx=0, node=NODE, size_bytes=50)],
    )

    # NODE died; the retried task produces on NODE_B while the bank still has
    # the actor registered on NODE.
    ref = _new_output(bank, actor_op, "retried-output", task_id=TASK, node=NODE_B)

    # Node-level accounting still charges NODE_B for the block.
    assert bank.live_object_store(node_id=NODE_B).num_pulled_output_blocks == 1

    bank.on_task_completed(op=actor_op, task_id=TASK)
    _consume(bank, ref)
    _assert_empty(bank.live_object_store(node_id=NODE_B))


def test_op_node_scope_is_joined_on_both_operator_and_node():
    bank = ResourceBank()
    op_a, op_b = _make_op(), _make_op()

    # op_a runs on both nodes; op_b only on NODE. Inputs are all produced on
    # NODE, so only the actors that live on NODE score a locality hit.
    bank.maybe_register_actor(op=op_a, actor_id="a-on-A", new_node_id=NODE)
    bank.maybe_register_actor(op=op_a, actor_id="a-on-B", new_node_id=NODE_B)
    bank.maybe_register_actor(op=op_b, actor_id="b-on-A", new_node_id=NODE)
    bank.on_task_submitted(
        actor_id="a-on-A", task_id=1, input_bms=[_bm(node=NODE, size_bytes=100)]
    )
    bank.on_task_submitted(
        actor_id="a-on-B", task_id=2, input_bms=[_bm(node=NODE, size_bytes=200)]
    )
    bank.on_task_submitted(
        actor_id="b-on-A", task_id=3, input_bms=[_bm(node=NODE, size_bytes=300)]
    )

    assert bank.op_node_ids() == {(op_a, NODE), (op_a, NODE_B), (op_b, NODE)}

    # Same operator, different node: one hit, one miss.
    a_on_node = bank.cumulative_object_store(op=op_a, node_id=NODE)
    assert (
        a_on_node.input_bytes_local.sum,
        a_on_node.input_bytes_local.num_samples,
    ) == (
        100,
        1,
    )
    assert a_on_node.input_bytes_remote.num_samples == 0

    a_on_node_b = bank.cumulative_object_store(op=op_a, node_id=NODE_B)
    assert (
        a_on_node_b.input_bytes_remote.sum,
        a_on_node_b.input_bytes_remote.num_samples,
    ) == (200, 1)
    assert a_on_node_b.input_bytes_local.num_samples == 0

    # Same node, different operator: not folded into op_a's scope.
    assert bank.cumulative_object_store(
        op=op_b, node_id=NODE
    ).input_bytes_local.sum == (300)

    # A pair that never existed reads as empty rather than raising.
    assert bank.cumulative_object_store(op=op_b, node_id=NODE_B).input_bytes.sum == 0


@pytest.mark.parametrize(
    "input_node", [NODE, NODE_B], ids=["inputs_on_node_a", "inputs_on_node_b"]
)
def test_op_node_scope_sums_to_the_operator_and_node_scopes(input_node):
    """Grafana derives the per-operator and per-node views by summing the
    (op, node) series over the other dimension, so the joined scope has to agree
    with both scopes it is derived from."""
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id="a-on-A", new_node_id=NODE)
    bank.maybe_register_actor(op=actor_op, actor_id="a-on-B", new_node_id=NODE_B)
    bank.on_task_submitted(
        actor_id="a-on-A", task_id=1, input_bms=[_bm(node=input_node, size_bytes=100)]
    )
    bank.on_task_submitted(
        actor_id="a-on-B", task_id=2, input_bms=[_bm(node=input_node, size_bytes=100)]
    )
    _new_output(bank, actor_op, "out-A", task_id=1, node=NODE, size_bytes=70)
    _new_output(bank, actor_op, "out-B", task_id=2, node=NODE_B, size_bytes=30)

    joined = [
        bank.cumulative_object_store(op=op, node_id=node)
        for op, node in bank.op_node_ids()
        if op is actor_op
    ]

    # Sum over nodes == the operator scope, for the locality split and outputs.
    op_scope = bank.cumulative_object_store(op=actor_op)
    for attr in ("input_bytes_local", "input_bytes_remote", "output_bytes"):
        assert sum(getattr(c, attr).sum for c in joined) == getattr(op_scope, attr).sum
        assert (
            sum(getattr(c, attr).num_samples for c in joined)
            == getattr(op_scope, attr).num_samples
        )

    # Sum over operators == the node scope (this dataset has one actor op).
    for node in (NODE, NODE_B):
        node_scope = bank.cumulative_object_store(node_id=node)
        op_on_node = bank.cumulative_object_store(op=actor_op, node_id=node)
        assert op_on_node.output_bytes.sum == node_scope.output_bytes.sum
        assert op_on_node.input_bytes_local.sum == node_scope.input_bytes_local.sum

    # Exactly one of the two actors is co-located with the input, whichever node
    # the input came from -- so the interesting part is WHICH scope took the hit.
    # A per-operator or per-node rollup alone cannot distinguish these.
    other_node = NODE_B if input_node is NODE else NODE
    colocated = bank.cumulative_object_store(op=actor_op, node_id=input_node)
    assert (
        colocated.input_bytes_local.num_samples,
        colocated.input_bytes_local.sum,
    ) == (
        1,
        100,
    )
    assert colocated.input_bytes_remote.num_samples == 0

    fetched_remotely = bank.cumulative_object_store(op=actor_op, node_id=other_node)
    assert (
        fetched_remotely.input_bytes_remote.num_samples,
        fetched_remotely.input_bytes_remote.sum,
    ) == (1, 100)
    assert fetched_remotely.input_bytes_local.num_samples == 0


def test_op_node_scope_outlives_the_actors_that_created_it():
    """The exported series has to stay monotonic for windowed Grafana queries:
    an operator's last actor leaving a node must not zero out its history."""
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(
        actor_id=ACTOR, task_id=TASK, input_bms=[_bm(node=NODE, size_bytes=100)]
    )
    bank.on_task_completed(op=actor_op, task_id=TASK)
    bank.deregister_actor(actor_id=ACTOR)

    assert (actor_op, NODE) in bank.op_node_ids()
    cumulative = bank.cumulative_object_store(op=actor_op, node_id=NODE)
    assert cumulative.input_bytes_local.sum == 100
    # The actor row is gone, so no live usage lingers on the joined scope.
    _assert_empty(bank.live_object_store(op=actor_op, node_id=NODE))


def test_op_node_scope_migrates_the_actor_row_but_not_its_history():
    """Relocation moves the actor to the new (op, node) scope. The blocks it
    already produced do not move with it, so the old scope keeps both its
    cumulative history and the live outputs still sitting on that node."""
    bank = ResourceBank()
    actor_op = _make_op()

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE)
    bank.on_task_submitted(
        actor_id=ACTOR, task_id=TASK, input_bms=[_bm(node=NODE, size_bytes=50)]
    )
    ref = _new_output(bank, actor_op, "pre-reloc", task_id=TASK, node=NODE)

    bank.maybe_register_actor(op=actor_op, actor_id=ACTOR, new_node_id=NODE_B)
    bank.on_task_submitted(
        actor_id=ACTOR, task_id=TASK + 1, input_bms=[_bm(node=NODE_B, size_bytes=80)]
    )

    old_scope = bank.cumulative_object_store(op=actor_op, node_id=NODE)
    new_scope = bank.cumulative_object_store(op=actor_op, node_id=NODE_B)
    # History stays where it was recorded...
    assert old_scope.input_bytes.sum == 50
    assert old_scope.output_bytes.sum == 100
    # ...and the relocated actor's new work lands on the new scope.
    assert new_scope.input_bytes.sum == 80
    assert new_scope.output_bytes.num_samples == 0

    # The output produced before relocation is still live on the old node.
    assert bank.live_object_store(
        op=actor_op, node_id=NODE
    ).num_pulled_output_bytes == (100)
    _consume(bank, ref)
    assert (
        bank.live_object_store(op=actor_op, node_id=NODE).num_pulled_output_bytes == 0
    )


def test_live_object_store_total_bytes_sums_all_live_components():
    object_store = LiveObjStore(
        num_input_bytes=100,
        num_pulled_output_bytes=200,
        num_prebuffered_output_bytes=300,
        num_dangling_output_bytes=400,
    )
    assert object_store.total_bytes() == 1000


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
