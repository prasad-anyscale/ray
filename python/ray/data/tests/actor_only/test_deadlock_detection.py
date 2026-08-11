import sys
import threading
from types import SimpleNamespace  # noqa: E402
from typing import TYPE_CHECKING, Any, Dict, List

import pytest
from typing_extensions import override  # noqa: E402

import ray
import ray.data._internal.experimental.execution.streaming_executor_state as sess  # noqa: E402
from ray import cloudpickle
from ray.data import ActorPoolStrategy
from ray.data._internal.execution.operators.base_physical_operator import (  # noqa: E402
    InternalQueueOperatorMixin,
)
from ray.data._internal.execution.resource_bank import ResourceBankBase  # noqa: E402
from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E402
    ExperimentalAPMO,
)

if TYPE_CHECKING:
    from ray._private.worker import BaseContext


Batch = Dict[str, Any]


class Consumer:
    def __call__(self, batch: Batch) -> Batch:
        return batch


# Pytest imports this module under a name the Ray workers can't import (the
# test dir is not an importable top-level package from a worker's cwd), so the
# UDF classes would fail to deserialize by reference. Force them to pickle by
# value. See ray/tests/rdt/test_rdt_custom.py for the same workaround.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _output_capped_map(num_rows: int) -> "ray.data.Dataset":
    """A range fed through an actor-pool map whose per-actor output is capped
    to a single block, forcing aggressive output backpressure upstream of the
    operator under test. Uses an identity UDF so row counts are preserved.

    The cap is what makes the downstream cases below prone to *deadlock* rather
    than mere throttling: the capped actor's output is only freed once a
    downstream operator launches a task that consumes it, but a downstream that
    must first buffer several rows (an equal split's balance, a large
    ``batch_size``) can't launch until it has them. Without deadlock detection
    upgrading the cap, the two wait on each other forever.
    """
    return ray.data.range(num_rows, override_num_blocks=num_rows).map_batches(
        Consumer,
        compute=ActorPoolStrategy(size=1, max_num_outputs_per_actor=1),
    )


def _drain_streaming_split(iterators: List[Any]) -> List[int]:
    """Consume every streaming-split iterator concurrently and return per-split
    row counts.

    ``streaming_split`` has an implicit barrier: all iterators must be consumed
    concurrently, otherwise the shared execution stalls -- so we drain them on
    separate threads.
    """
    lengths: List[int] = []
    lengths_lock = threading.Lock()

    def _consume(it) -> None:
        count = sum(1 for _ in it.iter_rows())
        with lengths_lock:
            lengths.append(count)

    threads = [threading.Thread(target=_consume, args=(it,)) for it in iterators]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return lengths


def test_small_output_cap_runs_e2e(ray_start_10_cpus_shared: "BaseContext") -> None:
    """A tiny ``max_num_outputs_per_actor`` must throttle, not deadlock."""
    ds = _output_capped_map(100)

    rows = ds.take_all()
    assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"


def test_map_then_min_rows_map_runs_e2e(
    ray_start_10_cpus_shared: "BaseContext",
) -> None:
    """map1 -> map2(min_rows_per_bundle).

    ``map2`` sets ``batch_size=64`` (its ``min_rows_per_bundle``), so it can't
    launch a task until 64 rows are buffered. The capped ``map1`` only produces
    one output at a time and won't produce more until a downstream task consumes
    it -- which never happens until ``map2`` launches. Detection must upgrade
    ``map1`` so ``map2`` can fill a bundle.
    """
    capped = _output_capped_map(100)
    ds = capped.map_batches(
        Consumer,
        batch_size=64,
        compute=ActorPoolStrategy(size=1),
    )

    rows = ds.take_all()
    assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"


# TODO(Justin): This deadlocks on preserve_order, because blocks
# must be reordered. During the reordering, the downstream blocks can't be consumed
# which can lead to deadlock.
@pytest.mark.parametrize("preserve_order", [False])
def test_union_output_capped_input_runs_e2e(
    preserve_order: bool,
    ray_start_10_cpus_shared: "BaseContext",
    restore_data_context,
) -> None:
    """map1 and map2 -> union, downstream of an output-capped actor pool.

    With ``preserve_order=True`` the union buffers later inputs while waiting on
    the (capped, slow) first input in round-robin order.

    Uses a 10-CPU cluster because the actor-only backend runs each read/map as
    its own actor pool: the two ``ReadRange`` branches plus ``MapBatches`` need
    three concurrent actors, which would otherwise starve on a 2-CPU cluster.
    """
    ray.data.DataContext.get_current().execution_options.preserve_order = preserve_order

    capped = _output_capped_map(100)
    other = ray.data.range(50, override_num_blocks=50)
    ds = capped.union(other)

    rows = ds.take_all()
    assert len(rows) == 150, f"expected 150 rows, got {len(rows)}"


def test_output_splitter_output_capped_input_runs_e2e(
    ray_start_10_cpus_shared: "BaseContext",
) -> None:
    """map -> OutputSplitter.

    An equal streaming split downstream of an output-capped actor pool must not
    deadlock. The splitter buffers input to balance the splits, which is the
    case where it can stall.

    Uses a 10-CPU cluster: ``ReadRange`` and ``MapBatches`` are each their own
    actor pool, plus the streaming-split coordinator actor.
    """
    capped = _output_capped_map(100)
    iterators = capped.streaming_split(2, equal=True)

    lengths = _drain_streaming_split(iterators)
    assert sorted(lengths) == [50, 50], lengths


def test_limit_then_output_splitter_runs_e2e(
    ray_start_10_cpus_shared: "BaseContext",
) -> None:
    """map -> limit -> OutputSplitter.

    The ``limit`` is a pass-through between the capped map and the splitter. It
    launches no tasks and holds no output of its own, so the splitter's row
    requirement has to propagate *through* it back to the capped map for
    detection to upgrade the right operator.
    """
    capped = _output_capped_map(100)
    limited = capped.limit(80)
    iterators = limited.streaming_split(2, equal=True)

    lengths = _drain_streaming_split(iterators)
    assert sorted(lengths) == [40, 40], lengths


def test_mix_output_capped_input_runs_e2e(
    ray_start_10_cpus_shared: "BaseContext",
) -> None:
    """Mix downstream of an output-capped actor pool must not deadlock.

    Mix pulls from whichever input is furthest behind its target ratio and
    waits rather than substituting a different input, so when that input is the
    capped one it stalls.

    Uses a 10-CPU cluster because the actor-only backend runs each read/map as
    its own actor pool (two ``ReadRange`` branches plus ``MapBatches``).
    """
    capped = _output_capped_map(100)
    other = ray.data.range(100, override_num_blocks=100)
    # dataset.mix defaults to STOP_ON_LONGEST_DROP, so all rows from both inputs
    # are emitted.
    ds = capped.mix(other, weights=[0.5, 0.5])

    rows = ds.take_all()
    assert len(rows) == 200, f"expected 200 rows, got {len(rows)}"


def test_mix_limit_and_from_blocks_runs_e2e(
    ray_start_10_cpus_shared: "BaseContext",
) -> None:
    """Map -> Limit and from_blocks -> Mix.

    One Mix input is a capped map behind a ``limit`` pass-through; the other is
    an in-memory ``from_blocks`` source. Mix must not deadlock waiting on the
    capped-and-limited branch.
    """
    import pyarrow as pa

    capped = _output_capped_map(100)
    limited = capped.limit(60)
    other = ray.data.from_blocks([pa.table({"id": list(range(40))})])
    # dataset.mix defaults to STOP_ON_LONGEST_DROP, so all rows from both inputs
    # are emitted.
    ds = limited.mix(other, weights=[0.5, 0.5])

    rows = ds.take_all()
    assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"


def test_union_then_map_then_splitter_runs_e2e(
    ray_start_10_cpus_shared: "BaseContext",
) -> None:
    """map1 -> limit and map2 -> union -> map3 -> OutputSplitter.

    The deepest topology: the splitter's requirement is reset at ``map3`` (a
    task-launcher) to ``map3``'s own bundle target, which then propagates
    through the union and the ``limit`` pass-through to the capped ``map1``, and
    independently to ``map2``. Every branch and hop must stay unblocked.
    """
    capped = _output_capped_map(100)  # map1
    limited = capped.limit(80)  # limit
    other = _output_capped_map(40)  # map2
    unioned = limited.union(other)  # union: 80 + 40 = 120 rows
    mapped = unioned.map_batches(
        Consumer, compute=ActorPoolStrategy(size=1, max_num_outputs_per_actor=1)
    )  # map3
    iterators = mapped.streaming_split(2, equal=True)  # OutputSplitter

    lengths = _drain_streaming_split(iterators)
    assert sorted(lengths) == [60, 60], lengths


def test_min_rows_map_then_splitter_runs_e2e(
    ray_start_10_cpus_shared: "BaseContext",
) -> None:
    """map1 -> map2(min_rows_per_bundle) -> OutputSplitter.

    Both a large-``batch_size`` map and an equal split sit downstream of the
    capped source. The splitter's requirement applies to ``map2`` (its direct
    producer), while ``map2``'s own bundle target propagates to the capped
    ``map1``.
    """
    capped = _output_capped_map(100)  # map1
    mapped = capped.map_batches(
        Consumer,
        batch_size=64,
        compute=ActorPoolStrategy(size=1),
    )  # map2 with min_rows_per_bundle
    iterators = mapped.streaming_split(2, equal=True)  # OutputSplitter

    lengths = _drain_streaming_split(iterators)
    assert sorted(lengths) == [50, 50], lengths


# TODO(Justin): The following below are unit tests. Lets move this out eventually
class _FakeQueueOp(InternalQueueOperatorMixin):
    """A non-APMO queueing operator (e.g. an OutputSplitter).

    A real ``InternalQueueOperatorMixin`` subclass (so the isinstance check in
    ``num_rows_required`` recognizes it) that overrides only the surface
    detection reads and skips the heavy real ``__init__``."""

    def __init__(self, name: str, inputs=(), *, rows_needed: int = 0):
        self._name = name
        self._input_deps = list(inputs)
        self._rows_needed = rows_needed

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def input_dependencies(self):
        return self._input_deps

    # Concrete stand-ins for the mixin's abstract queue accessors. detect_if_idle
    # never reads them; they exist only to make the class instantiable.
    @property
    @override
    def _input_queues(self):
        return ()

    @property
    @override
    def _output_queues(self):
        return ()

    @override
    def has_completed(self) -> bool:
        return False

    @override
    def min_num_rows_needed_to_make_progress(self) -> int:
        return self._rows_needed


class _FakeAPMO(_FakeQueueOp, ExperimentalAPMO):
    """A real ``ExperimentalAPMO`` stand-in. ``pulled_output_rows`` is what the
    resource bank reports as its held (pulled-but-unconsumed) output.

    Reuses ``_FakeQueueOp``'s shared surface (both are queueing ops) and adds the
    APMO-only methods detection calls. ``ExperimentalAPMO`` already subclasses
    ``InternalQueueOperatorMixin``, so this is a genuine instance of both."""

    def __init__(
        self,
        name: str,
        inputs=(),
        *,
        output_backpressured: bool,
        pulled_output_rows: int,
        has_budget: bool = True,
        active_tasks: int = 0,
        rows_needed: int = 0,
        upgrade_succeeds: bool = True,
    ):
        super().__init__(name, inputs, rows_needed=rows_needed)
        self.pulled_output_rows = pulled_output_rows
        self.upgrade_calls = 0
        self.obp_notifications: List[Any] = []
        self._output_backpressured = output_backpressured
        self._has_budget = has_budget
        self._active_tasks = active_tasks
        self._upgrade_succeeds = upgrade_succeeds

    @override
    def is_output_backpressured(self) -> bool:
        return self._output_backpressured

    @override
    def notify_in_task_output_backpressure(self, *, in_backpressure, policy_name):
        self.obp_notifications.append((in_backpressure, policy_name))

    @override
    def has_enough_budget_for_prebuffered_outputs(self) -> bool:
        return self._has_budget

    @override
    def upgrade_output_limit(self) -> bool:
        self.upgrade_calls += 1
        return self._upgrade_succeeds

    @override
    def get_actor_info(self):
        return SimpleNamespace(active=self._active_tasks)


class _FakeResourceBank(ResourceBankBase):
    """Reports each op's held output rows straight off the fake operator.

    ``detect_if_idle`` only reads ``live_object_store(op=...)`` from the bank, so
    that's the only method overridden; every other method inherits the base's
    ``raise NotImplementedError`` default, so an unexpected call fails loudly."""

    @override
    def live_object_store(self, op=None, node_id=None, actor_id=None):
        return SimpleNamespace(
            num_pulled_output_rows=getattr(op, "pulled_output_rows", 0)
        )


def test_no_upgrade_when_output_cap_meets_split_requirement() -> None:
    """APMO(output cap == split requirement) -> OutputSplitter(4): never upgrade.

    The map holds its full 4-row output cap and feeds an equal 4-way split whose
    requirement is exactly 4 rows. The cap already satisfies the splitter
    (``pulled == needed``), so even though the map is output-backpressured,
    detection must NOT fire ``upgrade_output_limit`` -- doing so would prematurely
    relax a cap that isn't actually wedging anything.
    """
    apmo = _FakeAPMO(
        "map",
        output_backpressured=True,
        pulled_output_rows=4,
        has_budget=True,
        active_tasks=0,
    )
    splitter = _FakeQueueOp("split", inputs=[apmo], rows_needed=4)  # num_output_splits

    upgraded = sess.detect_if_idle(
        output_operator=splitter, resource_bank=_FakeResourceBank()
    )

    assert upgraded is None
    assert apmo.upgrade_calls == 0
    # The map was still visited and evaluated (so the no-fire is a real decision,
    # not the traversal skipping it).
    assert apmo.obp_notifications == [(True, "Output")]


def test_upgrade_when_output_cap_below_split_requirement() -> None:
    """Contrast: cap below the split requirement DOES upgrade.

    Same topology, but the map only holds 3 of the 4 rows the split needs. With
    no downstream tasks to relieve it, the split is genuinely starved and
    detection must upgrade -- proving the no-fire test above isn't vacuous.
    """
    apmo = _FakeAPMO(
        "map",
        output_backpressured=True,
        pulled_output_rows=3,
        has_budget=True,
        active_tasks=0,
    )
    splitter = _FakeQueueOp("split", inputs=[apmo], rows_needed=4)

    upgraded = sess.detect_if_idle(
        output_operator=splitter, resource_bank=_FakeResourceBank()
    )

    assert upgraded is apmo
    assert apmo.upgrade_calls == 1


def test_detection_converges_and_does_not_scale_past_requirement() -> None:
    """ExperimentalAPMO -> OutputSplitter(2): detection converges, never runs away.

    Models a consumer paused for a long time on the last iteration: the map is
    output-backpressured the whole time (its outputs sit unconsumed) and the
    scheduling loop keeps ticking ``detect_if_idle``. During the stall the actor
    fills its unconsumed outputs up to its current cap, so ``pulled`` tracks the
    (upgraded) limit.

    The guarantee: once the map holds as many rows as the equal 2-way split needs
    (2), ``pulled < needed`` is false and NO further upgrade fires. The limit
    settles at 2 == ``num_output_splits`` and does not scale unboundedly, no
    matter how many detection ticks elapse during the pause. (A real 15s pause is
    just many such ticks; we loop instead -- the output limit lives on the APMO
    inside the remote SplitCoordinator, so an e2e run couldn't observe it anyway.)
    """

    class _StallingAPMO(_FakeAPMO):
        def upgrade_output_limit(self) -> bool:
            # Bump the cap by one and let the (stalled) actor fill it, so the next
            # detection tick sees the higher pulled count -- exactly the feedback
            # loop that makes detection converge.
            self.upgrade_calls += 1
            self.pulled_output_rows += 1
            return True

    apmo = _StallingAPMO(
        "map",
        output_backpressured=True,  # paused consumer -> outputs stay unconsumed
        pulled_output_rows=1,  # initial cap of 1, filled
        has_budget=True,
        active_tasks=0,
    )
    splitter = _FakeQueueOp("split", inputs=[apmo], rows_needed=2)  # equal 2-way

    # Many ticks stand in for the long pause. If detection didn't converge, this
    # would upgrade on every call and scale to ~1000.
    for _ in range(1000):
        sess.detect_if_idle(output_operator=splitter, resource_bank=_FakeResourceBank())

    assert apmo.pulled_output_rows == 2  # settled exactly at the requirement
    assert apmo.upgrade_calls == 1  # one upgrade (1 -> 2), then it stopped


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
