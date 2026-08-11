from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

import ray
from ray.data._internal.execution.bundle_queue.experimental import (
    DrainingActorPriorityBundleQueue,
)
from ray.data._internal.execution.interfaces import BlockEntry, RefBundle
from ray.data.block import BlockAccessor


def _create_bundle(data: Any) -> RefBundle:
    """A RefBundle with a single row, using an artificial (not ray.put) block ref."""
    block = pd.DataFrame({"data": [data]})
    block_ref = ray.ObjectRef(uuid4().hex[:28].encode())
    metadata = BlockAccessor.for_block(block).get_metadata()
    schema = BlockAccessor.for_block(block).schema()
    return RefBundle(
        [BlockEntry(block_ref, metadata)], owns_blocks=False, schema=schema
    )


def test_fifo_within_a_lane():
    queue = DrainingActorPriorityBundleQueue()
    b1, b2 = _create_bundle("a"), _create_bundle("b")
    queue.add(b1, actor_id="x")
    queue.add(b2, actor_id="y")

    assert queue.get_next_with_actor_id() == (b1, "x")
    assert queue.get_next_with_actor_id() == (b2, "y")
    assert not queue.has_next()


def test_get_next_returns_bundle_only():
    """Base-compatible get_next() still works (returns just the bundle)."""
    queue = DrainingActorPriorityBundleQueue()
    b1 = _create_bundle("a")
    queue.add(b1, actor_id="x")
    assert queue.get_next() is b1


def test_new_bundles_from_prioritized_actor_come_first():
    queue = DrainingActorPriorityBundleQueue()
    regular = _create_bundle("regular")
    queue.add(regular, actor_id="live")
    queue.prioritize_actor("drain")
    # Added AFTER prioritize -> routed to the priority lane.
    prio = _create_bundle("prio")
    queue.add(prio, actor_id="drain")

    assert queue.get_next_with_actor_id() == (prio, "drain")
    assert queue.get_next_with_actor_id() == (regular, "live")


def test_prioritize_moves_already_queued_bundles():
    queue = DrainingActorPriorityBundleQueue()
    # Interleave two actors' bundles into the regular lane BEFORE draining.
    d1 = _create_bundle("d1")
    queue.add(d1, actor_id="drain")
    queue.add(_create_bundle("l1"), actor_id="live")
    d2 = _create_bundle("d2")
    queue.add(d2, actor_id="drain")
    queue.add(_create_bundle("l2"), actor_id="live")

    # Draining "drain" pulls its already-queued bundles ahead (relative order kept).
    queue.prioritize_actor("drain")
    assert queue.get_next_with_actor_id() == (d1, "drain")
    assert queue.get_next_with_actor_id() == (d2, "drain")
    # Then the live actor's bundles, in their original order.
    assert [queue.get_next_with_actor_id()[1] for _ in range(2)] == ["live", "live"]


def test_deprioritize_stops_prioritizing_new_bundles():
    queue = DrainingActorPriorityBundleQueue()
    queue.prioritize_actor("a")
    assert "a" in queue.priority_actors()
    queue.deprioritize_actor("a")
    assert "a" not in queue.priority_actors()

    queue.add(_create_bundle("first"), actor_id="b")
    # "a" is no longer prioritized, so its new bundle stays behind "b"'s.
    a_bundle = _create_bundle("a-late")
    queue.add(a_bundle, actor_id="a")
    assert queue.get_next_with_actor_id()[1] == "b"
    assert queue.get_next_with_actor_id() == (a_bundle, "a")


def test_metrics_track_across_add_and_pop():
    queue = DrainingActorPriorityBundleQueue()
    assert len(queue) == 0
    queue.add(_create_bundle("a"), actor_id="x")
    queue.add(_create_bundle("b"), actor_id="y")
    assert len(queue) == 2
    assert queue.num_blocks() == 2
    queue.get_next_with_actor_id()
    assert len(queue) == 1
    queue.get_next()
    assert len(queue) == 0


def test_clear_resets_lanes_and_priority():
    queue = DrainingActorPriorityBundleQueue()
    queue.prioritize_actor("x")
    queue.add(_create_bundle("a"), actor_id="x")
    queue.add(_create_bundle("b"), actor_id="y")
    queue.clear()
    assert not queue.has_next()
    assert len(queue) == 0
    assert queue.priority_actors() == frozenset()


def test_peek_does_not_consume():
    queue = DrainingActorPriorityBundleQueue()
    b1 = _create_bundle("a")
    queue.add(b1, actor_id="x")
    assert queue.peek_next() is b1
    assert len(queue) == 1  # unchanged
    assert queue.get_next() is b1


def test_pop_from_empty_raises():
    queue = DrainingActorPriorityBundleQueue()
    with pytest.raises(ValueError):
        queue.get_next_with_actor_id()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
