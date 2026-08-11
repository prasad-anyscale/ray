import inspect

import pytest

from ray.data._internal.execution.execution_flags import (
    CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD,
    core_actor_backpressure_enabled,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    _configure_map_worker_for_core_backpressure,
    _MapWorker,
)
from ray.data._internal.execution.util import yield_block_with_stats
from ray.data.context import DataContext


def test_map_worker_submit_uses_grouped_yields():
    assert core_actor_backpressure_enabled()
    ctx = DataContext.get_current()
    worker_cls = _configure_map_worker_for_core_backpressure(
        type("MapWorker(test)", (_MapWorker,), {}),
        ctx,
    )
    assert hasattr(worker_cls.submit, "__ray_num_objects_per_yield__")
    assert (
        worker_cls.submit.__ray_num_objects_per_yield__
        == CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD
    )
    assert ctx._use_grouped_streaming_generator_yields


def test_yield_block_with_stats_uses_grouped_yield():
    ctx = DataContext.get_current()
    ctx._use_grouped_streaming_generator_yields = True

    def build_metadata(block_ser_time_s):
        return {"ser_time": block_ser_time_s}

    with DataContext.current(ctx):
        gen = yield_block_with_stats({"id": [1]}, build_metadata)
        assert inspect.isgenerator(gen)

        output = gen.send(None)
        assert isinstance(output, tuple)
        assert len(output) == CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD
        assert output[0] == {"id": [1]}
        assert isinstance(output[1], bytes)

        with pytest.raises(StopIteration):
            gen.send(None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
