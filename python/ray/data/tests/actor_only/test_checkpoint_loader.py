"""Tests that the checkpoint loader's internal dataset runs under the classic
backend when the actor-only backend is enabled.

The loader pipelines (``read_parquet -> [preprocess] -> repartition(1)`` for
``id_column``; plus ``groupby``/``sort`` for ``generated_id_column``) plan
AllToAll operators, which the actor-only executor rejects. The loader is an
internal bookkeeping dataset, so it must execute on the classic backend
regardless of the global flag.
"""

import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import ray
from ray.data import DataContext
from ray.data.checkpoint import CheckpointConfig
from ray.data.checkpoint.checkpoint_filter import IdColumnCheckpointManager

_AO_ENV = "RAY_DATA_ACTOR_ONLY_BACKEND"


@pytest.fixture(scope="module")
def ray_cluster():
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


@pytest.fixture
def actor_only_env():
    prev = os.environ.get(_AO_ENV)
    os.environ[_AO_ENV] = "1"
    yield
    if prev is None:
        del os.environ[_AO_ENV]
    else:
        os.environ[_AO_ENV] = prev


def _write_id_checkpoint(checkpoint_dir, ids):
    table = pa.table({"id": pa.array(ids, type=pa.int64())})
    pq.write_table(table, os.path.join(checkpoint_dir, "checkpoint.parquet"))


def test_actor_only_backend_disabled_scopes_backend_selection(actor_only_env):
    """Inside the scope every backend decision reads classic — executor class
    and compute promotion — and the flag is restored on exit, including the
    flag-was-unset case."""
    from ray.data._internal.compute import (
        TaskPoolStrategy,
        maybe_promote_compute_strategy,
    )
    from ray.data._internal.execution.execution_flags import (
        actor_only_backend_disabled,
        actor_only_backend_enabled,
    )
    from ray.data._internal.execution.streaming_executor import (
        StreamingExecutor,
        get_streaming_executor_cls,
    )

    assert actor_only_backend_enabled()
    assert get_streaming_executor_cls() is not StreamingExecutor

    with actor_only_backend_disabled():
        assert not actor_only_backend_enabled()
        assert get_streaming_executor_cls() is StreamingExecutor
        task_pool = TaskPoolStrategy()
        assert maybe_promote_compute_strategy(task_pool) is task_pool

    assert actor_only_backend_enabled()
    assert os.environ[_AO_ENV] == "1"

    del os.environ[_AO_ENV]
    with actor_only_backend_disabled():
        assert not actor_only_backend_enabled()
    assert _AO_ENV not in os.environ
    os.environ[_AO_ENV] = "1"  # the fixture's cleanup expects it set


def test_id_column_loader_runs_under_actor_only_backend(
    ray_cluster, actor_only_env, tmp_path
):
    """load_checkpoint must succeed with the actor-only backend enabled: its
    internal dataset repartitions (an AllToAll), which the actor-only executor
    rejects, so the loader must run classic."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    ids = [3, 1, 2, 5, 4]
    _write_id_checkpoint(str(checkpoint_dir), ids)

    manager = IdColumnCheckpointManager(
        checkpoint_config=CheckpointConfig(
            id_column="id", checkpoint_path=str(checkpoint_dir)
        ),
        data_context=DataContext.get_current(),
    )
    ids_ref, size_bytes = manager.load_checkpoint()

    assert size_bytes > 0
    loaded = ray.get(ids_ref)
    assert sorted(np.asarray(loaded).tolist()) == sorted(ids)
    # The loader's backend override is scoped: the global flag is restored.
    assert os.environ[_AO_ENV] == "1"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
