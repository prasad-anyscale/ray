"""Reads must attribute their driver-created listing blocks to a node.

Object-store accounting attributes every block to the node that produced it,
reading the node id off the block's execution stats. Listing blocks are built on
the driver rather than by a task, so the planner stamps them itself. When it
doesn't, the first block the resource bank sees has no stats and every read
raises ``AttributeError: 'NoneType' object has no attribute 'node_id'`` before a
single row is read.
"""

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data.context import DataContext

if TYPE_CHECKING:
    from ray._private.worker import BaseContext

N_ROWS = 64


def test_read_parquet_runs_under_actor_only_backend(
    ray_start_10_cpus_shared: "BaseContext", tmp_path
) -> None:
    pq.write_table(pa.table({"id": list(range(N_ROWS))}), str(tmp_path / "part-0.parquet"))

    ds = ray.data.read_parquet(str(tmp_path))

    assert ds.count() == N_ROWS


def test_listing_input_blocks_carry_exec_stats(
    ray_start_10_cpus_shared: "BaseContext", tmp_path
) -> None:
    import pyarrow.fs as pafs

    from ray.data._internal.datasource_v2.listing.file_indexer import (
        NonSamplingFileIndexer,
    )
    from ray.data._internal.logical.operators import ListFiles
    from ray.data._internal.planner.plan_list_files_op import _create_input_data_buffer

    paths = []
    for i in range(3):
        path = str(tmp_path / f"f{i}.parquet")
        pq.write_table(pa.table({"id": [i]}), path)
        paths.append(path)

    op = ListFiles(
        paths=paths,
        file_indexer=NonSamplingFileIndexer(ignore_missing_paths=False),
        filesystem=pafs.LocalFileSystem(),
        source_paths=paths,
        shuffle_config_factory=lambda: None,
    )
    buffer = _create_input_data_buffer(
        op, DataContext.get_current(), should_parallelize=True
    )

    bundles = buffer._input_data
    assert bundles
    for bundle in bundles:
        for entry in bundle.blocks:
            assert entry.metadata.exec_stats is not None
            assert entry.metadata.get_node_id()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
