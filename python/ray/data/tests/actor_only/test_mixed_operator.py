import sys
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray import cloudpickle

if TYPE_CHECKING:
    from ray._private.worker import BaseContext

N_ROWS = 100

Batch = Dict[str, Any]


class Producer:
    """Stamp a fixed data payload onto each row."""

    def __init__(self) -> None:
        self.data = np.zeros((4, 4), dtype=np.int8)

    def __call__(self, batch: Batch) -> Batch:
        n = len(batch["id"])
        return {"id": batch["id"], "data": [self.data] * n}


class Passthrough:
    def __call__(self, batch: Batch) -> Batch:
        return batch


# Pytest imports this module under a name the Ray workers can't import (the
# test dir is not an importable top-level package from a worker's cwd), so the
# UDF classes would fail to deserialize by reference. Force them to pickle by
# value. See ray/tests/rdt/test_rdt_custom.py for the same workaround.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def test_union_then_map(ray_start_10_cpus_shared: "BaseContext", tmp_path) -> None:
    """union(ds1, ds2) -> map_batches must yield every row from both inputs.

    ``ds1`` is an actor-pool ``map_batches`` (actor tasks); ``ds2`` is a
    ``read_parquet`` over a fixture file (actorless read tasks). The
    downstream map must drain both upstreams correctly.
    """
    # Drop a single-column Parquet file under ``tmp_path`` and let
    # ``read_parquet`` discover it as a directory dataset. ``override_num_blocks``
    # asks the read planner to split the file into multiple read tasks so we
    # exercise the actorless-task path with >1 task in flight.
    ds2_dir = tmp_path / "ds2"
    ds2_dir.mkdir()
    pq.write_table(
        pa.table({"id": list(range(N_ROWS))}),
        str(ds2_dir / "part-0.parquet"),
    )

    ds1 = ray.data.range(N_ROWS, override_num_blocks=N_ROWS).map_batches(Producer)
    ds2 = ray.data.read_parquet(str(ds2_dir), override_num_blocks=N_ROWS)

    unioned = ds2.union(ds1).map_batches(Passthrough)

    rows = unioned.take_all()
    assert len(rows) == 2 * N_ROWS, f"expected {2 * N_ROWS} rows, got {len(rows)}"
    # Both inputs are range(N_ROWS), so every id in [0, N_ROWS) appears twice.
    ids = sorted(r["id"] for r in rows)
    assert ids == sorted(list(range(N_ROWS)) * 2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
