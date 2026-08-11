import os

import pytest

import ray
from ray._common.test_utils import run_string_as_driver
from ray.util.annotations import RayDeprecationWarning


def test_write_file_retry_on_errors_emits_deprecation_warning(caplog):
    ctx = ray.data.DataContext.get_current()
    with pytest.warns(DeprecationWarning):
        ctx.write_file_retry_on_errors = []


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("scheduling_strategy", "DEFAULT"),
        ("scheduling_strategy_large_args", "SPREAD"),
        ("large_args_threshold", 1),
    ],
)
def test_scheduling_config_emits_deprecation_warning(attr, value):
    ctx = ray.data.DataContext()
    with pytest.warns(RayDeprecationWarning, match=rf"DataContext\.{attr}"):
        setattr(ctx, attr, value)


def test_data_context_current_context_manager():
    import copy

    from ray.data.context import DataContext

    original = DataContext.get_current()
    ctx1 = copy.deepcopy(original)
    ctx1.set_config("level", "1")

    ctx2 = copy.deepcopy(original)
    ctx2.set_config("level", "2")

    with pytest.raises(ValueError):
        with DataContext.current(ctx1):
            assert DataContext.get_current() is ctx1
            # Nested context manager
            with DataContext.current(ctx2):
                assert DataContext.get_current().get_config("level") == "2"

            assert DataContext.get_current().get_config("level") == "1"

            # Test that raising will reset context too
            raise ValueError("boom")

    assert DataContext.get_current() is original


@pytest.mark.parametrize(
    "env_value,expected",
    [(None, "False"), ("0", "False"), ("1", "True")],
)
def test_use_datasource_v2_env_var(env_value, expected):
    """``DataContext.use_datasource_v2`` defaults from the
    ``RAY_DATA_USE_DATASOURCE_V2`` environment variable (default off).

    The default is read at import time, so check it in a subprocess.
    """
    script = (
        "from ray.data.context import DataContext\n"
        "print(DataContext.get_current().use_datasource_v2)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "RAY_DATA_USE_DATASOURCE_V2"}
    if env_value is not None:
        env["RAY_DATA_USE_DATASOURCE_V2"] = env_value
    assert run_string_as_driver(script, env=env).strip().splitlines()[-1] == expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
