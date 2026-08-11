"""Shared fixtures for the actor-only-backend test suite."""
import os

import pytest

# Set the flag at import time (i.e. before any Ray cluster starts) so it lands
# in the environment the raylet/workers inherit. A per-test ``monkeypatch.setenv``
# only reaches the driver, and the shared Ray fixtures are module-scoped -- they
# start the cluster before any function-scoped fixture runs -- so a driver-only
# setting never reaches worker processes. That matters for execution that runs
# off-driver: e.g. ``streaming_split`` runs the executor inside a remote
# ``SplitCoordinator`` actor, which would otherwise fall back to the default
# (non-actor-only) backend and silently skip what these tests mean to exercise.
os.environ["RAY_DATA_ACTOR_ONLY_BACKEND"] = "1"


@pytest.fixture(autouse=True)
def actor_only_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the actor-only backend for each test in this directory."""
    monkeypatch.setenv("RAY_DATA_ACTOR_ONLY_BACKEND", "1")


@pytest.fixture(autouse=True)
def core_actor_backpressure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the core actor bp for each test in this directory."""
    monkeypatch.setenv("RAY_DATA_ENABLE_CORE_ACTOR_BACKPRESSURE", "1")
