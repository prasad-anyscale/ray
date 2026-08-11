"""Runtime flags and env-backed constants for the Ray Data execution engine.

Defined in a dedicated module to avoid circular imports between
``streaming_executor`` and operator packages (e.g. ``actor_pool_map_operator``).
"""

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from ray._common.utils import env_bool, env_float, env_integer
from ray.data._internal.experimental.execution.sizer.operator_ordering import (
    OrderingPolicy,
)


# Whether or not to enable operator fusion. Read at call time (not bound to a
# module constant at import) so tests can toggle it through the environment
# (``RAY_DATA_ENABLE_FUSION``) instead of patching a resolved symbol.
def fusion_enabled() -> bool:
    return env_bool("RAY_DATA_ENABLE_FUSION", True)


# Grouped streaming-generator yields (block + metadata) for map actors when
# core actor backpressure is enabled. Must match ``yield_block_with_stats``.
CORE_ACTOR_BACKPRESSURE_NUM_OBJECTS_PER_YIELD: int = 2

# Whether to simulate the move-semantic
RAY_DATA_MOVE_SEMANTIC: bool = env_bool("RAY_DATA_MOVE_SEMANTIC", False)

# The number of output bytes an actor can produce.
PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT: Optional[int] = env_integer(
    "RAY_DATA_PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT", None
)

# Fraction of each node's object-store memory that may be used for actor
# output buffering. Used to derive a default per-actor output-bytes
# backpressure limit when RAY_DATA_PER_ACTOR_OUTPUT_BYTES_BACKPRESSURE_LIMIT
# is unset.
OBJECT_STORE_OUTPUT_FRACTION: float = env_float(
    "RAY_DATA_OBJECT_STORE_OUTPUT_FRACTION", 0.5
)

# Fraction of a per-actor output-bytes budget used as the task's target max
# block size (0-1). Default 0.25 keeps ~4 sized blocks under the budget
# (e.g. 512MiB limit -> 128MiB blocks).
BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO: float = env_float(
    "RAY_DATA_BLOCK_SIZE_TO_OUTPUT_BYTES_RATIO", 0.25
)

# Hard [min, max] bounds on the per-actor object-store output-bytes budget
PER_ACTOR_OBJECT_STORE_MIN_BYTES: int = env_integer(
    "RAY_DATA_PER_ACTOR_OBJECT_STORE_MIN_BYTES", 16 * 1024 * 1024
)
PER_ACTOR_OBJECT_STORE_MAX_BYTES: int = env_integer(
    "RAY_DATA_PER_ACTOR_OBJECT_STORE_MAX_BYTES", 2 * 1024**3
)

# The number of output blocks an actor can produce. Note that if
# both the number of output bytes and blocks are specified, we take
# whichever hits the limit first (lowest limit wins)
# NOTE: We need to set a default to limit the # of object refs on the
# head node to avoid OOM. In pracice about 100K object refs can OOM
# We chose 20K as a reasonable default because pipelines usually don't
# have more than 5 operators. This can be made more sophisticated later.
PER_ACTOR_OUTPUT_BACKPRESSURE_LIMIT: Optional[int] = env_integer(
    "RAY_DATA_PER_ACTOR_OUTPUT_BACKPRESSURE_LIMIT", 20000
)

# The number of input tasks in flight per actor. Each task
# may accept multiple blocks, but the # of tasks cannot change.
# This is to allow for good pipelining.
PER_ACTOR_INPUT_BACKPRESSURE_LIMIT: int = env_integer(
    "RAY_DATA_PER_ACTOR_INPUT_BACKPRESSURE_LIMIT",
    2,  # roughly 2 * 128MiB = 256 MiB
)

# The number of input bytes. Each task may accept multiple blocks
# so we aggregate across all ongoing tasks.
PER_ACTOR_INPUT_BYTES_BACKPRESSURE_LIMIT: Optional[int] = env_integer(
    "RAY_DATA_PER_ACTOR_INPUT_BYTES_BACKPRESSURE_LIMIT", None
)

# If True, perform some extra checks to assert that actor-only
# backend is working.
ACTOR_ONLY_DEBUG: bool = env_bool("RAY_DATA_ACTOR_ONLY_DEBUG", False)

# If True, will update the actor's state if restarting. This is an expensive
# operation and should only be set in spot preemption, or if you expect node
# OOMs or Actor OOMs to occur.
REFRESH_ACTOR_STATE: bool = env_bool("RAY_DATA_REFRESH_ACTOR_STATE", True)

# If True, will use the new operator sizer to control the # of actors per
# operator. Otherwise we will default to the old resource allocator.
ENABLE_OPERATOR_SIZER: bool = env_bool("RAY_DATA_ENABLE_OPERATOR_SIZER", True)

# How often the operator sizer renews its resource request on the
# AutoscalingCoordinator so the allocation doesn't expire.
SIZER_ALLOCATION_RENEW_PERIOD_S: int = env_integer(
    "RAY_DATA_SIZER_ALLOCATION_RENEW_PERIOD_S", 30
)

# If true, will use ray core SPREAd strategy to place actors instead of data's locality aware strategy. This is useful for testing and debugging the sizer without the confounding factor of locality-aware placement.
SIZER_RAY_CORE_SPREAD: bool = env_bool("RAY_DATA_SIZER_RAY_CORE_SPREAD", False)

# If True, the sizer understands per-operator scheduling constraints (label
# selectors, placement groups and custom resource requirements).
SIZER_CONSTRAINT_AWARE_PLACEMENT: bool = env_bool(
    "RAY_DATA_SIZER_CONSTRAINT_AWARE_PLACEMENT", True
)

# Order in which the sizer grants (scale_how_many) and places (scale_where)
# operators; both phases share it so grants match placements. See OrderingPolicy.
# Default scarcity-first: pairs with constraint-aware placement (place the
# scarcest / most-constrained operators first).
SIZER_ORDERING_POLICY: "OrderingPolicy" = OrderingPolicy.from_env(
    os.environ.get("RAY_DATA_SIZER_ORDERING_POLICY", "scarcity_first")
)

# If True, the sizer's locality-aware placement weights an operator's upstream
# vs downstream neighbors by each edge's measured share of the op's byte flow
# (colocating near producers saves input bytes on the wire; near consumers,
# output bytes) instead of treating both sides the same. Neutral 0.5/0.5
# weights (flag off, or no bytes observed yet) reproduce the unweighted rank
# exactly.
SIZER_EDGE_WEIGHTED_COLOCATION: bool = env_bool("RAY_DATA_SIZER_EDGE_COLOCATION", True)

# Pipeline optimizer: cross-operator sizing corrections (capacity
# transfers, silent-bottleneck grants), run as its own phase before
# scale_how_many / scale_where each tick.
SIZER_ENABLE_OPTIMIZER: bool = env_bool("RAY_DATA_SIZER_ENABLE_OPTIMIZER", True)
# Cancel a transfer whose claimed pending actors have not landed by then.
SIZER_TRANSFER_TIMEOUT_S: float = float(
    os.environ.get("RAY_DATA_SIZER_TRANSFER_TIMEOUT_S", "60")
)
# Length (in samples, recorded at most ~2x/s) of the optimizer's per-op
# signal windows.
SIZER_SIGNAL_WINDOW_SAMPLES: int = int(
    os.environ.get("RAY_DATA_SIZER_SIGNAL_WINDOW_SAMPLES", "20")
)
# Corrective transfers wait until the node allocation has been unchanged
# this long: while the cluster is still resizing, growth is the
# autoscaler's move.
SIZER_CLUSTER_SETTLE_S: float = float(
    os.environ.get("RAY_DATA_SIZER_CLUSTER_SETTLE_S", "15")
)
# Consecutive sizing ticks with an unplaced ask before an op qualifies as
# a transfer recipient.
SIZER_SHORTFALL_TICKS: int = int(os.environ.get("RAY_DATA_SIZER_SHORTFALL_TICKS", "3"))

# Seconds a PENDING actor may stay not-ready before the pool releases it,
# kills the handle and frees its claim for re-placement. Covers pendings that
# can never land: actors wedged in failed restarts after losing constructor
# args (ray#53727), stuck runtime-env installs, ghosts. Generous by design --
# ordinary starts (model download included) finish well inside it, and during
# a provisioning drought an expiry cycle is no worse than the held gate.
SIZER_PENDING_ACTOR_EXPIRY_S: int = env_integer(
    "RAY_DATA_SIZER_PENDING_ACTOR_EXPIRY_S", 300
)

# Seconds a downscaled actor may stay in the draining state (no new tasks, but
# in-flight tasks still finishing or output blocks still unconsumed) before the
# pool emits a (debounced, aggregated) WARNING that it is holding node capacity.
SIZER_DRAIN_STALL_WARN_S: int = env_integer("RAY_DATA_SIZER_DRAIN_STALL_WARN_S", 120)

# How a fully-drained (or pending) actor is torn down.
#   False (default): explicit ``ray.kill`` -- immediate, but irreversible; Ray
#     core will NOT restart the actor, so its already-produced objects can no
#     longer be lineage-reconstructed.
#   True: just drop our references to the actor and let Ray core garbage-collect
#     it. A restartable actor (``max_restarts`` > 0) can then be restarted by Ray
#     core to reconstruct a lost object, preserving lineage-based fault tolerance.
# Only affects DELIBERATE teardown (drain kill / pending-victim removal); external
# deaths (OOM, preemption, lost node) are always handled by Ray core regardless.
RAY_CORE_FAULT_TOLERANCE: bool = env_bool("RAY_DATA_RAY_CORE_FAULT_TOLERANCE", True)

# When True, emit the per-tick "Before/After pulling outputs" ResourceBank
# dumps for every operator on every debug tick. When False (default), only
# emit them for actor-pool operators that are currently at full output
# capacity -- the case these dumps were added to diagnose. Off by default
# because the per-op INFO line per tick floods the log and adds non-trivial
# overhead on the executor thread.
VERBOSE_RESOURCE_BANK_LOGGING: bool = env_bool(
    "RAY_DATA_VERBOSE_RESOURCE_BANK_LOGGING", False
)

# Concurrency for read-source actors (c8). When > 1, each read actor runs this
# many read tasks concurrently (max_concurrency and max_tasks_in_flight both
# set to this), its output-bytes cap and core-actor-backpressure valve are
# scaled by it, and its memory reservation grows per slot -- which can exceed
# small nodes' schedulable memory (e.g. 8 slots ~ 20.6GiB vs an m5.2xlarge's
# 20GiB). 1 (default) keeps single-task read actors, whose in-task IO
# parallelism comes from make_async_gen() threading instead.
READ_ACTOR_CONCURRENCY: int = env_integer("RAY_DATA_READ_ACTOR_CONCURRENCY", 1)

# when enabled, sets default memory limits for operators that don't have explicit memory limits set. This is useful for
ENABLE_DEFAULT_MEMORY_LIMITS: int = env_bool(
    "RAY_DATA_ENABLE_DEFAULT_MEMORY_LIMITS", False
)

# Assumed physical memory per CPU core (bytes) for read-concurrency (c8/mtf8)
# actors. Defaults to ~4 GiB/core, the common hyperscaler ratio. The usable heap
# per core is derived from this (see _usable_memory_per_cpu) by subtracting Ray's
# object-store (~30%) and system (~10%) memory reservations; an N-way read actor
# then reserves usable x num_cpus x max_concurrency, so the sizer (and Ray Core)
# place it by memory instead of packing nodes by CPU alone and OOMing their RAM.
# An explicit user memory= always takes precedence.
ACTOR_MEMORY_PER_CPU_BYTES: int = env_integer(
    "RAY_DATA_SIZER_ACTOR_MEMORY_PER_CPU_BYTES", 4 * 1024**3
)


def _usable_memory_per_cpu() -> int:
    """Usable heap per CPU core in bytes.

    Derived statically from ``ACTOR_MEMORY_PER_CPU_BYTES`` (assumed physical memory
    per core, ~4 GiB) by subtracting Ray's object-store and system-memory
    reservation proportions (read from ``ray_constants`` so env overrides are
    honored). The remainder is the heap a task can actually use per core. This
    deliberately does not read ``ray.cluster_resources()`` -- that's empty before
    any node is up, and the static ratio is stable across the run.
    """
    # Imported lazily: this module is intentionally import-light to avoid the
    # circular imports it was split out to break.
    from ray._private import ray_constants

    usable_fraction = max(
        0.0,
        1.0
        - ray_constants.DEFAULT_OBJECT_STORE_MEMORY_PROPORTION
        - ray_constants.DEFAULT_SYSTEM_RESERVED_MEMORY_PROPORTION,
    )
    return int(ACTOR_MEMORY_PER_CPU_BYTES * usable_fraction)


def actor_only_backend_enabled() -> bool:
    """Return whether the actor-only Ray Data backend is enabled.

    Reads ``RAY_DATA_ACTOR_ONLY_BACKEND`` on every call (not cached) so tests can
    toggle the backend through the environment without clearing a cache.
    """
    return env_bool("RAY_DATA_ACTOR_ONLY_BACKEND", False)


@contextmanager
def actor_only_backend_disabled() -> Iterator[None]:
    """Scope in which the actor-only backend reads as disabled.

    For internal bookkeeping datasets (e.g. the checkpoint loader) whose plans
    use operators the actor-only executor rejects (AllToAll from
    ``repartition``/``groupby``/``sort``). Because the flag is re-read from the
    environment on every call, flipping the env var scopes every backend
    decision made while planning and executing inside the block: compute
    promotion, operator selection, and executor selection.

    NOTE: the override is process-global. A dataset planned concurrently in
    another thread during this scope would also see the backend disabled.
    Acceptable for short-lived internal datasets; the durable mechanism is
    per-dataset configuration (see ray-project/ray#54520).
    """
    prev = os.environ.get("RAY_DATA_ACTOR_ONLY_BACKEND")
    os.environ["RAY_DATA_ACTOR_ONLY_BACKEND"] = "0"
    try:
        yield
    finally:
        if prev is None:
            del os.environ["RAY_DATA_ACTOR_ONLY_BACKEND"]
        else:
            os.environ["RAY_DATA_ACTOR_ONLY_BACKEND"] = prev


def core_actor_backpressure_enabled() -> bool:
    """Return whether Ray Data should use actor-wide core generator backpressure."""

    return (
        # If True, will use `_actor_generator_backpressure_num_objects`, which
        # is a actor-level backpressure buffer. If False, will use
        # `_generator_backpressure_num_objects`, which is task-level. The
        # difference being that actors with max_concurrency > 1 will have
        # a buffer limit of max_concurrency * num_tasks_running, but
        # `_actor_generator_backpressure_num_objects` will always be fixed
        env_bool("RAY_DATA_ENABLE_CORE_ACTOR_BACKPRESSURE", True)
        and actor_only_backend_enabled()
    )
