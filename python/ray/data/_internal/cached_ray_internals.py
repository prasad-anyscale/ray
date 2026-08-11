import time
from typing import Dict, Set, Tuple

import ray
from ray.data._internal.cache import timed_cache
from ray.data._internal.execution.interfaces import ExecutionResources, NodeIdStr
from ray.data._internal.execution.node_trackers.actor_location import (
    get_or_create_actor_location_tracker,
)

# If we submit a task immediately before the deadline,
# Ray Core might not have enough time to launch the
# task and fetch objects before the node is terminated.
# To avoid this, we stop using these inputs some time before the deadline.
DRAIN_DEADLINE_BUFFER_TIME_MS = 5000


@timed_cache(ttl=60)
def get_local_ongoing_lineage_reconstruction_tasks():
    return ray._private.internal_api.get_local_ongoing_lineage_reconstruction_tasks()


@timed_cache(ttl=1)
def get_draining_nodes() -> Dict[str, int]:
    return ray._private.state.state.get_draining_nodes()


def get_alive_nodes_uncached() -> Dict[NodeIdStr, ExecutionResources]:
    """Fresh ``ray.nodes()`` snapshot; no TTL cache.

    Prefer ``get_alive_nodes`` for hot paths. Use this when a miss against a
    stale/incomplete view must see nodes that just joined (e.g. actor landed
    before the coordinator allocation cache warmed).
    """
    node_resources: Dict[NodeIdStr, ExecutionResources] = {}
    for node_info in ray.nodes():
        if not node_info["alive"]:
            continue
        node_id = node_info["NodeID"]
        raw = node_info["Resources"]
        resources = ExecutionResources.from_resource_dict(raw)
        node_resources[node_id] = resources
    return node_resources


@timed_cache(ttl=5)
def get_alive_nodes() -> Dict[NodeIdStr, ExecutionResources]:
    return get_alive_nodes_uncached()


@timed_cache(ttl=5)
def get_alive_node_labels() -> Dict[NodeIdStr, Dict[str, str]]:
    node_labels: Dict[NodeIdStr, Dict[str, str]] = {}
    for node_info in ray.nodes():
        if not node_info["alive"]:
            continue
        node_labels[node_info["NodeID"]] = node_info.get("Labels") or {}
    return node_labels


@timed_cache(ttl=1)
def get_actor_locations(logical_actor_ids: Tuple[str, ...]) -> Dict[str, str]:
    """Get the actor locations from logical actor ids.
    NOTE: This function is not thread-safe"""
    return ray.get(
        get_or_create_actor_location_tracker().get_actor_locations.remote(
            logical_actor_ids
        )
    )


def get_drained_nodes() -> Set[str]:
    """Returns the set of nodes that are draining and have passed its deadline."""
    now = time.time()
    return {
        node_id
        for node_id, deadline in get_draining_nodes().items()
        # deadline is in ms
        if deadline - DRAIN_DEADLINE_BUFFER_TIME_MS < now * 1000
    }
