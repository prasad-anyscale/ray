def get_task_pool_map_operator_cls():
    from ray.data._internal.execution.operators.task_pool_map_operator import (
        TaskPoolMapOperator,
    )

    return TaskPoolMapOperator


def get_actor_pool_map_operator_cls():
    from ray.data._internal.execution.execution_flags import (
        actor_only_backend_enabled,
    )

    # The actor-only backend requires every actor-pool op to be the
    # experimental operator (its executor asserts on the type), so the class is
    # selected here, at the single construction chokepoint.
    if actor_only_backend_enabled():
        from ray.data._internal.experimental.execution.operators.actor_pool_map_operator import (  # noqa: E501
            ExperimentalAPMO,
        )

        return ExperimentalAPMO

    from ray.data._internal.execution.operators.actor_pool_map_operator import (
        ActorPoolMapOperator,
    )

    return ActorPoolMapOperator
