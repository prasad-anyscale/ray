from ray.util.metrics import Counter


class OptimizerMetrics:
    """Counter-only telemetry for the pipeline optimizer.

    Counters, not gauges: optimizer events are short-lived and easily
    missed between scrapes; dashboards derive views with ``increase()``.
    Nothing is recorded when a rule simply has nothing to do.
    """

    actions = Counter(
        "data_pipeline_optimizer_actions",
        description=(
            "Optimizer actions by stage. capacity_transfer: fired ->"
            " landed | timeout. silent_bottleneck: granted."
        ),
        tag_keys=("dataset", "optimization", "recipient", "result"),
    )
    corrective_transfers = Counter(
        "data_pipeline_optimizer_corrective_transfers",
        description=(
            "Actors moved by corrective transfers, as donor -> recipient"
            " edges (incremented by the actor count of each transfer)."
        ),
        tag_keys=("dataset", "donor", "recipient"),
    )
    refusals = Counter(
        "data_pipeline_optimizer_refusals",
        description=(
            "Why an optimization refused to act: warmup | cluster_settling |"
            " no_donor_surplus | capacity_already_free |"
            " victims_insufficient | thrashing."
        ),
        tag_keys=("dataset", "optimization", "reason"),
    )

    def __init__(self, dataset_id: str):
        self._dataset_id = dataset_id

    def record_action(self, optimization: str, recipient: str, result: str) -> None:
        self.actions.inc(
            1,
            tags={
                "dataset": self._dataset_id,
                "optimization": optimization,
                "recipient": recipient,
                "result": result,
            },
        )

    def record_corrective_transfer(
        self, donor: str, recipient: str, num_actors: int
    ) -> None:
        self.corrective_transfers.inc(
            num_actors,
            tags={
                "dataset": self._dataset_id,
                "donor": donor,
                "recipient": recipient,
            },
        )

    def record_refusal(self, optimization: str, reason: str) -> None:
        self.refusals.inc(
            1,
            tags={
                "dataset": self._dataset_id,
                "optimization": optimization,
                "reason": reason,
            },
        )
