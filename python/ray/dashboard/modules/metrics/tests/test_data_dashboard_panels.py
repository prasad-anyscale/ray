# ABOUTME: Tests the Ray Data Grafana dashboard JSON generation for the
# ABOUTME: OperatorSizer actor-delta bar-chart panels.
import json
import sys

import pytest

from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
    generate_data_grafana_dashboard,
)


def _all_panels(dashboard):
    for panel in dashboard["panels"]:
        yield panel
        for nested in panel.get("panels", []):
            yield nested


def _panel_by_title(dashboard, title):
    panels = [p for p in _all_panels(dashboard) if p["title"] == title]
    assert len(panels) == 1, f"expected exactly one panel titled {title!r}"
    return panels[0]


@pytest.mark.parametrize(
    "title, metric, per_interval",
    [
        ("Sizer How Many Up/Down", "sizer_actor_delta_how_many", False),
        ("Sizer Where Up/Down", "sizer_actor_delta_where", False),
        (
            "Sizer How Many Up/Down (change per interval)",
            "sizer_actor_delta_how_many",
            True,
        ),
        (
            "Sizer Where Up/Down (change per interval)",
            "sizer_actor_delta_where",
            True,
        ),
    ],
)
def test_sizer_actor_delta_panels(title, metric, per_interval):
    content, _ = generate_data_grafana_dashboard()
    panel = _panel_by_title(json.loads(content), title)

    assert panel["type"] == "barchart"
    assert panel["options"]["stacking"] == "normal"
    assert panel["fieldConfig"]["defaults"]["custom"]["axisCenteredZero"] is True

    # Fixed colors keyed on the operator's position suffix pair each
    # operator's up and down series regardless of the pipeline's names.
    override_regexes = [
        override["matcher"]["options"] for override in panel["fieldConfig"]["overrides"]
    ]
    assert "/_0$/" in override_regexes
    assert "/_5$/" in override_regexes

    up, down = panel["targets"]
    assert f"ray_data_{metric}_up" in up["expr"]
    assert down["expr"].startswith("-1 * ")
    assert f"ray_data_{metric}_down" in down["expr"]
    for target in (up, down):
        assert target["legendFormat"] == "{{operator}}"
        assert target["range"] is True
        assert target["instant"] is False
        assert ("offset $__interval" in target["expr"]) is per_interval
        if per_interval:
            # The offset-subtraction form counts a series' first sample as a
            # jump from zero (unlike increase()), so scaling that finishes
            # before the first export — e.g. fixed-size pools placed at
            # bootstrap — still shows in its first bucket.
            assert "or 0 * sum(" in target["expr"]
            # Floor $__interval above the scrape interval so windows always
            # span multiple samples (Explore and zoomed-in views ignore the
            # panel's maxDataPoints).
            assert target["interval"] == "1m"

    if per_interval:
        # Coarse buckets keep per-interval bars readable, with the change
        # printed on each bar.
        assert panel["maxDataPoints"] == 30
        assert panel["options"]["showValue"] == "auto"


def test_sizer_tick_duration_panel():
    content, _ = generate_data_grafana_dashboard()
    panel = _panel_by_title(json.loads(content), "Sizer Tick Duration by Phase")

    (target,) = panel["targets"]
    assert "ray_data_sizer_tick_duration_s" in target["expr"]
    # The gauge is per-dataset, not per-operator, so it must not be filtered by
    # $Operator -- that would drop every series.
    assert "$Operator" not in target["expr"]
    assert "by (dataset, phase)" in target["expr"]
    assert target["legendFormat"] == "{{phase}}: {{dataset}}"

    # Every phase the sizer records should be named in the description, so the
    # panel explains what "optimize" covers without reading the source.
    for phase in ("optimize", "how_many", "where", "apply", "e2e"):
        assert phase in panel["description"]


def test_sizer_actor_delta_panels_point_at_the_optimizer():
    """The delta panels count only the sizer's own passes; the descriptions have
    to say so, or a pool that changed size under a transfer reads as a bug."""
    dashboard = json.loads(generate_data_grafana_dashboard()[0])
    for title in (
        "Sizer How Many Up/Down",
        "Sizer How Many Up/Down (change per interval)",
        "Sizer Where Up/Down",
        "Sizer Where Up/Down (change per interval)",
    ):
        description = _panel_by_title(dashboard, title)["description"]
        assert "Pipeline Optimizer" in description


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
