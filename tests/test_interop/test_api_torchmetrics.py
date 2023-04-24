import pytest

import jatic_toolbox

METRICS = jatic_toolbox.list_metrics(provider="torchmetrics")


@pytest.mark.parametrize("metric", METRICS + ["MeanAveragePrecision"])
def test_api_torchmetrics_load_metric(metric):
    m = jatic_toolbox.load_metric(
        provider="torchmetrics", metric_name=metric, task="multiclass", num_classes=10
    )
    assert m is not None
    assert hasattr(m, "update")
    assert hasattr(m, "compute")
