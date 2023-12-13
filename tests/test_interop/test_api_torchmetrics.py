# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest

import maite

METRICS = list(maite.list_metrics(provider="torchmetrics"))


@pytest.mark.parametrize("metric", METRICS + ["MeanAveragePrecision"])
def test_api_torchmetrics_load_metric(metric):
    m = maite.load_metric(
        provider="torchmetrics", metric_name=metric, task="multiclass", num_classes=10
    )
    assert m is not None
    assert hasattr(m, "update")
    assert hasattr(m, "compute")
