# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest

import maite

METRICS = maite.list_metrics(provider="torcheval")


@pytest.mark.parametrize("metrics_name", METRICS)
def test_api_torcheval_load_metric(metrics_name):
    import importlib

    tm_clazz = importlib.import_module("torcheval.metrics")
    clazz = getattr(tm_clazz, metrics_name)

    kwargs = {}
    if "classification" in clazz.__module__:
        kwargs = {}

    if "Multiclass" in metrics_name:
        kwargs.update({"num_classes": 10})

    if "Multilabel" in metrics_name and (
        metrics_name != "MultilabelAccuracy"
        and metrics_name != "TopKMultilabelAccuracy"
    ):
        kwargs.update({"num_labels": 10})

    if metrics_name in (
        "BinaryRecallAtFixedPrecision",
        "MultilabelRecallAtFixedPrecision",
    ):
        kwargs.update({"min_precision": 0.5})

    if metrics_name == "BLEUScore":
        kwargs.update({"n_gram": 4})

    if metrics_name == "TopKMultilabelAccuracy":
        kwargs.update({"k": 3})

    m = maite.load_metric(provider="torcheval", metric_name=metrics_name, **kwargs)
    assert m is not None
    assert hasattr(m, "update")
    assert hasattr(m, "compute")
