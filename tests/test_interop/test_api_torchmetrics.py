# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest

import maite
from maite._internals.interop.registry import METRIC_REGISTRY

METRICS = list(maite.list_metrics(provider="torchmetrics"))
METRICS_WHITELIST = [
    "Accuracy",
    "AUROC",
    "AveragePrecision",
    "CalibrationError",
    "CohenKappa",
    "ConfusionMatrix",
    "CosineSimilarity",
    "ExactMatch",
    "F1Score",
    "FBetaScore",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "KLDivergence",
    "MatthewsCorrCoef",
    "MeanAbsoluteError",
    "mean_average_precision",
    "multiclass_accuracy",
    "Precision",
    "PrecisionRecallCurve",
    "Recall",
    "ROC",
    "Specificity",
    "StatScores",
]


@pytest.mark.parametrize("metric", METRICS_WHITELIST)
def test_api_torchmetrics_load_metric(metric):
    default_config = {
        "provider": "torchmetrics",
        "metric_name": metric,
        "task": "multiclass",
        "num_classes": 10,
    }

    if metric in METRIC_REGISTRY.keys():
        config = METRIC_REGISTRY[metric]

        if metric == "multiclass_accuracy":
            config[
                "num_classes"
            ] = 10  # num_classes is a required value, needs to be supplied by user

    elif metric in ["CosineSimilarity", "KLDivergence", "MeanAbsoluteError"]:
        config = {"provider": "torchmetrics", "metric_name": metric}

    else:
        config = default_config

    m = maite.load_metric(**config)
    assert m is not None
    assert hasattr(m, "update")
    assert hasattr(m, "compute")
