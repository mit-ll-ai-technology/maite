# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest
import torch
import torchmetrics
import torchmetrics.classification
from numpy import ndarray

import maite.protocols.image_classification as ic
from maite._internals.interop.metrics.torchmetrics import (
    TM_CLASSIFICATION_METRIC_WHITELIST,
    TMClassificationMetric,
    _get_valid_classification_metrics,
)
from maite.protocols import ArrayLike


def make_metadata_batch(xs: Sequence[Any]):
    r: list[ic.DatumMetadataType] = [{"id": i} for i, _ in enumerate(xs)]
    return r


def test__get_valid_classification_metrics():
    # _get_valid_classification_metrics is a developer only utility to help
    # determine which metrics should be in the whitelist.
    # just minimal testing to check that it runs.
    ms = _get_valid_classification_metrics("Multiclass")
    assert ms
    ms = _get_valid_classification_metrics("Binary")
    assert ms
    ms = _get_valid_classification_metrics("Multilabel")
    assert ms


metric_extra_params = {
    "MulticlassRecallAtFixedPrecision": {"min_precision": 0.1},
    "MulticlassFBetaScore": {"beta": 0.1},
}


def one_hot_encode_matrix(labels, num_classes):
    labels = np.array(labels)
    one_hot_encoded = np.zeros((labels.size, num_classes))
    one_hot_encoded[np.arange(labels.size), labels] = 1
    return one_hot_encoded


def list_to_one_hot_list(labels, num_classes) -> list[ndarray]:
    one_hot_list = []
    for label in labels:
        one_hot = np.zeros(num_classes, dtype=int)
        one_hot[label] = 1
        one_hot_list.append(one_hot)
    return one_hot_list


def assert_metrics_equal(
    dict1,
    dict2,
    rtol=1e-05,
    atol=1e-08,
):
    k1 = list(dict1.keys())
    k2 = list(dict2.keys())
    k1.sort()
    k2.sort()
    assert k1 == k2
    for k in k1:
        v1 = dict1[k]
        v2 = dict2[k]
        assert type(v1) is type(v2), (
            f"Values for key {k} has different types: {type(v1)} and {type(v2)}"
        )
        if isinstance(v1, torch.Tensor):
            assert torch.isclose(v1, v2, rtol=1e-05, atol=1e-08), (
                f"Values for key {k} are different: {v1} and {v2}"
            )
        else:
            assert v1 == v2, f"Values for key {k} are different: {v1} and {v2}"


def test_multiclass_accuracy():
    # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#multiclassaccuracy
    targets = np.array([2, 1, 0, 0])
    targets = list_to_one_hot_list(targets, num_classes=3)
    preds = np.array(
        [[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]]
    )

    mca = torchmetrics.classification.MulticlassAccuracy(num_classes=3)
    wrapper = TMClassificationMetric(mca)
    wrapper.reset()  # Not necessary
    wrapper.update(preds.tolist(), targets, make_metadata_batch(targets))

    assert_metrics_equal(
        wrapper.compute(),
        {"MulticlassAccuracy": torch.tensor(0.8333333)},
    )


def test_output_transform():
    targets = np.array([1])
    targets = list_to_one_hot_list(targets, num_classes=3)
    preds = np.array([[0.01, 0.98, 0.01]])

    mca = torchmetrics.classification.MulticlassAccuracy(num_classes=3)

    # Adds key, converts to float, and puts result in list
    def output_transform(metric_result_tensor):
        return {"metric_result": [float(metric_result_tensor)]}

    wrapper = TMClassificationMetric(mca, output_transform=output_transform)
    wrapper.update(preds.tolist(), targets, make_metadata_batch(targets))

    assert_metrics_equal(
        {"metric_result": [1.0]},
        wrapper.compute(),
    )


def test_output_key():
    targets = np.array([1])
    targets = list_to_one_hot_list(targets, num_classes=3)
    preds = np.array([[0.01, 0.98, 0.01]])

    mca = torchmetrics.classification.MulticlassAccuracy(num_classes=3)
    wrapper = TMClassificationMetric(mca, output_key="metric_result")
    wrapper.update(preds.tolist(), targets, make_metadata_batch(targets))

    assert_metrics_equal(
        {"metric_result": torch.tensor(1.0)},
        wrapper.compute(),
    )


def test_both_output_key_and_output_transform():
    # Adds key, converts to float, and puts result in list
    def output_transform(metric_result_tensor):
        return {"transform_key": [float(metric_result_tensor)]}

    mca = torchmetrics.classification.MulticlassAccuracy(num_classes=3)
    # Should fail since can't have both output_key and output_transform
    with pytest.raises(ValueError):
        TMClassificationMetric(
            mca, output_key="metric_result", output_transform=output_transform
        )


def do_batches(
    metric_name, batches: list[tuple[Sequence[ArrayLike], Sequence[ArrayLike]]]
):
    num_classes = 3
    extra_params = metric_extra_params.get(metric_name, {})
    metric = TM_CLASSIFICATION_METRIC_WHITELIST[metric_name](
        num_classes=num_classes, **extra_params
    )
    tm_metric = TM_CLASSIFICATION_METRIC_WHITELIST[metric_name](
        num_classes=num_classes, **extra_params
    )

    maite_metric = TMClassificationMetric(tm_metric, output_key="metric_result")

    for predicted, targets in batches:
        # List of (Cl,) to (N, Cl)
        p = torch.stack([torch.as_tensor(p) for p in predicted])
        # List of (Cl,) one hot targets to (N,)
        t = torch.stack([torch.as_tensor(t) for t in targets]).argmax(dim=1)
        metric.update(p, t)

        maite_metric.update(predicted, targets, make_metadata_batch(targets))

    metric_results = metric.compute()
    d = maite_metric.compute()
    assert isinstance(d, dict)
    assert isinstance(d["metric_result"], torch.Tensor), f"{d}"
    assert isinstance(metric_results, torch.Tensor), f"{d}"
    assert torch.isclose(metric_results, d["metric_result"], equal_nan=True).all()


@pytest.mark.parametrize("metric_name", list(TM_CLASSIFICATION_METRIC_WHITELIST.keys()))
def test_one_batch_probs(metric_name):
    predicted = [torch.tensor([0.1, 0.1, 0.8])]
    targets = [torch.tensor([0, 0, 1])]
    do_batches(metric_name, [(predicted, targets)])


@pytest.mark.parametrize("metric_name", list(TM_CLASSIFICATION_METRIC_WHITELIST.keys()))
def test_two_batch_probs(metric_name):
    batch1 = (
        [np.array([0.16, 0.26, 0.58]), np.array([0.22, 0.61, 0.17])],
        list_to_one_hot_list([2, 1], num_classes=3),
    )
    batch2 = (
        [np.array([0.71, 0.09, 0.20]), np.array([0.05, 0.82, 0.13])],
        list_to_one_hot_list([0, 0], num_classes=3),
    )
    do_batches(metric_name, [batch1, batch2])


@pytest.mark.parametrize("metric_name", list(TM_CLASSIFICATION_METRIC_WHITELIST.keys()))
def test_two_batch_logits(metric_name):
    batch1 = (
        [np.array([0.16, 1.26, 2.58]), np.array([1.22, 2.61, 0.17])],
        list_to_one_hot_list([2, 1], num_classes=3),
    )
    batch2 = (
        [np.array([10.71, 10.09, 10.20]), np.array([20.05, 20.82, 20.13])],
        list_to_one_hot_list([0, 0], num_classes=3),
    )
    do_batches(metric_name, [batch1, batch2])
