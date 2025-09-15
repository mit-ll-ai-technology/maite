# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import maite._internals.protocols.image_classification as ic
from maite._internals.protocols.generic import MetricComputeReturnType, MetricMetadata
from maite._internals.protocols.image_classification import ArrayLike, DatumMetadataType
from maite.tasks import evaluate_from_predictions


class SimpleAccuracyMetric:
    metadata: MetricMetadata = {"id": "A simple accuracy metric"}

    def __init__(self) -> None:
        self._total = 0
        self._correct = 0
        self._datum_correct = {}

    def reset(self) -> None:
        self._total = 0
        self._correct = 0

    def update(
        self,
        pred_batch: Sequence[ArrayLike],
        target_batch: Sequence[ArrayLike],
        metadata_batch: Sequence[DatumMetadataType],
    ) -> None:
        model_probs = [np.array(r) for r in pred_batch]
        true_onehot = [np.array(r) for r in target_batch]

        # Stack into single array, convert to class indices
        model_classes = np.vstack(model_probs).argmax(axis=1)
        truth_classes = np.vstack(true_onehot).argmax(axis=1)

        # Compare classes and update running counts
        same = model_classes == truth_classes

        for correct, metadata in zip(same, metadata_batch):
            self._datum_correct[metadata["id"]] = correct

        self._total += len(same)
        self._correct += same.sum()

    def compute(self) -> dict[str, Any]:
        if self._total > 0:
            return {
                "accuracy": self._correct / self._total,
                "datum_correct": self._datum_correct,
            }
        else:
            raise Exception("No batches processed yet.")


@pytest.fixture
def simple_ic_metric() -> ic.Metric:
    return SimpleAccuracyMetric()


@pytest.fixture
def mock_prediction_batches() -> Sequence[Sequence[ic.TargetType]]:
    target_batch_01: Sequence[ic.TargetType] = [
        np.array([0.8, 0.1, 0.0, 0.1]),  # 0 correct
        np.array([0.0, 0.1, 0.8, 0.1]),  # 2 correct
    ]
    target_batch_02: Sequence[ic.TargetType] = [
        np.array([0.1, 0.2, 0.6, 0.1]),  # 2 incorrect
        np.array([0.1, 0.1, 0.7, 0.1]),  # 2 correct
    ]
    return [target_batch_01, target_batch_02]


@pytest.fixture
def mock_target_batches() -> Sequence[Sequence[ic.TargetType]]:
    target_batch_01: Sequence[ic.TargetType] = [
        np.array([1.0, 0.0, 0.0, 0.0]),  # 0
        np.array([0.0, 0.0, 1.0, 0.0]),  # 2
    ]
    target_batch_02: Sequence[ic.TargetType] = [
        np.array([0.0, 1.0, 0.0, 0.0]),  # 1
        np.array([0.0, 0.0, 1.0, 0.0]),  # 2
    ]
    return [target_batch_01, target_batch_02]


@pytest.fixture
def mock_metadata_batch() -> Sequence[Sequence[ic.DatumMetadataType]]:
    return [[{"id": 100}, {"id": 101}], [{"id": 102}, {"id": 103}]]


def test_simple_ic_evaluate_from_predictions(
    simple_ic_metric, mock_prediction_batches, mock_target_batches, mock_metadata_batch
) -> None:
    metric: ic.Metric = simple_ic_metric
    predictions: Sequence[Sequence[ic.TargetType]] = mock_prediction_batches
    targets: Sequence[Sequence[ic.TargetType]] = mock_target_batches
    metadatas: Sequence[Sequence[ic.DatumMetadataType]] = mock_metadata_batch

    metric_return: MetricComputeReturnType = evaluate_from_predictions(
        metric=metric,
        pred_batches=predictions,
        target_batches=targets,
        metadata_batches=metadatas,
    )
    print(f"type(metric_return): {type(metric_return)}, metric_return: {metric_return}")
    assert metric_return["accuracy"] == 0.75
    assert metric_return["datum_correct"] == {
        100: True,
        101: True,
        102: False,
        103: True,
    }
