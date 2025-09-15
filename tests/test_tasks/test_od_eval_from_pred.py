# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from maite._internals.protocols.generic import MetricComputeReturnType, MetricMetadata
from maite.protocols import object_detection as od
from maite.tasks import evaluate_from_predictions

N_CLASSES: int = 2  # Number pf possible classes that can be detected


@dataclass
class MyObjectDetectionTarget:
    """
    Implements the od.ObjectDetectionTarget protocol.
    """

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class MatchingBoxesPercentageMetric:
    """
    Implements the od.Metric prototype.

    It will match the exact location and dimensions of a predicted box vs. a target box (i.e., the truth).
    The Metric represents the percentage of correct matches being compared.
    """

    metric_label: str = "matched percentage"

    def __init__(self, metric_id: str) -> None:
        self._prediction_boxes = []
        self._target_boxes = []
        self.metadata = MetricMetadata(id=metric_id)

    def reset(self) -> None:
        self._prediction_boxes = []
        self._target_boxes = []

    def update(
        self,
        pred_batch: Sequence[od.ObjectDetectionTarget],
        target_batch: Sequence[od.ObjectDetectionTarget],
        metadata_batch: Sequence[od.DatumMetadataType],
    ) -> None:
        self._prediction_boxes.extend(pred_batch)
        self._target_boxes.extend(target_batch)

    def compute(self) -> dict[str, Any]:
        exact_matches_per_image: list[bool] = []
        for prediction_target, truth_target in zip(
            self._prediction_boxes, self._target_boxes
        ):
            for pairing_match in prediction_target.boxes == truth_target.boxes:
                exact_matches_per_image.append(True if np.all(pairing_match) else False)
        bool_array = np.array(exact_matches_per_image)
        true_count = np.count_nonzero(bool_array)
        return {f"{self.metric_label}": (true_count / len(bool_array)) * 100}


@pytest.fixture
def matching_boxes_percentage_metric() -> od.Metric:
    return MatchingBoxesPercentageMetric(metric_id="MatchingBoxesPercentageMetric")


@pytest.fixture
def mock_prediction_batches() -> Sequence[Sequence[od.TargetType]]:
    prediction_boxes: list[tuple[int, int, int, int]] = [
        (1, 1, 12, 12),
        (100, 100, 120, 120),
        (180, 180, 270, 270),
    ]
    return [_create_od_target_batch(prediction_boxes)]


@pytest.fixture
def mock_target_batches() -> Sequence[Sequence[od.TargetType]]:
    target_boxes: list[tuple[int, int, int, int]] = [
        (1, 1, 12, 12),
        (90, 90, 120, 120),
        (180, 180, 270, 270),
    ]
    return [_create_od_target_batch(target_boxes)]


@pytest.fixture
def mock_metadata_batches() -> Sequence[Sequence[od.DatumMetadataType]]:
    return [[{"id": 1}]]


def _create_od_target_batch(
    boxes: list[tuple[int, int, int, int]],
) -> Sequence[od.TargetType]:
    """
    Creates a single instance of an ObjectDetectionTarget, from the input boxes.
    Args:
        boxes (list[tuple[int, int, int, int]]): Boxes associated with a target (i.e., a single image)

    Returns:
        Sequence[od.TargetType]: A batch with a single object detection target
    """
    num_boxes = len(boxes)
    fake_labels = np.random.randint(0, N_CLASSES, num_boxes)
    fake_scores = np.zeros(num_boxes)
    batch = [
        MyObjectDetectionTarget(
            boxes=np.array(boxes), labels=fake_labels, scores=fake_scores
        )
    ]
    return batch


def test_simple_od_evaluate_from_predictions(
    matching_boxes_percentage_metric,
    mock_prediction_batches,
    mock_target_batches,
    mock_metadata_batches,
) -> None:
    metric: od.Metric = matching_boxes_percentage_metric
    predictions: Sequence[Sequence[od.TargetType]] = mock_prediction_batches
    targets: Sequence[Sequence[od.TargetType]] = mock_target_batches
    metadata_batches = mock_metadata_batches
    metric_return: MetricComputeReturnType = evaluate_from_predictions(
        metric=metric,
        pred_batches=predictions,
        target_batches=targets,
        metadata_batches=metadata_batches,
    )

    # Evaluate the results.
    print(f"type(metric_return): {type(metric_return)}, metric_return: {metric_return}")
    assert metric_return == {
        f"{MatchingBoxesPercentageMetric.metric_label}": 66.66666666666666
    }
