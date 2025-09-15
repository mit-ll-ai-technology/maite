# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torchmetrics.detection import (
    CompleteIntersectionOverUnion,
    MeanAveragePrecision,
    PanopticQuality,
)

from maite._internals.interop.metrics.torchmetrics_detection import (
    TM_DETECTION_METRIC_WHITELIST,
    TMDetectionMetric,
)
from maite.protocols import ArrayLike
from maite.protocols.object_detection import DatumMetadataType


@dataclass
class ODTgt:
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike


# Test case from https://lightning.ai/docs/torchmetrics/stable/detection/complete_intersection_over_union.html
cioutest_preds = [
    ODTgt(
        torch.tensor(
            [[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]
        ).numpy(),
        torch.tensor([4, 5]).numpy(),
        torch.tensor([0.236, 0.56]).numpy(),
    )
]
cioutest_target = [
    ODTgt(
        torch.tensor([[300.00, 100.00, 315.00, 150.00]]).numpy(),
        torch.tensor([5]).numpy(),
        torch.tensor([1]).numpy(),
    )
]
cioutest_metadata: list[DatumMetadataType] = [
    {"id": i} for i, _ in enumerate(cioutest_preds)
]


def test_ciou():
    tm_metric = CompleteIntersectionOverUnion()
    wrapper = TMDetectionMetric(tm_metric, device="cpu")
    wrapper.reset()
    wrapper.update(cioutest_preds, cioutest_target, cioutest_metadata)
    out = wrapper.compute()
    assert torch.isclose(out["ciou"], torch.tensor(0.861140727))


def test_output_transform():
    tm_metric = CompleteIntersectionOverUnion()
    wrapper = TMDetectionMetric(
        tm_metric,
        device="cpu",
        output_transform=lambda res: f"complete_iou={res['ciou']:0.3}",
    )
    wrapper.reset()
    wrapper.update(cioutest_preds, cioutest_target, cioutest_metadata)
    out = wrapper.compute()
    assert out == {"CompleteIntersectionOverUnion": "complete_iou=0.861"}


def test_output_key():
    tm_metric = CompleteIntersectionOverUnion()
    wrapper = TMDetectionMetric(tm_metric, device="cpu", output_key="output_key")
    wrapper.reset()
    wrapper.update(cioutest_preds, cioutest_target, cioutest_metadata)
    out = wrapper.compute()
    assert torch.isclose(out["output_key"]["ciou"], torch.tensor(0.861140727))


@pytest.mark.parametrize("metric_name", list(TM_DETECTION_METRIC_WHITELIST.keys()))
def test_wrapper(metric_name):
    tm_metric = TM_DETECTION_METRIC_WHITELIST[metric_name]()

    wrapper = TMDetectionMetric(tm_metric, device="cpu")
    wrapper.update(cioutest_preds, cioutest_target, cioutest_metadata)
    wrapper.compute()


def test_invalid_metric():
    tm_metric = PanopticQuality(things={0, 1}, stuffs={6, 7})
    with pytest.raises(ValueError):
        TMDetectionMetric(tm_metric)


def test_invalid_map_metric():
    with pytest.raises(ValueError):
        tm_metric = MeanAveragePrecision(iou_type="segm")
        TMDetectionMetric(tm_metric)
