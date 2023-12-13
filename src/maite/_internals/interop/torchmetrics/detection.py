# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Dict, List, Union

import torch as tr
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision as _MeanAveragePrecision

from maite.protocols import HasDetectionPredictions


class MeanAveragePrecision(_MeanAveragePrecision):
    def update(
        self,
        preds: Union[List[Dict[str, Tensor]], HasDetectionPredictions],
        target: List[Dict[str, Tensor]],
    ) -> None:
        """Compute the mean average precision for object detection task."""

        if isinstance(preds, HasDetectionPredictions):
            if TYPE_CHECKING:
                assert preds.labels is not None
                assert isinstance(preds.labels, List)
                assert isinstance(preds.boxes, list)
                assert isinstance(preds.scores, list)

            _preds = [
                dict(
                    boxes=tr.as_tensor(preds.boxes[i], device=self.device),
                    labels=tr.as_tensor(preds.labels[i], device=self.device),
                    scores=tr.as_tensor(preds.scores[i], device=self.device),
                )
                for i in range(len(preds.boxes))
            ]

            _target = [
                dict(
                    boxes=tr.as_tensor(target[i]["boxes"], device=self.device),
                    labels=tr.as_tensor(target[i]["labels"], device=self.device),
                )
                for i in range(len(target))
            ]
            return super().update(_preds, _target)

        return super().update(preds, target)
