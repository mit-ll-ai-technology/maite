# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
import torchmetrics
import torchmetrics.detection

import maite.protocols.object_detection as od
from maite.protocols import ArrayLike, MetricMetadata


def _arraylike_as_tensor(
    arr: ArrayLike, device: Any | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Safe bridging of `maite.ArrayLike` to `torch.Tensor`.

    This is a naive bridging attempt. Direct bridging to `torch.Tensor` is attempted
    first. If this fails, then bridging to a `numpy.ndarray` is attempted as an
    intermediate step prior to attempting to bridge to a `torch.Tensor` again.

    Note that this approach may fail for a variety reasons, such as incompatible device
    placement of data (e.g., CPU vs. GPU), or inherent differences in the library used
    to instantiate the ArrayLike (e.g., tensorflow.Tensor vs. torch.Tensor).
    """
    try:
        return torch.as_tensor(arr, device=device, dtype=dtype)
    except Exception as e1:
        try:
            arr = np.asarray(arr)
            return torch.as_tensor(arr, device=device, dtype=dtype)
        except Exception as e2:
            raise Exception(
                (
                    f"Unable to bridge data of type {type(arr)} directly to torch.Tensor due to the following error: {e1}."
                    f"Attempt to bridge to numpy.ndarray as an intermediary also failed."
                )
            ) from e2


TM_DETECTION_METRIC_WHITELIST = {
    "CompleteIntersectionOverUnion": torchmetrics.detection.CompleteIntersectionOverUnion,
    "DistanceIntersectionOverUnion": torchmetrics.detection.DistanceIntersectionOverUnion,
    "GeneralizedIntersectionOverUnion": torchmetrics.detection.GeneralizedIntersectionOverUnion,
    "IntersectionOverUnion": torchmetrics.detection.IntersectionOverUnion,
    "MeanAveragePrecision": torchmetrics.detection.MeanAveragePrecision,
}


class TMDetectionMetric:
    """MAITE-compliant Wrapper for Torchmetrics Detection Metrics.

    `TMDetectionMetric` is a wrapper around `torchmetrics.detection` metrics adhering
    to MAITE's `maite.protocols.object_detection.Metric` protocol.

    `~torchmetrics.detection.ciou.CompleteIntersectionOverUnion`,
    `~torchmetrics.detection.diou.DistanceIntersectionOverUnion`,
    `~torchmetrics.detection.giou.GeneralizedIntersectionOverUnion`,
    `~torchmetrics.detection.iou.IntersectionOverUnion`, and
    `~torchmetrics.detection.mean_ap.MeanAveragePrecision` with `iou_type='bbox'` are supported.

    Examples
    --------

    >>> from typing import Sequence
    >>> import torch
    >>> import torchmetrics.detection
    >>> from maite.protocols import object_detection as od
    >>> from maite.interop.torchmetrics import TMDetectionMetric
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class ObjectDetectionTarget_Impl:
    ...     boxes: torch.Tensor
    ...     labels: torch.Tensor
    ...     scores: torch.Tensor
    >>> detection_metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox")
    >>> output_transform = lambda x: x["map_50"]
    >>> wrapped_detect_metric: od.Metric = TMDetectionMetric(
    ...     detection_metric, output_key="MAP", output_transform=output_transform
    ... )
    >>> preds: Sequence[od.TargetType] = [
    ...     ObjectDetectionTarget_Impl(
    ...         boxes=torch.Tensor([[258.0, 41.0, 606.0, 285.0]]),
    ...         labels=torch.Tensor([0]),
    ...         scores=torch.Tensor([0.536]),
    ...     )
    ... ]
    >>> targets: Sequence[od.TargetType] = [
    ...     ObjectDetectionTarget_Impl(
    ...         boxes=torch.Tensor([[214.0, 41.0, 562.0, 285.0]]),
    ...         labels=torch.Tensor([0]),
    ...         scores=torch.Tensor([1.0]),
    ...     )
    ... ]
    >>> metadatas: Sequence[od.DatumMetadataType] = [{"id": 1}]
    >>> wrapped_detect_metric.update(preds, targets, metadatas)
    >>> results = wrapped_detect_metric.compute()
    >>> print(results)
    {'MAP': tensor(1.)}
    """

    def __init__(
        self,
        metric: torchmetrics.Metric,
        output_key: str | None = None,
        output_transform: Callable[[dict[str, torch.Tensor]], Any] | None = None,
        device: Any | None = None,
        dtype: torch.dtype | None = None,
        metadata: MetricMetadata | None = None,
    ):
        """
        Parameters
        ----------
        metric : `torchmetrics.Metric`
            Metric being wrapped. The metric must be a `torchmetrics.detection` metric.
        output_key : str, optional
            Outermost key returned from calling `TMDetectionMetric.compute()`. If none
            is provided, and the result of `torchmetrics.Metric.compute()` or
            `output_transform` (if provided) is not a dictionary with a string as its
            key, then the value will default to the name of the `torchmetrics.Metric`.
        output_transform : Callable[dict[str, torch.Tensor]]
            Function that takes the output of `torchmetrics.Metric.compute()` as input
            and returns a modified version of it.
        device : Any, optional
            Torch device type on which a torch.Tensor is or will be allocated. If none
            is passed, then the device will be inferred by torch.
        dtype : `torch.dtype`, optional
            Torch data type. If none is passed, then data type will be inferred by
            torch.
        metadata : `MetricMetadata`
            A typed dictionary containing at least an 'id' field of type str.

        Raises
        ------
            ValueError
                If an unsupported model is given.
        """
        TMDetectionMetric._assert_valid_detection_metric(metric)
        TMDetectionMetric._assert_valid_detection_metric_parameters(metric)

        self.metric = metric
        self.output_key = output_key
        self.output_transform = output_transform
        self.device = device
        self.dtype = dtype

        if metadata is None:
            metadata = {"id": metric._get_name()}

        self.metadata = metadata

        if self.device is not None:
            self.metric.to(device)

    @staticmethod
    def _assert_valid_detection_metric(metric: torchmetrics.Metric):
        if type(metric) not in list(TM_DETECTION_METRIC_WHITELIST.values()):
            raise ValueError(
                f"Unsupported `metric` supplied: {type(metric)}. Must be one of {list(TM_DETECTION_METRIC_WHITELIST.keys())}."
            )

    @staticmethod
    def _assert_valid_detection_metric_parameters(metric: torchmetrics.Metric):
        """Check if `metric` has valid parameters.

        This ensures that a valid metric solely performs the object detection task.
        """
        if isinstance(metric, torchmetrics.detection.MeanAveragePrecision):
            if "segm" in metric.iou_type:
                raise ValueError(
                    "torchmetrics.detection.MeanAveragePrecision with 'segm' iou_type not supported."
                )

    def _format_for_tm(
        self,
        odt: od.ObjectDetectionTarget,
    ) -> dict[str, torch.Tensor]:
        """Format `odt` to expected `torchmetrics.detection` metric input type.

        Convert an `ObjectDetectionTarget` into a dictionary with "boxes", "scores", and
        "labels" as keys, with values of type `torch.Tensor`. This is the expected input
        type of the underlying `torchmetrics.detection` metric's `update()` function.

        Notes
        -----
        Label *must* be an `int` type for `torchmetrics.detection` metrics.
        """
        kwargs = {"device": self.device, "dtype": self.dtype}
        return {
            "boxes": _arraylike_as_tensor(odt.boxes, **kwargs),
            "scores": _arraylike_as_tensor(odt.scores, **kwargs),
            "labels": _arraylike_as_tensor(
                odt.labels, device=self.device, dtype=torch.int32
            ),
        }

    def reset(self) -> None:
        # doc-ignore: EX01
        """
        Reset the metric's internal state.

        This method is used to reset the metric's internal state, such as counters or
        accumulators, to their initial values.
        """
        self.metric.reset()

    def update(
        self,
        pred_batch: Sequence[od.TargetType],
        target_batch: Sequence[od.TargetType],
        metadata_batch: Sequence[od.DatumMetadataType],
    ) -> None:
        # doc-ignore: EX01
        """
        Update the internal state of the metric with new predictions and targets.

        MAITE's `object_detection.Metric` protocol requires the elements of `preds` and
        `targets` to adhere to MAITE's `object_detection.ObjectDetectionTarget` protocol.
        Each element will then formatted into the expected input type for the underlying
        `torchmetrics.detection` metric.

        Parameters
        ----------
        pred_batch : Sequence[od.TargetType]
            Batch of predicted object detection targets.
        target_batch : Sequence[od.TargetType]
            Batch of ground truth object detection targets.
        metadata_batch : Sequence[ic.DatumMetadataType]
            Batch of metadata.
        """
        preds_tm = [self._format_for_tm(arr) for arr in pred_batch]
        targets_tm = [self._format_for_tm(arr) for arr in target_batch]
        self.metric.update(preds_tm, targets_tm)

    def compute(self) -> dict[str, Any]:
        # doc-ignore: EX01
        """
        Compute the torchmetric metric result and format it.

        If `TMDetectionMetric.output_key` is provided, the result will be wrapped in a
        dictionary with that key. Otherwise, if the result is not already a dictionary,
        it will be wrapped in a dictionary with a key derived from the metric's name.

        If `TMDetectionMetric.output_transform` is provided, the result will be
        passed to the `output_transform` function for additional post-processing.

        Returns
        -------
        Dict[str, Any]
            The formatted metric result.
        """
        results = self.metric.compute()

        if self.output_transform:
            results = self.output_transform(results)

        if self.output_key:
            return {self.output_key: results}

        if not isinstance(results, dict):
            # ensure dict[str, Any]
            key = self.metric._get_name()
            results = {key: results}

        return results
