# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, Type

import numpy as np
import torch
import torchmetrics
import torchmetrics.classification

import maite.protocols.image_classification as ic
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


def _get_valid_classification_metrics(
    allowed_modules: (
        Literal["Multiclass", "Binary", "Multilabel"]
        | list[Literal["Multiclass", "Binary", "Multilabel"]]
        | None
    ) = None,
) -> dict[str, torchmetrics.Metric]:
    """Check if a classification metric is supported by `TMClassificationMetric`.

    Metrics are gathered dynamically from `torchmetrics.classification` and must be of
    type, or subclass of, `torchmetrics.Metric`.

    Parameters
    ----------
    allowed_modules : Literal["Multiclass", "Binary", "Multilabel"] | list[Literal["Multiclass", "Binary", "Multilabel"]], default=None
        Allowed module names. If provided, only metrics from these modules will be
        returned. If none are provided, all classification metrics are returned.
        Elements of the list must be either "Multiclass", "Binary", or "Multilabel".

    Notes
    -----
    Post-filtering is required to ensure the metrics support the ic.Metrics protocol.
    """
    if allowed_modules is None:
        allowed_modules = ["Multiclass", "Binary", "Multilabel"]
    if isinstance(allowed_modules, str):
        allowed_modules = [allowed_modules]
    modules = {}
    for module in vars(torchmetrics.classification):
        for allowed_module_type in allowed_modules:
            attr = getattr(torchmetrics.classification, module)
            if allowed_module_type in module:
                try:
                    if not issubclass(attr, torchmetrics.Metric):
                        continue
                except TypeError:
                    continue
                modules[attr.__name__] = attr
    return modules


# TM_CLASSIFICATION_METRIC_WHITELIST = _get_valid_classification_metrics("Multiclass")
TM_CLASSIFICATION_METRIC_WHITELIST: dict[str, Type[torchmetrics.Metric]] = {
    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy,
    "MulticlassCohenKappa": torchmetrics.classification.MulticlassCohenKappa,
    "MulticlassConfusionMatrix": torchmetrics.classification.MulticlassConfusionMatrix,
    "MulticlassExactMatch": torchmetrics.classification.MulticlassExactMatch,
    "MulticlassF1Score": torchmetrics.classification.MulticlassF1Score,
    "MulticlassFBetaScore": torchmetrics.classification.MulticlassFBetaScore,
    "MulticlassHammingDistance": torchmetrics.classification.MulticlassHammingDistance,
    "MulticlassJaccardIndex": torchmetrics.classification.MulticlassJaccardIndex,
    "MulticlassMatthewsCorrCoef": torchmetrics.classification.MulticlassMatthewsCorrCoef,
    "MulticlassNegativePredictiveValue": torchmetrics.classification.MulticlassNegativePredictiveValue,
    "MulticlassPrecision": torchmetrics.classification.MulticlassPrecision,
    "MulticlassRecall": torchmetrics.classification.MulticlassRecall,
    "MulticlassSpecificity": torchmetrics.classification.MulticlassSpecificity,
    "MulticlassStatScores": torchmetrics.classification.MulticlassStatScores,
}


class TMClassificationMetric:
    """
    MAITE-compliant Wrapper for TorchMetrics Classification Metrics.

    `TMClassificationMetric` is a wrapper around `torchmetrics.classification` metrics
    adhering to MAITE's `image_classification.Metric` protocol.

    Notes
    -----
    Only `Multiclass` metrics are currently supported. Please refer to the `torchmetrics`
    documentation for more information: https://lightning.ai/docs/torchmetrics/stable/.

    Supported metrics:
        * `~torchmetrics.classification.MulticlassAccuracy`
        * `~torchmetrics.classification.MulticlassCohenKappa`
        * `~torchmetrics.classification.MulticlassConfusionMatrix`
        * `~torchmetrics.classification.MulticlassExactMatch`
        * `~torchmetrics.classification.MulticlassF1Score`
        * `~torchmetrics.classification.MulticlassFBetaScore`
        * `~torchmetrics.classification.MulticlassHammingDistance`
        * `~torchmetrics.classification.MulticlassJaccardIndex`
        * `~torchmetrics.classification.MulticlassMatthewsCorrCoef`
        * `~torchmetrics.classification.MulticlassNegativePredictiveValue`
        * `~torchmetrics.classification.MulticlassPrecision`
        * `~torchmetrics.classification.MulticlassRecall`
        * `~torchmetrics.classification.MulticlassSpecificity`
        * `~torchmetrics.classification.MulticlassStatScores`

    (Unsupported metrics either require `preds` and `targets` to be of varying different
    shapes or require additional parameters in their `update()` function signature.)

    Examples
    --------

    >>> import torch
    >>> import torchmetrics.classification
    >>> from typing_extensions import Sequence
    >>> from maite.protocols import image_classification as ic, MetricMetadata
    >>> from maite.interop.metrics.torchmetrics import TMClassificationMetric
    >>>
    >>> preds: Sequence[ic.TargetType] = [
    ...     torch.tensor([0.1, 0.8, 0.1]),
    ...     torch.tensor([0.6, 0.2, 0.2]),
    ...     torch.tensor([0.4, 0.3, 0.3]),
    ... ]
    >>> target: Sequence[ic.TargetType] = [
    ...     torch.tensor([0, 1, 0]),
    ...     torch.tensor([1, 0, 0]),
    ...     torch.tensor([0, 0, 1]),
    ... ]
    >>> metadatas: Sequence[ic.DatumMetadataType] = [{"id": 1}, {"id": 2}, {"id": 3}]
    >>> # Create native TorchMetrics metric
    >>> classification_metric = torchmetrics.classification.MulticlassAccuracy(
    ...     num_classes=3
    ... )
    >>>
    >>> # Add additional field to base MetricMetadata
    >>> class MyMetricMetadata(MetricMetadata):
    ...     num_classes: int
    >>> metadata: MyMetricMetadata = {"id": "Multiclass Accuracy", "num_classes": 3}
    >>>
    >>> # Wrap metric and apply to sample data
    >>> wrapped_classification_metric: ic.Metric = TMClassificationMetric(
    ...     classification_metric, metadata=metadata
    ... )
    >>> wrapped_classification_metric.update(preds, target, metadatas)
    >>> result = wrapped_classification_metric.compute()
    >>> result  # doctest: +SKIP
    {'MulticlassAccuracy': tensor(0.6667)}
    >>> print(
    ...     f"{result['MulticlassAccuracy'].item():0.3f}"
    ... )  # consistent formatting for doctest
    0.667
    """

    def __init__(
        self,
        metric: torchmetrics.Metric,
        output_key: str | None = None,
        output_transform: Callable[[torch.Tensor], dict[str, Any]] | None = None,
        device: Any | None = None,
        dtype: torch.dtype | None = None,
        metadata: MetricMetadata | None = None,
    ):
        """
        Parameters
        ----------
        metric : `torchmetrics.Metric`
            Metric being wrapped. The metric must be a `torchmetrics.classification`
            metric.
        output_key : str, optional
            Outermost key returned from calling `TMClassificationMetric.compute()`. If
            neither output_key nor output_transform is provided and the metric result is
            not already a dictionary with str keys, then the name of the
            `torchmetrics.Metric` will be used as the default outmost key. Note: At most
            one of output_key and output_transform may be provided.
        output_transform : Callable[[torch.Tensor], dict[str, Any]], optional
            Function that takes the output of `torchmetrics.Metric.compute()` as input
            and returns a modified version of it to be returned by
            `TMClassificationMetric.compute()`. Note: At most one of output_key and
            output_transform may be provided.
        device : Any, optional
            Torch device type on which a torch.Tensor is or will be allocated. If none
            is passed, then the device will be inferred by torch.
        dtype : `torch.dtype`, optional
            Torch data type. If none is passed, then data type will be inferred by
            torch.
        metadata : `MetricMetadata`
            A typed dictionary containing at least an 'id' field of type str.
        """
        TMClassificationMetric._assert_valid_classification_metric(metric)

        if output_key is not None and output_transform is not None:
            raise ValueError(
                "Only one of `output_key` and `output_transform` may be provided"
            )

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
    def _assert_valid_dims(tensors: list[torch.Tensor]):
        """Ensure tensor elements are of shape `(Cl,)`.

        Note this is required by MAITE's `image_classification.Metric` protocol, where
        `Cl` refers to the classes, or labels, of the data.
        """
        if not all(tensor.ndim == 1 for tensor in tensors):
            raise ValueError(
                "Invalid dimensions for `preds` or `targets` elements. Expected 1-dimensional `ArrayLike` elements of shape `(Cl,)`"
            )

    @staticmethod
    def _assert_valid_classification_metric(metric: torchmetrics.Metric):
        if type(metric) not in list(TM_CLASSIFICATION_METRIC_WHITELIST.values()):
            raise ValueError(
                f"Invalid `metric` supplied: {type(metric)}. Must be one of {list(TM_CLASSIFICATION_METRIC_WHITELIST.keys())}."
            )

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
        pred_batch: Sequence[ic.TargetType],
        target_batch: Sequence[ic.TargetType],
        metadata_batch: Sequence[ic.DatumMetadataType],
    ) -> None:
        # doc-ignore: EX01
        """
        Update the internal state of the metric with new predictions and targets.

        MAITE's `image_classification.Metric` protocol requires the elements of `preds`
        and `targets` to be of shape `(Cl,)`, where `Cl` refers to the classes, or
        labels, of the data.

        Parameters
        ----------
        pred_batch : ic.TargetBatchType
            Batch of predicted classificaton values.
        target_batch : ic.TargetBatchType
            Batch of ground truth classification values.
        metadata_batch : Sequence[ic.DatumMetadataType]
            Batch of metadata.
        """
        preds_tm = [
            _arraylike_as_tensor(arr, device=self.device, dtype=self.dtype)
            for arr in pred_batch
        ]
        targets_tm = [
            _arraylike_as_tensor(arr, device=self.device, dtype=self.dtype)
            for arr in target_batch
        ]

        self._assert_valid_dims(preds_tm)
        self._assert_valid_dims(targets_tm)

        preds_tm = torch.stack(preds_tm).to(self.device)
        targets_tm = torch.stack(targets_tm).argmax(dim=1).to(self.device)
        self.metric.update(preds_tm, targets_tm)

    def compute(self) -> dict[str, Any]:
        # doc-ignore: EX01
        """
        Compute the TorchMetric metric result and format it.

        If `TMClassificationMetric.output_transform` is provided, the result will be
        passed to the `output_transform` function for additional post-processing.

        If `TMClassificationMetric.output_key` is provided, the
        result will be wrapped in a dictionary with that key. Otherwise, if the result is
        not already a dictionary, it will be wrapped in a dictionary with a key derived
        from the metric's name.

        Returns
        -------
        Dict[str, Any]
            The formatted metric result.
        """
        results = self.metric.compute()

        if self.output_transform is not None:
            # User wants full control over reformatting the results that get returned
            return self.output_transform(results)

        if self.output_key is not None:
            # User wants the output with a top-level key
            return {self.output_key: results}

        if not isinstance(results, dict) or any(
            not isinstance(k, str) for k in results.keys()
        ):
            # Ensure dict[str, Any]
            key = self.metric._get_name()
            results = {key: results}

        return results
