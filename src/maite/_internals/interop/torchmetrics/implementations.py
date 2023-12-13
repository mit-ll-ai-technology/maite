# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any

from typing_extensions import Self

import maite

from ...protocols.typing import Metric


class ClassificationReport(Metric):
    def __init__(self, num_classes: int, average: str = "none") -> None:
        """
        Provides a dictionary with the following metrics:
        - accuracy
        - f1score
        - precision
        - recall

        Parameters
        ----------
        num_classes : int
            Number of classes
        average : str
            Average method, by default "none"

        Returns
        -------
        Dict[str, Metric]
            Dictionary with the metrics

        Example
        -------
        >>> import maite as jtb
        >>> classification_report = jtb.load_metric(provider="torchmetrics", metric_name="classification_report", num_classes=10, average="macro")
        """
        super().__init__()
        self.metrics = dict(
            accuracy=maite.load_metric(
                provider="torchmetrics",
                metric_name="Accuracy",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
            f1score=maite.load_metric(
                provider="torchmetrics",
                metric_name="F1Score",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
            precision=maite.load_metric(
                provider="torchmetrics",
                metric_name="Precision",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
            recall=maite.load_metric(
                provider="torchmetrics",
                metric_name="Recall",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
        )

    def to(self, *args: Any, **kwargs: Any) -> Self:
        for metric in self.metrics.values():
            metric.to(*args, **kwargs)
        return self

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def update(self, *args: Any, **kwargs: Any) -> None:
        for metric in self.metrics.values():
            metric.update(*args, **kwargs)

    def compute(self) -> Any:
        return {k: v.compute() for k, v in self.metrics.items()}
