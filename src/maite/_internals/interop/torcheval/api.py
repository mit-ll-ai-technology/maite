# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Callable, Iterable

from maite.protocols import Metric

from ...import_utils import is_torcheval_available


class TorchEvalAPI:
    def list_metrics(self) -> Iterable[Any]:
        """
        List all metrics from TorchEval.

        Returns
        -------
        Iterable[Any]
            The metrics.

        Examples
        --------
        >>> from maite._internals.interop.torcheval.api import TorchEvalAPI
        >>> api = TorchEvalAPI()
        >>> api.list_metrics()
        [...]
        """
        if not is_torcheval_available():  # pragma: no cover
            raise ImportError("TorchEval is not installed.")

        import torcheval.metrics as metrics

        return [
            m
            for m in metrics.__all__
            if m not in ("functional", "Metric", "FrechetAudioDistance")
        ]

    def load_metric_builder(self, metrics_name: str) -> Callable[..., Metric]:
        """
        Load a metric builder from TorchEval.

        Parameters
        ----------
        metrics_name : str
            The name of the metric to load.

        Returns
        -------
        Type[Metric]
            The metric builder.

        Examples
        --------
        >>> from maite._internals.interop.torcheval.api import TorchEvalAPI
        >>> api = TorchEvalAPI()
        >>> api.load_metric_builder("Accuracy")
        <function TorchEvalAPI.load_metric.<locals>.MetricBuilder(self, **kwargs: Any) -> Metric>
        """
        if not is_torcheval_available():  # pragma: no cover
            raise ImportError("TorchEval is not installed.")

        import torcheval.metrics as metrics

        assert metrics_name in metrics.__all__, f"{metrics_name} not found in torcheval"

        def MetricBuilder(**kwargs: Any) -> Metric:
            import importlib

            tm_clazz = importlib.import_module("torcheval.metrics")
            clazz = getattr(tm_clazz, metrics_name)
            return clazz(**kwargs)

        return MetricBuilder

    def load_metric(self, metric_name: str, **kwargs: Any) -> Metric:
        """
        Load a metric from TorchEval.

        Parameters
        ----------
        metric_name : str
            The name of the metric to load.
        **kwargs : Any
            The arguments to pass to the metric.

        Returns
        -------
        Metric
            The metric.

        Examples
        --------
        >>> from maite._internals.interop.torcheval.api import TorchEvalAPI
        >>> api = TorchEvalAPI()
        >>> api.load_metric("Accuracy")
        Accuracy()
        """
        if not is_torcheval_available():  # pragma: no cover
            raise ImportError("TorchEval is not installed.")

        builder = self.load_metric_builder(metric_name)
        return builder(**kwargs)
