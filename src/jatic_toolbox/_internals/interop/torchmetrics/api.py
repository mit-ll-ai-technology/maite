from typing import Any, Iterable, Type

from jatic_toolbox.protocols import Metric

from ...import_utils import is_torchmetrics_available


class TorchMetricsAPI:
    def list_metrics(self) -> Iterable[Any]:
        """
        List all metrics from TorchMetrics.

        Returns
        -------
        Iterable[Any]
            The metrics.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.torchmetrics.api import TorchMetricsAPI
        >>> api = TorchMetricsAPI()
        >>> api.list_metrics()
        [...]
        """
        if not is_torchmetrics_available():
            raise ImportError("TorchMetrics is not installed.")

        import torchmetrics

        return [
            m
            for m in torchmetrics.__all__
            if m not in ("functional", "Metric", "MetricCollection")
        ]

    def load_metric_builder(self, metrics_name, **kwargs: Any) -> Type[Metric]:
        """
        Load a metric builder from TorchMetrics.

        Parameters
        ----------
        metrics_name : str
            The name of the metric to load.
        **kwargs : Any
            The arguments to pass to the metric.

        Returns
        -------
        Type[Metric]
            The metric builder.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.torchmetrics.api import TorchMetricsAPI
        >>> api = TorchMetricsAPI()
        >>> api.load_metric_builder("Accuracy")
        <class 'torchmetrics.classification.accuracy.Accuracy'>
        """
        if not is_torchmetrics_available():
            raise ImportError("TorchMetrics is not installed.")

        import torchmetrics

        fn = getattr(torchmetrics, metrics_name)
        return fn

    def load_metric(self, metric_name: str, **kwargs: Any) -> Metric:
        """
        Load a metric from TorchMetrics.

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
        >>> from jatic_toolbox._internals.interop.torchmetrics.api import TorchMetricsAPI
        >>> api = TorchMetricsAPI()
        >>> api.load_metric("Accuracy")
        Accuracy()
        """
        if not is_torchmetrics_available():
            raise ImportError("TorchMetrics is not installed.")

        import importlib

        tm_clazz = importlib.import_module("torchmetrics")
        clazz = getattr(tm_clazz, metric_name)
        return clazz(**kwargs)
