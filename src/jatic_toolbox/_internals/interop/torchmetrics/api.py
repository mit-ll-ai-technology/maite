from typing import Any, Callable, Iterable

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

    def load_metric_builder(self, metrics_name) -> Callable[..., Metric[Any]]:
        """
        Load a metric builder from TorchMetrics.

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
        >>> from jatic_toolbox._internals.interop.torchmetrics.api import TorchMetricsAPI
        >>> api = TorchMetricsAPI()
        >>> api.load_metric_builder("Accuracy")
        <function TorchMetricsAPI.load_metric.<locals>.MetricBuilder(self, **kwargs: Any) -> Metric>
        """
        if not is_torchmetrics_available():
            raise ImportError("TorchMetrics is not installed.")

        import torchmetrics

        if metrics_name == "MeanAveragePrecision":

            def CustomMAPBuilder(**kwargs: Any) -> Metric:
                from .detection import MeanAveragePrecision

                # TODO: Fix Protocol
                return MeanAveragePrecision(**kwargs)  # type: ignore

            return CustomMAPBuilder
        else:
            assert (
                metrics_name in torchmetrics.__all__
            ), f"{metrics_name} not found in torchmetrics"

            def MetricBuilder(**kwargs: Any) -> Metric:
                import importlib

                tm_clazz = importlib.import_module("torchmetrics")
                clazz = getattr(tm_clazz, metrics_name)
                return clazz(**kwargs)

            return MetricBuilder

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

        builder = self.load_metric_builder(metric_name)
        return builder(**kwargs)