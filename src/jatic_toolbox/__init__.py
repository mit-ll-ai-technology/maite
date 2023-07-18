from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str


from jatic_toolbox._internals.interop.api import (
    list_datasets,
    list_metrics,
    list_models,
    load_dataset,
    load_metric,
    load_model,
)
from jatic_toolbox._internals.interop.evaluate import evaluate
from jatic_toolbox._internals.interop.registry import (
    DATASET_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
)

__all__ = [
    "list_models",
    "load_model",
    "list_datasets",
    "load_dataset",
    "list_metrics",
    "load_metric",
    "evaluate",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "METRIC_REGISTRY",
]
