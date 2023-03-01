from jatic_toolbox._internals.import_utils import is_numpy_available, is_torch_available
from jatic_toolbox._internals.protocols import (
    ArrayLike,
    Augmentation,
    Classifier,
    ClassifierDataLoader,
    ClassifierDataset,
    ClassifierWithParameters,
    DataLoader,
    Dataset,
    HasLogits,
    HasObjectDetections,
    HasProbs,
    HasTarget,
    Metric,
    MetricCollection,
    Model,
    ModelOutput,
    ObjectDetector,
    ShapedArray,
    SupportsClassification,
    TypedCollection,
)

__all__ = [
    "ArrayLike",
    "Augmentation",
    "Classifier",
    "ClassifierDataLoader",
    "ClassifierDataset",
    "ClassifierWithParameters",
    "DataLoader",
    "Dataset",
    "HasLogits",
    "HasObjectDetections",
    "HasProbs",
    "HasTarget",
    "Metric",
    "MetricCollection",
    "Model",
    "ModelOutput",
    "ObjectDetector",
    "ShapedArray",
    "SupportsClassification",
    "TypedCollection",
]

if is_numpy_available():
    from jatic_toolbox._internals.protocols import (  # noqa: F401
        NumPyClassifier,
        NumPyClassifierWithParameters,
        NumPyDataLoader,
        NumPyDataset,
        NumPyMetric,
        NumPyMetricCollection,
    )

    __all__.extend(
        [
            "NumPyClassifier",
            "NumPyClassifierWithParameters",
            "NumPyDataLoader",
            "NumPyDataset",
            "NumPyMetric",
            "NumPyMetricCollection",
        ]
    )


if is_torch_available():
    from jatic_toolbox._internals.protocols import (  # noqa: F401
        TorchClassifier,
        TorchClassifierWithParameters,
        TorchDataLoader,
        TorchDataset,
        TorchMetric,
        TorchMetricCollection,
    )

    __all__.extend(
        [
            "TorchClassifier",
            "TorchClassifierWithParameters",
            "TorchDataLoader",
            "TorchDataset",
            "TorchMetric",
            "TorchMetricCollection",
            "TypedCollection",
        ]
    )
