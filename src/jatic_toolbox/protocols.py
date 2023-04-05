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
    DatasetDict,
    HasDetectionLogits,
    HasDetectionProbs,
    HasLogits,
    HasObjectDetections,
    HasProbs,
    HasTarget,
    Metric,
    MetricCollection,
    Model,
    ModelOutput,
    ObjectDetection,
    ObjectDetectionDataset,
    ObjectDetector,
    ShapedArray,
    SupportsClassification,
    SupportsImageClassification,
    SupportsObjectDetection,
    TypedCollection,
    VisionDataset,
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
    "DatasetDict",
    "HasDetectionLogits",
    "HasDetectionProbs",
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
    "VisionDataset",
    "SupportsImageClassification",
    "ObjectDetectionDataset",
    "SupportsObjectDetection",
    "ObjectDetection",
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
