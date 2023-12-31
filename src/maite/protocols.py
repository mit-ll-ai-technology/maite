# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from maite._internals.protocols.type_guards import (
    is_list_dict,
    is_list_of_type,
    is_typed_dict,
)
from maite._internals.protocols.typing import (
    ArrayLike,
    ArtifactHubEndpoint,
    Augmentation,
    DataLoader,
    Dataset,
    DatasetProvider,
    DatumMetadata,
    HasDataBoxes,
    HasDataBoxesLabels,
    HasDataImage,
    HasDataLabel,
    HasDataMetadata,
    HasDataObjects,
    HasDetectionLogits,
    HasDetectionPredictions,
    HasDetectionProbs,
    HasLogits,
    HasProbs,
    HasScores,
    ImageClassifier,
    Metric,
    MetricProvider,
    Model,
    ModelMetadata,
    ModelProvider,
    ObjectDetectionDataLoader,
    ObjectDetectionDataset,
    ObjectDetector,
    SupportsArray,
    SupportsImageClassification,
    SupportsObjectDetection,
    TaskName,
    VisionDataLoader,
    VisionDataset,
)

__all__ = [
    "ArrayLike",
    "ArtifactHubEndpoint",
    "ImageClassifier",
    "ObjectDetector",
    "Dataset",
    "DatasetProvider",
    "DatumMetadata",
    "HasLogits",
    "HasProbs",
    "SupportsImageClassification",
    "SupportsObjectDetection",
    "VisionDataset",
    "HasDataObjects",
    "ModelMetadata",
    "Metric",
    "MetricProvider",
    "HasDetectionLogits",
    "HasDetectionPredictions",
    "Augmentation",
    "DataLoader",
    "VisionDataLoader",
    "HasScores",
    "is_list_dict",
    "ObjectDetectionDataLoader",
    "ObjectDetectionDataset",
    "is_typed_dict",
    "HasDetectionProbs",
    "HasDataImage",
    "HasDataLabel",
    "HasDataBoxes",
    "SupportsArray",
    "HasDataObjects",
    "is_list_of_type",
    "HasDataBoxesLabels",
    "HasDataMetadata",
    "Model",
    "ModelProvider",
    "TaskName",
]
