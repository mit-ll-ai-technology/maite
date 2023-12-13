# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

DATASET_REGISTRY: dict[str, dict[str, Any]]

DATASET_REGISTRY = {
    "cifar10-test": dict(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="test",
    ),
    "fashion_mnist-test": dict(
        provider="huggingface",
        dataset_name="fashion_mnist",
        task="image-classification",
        split="test",
    ),
    "coco-val": dict(
        provider="huggingface",
        dataset_name="detection-datasets/coco",
        task="object-detection",
        split="val",
    ),
}

MODEL_REGISTRY: dict[str, dict[str, Any]]

MODEL_REGISTRY = {
    "vit_for_cifar10": dict(
        provider="huggingface",
        task="image-classification",
        model_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    ),
    "fasterrcnn_resnet50_fpn": dict(
        provider="torchvision",
        model_name="fasterrcnn_resnet50_fpn",
        task="object-detection",
    ),
}

METRIC_REGISTRY: dict[str, dict[str, Any]]

METRIC_REGISTRY = {
    "multiclass_accuracy": dict(
        provider="torchmetrics",
        metric_name="Accuracy",
        task="multiclass",
    ),
    "mean_average_precision": dict(
        provider="torchmetrics",
        metric_name="MeanAveragePrecision",
        box_format="xywh",
        iou_thresholds=[0.5],
        rec_thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        max_detection_thresholds=[1, 10, 100],
    ),
    "classification_report": dict(
        provider="torchmetrics",
        metric_name="ClassificationReport",
    ),
}
