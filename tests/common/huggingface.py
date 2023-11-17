# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import UserDict
from dataclasses import dataclass
from typing import Any, Dict

import datasets
import numpy as np
import torch as tr
from datasets import Dataset, Features
from PIL.Image import Image
from transformers.utils import ModelOutput

from .utils import create_random_image


class BatchFeatures(UserDict):
    def to(self, device):
        return self


@dataclass
class ObjectDetectionWithLogits(ModelOutput):
    logits: tr.Tensor = None  # type: ignore[assignment]
    pred_boxes: tr.Tensor = None  # type: ignore[assignment]


@dataclass
class ObjectDetectionOutput(ModelOutput):
    boxes: tr.Tensor = None  # type: ignore[assignment]
    scores: tr.Tensor = None  # type: ignore[assignment]
    labels: tr.Tensor = None  # type: ignore[assignment]


@dataclass
class Meta:
    id2label: dict
    num_labels: int


def get_test_vision_dataset(image_key="image", label_key="label"):
    """
    Creates a test dataset for testing vision datasets.
    """

    data = {
        "col_1": [3, 2, 1, 0],
        "col_2": ["a", "b", "c", "d"],
        "col_3": [False, True, False, True],
    }
    features = Features(
        {
            "col_1": datasets.Value("int64"),
            "col_2": datasets.Value("string"),
            "col_3": datasets.Value("bool"),
        }
    )

    if image_key is not None:
        data[image_key] = [create_random_image() for i in range(4)]
        features[image_key] = datasets.Image()

    if label_key is not None:
        data[label_key] = [3, 2, 1, 0]
        features[label_key] = datasets.ClassLabel(4, ["w", "x", "y", "z"])

    return Dataset.from_dict(data, features=features)


def get_test_detection_dataset(
    image_key="image",
    object_key="objects",
    category_key="category",
    bbox_key="bbox",
    objects_as_list: bool = False,
):
    """
    Creates a test dataset for testing object detection datasets.
    """

    data = {
        "col_1": [3],
        "col_2": ["a"],
        "col_3": [False],
    }
    features = Features(
        {
            "col_1": datasets.Value("int64"),
            "col_2": datasets.Value("string"),
            "col_3": datasets.Value("bool"),
        }
    )

    if image_key is not None:
        data[image_key] = [create_random_image()]
        features[image_key] = datasets.Image()

    if object_key and (category_key or bbox_key):
        data[object_key] = [{"obj_col": ["1", "2", "3", "4"]}]

        obj_feature: Dict[str, Any] = {"obj_col": datasets.Value("string")}
        if category_key is not None:
            data[object_key][0][category_key] = [3, 2, 1, 0]
            obj_feature[category_key] = datasets.ClassLabel(names=["w", "x", "y", "z"])

        if bbox_key:
            data[object_key][0][bbox_key] = [
                [1.0, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ]
            obj_feature[bbox_key] = datasets.Sequence(
                feature=datasets.Value("float32"), length=4
            )

        features[object_key] = datasets.Sequence(feature=obj_feature)

    ds = Dataset.from_dict(data, features=features)
    if objects_as_list:
        # some datasets don't use "Sequence"
        ds.features[object_key] = [ds.features[object_key].feature]

    return ds


def get_test_vision_model():
    @dataclass
    class VisionOutput:
        logits: tr.Tensor

    class Processor:
        def __call__(self, images, return_tensors):
            if isinstance(images[0], Image):
                images = [np.asarray(i) for i in images]

            images = [tr.as_tensor(i) for i in images]
            return BatchFeatures(pixel_values=tr.stack(images))

    class Model(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "cpu"
            self.config = Meta(
                id2label={i: f"label_{i}" for i in range(10)}, num_labels=10
            )
            self.linear = tr.nn.Linear(10, 10)

        def forward(self, *args, **kwargs):
            logits = tr.randn(1, 10)
            return VisionOutput(logits)

    return Processor(), Model()


def get_test_object_detection_model(output_as_list=False):
    class Processor:
        def __call__(self, images, return_tensors):
            if isinstance(images[0], Image):
                images = [np.asarray(i) for i in images]

            images = [tr.as_tensor(i) for i in images]
            return BatchFeatures(pixel_values=tr.stack(images))

        def post_process_object_detection(self, outputs, threshold, target_sizes):
            boxes = tr.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
            scores = tr.tensor([0.5, 0.5])
            labels = tr.tensor([0, 1])
            if output_as_list:
                return [
                    dict(boxes=b, scores=s, labels=l)
                    for b, s, l in zip(boxes, scores, labels)
                ]
            return ObjectDetectionOutput(boxes, scores, labels)

    class Model(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "cpu"
            self.config = Meta(
                id2label={i: f"label_{i}" for i in range(10)}, num_labels=10
            )
            self.linear = tr.nn.Linear(10, 10)

        def forward(self, *args, **kwargs):
            boxes = tr.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
            logits = tr.randn(1, 10)
            return ObjectDetectionWithLogits(logits, boxes)

    return Processor(), Model()
