from dataclasses import dataclass

import datasets
import torch as tr
from datasets import Dataset, Features

from .utils import create_random_image


def get_test_vision_dataset(image_key, label_key):
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
    image_key,
    object_key,
    category_key,
    bbox_key,
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

        obj_feature = {"obj_col": datasets.Value("string")}
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

    class Features(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, images, return_tensors):
            return Features(pixel_values=tr.stack(images))

    class Model(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "cpu"

        def forward(self, *args, **kwargs):
            logits = tr.randn(1, 10)
            return VisionOutput(logits)

    return Processor(), Model()


def get_test_object_detection_model(with_post_processor, output_as_list):
    @dataclass
    class ObjectDetectionWithLogits:
        logits: tr.Tensor
        pred_boxes: tr.Tensor

    @dataclass
    class ObjectDetectionOutput:
        boxes: tr.Tensor
        scores: tr.Tensor
        labels: tr.Tensor

    class Features(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, images, return_tensors):
            return Features(pixel_values=tr.stack(images))

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

        def forward(self, *args, **kwargs):
            boxes = tr.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
            scores = tr.tensor([0.5, 0.5])
            labels = tr.tensor([0, 1])
            logits = tr.randn(1, 10)

            if with_post_processor:
                return ObjectDetectionWithLogits(logits, boxes)
            return ObjectDetectionOutput(boxes, scores, labels)

    return Processor(), Model()
