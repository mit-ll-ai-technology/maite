# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch as tr
from torch.utils.data import Dataset
from torchvision.models._api import WeightsEnum
from torchvision.transforms import ToTensor

from .utils import create_random_image


def get_test_vision_dataset(has_split: bool = True, has_train=False):
    """
    Creates a test dataset for testing torchvision datasets.
    """

    class TVDataset(Dataset):
        def __init__(self, *, split=None, train=None):
            if not has_split and split is not None:
                raise TypeError("split is not supported")

            if not has_train and train is not None:
                raise TypeError("train is not supported")

            self.classes = ["a", "b", "c", "d"]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            image = create_random_image()
            label = 3
            return image, label

    return TVDataset


class Weights:
    def __init__(self):
        self.weights = tr.randn(1, 3, 10, 10)
        self.meta = {"categories": ["a", "b", "c", "d"]}

    def transforms(self):
        return ToTensor()


class TVWeights(WeightsEnum):
    DEFAULT = Weights()


def get_test_vision_model():
    """Creates a test model for testing torchvision image classification models."""

    class TVModel(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = tr.nn.Linear(1, 1)

        def forward(self, x):
            return x

    return TVWeights, TVModel()


def get_test_object_detection_model():
    """Creates a test model for testing torchvision object detectionmodels."""

    class TVModel(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = tr.nn.Linear(1, 1)

        def forward(self, x):
            boxes = tr.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
            scores = tr.tensor([0.5, 0.5])
            labels = tr.tensor([0, 1])
            return [
                dict(boxes=b, scores=s, labels=l)
                for b, s, l in zip(boxes, scores, labels)
            ]

    return TVWeights, TVModel()
