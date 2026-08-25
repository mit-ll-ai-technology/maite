"""minimal dummy protocol implementers for object_detection AI problem"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import maite.protocols.object_detection as od
from maite.protocols import (
    ArrayLike,
    AugmentationMetadata,
    DatasetMetadata,
    DatumMetadata,
    MetricMetadata,
    ModelMetadata,
)

IMG = np.zeros((3, 4, 4), dtype=np.float32)
BADIMG = np.zeros((4, 4), dtype=np.float32)  # use 2D shape

Batch = tuple[Sequence[ArrayLike], Sequence[od.ObjectDetectionTarget], Sequence[DatumMetadata]]


@dataclass
class Target:  # satisfy od.ObjectDetectionTarget
    boxes: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0, 1.0, 1.0]]))
    labels: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    scores: np.ndarray = field(default_factory=lambda: np.array([1.0]))


def make_target() -> Target:
    """Make object_detection target: a vector of non-negative values that sum to 1"""
    return Target()


def make_bad_target() -> Target:
    return Target(scores=np.array([0.0, 1.0, 1.0]))  # mismatched 'scores' attribute cardinality


class Dataset:
    metadata: DatasetMetadata = {"id": "dummy-od-dataset"}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int, /) -> tuple[ArrayLike, Target, DatumMetadata]:
        if i > (len(self) - 1):
            raise IndexError("index out of range")
        return IMG, make_target(), {"id": i}


class DataLoader:
    def __iter__(self) -> Iterator[Batch]:
        yield [IMG], [make_target()], [{"id": 0}]


class Augmentation:
    metadata: AugmentationMetadata = {"id": "dummy-od-augmentation"}

    def __call__(self, batch: Batch) -> Batch:
        return batch


class Model:
    metadata: ModelMetadata = {"id": "dummy-od-model"}

    def __call__(self, xs: Sequence[ArrayLike]) -> Sequence[Target]:
        return [make_target() for _ in xs]


class Metric:
    metadata: MetricMetadata = {"id": "dummy-od-metric"}

    def __init__(self) -> None:
        self.n = 0

    def update(
        self,
        preds: Sequence[od.ObjectDetectionTarget],
        targets: Sequence[od.ObjectDetectionTarget],  # noqa: ARG002, intentionally unused
        mds: Sequence[DatumMetadata],  # noqa: ARG002, intentionally unused
    ) -> None:
        self.n += len(preds)

    def reset(self) -> None:
        self.n = 0

    def compute(self) -> Mapping[str, Any]:
        return {"n_seen": self.n}


class BadDataset(Dataset):
    def __getitem__(self, i: int, /) -> tuple[ArrayLike, Target, DatumMetadata]:
        good_datum = super().__getitem__(i)
        return (BADIMG, good_datum[1], good_datum[2])


class BadDataLoader(DataLoader):
    def __iter__(self) -> Iterator[Batch]:
        yield [IMG], [make_bad_target()], [{"id": 0}]


class BadAugmentation(Augmentation):
    def __call__(self, batch: Batch) -> Batch:
        return [BADIMG for _ in range(len(batch[0]))], batch[1], batch[2]


class BadModel(Model):
    def __call__(self, xs: Sequence[ArrayLike]) -> Sequence[Target]:
        return [make_bad_target() for _ in xs]


# Show statically that above impls satisfy typing
dataset: od.Dataset = Dataset()
dataloader: od.DataLoader = DataLoader()
augmentation: od.Augmentation = Augmentation()
model: od.Model = Model()
metric: od.Metric = Metric()

# Show statically that "bad" impls satisfy static type checker
# (hard to make semantically bad Metric at present)
bad_dataset: od.Dataset = BadDataset()
bad_dataloader: od.Dataset = BadDataset()
bad_augmentation: od.Augmentation = BadAugmentation()
bad_model: od.Model = BadModel()
