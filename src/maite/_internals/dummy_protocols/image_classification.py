"""minimal dummy protocol implementers for image_classification AI problem"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

import maite.protocols.image_classification as ic
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

Batch = tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadata]]


def make_target() -> np.ndarray:
    """Make valid classification target: a onehot probability vector"""
    return np.eye(2, dtype=np.float32)[0]


def make_bad_target() -> np.ndarray:
    """Make invalid classification target: a non-onehot probability vector"""
    return np.ones(2, dtype=np.float32)[0]  # non-onehot model output


class Dataset:
    metadata: DatasetMetadata = {"id": "dummy-ic-dataset"}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int, /) -> tuple[ArrayLike, ArrayLike, DatumMetadata]:
        if i > len(self):
            raise IndexError("Index out of range")
        return IMG, make_target(), {"id": i}


class DataLoader:
    def __iter__(self) -> Iterator[Batch]:
        yield [IMG], [make_target()], [{"id": 0}]


class Augmentation:
    metadata: AugmentationMetadata = {"id": "dummy-ic-augmentation"}

    def __call__(self, batch: Batch) -> Batch:
        return batch


class Model:
    metadata: ModelMetadata = {"id": "dummy-ic-model"}

    def __call__(self, xs: Sequence[ArrayLike]) -> Sequence[ArrayLike]:
        return [make_target() for _ in xs]


class Metric:
    metadata: MetricMetadata = {"id": "dummy-ic-metric"}

    def __init__(self) -> None:
        self.n = 0

    def update(
        self,
        preds: Sequence[ArrayLike],
        targets: Sequence[ArrayLike],  # noqa: ARG002, intentionally unused
        mds: Sequence[DatumMetadata],  # noqa: ARG002, intentionally unused
    ) -> None:
        self.n += len(preds)

    def reset(self) -> None:
        self.n = 0

    def compute(self) -> Mapping[str, Any]:
        return {"n_seen": self.n}


class BadDataset(Dataset):
    def __getitem__(self, i: int, /) -> tuple[ArrayLike, ArrayLike, DatumMetadata]:
        good_datum = super().__getitem__(i)
        return (BADIMG, good_datum[1], good_datum[2])


class BadDataLoader(DataLoader):
    def __iter__(self) -> Iterator[Batch]:
        yield [IMG], [make_bad_target()], [{"id": 0}]


class BadAugmentation(Augmentation):
    def __call__(self, batch: Batch) -> Batch:
        return [BADIMG for _ in range(len(batch[0]))], batch[1], batch[2]


class BadModel(Model):
    def __call__(self, xs: Sequence[ArrayLike]) -> Sequence[ArrayLike]:
        return [make_bad_target() for _ in xs]


# Show statically that above impls satisfy static type checker
dataset: ic.Dataset = Dataset()
dataloader: ic.DataLoader = DataLoader()
augmentation: ic.Augmentation = Augmentation()
model: ic.Model = Model()
metric: ic.Metric = Metric()

# Show statically that "bad" impls satisfy static type checker
# (hard to make semantically bad Metric at present)
bad_dataset: ic.Dataset = BadDataset()
bad_dataloader: ic.Dataset = BadDataset()
bad_augmentation: ic.Augmentation = BadAugmentation()
bad_model: ic.Model = BadModel()
