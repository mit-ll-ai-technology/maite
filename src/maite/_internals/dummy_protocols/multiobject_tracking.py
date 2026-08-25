"""minimal dummy protocol implementers for multiobject_tracking AI problem"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

import maite.protocols.multiobject_tracking as mot
from maite.protocols import (
    AugmentationMetadata,
    DatasetMetadata,
    MetricMetadata,
    ModelMetadata,
)

H = W = 4

Batch = tuple[Sequence[mot.VideoStream], Sequence[mot.MultiobjectTrackingTarget], Sequence[mot.DatumMetadata]]


@dataclass
class Frame:  # satisfy mot.VideoFrame
    pixels: np.ndarray
    time_s: float
    pts: int
    frame_index: int


@dataclass
class FrameTracks:  # satisfy mot.SingleFrameObjectTrackingTarget
    boxes: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0, 1.0, 1.0]]))
    labels: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    scores: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    track_ids: np.ndarray = field(default_factory=lambda: np.array([0]))


@dataclass
class Target:  # satisfies mot.MultiobjectTrackingTarget
    frame_tracks: Sequence[FrameTracks] = field(default_factory=lambda: [FrameTracks(), FrameTracks()])


def make_stream(n_frames: int = 2) -> list[Frame]:  # a list[Frame] is a VideoStream
    return [Frame(np.zeros((3, H, W), dtype=np.float32), time_s=i / 30, pts=i, frame_index=i) for i in range(n_frames)]


def make_bad_stream(n_frames: int = 2) -> list[Frame]:
    return [
        Frame(
            np.zeros((3, H, W), dtype=np.float32),
            time_s=-i / 30,  # negative time value
            pts=i,
            frame_index=i,
        )
        for i in range(n_frames)
    ]


def make_target(n_frames: int = 2) -> Target:
    return Target(frame_tracks=[FrameTracks() for _ in range(n_frames)])


def make_bad_target(n_frames: int = 2) -> Target:
    return Target(
        frame_tracks=[
            FrameTracks(labels=np.array([]))  # mismatched 'labels' attribute cardinality (zero-length)
            for _ in range(n_frames)
        ]
    )


def make_md(i: int) -> mot.DatumMetadata:
    return {"id": i, "height": H, "width": W, "time_base": Fraction(1, 30), "size": 0}


class Dataset:
    metadata: DatasetMetadata = {"id": "dummy-mot-dataset"}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int, /) -> tuple[mot.VideoStream, Target, mot.DatumMetadata]:
        return make_stream(), make_target(), make_md(0)


class DataLoader:
    def __iter__(self) -> Iterator[Batch]:
        yield [make_stream()], [make_target()], [make_md(0)]


class Augmentation:
    metadata: AugmentationMetadata = {"id": "dummy-mot-augmentation"}

    def __call__(self, batch: Batch) -> Batch:
        return batch


class Model:
    metadata: ModelMetadata = {"id": "dummy-mot-model"}

    def __call__(self, xs: Sequence[mot.VideoStream]) -> Sequence[Target]:
        # one FrameTracks per frame: output length honors input stream length
        return [Target(frame_tracks=[FrameTracks() for _ in stream]) for stream in xs]


class Metric:
    metadata: MetricMetadata = {"id": "dummy-mot-metric"}

    def __init__(self) -> None:
        self.n = 0

    def update(
        self,
        preds: Sequence[mot.MultiobjectTrackingTarget],
        targets: Sequence[mot.MultiobjectTrackingTarget],  # noqa: ARG002, intentionally unused
        mds: Sequence[mot.DatumMetadata],  # noqa: ARG002, intentionally unused
    ) -> None:
        self.n += len(preds)

    def reset(self) -> None:
        self.n = 0

    def compute(self) -> Mapping[str, Any]:
        return {"n_seen": self.n}


class BadDataset(Dataset):
    def __getitem__(self, i: int, /) -> tuple[mot.VideoStream, Target, mot.DatumMetadata]:
        if i > 2:
            raise IndexError("Index out of range")
        return make_stream(), make_bad_target(), make_md(0)


class BadDataLoader(DataLoader):
    def __iter__(self) -> Iterator[Batch]:
        yield [make_stream()], [make_bad_target()], [make_md(0)]


class BadAugmentation(Augmentation):
    def __call__(self, batch: Batch) -> Batch:
        return batch[0], [make_bad_target() for _ in range(len(batch))], batch[2]


class BadModel(Model):
    def __call__(self, xs: Sequence[mot.VideoStream]) -> Sequence[Target]:
        return [make_bad_target() for _ in xs]


# Show statically that above impls satisfy typing
dataset: mot.Dataset = Dataset()
dataloader: mot.DataLoader = DataLoader()
augmentation: mot.Augmentation = Augmentation()
model: mot.Model = Model()
metric: mot.Metric = Metric()

# Show statically that "bad" impls satisfy static type checker
# (hard to make semantically bad Metric at present)
bad_dataset: mot.Dataset = BadDataset()
bad_dataloader: mot.Dataset = BadDataset()
bad_augmentation: mot.Augmentation = BadAugmentation()
bad_model: mot.Model = BadModel()
