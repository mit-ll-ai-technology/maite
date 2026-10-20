# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from maite.protocols import (
    DatasetMetadata,
    MetricMetadata,
    ModelMetadata,
)
from maite.protocols.multiobject_tracking import (
    DatumMetadataType,
    InputType,
    TargetType,
)

# test lightweight implementations for the multi-object tracking domain
#
# mock up a set of components such that:
#
# InputType is VideoStream (i.e. Iterable[VideoFrame]) -- crucially, an input datum here
#   is genuinely an iterable, and we yield it as a one-shot generator so that it iterates
#   destructively. This exercises `evaluate`'s materialization of iterator-typed data.
# TargetType is MultiobjectTrackingTarget
# DatumMetadataType is DatumMetadata (i.e. a specialized TypedDict)


class VideoFrameImpl:
    """Minimal structural `mot.VideoFrame`."""

    def __init__(self, frame_index: int) -> None:
        self.pixels = frame_index * np.ones((3, 2, 2))
        self.time_s = float(frame_index)
        self.pts = frame_index
        self.frame_index = frame_index


class SingleFrameTargetImpl:
    """Minimal structural `mot.SingleFrameObjectTrackingTarget` (no detections)."""

    def __init__(self) -> None:
        self.boxes = np.zeros((0, 4))
        self.labels = np.zeros((0,))
        self.scores = np.zeros((0,))
        self.track_ids = np.zeros((0,))


class TrackingTargetImpl:
    """Minimal structural `mot.MultiobjectTrackingTarget` over `n_frames` frames."""

    def __init__(self, n_frames: int) -> None:
        self.frame_tracks = [SingleFrameTargetImpl() for _ in range(n_frames)]


class DatasetImpl:
    """MOT dataset whose input datum is a generator of frames (a destructive VideoStream).

    Video i yields `n_frames` frames whose `frame_index` runs 0..n_frames-1. Because the
    datum is a one-shot generator, consuming it once (e.g. to build augmented data) would
    leave a spent iterator for the model/metric unless `evaluate` materializes it.
    """

    def __init__(self, size: int = 4, n_frames: int = 3) -> None:
        self.size = size
        self.n_frames = n_frames
        self.metadata = DatasetMetadata(id="mot_simple_dataset")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, ind: int) -> tuple[InputType, TargetType, DatumMetadataType]:
        if not (0 <= ind < self.size):
            raise IndexError
        video_stream = (VideoFrameImpl(f) for f in range(self.n_frames))  # one-shot stream
        target = TrackingTargetImpl(self.n_frames)
        metadata: DatumMetadataType = {
            "id": ind,
            "height": 2,
            "width": 2,
            "time_base": Fraction(1, 30),
            "size": 0,
        }
        return video_stream, target, metadata


class ModelImpl:
    """Predicts a per-video track by consuming the whole input stream.

    Counting frames forces a full iteration of each input datum, so a spent-iterator
    input would yield a (wrong) zero-frame prediction instead of `n_frames` frames.
    """

    def __init__(self) -> None:
        self.metadata = ModelMetadata(id="mot_simple_model")

    def __call__(self, batch: Sequence[InputType]) -> Sequence[TargetType]:
        return [TrackingTargetImpl(sum(1 for _ in video_stream)) for video_stream in batch]


class MetricImpl:
    """Agreement metric: fraction of predictions whose frame count matches the target's."""

    metadata: MetricMetadata = {"id": "mot_simple_metric"}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(
        self,
        pred_batch: Sequence[TargetType],
        target_batch: Sequence[TargetType],
        _metadata_batch: Sequence[DatumMetadataType],
    ) -> None:
        for pred, target in zip(pred_batch, target_batch, strict=True):
            self._correct += int(len(pred.frame_tracks) == len(target.frame_tracks))
            self._total += 1

    def compute(self) -> Mapping[str, Any]:
        return {"frame_count_agreement": self._correct / self._total}
