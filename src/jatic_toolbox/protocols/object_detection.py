from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol, Sequence

from typing_extensions import TypeAlias

from .array import ArrayLike

__all__ = ["ClassScores", "BoundingBox", "ObjectDetectionOutput", "ObjectDetection"]


ClassScores: TypeAlias = Dict[Any, float]


class BoundingBox(Protocol):
    """A bounding box protocol."""

    min_vertex: Sequence[float]
    max_vertex: Sequence[float]


@dataclass
class ObjectDetectionOutput:
    """An object detection output protocol."""

    # doc-ignore: NOQA
    boxes: Iterable[BoundingBox]
    scores: Iterable[ClassScores]


class ObjectDetection(Protocol):
    """An object-detection protocol."""

    def __call__(
        self, img_iter: Iterable[ArrayLike]
    ) -> Iterable[ObjectDetectionOutput]:
        ...
