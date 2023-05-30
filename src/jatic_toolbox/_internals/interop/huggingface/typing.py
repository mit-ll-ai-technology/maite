from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch
from typing_extensions import (
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

from jatic_toolbox.protocols import ArrayLike, Dataset, HasDetectionLogits, HasLogits

T = TypeVar("T", bound=ArrayLike)


@runtime_checkable
class HuggingFaceDataset(Dataset[Mapping[str, Any]], Protocol):
    features: Mapping[str, Any]

    def set_transform(
        self, transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ) -> None:
        ...


class HasImagesDict(TypedDict):
    image: ArrayLike


@dataclass
class HuggingFacePostProcessedImages:
    probs: Union[ArrayLike, Sequence[ArrayLike]]
    labels: Optional[Union[ArrayLike, Sequence[Sequence[str]]]] = None


class HuggingFacePostProcessedDetections(TypedDict):
    scores: ArrayLike
    boxes: ArrayLike
    labels: ArrayLike


@dataclass
class BatchedHuggingFaceObjectDetectionOutput:
    boxes: Sequence[ArrayLike]
    scores: Sequence[ArrayLike]
    labels: Sequence[ArrayLike]


@dataclass
class HuggingFaceObjectDetectionOutput(Dict[str, ArrayLike]):
    boxes: ArrayLike
    scores: ArrayLike
    labels: ArrayLike


HFProcessedDetection: TypeAlias = List[Dict[Literal["scores", "labels", "boxes"], T]]


@runtime_checkable
class HFOutput(Protocol):
    logits: ArrayLike
    boxes: ArrayLike


class BatchFeature(Dict[str, ArrayLike]):
    def to(self, device: Union[str, int]) -> Self:  # pragma: no cover
        ...


@runtime_checkable
class HuggingFaceProcessor(Protocol):
    def __call__(
        self,
        images: Sequence[ArrayLike],
        return_tensors: Union[bool, str] = False,
        **kwargs: Any,
    ) -> BatchFeature:
        ...


@runtime_checkable
class HuggingFaceObjectDetectionPostProcessor(Protocol):
    def __call__(
        self,
        outputs: HasDetectionLogits,
        threshold: float,
        target_sizes: Any,
    ) -> Union[
        BatchedHuggingFaceObjectDetectionOutput,
        Sequence[HuggingFaceObjectDetectionOutput],
    ]:
        ...


@runtime_checkable
class HuggingFaceModule(Protocol):
    config: Any

    def to(self, device: Optional[Union[int, torch.device]]) -> Self:
        ...


@runtime_checkable
class HuggingFaceWithLogits(HuggingFaceModule, Protocol):
    def __call__(self, pixel_values: ArrayLike, **kwargs: Any) -> HasLogits:
        ...


@runtime_checkable
class HuggingFaceWithDetection(HuggingFaceModule, Protocol):
    def __call__(self, pixel_values: ArrayLike, **kwargs: Any) -> HasDetectionLogits:
        ...


@runtime_checkable
class HuggingFaceWrapper(Protocol):
    _dataset: HuggingFaceDataset

    def set_transform(
        self, transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ) -> None:
        self._dataset.set_transform(transform)
