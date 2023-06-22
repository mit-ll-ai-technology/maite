from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TypeVar, Union

import torch
from typing_extensions import Protocol, Self, runtime_checkable

from jatic_toolbox.protocols import ArrayLike, Dataset, HasLogits

T = TypeVar("T", bound=ArrayLike)


@runtime_checkable
class HuggingFaceDataset(Dataset[Mapping[str, Any]], Protocol):
    features: Mapping[str, Any]

    def set_transform(
        self, transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ) -> None:
        ...


@dataclass
class HuggingFacePostProcessedImages:
    probs: Union[ArrayLike, Sequence[ArrayLike]]
    labels: Optional[Union[ArrayLike, Sequence[Sequence[str]]]] = None


@dataclass
class HuggingFaceDetectorOutput(Dict[str, Any]):
    logits: Union[ArrayLike, Sequence[ArrayLike]]
    boxes: Union[ArrayLike, Sequence[ArrayLike]]


@dataclass
class HuggingFaceObjectDetectionPostProcessedOutput(Dict[str, Any]):
    boxes: Union[ArrayLike, Sequence[ArrayLike]]
    scores: Union[ArrayLike, Sequence[ArrayLike]]
    labels: Union[ArrayLike, Sequence[ArrayLike]]


@dataclass
class HuggingFacePostProcessorInput(Dict[str, ArrayLike]):
    logits: Union[ArrayLike, Sequence[ArrayLike]]
    pred_boxes: Union[ArrayLike, Sequence[ArrayLike]]


@runtime_checkable
class HFOutput(Protocol):
    logits: Union[ArrayLike, Sequence[ArrayLike]]
    pred_boxes: Union[ArrayLike, Sequence[ArrayLike]]


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
        outputs: HFOutput,
        threshold: float,
        target_sizes: Any,
    ) -> Union[
        HuggingFaceObjectDetectionPostProcessedOutput,
        Sequence[HuggingFaceObjectDetectionPostProcessedOutput],
    ]:
        ...


@runtime_checkable
class HuggingFaceModule(Protocol):
    config: Any

    def to(self, device: Optional[Union[int, torch.device]]) -> Self:
        ...


@runtime_checkable
class HuggingFaceWithLogits(HuggingFaceModule, Protocol):
    def __call__(
        self, pixel_values: Union[ArrayLike, Sequence[ArrayLike]], **kwargs: Any
    ) -> HasLogits:
        ...


@runtime_checkable
class HuggingFaceWithDetection(HuggingFaceModule, Protocol):
    def __call__(
        self, pixel_values: Union[ArrayLike, Sequence[ArrayLike]], **kwargs: Any
    ) -> HFOutput:
        ...


@runtime_checkable
class HuggingFaceWrapper(Protocol):
    _dataset: HuggingFaceDataset

    def set_transform(
        self, transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ) -> None:
        self._dataset.set_transform(transform)
