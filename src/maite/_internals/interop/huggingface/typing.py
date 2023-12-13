# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from torch import Tensor
from typing_extensions import Protocol, Self, runtime_checkable

from maite.protocols import ArrayLike, HasLogits

T = TypeVar("T", bound=ArrayLike)
T1 = TypeVar("T1")


@runtime_checkable
class HuggingFaceDataset(Protocol):
    @property
    def features(self) -> Mapping[str, Any]:
        ...

    def set_transform(self, transform: Callable[[Any], Any]) -> None:
        ...

    def __getitem__(self, key: Union[int, slice, Iterable[int]]) -> Dict[str, Any]:
        ...

    def __len__(self) -> int:
        ...


@dataclass
class HuggingFaceProbs:
    probs: Tensor


@dataclass
class HuggingFacePredictions:
    scores: Tensor
    labels: Optional[Union[Tensor, Sequence[Sequence[str]]]] = None


@dataclass
class HuggingFaceDetectorOutput(Dict[str, Any]):
    logits: Tensor
    boxes: Tensor


@dataclass
class HuggingFaceDetectorPredictions(Dict[str, Any]):
    scores: Sequence[Tensor]
    boxes: Sequence[Tensor]
    labels: Sequence[Tensor]


@dataclass
class HuggingFacePostProcessorInput(Dict[str, Any]):
    logits: Tensor
    pred_boxes: Tensor


@runtime_checkable
class HFOutput(Protocol):
    logits: Tensor
    pred_boxes: Tensor


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
        HuggingFaceDetectorPredictions,
        Sequence[HuggingFaceDetectorPredictions],
    ]:
        ...


@runtime_checkable
class HuggingFaceModule(Protocol):
    config: Any

    def parameters(self) -> Iterable[Tensor]:
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
