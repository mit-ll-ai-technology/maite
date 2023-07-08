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


@runtime_checkable
class HuggingFaceWrapper(Protocol):
    _dataset: HuggingFaceDataset

    def set_transform(
        self, transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ) -> None:
        self._dataset.set_transform(transform)
