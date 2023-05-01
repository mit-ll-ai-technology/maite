from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, Union

from typing_extensions import (
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

from jatic_toolbox._internals.protocols import HasLogits
from jatic_toolbox.protocols import ArrayLike, HasDetectionLogits, HasObjectDetections

T = TypeVar("T", bound=ArrayLike)


class HasImagesDict(TypedDict):
    image: ArrayLike


@dataclass
class HuggingFacePostProcessedImages:
    probs: Union[ArrayLike, Sequence[ArrayLike]]
    labels: Optional[Union[ArrayLike, Sequence[ArrayLike]]]


class HuggingFacePostProcessedDetections(TypedDict):
    scores: ArrayLike
    labels: ArrayLike
    boxes: ArrayLike


@dataclass
class HuggingFaceObjectDetectionOutput(Generic[T]):
    boxes: Union[T, Sequence[T]]
    scores: Union[T, Sequence[T]]
    labels: Optional[Union[T, Sequence[T]]]


HFProcessedDetection: TypeAlias = List[Dict[Literal["scores", "labels", "boxes"], T]]


@runtime_checkable
class HFOutput(Protocol):
    logits: ArrayLike
    pred_boxes: ArrayLike


class BatchFeature(Dict[str, T]):
    def to(self, device: Union[str, int]) -> Self:  # pragma: no cover
        ...


class HuggingFaceProcessor(Protocol):
    def __call__(
        self,
        images: Sequence[ArrayLike],
        return_tensors: Union[bool, str] = False,
        **kwargs: Any,
    ) -> BatchFeature[ArrayLike]:
        ...


class HuggingFaceObjectDetectionPostProcessor(Protocol):
    def __call__(
        self,
        outputs: HasDetectionLogits[ArrayLike],
        threshold: float,
        target_sizes: Any,
    ) -> Union[
        HasObjectDetections[ArrayLike], Sequence[HuggingFacePostProcessedDetections]
    ]:
        ...


class HuggingFaceWithLogits(Protocol):
    device: Union[int, str]
    config: Any

    def to(self, device: Union[str, int]) -> Self:
        ...

    def __call__(self, pixel_values: ArrayLike, **kwargs: Any) -> HasLogits[ArrayLike]:
        ...


class HuggingFaceWithDetection(Protocol):
    device: Union[int, str]
    config: Any

    def to(self, device: Union[str, int]) -> Self:
        ...

    def __call__(
        self, pixel_values: ArrayLike, **kwargs: Any
    ) -> HasDetectionLogits[ArrayLike]:
        ...
