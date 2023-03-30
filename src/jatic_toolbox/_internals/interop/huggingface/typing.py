from typing import Any, Dict, List, Sequence, TypeVar, Union

from typing_extensions import Literal, Protocol, Self, TypeAlias, runtime_checkable

from jatic_toolbox._internals.protocols import HasLogits
from jatic_toolbox.protocols import ArrayLike, HasDetectionLogits

T = TypeVar("T", bound=ArrayLike)


HFProcessedDetection: TypeAlias = List[Dict[Literal["scores", "labels", "boxes"], T]]


@runtime_checkable
class HFOutput(Protocol):
    logits: ArrayLike
    pred_boxes: ArrayLike


class BatchFeature(Dict[str, T]):
    def to(self, device: Union[str, int]) -> Self:
        ...


class HuggingFaceProcessor(Protocol):
    def __call__(
        self,
        images: Sequence[ArrayLike],
        return_tensors: Union[bool, str] = False,
        **kwargs: Any,
    ) -> BatchFeature[ArrayLike]:
        ...

    def post_process_object_detection(
        self,
        outputs: HasDetectionLogits[ArrayLike],
        threshold: float,
        target_sizes: Any,
    ) -> HFProcessedDetection[ArrayLike]:
        ...


class HuggingFaceWithLogits(Protocol):
    device: Union[int, str]

    def to(self, device: Union[str, int]) -> Self:
        ...

    def __call__(self, pixel_values: ArrayLike, **kwargs: Any) -> HasLogits[ArrayLike]:
        ...


class HuggingFaceWithDetection(Protocol):
    device: Union[int, str]

    def to(self, device: Union[str, int]) -> Self:
        ...

    def __call__(
        self, pixel_values: ArrayLike, **kwargs: Any
    ) -> HasDetectionLogits[ArrayLike]:
        ...
