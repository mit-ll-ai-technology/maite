from typing import Any, Dict, List, Sequence, TypeVar, Union

from torch import Tensor
from typing_extensions import Literal, Protocol, TypeAlias, runtime_checkable

from jatic_toolbox._internals.protocols import HasLogits
from jatic_toolbox.protocols import ArrayLike, HasDetectionLogits

T = TypeVar("T", bound=ArrayLike)


HFProcessedDetection: TypeAlias = List[Dict[Literal["scores", "labels", "boxes"], T]]


@runtime_checkable
class HFOutput(Protocol):
    logits: Tensor
    pred_boxes: Tensor


class BatchFeature(Dict[str, T]):
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
        self, outputs: HasDetectionLogits[Tensor], threshold: float, target_sizes: Any
    ) -> HFProcessedDetection[Tensor]:
        ...


class HuggingFaceWithLogits(Protocol):
    def __call__(self, pixel_values: ArrayLike, **kwargs: Any) -> HasLogits[Tensor]:
        ...


class HuggingFaceWithDetection(Protocol):
    def __call__(
        self, pixel_values: ArrayLike, **kwargs: Any
    ) -> HasDetectionLogits[Tensor]:
        ...
