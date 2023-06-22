from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import (
    ClassVar,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

from ..import_utils import is_pil_available

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_cont = TypeVar("T_cont", contravariant=True)
T2 = TypeVar("T2")
P = ParamSpec("P")


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
        ...


if is_pil_available():
    from PIL.Image import Image

else:

    class Image(Protocol):
        format = None
        format_description = None

        @property
        def __array_interface__(self):
            ...


A = TypeVar("A", bound=ArrayLike)


if TYPE_CHECKING:
    from dataclasses import Field  # provided by typestub but not generic at runtime
else:

    class Field(Protocol[T2]):
        name: str
        type: Type[T2]
        default: T2
        default_factory: Callable[[], T2]
        repr: bool
        hash: Optional[bool]
        init: bool
        compare: bool
        metadata: Mapping[str, Any]


class DataClass_(Protocol):
    # doesn't provide __init__, __getattribute__, etc.
    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]


@runtime_checkable
class DataClass(DataClass_, Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __getattribute__(self, __name: str) -> Any:
        ...

    def __setattr__(self, __name: str, __value: Any) -> None:
        ...


#
# Data Structures
#
# These protocols are TypedDicts.


class HasDataImage(TypedDict):
    image: Union[ArrayLike, Sequence[ArrayLike]]


class HasDataLabel(TypedDict):
    label: Union[int, ArrayLike, Sequence[ArrayLike], Sequence[int]]


class HasDataBoxes(TypedDict):
    boxes: Union[ArrayLike, Sequence[ArrayLike]]


class ObjectData(HasDataBoxes):
    # TODO: Should this be "label" or "labels"?
    labels: Union[Sequence[int], ArrayLike, Sequence[ArrayLike]]


class SupportsImageClassification(HasDataImage, HasDataLabel):
    ...


class SupportsObjectDetection(HasDataImage):
    objects: Union[ObjectData, Sequence[ObjectData]]


@runtime_checkable
class Dataset(Protocol[T_co]):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: Any) -> T_co:
        ...

    # def set_transform(self, transform: Callable[[T], T]) -> None:
    #     ...


VisionDataset: TypeAlias = Dataset[SupportsImageClassification]
ObjectDetectionDataset: TypeAlias = Dataset[SupportsObjectDetection]


"""
DataLoading
"""


class _DataLoaderIterator(Protocol[T_co]):
    def __next__(self) -> T_co:
        ...


@runtime_checkable
class DataLoader(Protocol[T_co]):
    def __iter__(self) -> _DataLoaderIterator[T_co]:
        ...


VisionDataLoader: TypeAlias = DataLoader[SupportsImageClassification]
ObjectDetectionDataLoader: TypeAlias = DataLoader[SupportsObjectDetection]


Augmentation: TypeAlias = Callable[[T], T]
"""
Augmentation Protocol.

Supports simple augmentations and adversarial attacks.

Parameters
----------
input: T

Returns
-------
output: T
"""


#
# Output Data Structures
#


@runtime_checkable
class HasLabel(Protocol[T]):
    label: T


@runtime_checkable
class HasObject(Protocol):
    object: Sequence[ObjectData]


@runtime_checkable
class HasBoxes(Protocol[T]):
    boxes: T


@runtime_checkable
class HasLogits(Protocol[T]):
    logits: T


@runtime_checkable
class HasProbs(Protocol[T]):
    probs: T


@runtime_checkable
class HasScores(Protocol[T]):
    scores: T
    labels: T


@runtime_checkable
class HasDetectionLogits(Protocol[T]):
    logits: T
    boxes: T


@runtime_checkable
class HasDetectionProbs(HasProbs[T], HasBoxes[T], Protocol[T]):
    ...


@runtime_checkable
class HasDetectionPredictions(Protocol[T]):
    scores: T
    boxes: T
    labels: T


"""
Models
"""


@runtime_checkable
class Model(Protocol):
    def get_labels(self) -> Sequence[str]:
        ...


@runtime_checkable
class ImageClassifier(Model, Protocol[T]):
    def __call__(
        self, data: HasDataImage
    ) -> Union[HasLogits[T], HasProbs[T], HasScores[T]]:
        ...


@runtime_checkable
class ObjectDetector(Model, Protocol[T]):
    def __call__(
        self, data: HasDataImage
    ) -> Union[HasDetectionLogits[T], HasDetectionProbs[T], HasDetectionPredictions[T]]:
        ...


"""
Metric protocol is based off of:
  - `torchmetrics`
  - `torcheval`
"""


@runtime_checkable
class Metric(Protocol):
    def reset(self) -> None:
        ...

    def update(self, *args: Any, **kwargs: Any) -> None:
        ...

    def compute(self) -> Any:
        ...

    def to(self, *args: Any, **kwargs: Any) -> Self:
        ...
