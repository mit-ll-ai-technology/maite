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

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_cont = TypeVar("T_cont", contravariant=True)
T2 = TypeVar("T2")
P = ParamSpec("P")


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
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


Preprocessor: TypeAlias = Callable[[T], T]
"""
PreProcessor Protocol.

Preprocessors are functions that take in a single input and return a single output.

Parameters
----------
input: Sequence[T]

Returns
-------
output: Sequence[T]
"""


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
class HasLabel(Protocol):
    label: Union[ArrayLike, Sequence[ArrayLike]]


@runtime_checkable
class HasObject(Protocol):
    object: Sequence[ObjectData]


@runtime_checkable
class HasBoxes(Protocol):
    boxes: Union[ArrayLike, Sequence[ArrayLike]]


@runtime_checkable
class HasLogits(Protocol):
    logits: Union[ArrayLike, Sequence[ArrayLike]]


@runtime_checkable
class HasProbs(Protocol):
    probs: Union[ArrayLike, Sequence[ArrayLike]]


@runtime_checkable
class HasScores(Protocol):
    scores: Union[ArrayLike, Sequence[ArrayLike]]
    label: Union[ArrayLike, Sequence[ArrayLike]]


@runtime_checkable
class HasDetectionLogits(HasLogits, HasBoxes, Protocol):
    ...


@runtime_checkable
class HasDetectionProbs(HasProbs, HasBoxes, Protocol):
    ...


@runtime_checkable
class HasDetectionPredictions(Protocol):
    scores: Union[ArrayLike, Sequence[ArrayLike]]
    boxes: Union[ArrayLike, Sequence[ArrayLike]]
    labels: Union[ArrayLike, Sequence[ArrayLike]]


"""
Post-Processing
"""

ClassifierPostProcessor: TypeAlias = Callable[
    [Union[HasLogits, HasProbs]], Union[HasProbs, HasScores]
]

DetectorPostProcessor: TypeAlias = Callable[
    [Union[HasDetectionLogits, HasDetectionProbs]],
    Union[HasProbs, HasDetectionPredictions],
]

PostProcessor: TypeAlias = Union[ClassifierPostProcessor, DetectorPostProcessor]


"""
Models
"""


@runtime_checkable
class Model(Protocol):
    def get_labels(self) -> Sequence[str]:
        ...


@runtime_checkable
class ModelWithPostProcessor(Model, Protocol):
    post_processor: PostProcessor


@runtime_checkable
class ModelWithPreProcessor(Model, Protocol):
    preprocessor: Preprocessor[
        Union[SupportsImageClassification, SupportsObjectDetection]
    ]


@runtime_checkable
class ImageClassifier(Model, Protocol):
    def __call__(self, data: HasDataImage) -> Union[HasLogits, HasProbs, HasScores]:
        ...


@runtime_checkable
class ObjectDetector(Model, Protocol):
    def __call__(
        self, data: HasDataImage
    ) -> Union[HasDetectionLogits, HasDetectionProbs, HasDetectionPredictions]:
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
