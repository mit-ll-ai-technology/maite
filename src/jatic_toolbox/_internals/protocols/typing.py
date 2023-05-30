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

    @runtime_checkable
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


@runtime_checkable
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


class DataHasImage(TypedDict):
    image: ArrayLike


class DataHasLabel(TypedDict):
    label: int


class ImageClassifierData(DataHasImage, DataHasLabel):
    ...


class ObjectData(TypedDict):
    boxes: Sequence[ArrayLike]
    labels: Sequence[int]


class ObjectDetectionData(DataHasImage):
    objects: ObjectData


@runtime_checkable
class Dataset(Protocol[T_co]):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: Any) -> T_co:
        ...


VisionDataset: TypeAlias = Dataset[ImageClassifierData]
ObjectDetectionDataset: TypeAlias = Dataset[ObjectDetectionData]


"""
DataLoading
"""


class BatchedImages(TypedDict):
    image: Union[ArrayLike, Sequence[ArrayLike]]


class BatchedLabels(TypedDict):
    label: Union[ArrayLike, Sequence[ArrayLike], Sequence[int]]


class BatchedObjects(TypedDict):
    objects: Sequence[ObjectData]


class SupportsImageClassification(BatchedImages, BatchedLabels):
    ...


class SupportsObjectDetection(BatchedImages, BatchedObjects):
    ...


@runtime_checkable
class _DataLoaderIterator(Protocol[T_co]):
    def __next__(self) -> T_co:
        ...


@runtime_checkable
class DataLoader(Protocol[T_co]):
    def __iter__(self) -> _DataLoaderIterator[T_co]:
        ...


VisionDataLoader: TypeAlias = DataLoader[SupportsImageClassification]
ObjectDetectionDataLoader: TypeAlias = DataLoader[SupportsObjectDetection]
BatchedData = TypeVar(
    "BatchedData", SupportsImageClassification, SupportsObjectDetection
)


Preprocessor: TypeAlias = Callable[[Sequence[T]], Sequence[T]]
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


"""
Output Data Structures
"""


@runtime_checkable
class HasLabel(Protocol):
    label: ArrayLike


@runtime_checkable
class HasObject(Protocol):
    object: Sequence[ObjectData]


@runtime_checkable
class HasPredBoxes(Protocol):
    pred_boxes: ArrayLike


@runtime_checkable
class HasBoxes(Protocol):
    boxes: ArrayLike


@runtime_checkable
class HasLogits(Protocol):
    logits: ArrayLike


@runtime_checkable
class HasProbs(Protocol):
    probs: ArrayLike


@runtime_checkable
class HasScores(Protocol):
    scores: ArrayLike


@runtime_checkable
class HasDetectionLogits(HasLogits, HasPredBoxes, Protocol):
    ...


@runtime_checkable
class HasDetectionProbs(HasProbs, HasPredBoxes, Protocol):
    ...


@runtime_checkable
class HasLabelPredictions(Protocol):
    labels: ArrayLike


@runtime_checkable
class HasLogitsPredictions(HasLogits, HasLabelPredictions, Protocol):
    ...


@runtime_checkable
class HasProbPredictions(HasProbs, HasLabelPredictions, Protocol):
    ...


@runtime_checkable
class HasScorePredictions(HasScores, HasLabelPredictions, Protocol):
    ...


@runtime_checkable
class HasDetectionProbPredictions(Protocol):
    probs: Sequence[ArrayLike]
    boxes: Sequence[ArrayLike]


@runtime_checkable
class HasDetectionScorePredictions(Protocol):
    scores: Sequence[ArrayLike]
    boxes: Sequence[ArrayLike]
    labels: Sequence[ArrayLike]


"""
Post-Processing
"""

LP = Union[HasLogits, HasProbs]
LDP = Union[HasDetectionLogits, HasDetectionProbs]

ClassifierPostProcessor: TypeAlias = Callable[
    [Union[HasLogits, HasProbs]], Union[HasProbs, HasScorePredictions]
]

DetectorPostProcessor: TypeAlias = Callable[
    [LDP],
    Union[HasProbs, HasDetectionScorePredictions],
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
class ModelWithPostProcessor(Protocol):
    def get_labels(self) -> Sequence[str]:
        ...

    post_processor: PostProcessor


@runtime_checkable
class ModelWithPreProcessor(Protocol):
    def get_labels(self) -> Sequence[str]:
        ...

    preprocessor: Preprocessor[Union[ImageClassifierData, ObjectDetectionData]]


@runtime_checkable
class ImageClassifier(Model, Protocol):
    def __call__(
        self, data: SupportsImageClassification
    ) -> Union[HasLogits, HasProbs, HasScorePredictions]:
        ...


@runtime_checkable
class ObjectDetector(Model, Protocol):
    def __call__(
        self, data: SupportsObjectDetection
    ) -> Union[HasDetectionLogits, HasDetectionProbs, HasDetectionScorePredictions]:
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
