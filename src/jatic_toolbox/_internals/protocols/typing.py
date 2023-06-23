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
    # minimum protocol for pillow-like Image?
    class Image(Protocol):
        format = None
        format_description = None

        @property
        def __array_interface__(self):
            ...


SupportsArray: TypeAlias = Union[ArrayLike, Sequence[ArrayLike]]


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


class HasDataImage(TypedDict):
    image: SupportsArray


class HasDataLabel(TypedDict):
    label: Union[int, SupportsArray, Sequence[int]]


class HasDataBoxes(TypedDict):
    boxes: SupportsArray


class ObjectDetectionData(HasDataBoxes):
    labels: Union[Sequence[int], SupportsArray]


class SupportsImageClassification(HasDataImage, HasDataLabel):
    ...


class SupportsObjectDetection(HasDataImage):
    objects: Union[ObjectDetectionData, Sequence[ObjectDetectionData]]


@runtime_checkable
class Dataset(Protocol[T_co]):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: Any) -> T_co:
        ...


VisionDataset: TypeAlias = Dataset[SupportsImageClassification]
ObjectDetectionDataset: TypeAlias = Dataset[SupportsObjectDetection]


#
# DataLoading
#


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
class HasLabel(Protocol):
    label: SupportsArray


@runtime_checkable
class HasBoxes(Protocol):
    boxes: SupportsArray


@runtime_checkable
class HasLogits(Protocol):
    logits: SupportsArray


@runtime_checkable
class HasProbs(Protocol):
    probs: SupportsArray


@runtime_checkable
class HasScores(Protocol):
    """
    Scores are predictions for either an image or detection box.

    ```python
    >>> import torch as tr
    >>> logits = tr.rand(3, 10) # batch size 3, 10 classes
    >>> probs = logits.softmax(dim=1)  # sums to 1 along dim=1
    >>> labels = probs.argmax(dim=1)  # predicted label for each score
    >>> scores = probs[:, labels]  # probability of the predicted label
    ```

    Attributes
    ----------
    scores : SupportsArray
        Scores are predictions for a single class. For example, in binary classification,
        scores are the probability of the positive class.

    labels : SupportsArray
        Labels are predicted label for each score. For example, in binary classification,
        labels are either 0 or 1.
    """

    scores: SupportsArray
    labels: SupportsArray


@runtime_checkable
class HasDetectionLogits(HasBoxes, HasLogits, Protocol):
    ...


@runtime_checkable
class HasDetectionProbs(HasProbs, HasBoxes, Protocol):
    ...


@runtime_checkable
class HasDetectionPredictions(HasBoxes, HasScores, Protocol):
    ...


"""
Models
"""


@runtime_checkable
class Model(Protocol):
    def get_labels(self) -> Sequence[str]:
        ...


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

# TODO: Add updates to support our protocols


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
