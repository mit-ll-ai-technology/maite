from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from typing_extensions import ParamSpec, Self, TypeAlias, TypedDict

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
    """
    A dictionary that contains an image.

    Attributes
    ----------
    image : SupportsArray
        An image or batch of images.

    Examples
    --------

    Create a fake image and add it to a dictionary.

    >>> import numpy as np
    >>> image = np.ones((3, 224, 224))
    >>> data: HasDataImage = {'image': image}

    Validate with a type checker.

    >>> def supports_has_data_image(data: HasDataImage) -> None:
    ...     pass
    >>> supports_has_data_image(data)  # passes
    """

    image: SupportsArray


class HasDataLabel(TypedDict):
    """
    A dictionary that contains a label.

    Attributes
    ----------
    label : Union[int, SupportsArray, Sequence[int]]
        A label or batch of labels.

    Examples
    --------
    Create a fake label and add it to a dictionary.

    >>> import numpy as np
    >>> label = np.array(1)
    >>> data: HasDataLabel = {'label': label}

    Validate with a type checker.

    >>> def supports_has_data_label(data: HasDataLabel) -> None:
    ...     pass
    >>> supports_has_data_label(data)  # passes
    """

    label: Union[int, SupportsArray, Sequence[int]]


class HasDataBoxes(TypedDict):
    """
    A dictionary that contains boxes.

    Attributes
    ----------
    boxes : SupportsArray
        A batch of boxes.

    Examples
    --------
    Create a fake boxes and add it to a dictionary.

    >>> import numpy as np
    >>> boxes = np.array([[0, 0, 1, 1]])
    >>> data: HasDataBoxes = {'boxes': boxes}

    Validate with a type checker.

    >>> def supports_has_data_boxes(data: HasDataBoxes) -> None:
    ...     pass
    >>> supports_has_data_boxes(data)  # passes
    """

    boxes: SupportsArray


class HasDataBoxesLabels(HasDataBoxes):
    """
    A dictionary that contains boxes and labels.
    Typically this is used for object detection.

    Attributes
    ----------
    boxes : SupportsArray
        A batch of boxes.
    labels : Sequence[int] | SupportsArray
        A batch of labels.

    Examples
    --------
    Create fake boxes and labels and add them to a dictionary.

    >>> import numpy as np
    >>> boxes = np.array([[0, 0, 1, 1]])
    >>> labels = np.array([1])
    >>> data: HasDataBoxesLabels = {'boxes': boxes, 'labels': labels}

    Validate with a type checker.

    >>> def supports_has_data_boxes_labels(data: HasDataBoxesLabels) -> None:
    ...     pass
    >>> supports_has_data_boxes_labels(data)  # passes
    """

    labels: Union[Sequence[int], SupportsArray]


class HasDataObjects(TypedDict):
    """
    A dictionary that contains objects (boxes and labels) for object detection.

    Attributes
    ----------
    objects : HasDataBoxesLabels | Sequence[HasDataBoxesLabels]

    Examples
    --------
    Create fake boxes and labels and add them to a dictionary.

    >>> import numpy as np
    >>> boxes = np.array([[0.0, 0., 1., 1.]])
    >>> labels = np.array([1])
    >>> data: HasDataBoxesLabels = {'boxes': boxes, 'labels': labels}
    >>> objects: HasDataObjects = {'objects': data}

    Validate with a type checker.

    >>> def supports_has_data_objects(data: HasDataObjects) -> None:
    ...     pass
    >>> supports_has_data_objects(objects)  # passes
    """

    objects: Union[HasDataBoxesLabels, Sequence[HasDataBoxesLabels]]


class SupportsImageClassification(HasDataImage, HasDataLabel):
    """
    A dictionary that contains an image and label.

    Attributes
    ----------
    image : SupportsArray
        An image or batch of images.
    label : int | SupportsArray, Sequence[int]
        A label or batch of labels.

    Examples
    --------
    Create a fake image and label and add them to a dictionary.

    >>> import numpy as np
    >>> image = np.zeros((3, 224, 224))
    >>> label = np.array([1])
    >>> data: SupportsImageClassification = {'image': image, 'label': label}

    Validate with a type checker.

    >>> def supports_has_data_image_classification(data: SupportsImageClassification) -> None:
    ...     pass
    >>> supports_has_data_image_classification(data)  # passes
    """

    ...


class SupportsObjectDetection(HasDataImage, HasDataObjects):
    """
    A dictionary that contains an image and objects (boxes and labels).

    Attributes
    ----------
    image : SupportsArray
        An image or batch of images.
    objects : HasDataBoxesLabels | Sequence[HasDataBoxesLabels]
        A batch of objects.

    Examples
    --------
    Create fake boxes and labels and add them to a dictionary.

    >>> import numpy as np
    >>> image = np.zeros((3, 224, 224))
    >>> boxes = np.array([[0.0, 0., 1., 1.]])
    >>> labels = np.array([1])
    >>> data: HasDataBoxesLabels = {'boxes': boxes, 'labels': labels}
    >>> batch: SupportsObjectDetection = {'image': image, 'objects': data}

    Validate with a type checker.

    >>> def supports_has_data_objects(data: SupportsObjectDetection) -> None:
    ...     pass
    >>> supports_has_data_objects(batch)  # passes
    """

    ...


@runtime_checkable
class Dataset(Protocol[T_co]):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: Any) -> T_co:
        ...


@runtime_checkable
class VisionDataset(Dataset[SupportsImageClassification], Protocol):
    ...


@runtime_checkable
class ObjectDetectionDataset(Dataset[SupportsObjectDetection], Protocol):
    ...


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
    """
    A protocol for a data structure that must contain a label.

    Attributes
    ----------
    label : SupportsArray
        A label or batch of labels.

    Examples
    --------
    Create a fake label and add it to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     label: SupportsArray
    >>> data = FakeData(np.array([1]))

    Validate with a type checker.

    >>> def supports_has_label(data: HasLabel) -> None:
    ...     pass
    >>> supports_has_label(data)  # passes
    """

    label: SupportsArray


@runtime_checkable
class HasBoxes(Protocol):
    """
    A protocol for a data structure that must contain boxes.

    Attributes
    ----------
    boxes : SupportsArray
        A batch of boxes.

    Examples
    --------
    Create a fake boxes and add it to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     boxes: SupportsArray
    >>> data = FakeData(np.array([[0, 0, 1, 1]]))

    Validate with a type checker.

    >>> def supports_has_boxes(data: HasBoxes) -> None:
    ...     pass
    >>> supports_has_boxes(FakeData)  # passes
    """

    boxes: SupportsArray


@runtime_checkable
class HasLogits(Protocol):
    """
    A protocol for a data structure that must contain logits.

    Attributes
    ----------
    logits : SupportsArray
        A batch of logits.

    Examples
    --------
    Create a fake logits and add it to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     logits: SupportsArray
    >>> data = FakeData(np.array([0.5]))

    Validate with a type checker.

    >>> def supports_has_logits(data: HasLogits) -> None:
    ...     pass
    >>> supports_has_logits(data)  # passes
    """

    logits: SupportsArray


@runtime_checkable
class HasProbs(Protocol):
    """
    A protocol for a data structure that must contain probabilities.

    Attributes
    ----------
    probs : SupportsArray
        A batch of probabilities.

    Examples
    --------
    Create a fake probabilities and add it to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     probs: SupportsArray
    >>> data = FakeData(np.array([0.5]))

    Validate with a type checker.

    >>> def supports_has_probs(data: HasProbs) -> None:
    ...     pass
    >>> supports_has_probs(data)  # passes
    """

    probs: SupportsArray


@runtime_checkable
class HasScores(Protocol):
    """
    Scores are predictions for either an image or detection box.

    Attributes
    ----------
    scores : SupportsArray
        Scores are predictions for a single class. For example, in binary classification,
        scores are the probability of the positive class.

    labels : SupportsArray
        Labels are predicted label for each score. For example, in binary classification,
        labels are either 0 or 1.

    Examples
    --------
    Create fake scores and labels and add them to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     scores: SupportsArray
    ...     labels: SupportsArray
    >>> data = FakeData(np.array([0.5]), np.array([1]))

    Validate with a type checker.

    >>> def supports_has_scores(data: HasScores) -> None:
    ...     pass
    >>> supports_has_scores(data)  # passes
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
    """
    A protocol for models.

    Methods
    -------
    get_labels
        Returns the labels for the model.
    """

    def get_labels(self) -> Sequence[str]:
        """Returns the labels for the model."""
        ...


@runtime_checkable
class ImageClassifier(Model, Protocol):
    """A protocol for image classifiers."""

    def __call__(self, data: SupportsArray) -> Union[HasLogits, HasProbs, HasScores]:
        """
        Call the model.

        Parameters
        ----------
        data : SupportsArray
            An image or batch of images.

        Returns
        -------
        output : HasLogits | HasProbs | HasScores
            The output of the model.
        """
        ...


@runtime_checkable
class ObjectDetector(Model, Protocol):
    """A protocol for object detectors."""

    def __call__(
        self, data: SupportsArray
    ) -> Union[HasDetectionLogits, HasDetectionProbs, HasDetectionPredictions]:
        """
        Call the model.

        Parameters
        ----------
        data : SupportsArray
            An image or batch of images.

        Returns
        -------
        output : HasDetectionLogits | HasDetectionProbs | HasDetectionPredictions
            The output of the model.
        """
        ...


"""
Metric protocol is based off of:
  - `torchmetrics`
  - `torcheval`
"""

# TODO: Add updates to support our protocols


@runtime_checkable
class Metric(Protocol):
    """A protocol for metrics."""

    def reset(self) -> None:
        """Resets the metric."""
        ...

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Updates the metric.

        Parameters
        ----------
        *args : Any
            Positional arguments.
        **kwargs : Any
            Keyword arguments.
        """
        ...

    def compute(self) -> Any:
        """
        Computes the metric.

        Returns
        -------
        output : Any
            The computed metric.
        """
        ...

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """
        Transfers the metric to a device.

        Parameters
        ----------
        *args : Any
            Positional arguments.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        output : Self
            The metric on the desired device.
        """
        ...
