from typing import Any, Iterator, Protocol, Sequence, TypeVar, Union, runtime_checkable

from typing_extensions import ParamSpec, Self, TypeAlias, TypedDict

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_cont = TypeVar("T_cont", contravariant=True)
T2 = TypeVar("T2")
P = ParamSpec("P")


@runtime_checkable
class ArrayLike(Protocol):
    """
    A protocol for an array-like object.

    Examples
    --------
    Create a fake array-like object.

    >>> import numpy as np
    >>> array_like: ArrayLike = np.ones((3, 224, 224))

    Validate with a type checker.

    >>> def supports_array_like(array_like: ArrayLike) -> None:
    ...     pass
    >>> supports_array_like(array_like)  # passes
    """

    def __array__(self) -> Any:
        ...


SupportsArray: TypeAlias = Union[ArrayLike, Sequence[ArrayLike]]


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
    """
    A protocol for vision datasets providing images and labels.

    Examples
    --------
    Create a fake dataset.

    >>> import numpy as np
    >>> data: SupportsImageClassification = {
    ...     "image": np.zeros((3, 224, 224)),
    ...     'label': np.array([1]),
    ... }

    Create a dataset with the fake data.

    >>> class FakeDataset:
    ...     def __len__(self) -> int:
    ...         return 1
    ...     def __getitem__(self, index: int) -> SupportsImageClassification:
    ...         return data
    >>> dataset: VisionDataset = FakeDataset()

    Validate with a type checker.

    >>> def supports_vision_dataset(dataset: VisionDataset) -> None:
    ...     pass
    >>> supports_vision_dataset(dataset)  # passes
    """

    ...


@runtime_checkable
class ObjectDetectionDataset(Dataset[SupportsObjectDetection], Protocol):
    """
    A protocol for object detection datasets providing images and detectuib objects with boxes and labels.

    Examples
    --------
    Create a fake dataset.

    >>> import numpy as np
    >>> data: SupportsObjectDetection = {
    ...     "image": np.zeros((3, 224, 224)),
    ...     "objects": {
    ...         "boxes": np.array([[0.0, 0.0, 1.0, 1.0]]),
    ...         "labels": np.array([1])
    ...     },
    ... }

    Create a dataset with the fake data.

    >>> class FakeDataset:
    ...     def __len__(self) -> int:
    ...         return 1
    ...     def __getitem__(self, index: int) -> SupportsObjectDetection:
    ...         return data
    >>> dataset: ObjectDetectionDataset = FakeDataset()

    Validate with a type checker.

    >>> def supports_object_detection_dataset(dataset: ObjectDetectionDataset) -> None:
    ...     pass
    >>> supports_object_detection_dataset(dataset)  # passes
    """

    ...


#
# DataLoading
#


@runtime_checkable
class DataLoader(Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        ...


@runtime_checkable
class VisionDataLoader(DataLoader[SupportsImageClassification], Protocol):
    ...


@runtime_checkable
class ObjectDetectionDataLoader(DataLoader[SupportsObjectDetection], Protocol):
    ...


@runtime_checkable
class Augmentation(Protocol[T]):
    def __call__(self, input: T) -> T:
        """
        Augmentation Protocol.

        Parameters
        ----------
        input : T
            Input data.

        Returns
        -------
        output : T
            Augmented data.

        Examples
        --------
        Create a fake augmentation.

        >>> def augment(data: SupportsImageClassification) -> SupportsImageClassification:
        ...     return data

        Validate with a type checker.

        >>> def supports_augmentation(augmentation: Augmentation) -> None:
        ...     pass
        >>> supports_augmentation(augment)  # passes
        """
        ...


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
    """
    Detection logits are logits for detection boxes.

    Attributes
    ----------
    logits : SupportsArray
        A batch of logits.
    boxes : SupportsArray
        A batch of boxes.

    Examples
    --------
    Create fake logits and boxes and add them to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     logits: SupportsArray
    ...     boxes: SupportsArray
    >>> data = FakeData(np.array([0.5]), np.array([[0, 0, 1, 1]]))

    Validate with a type checker.

    >>> def supports_has_detection_logits(data: HasDetectionLogits) -> None:
    ...     pass
    >>> supports_has_detection_logits(data)  # passes
    """

    ...


@runtime_checkable
class HasDetectionProbs(HasProbs, HasBoxes, Protocol):
    """
    Detection probabilities are probabilities for detection boxes.

    Attributes
    ----------
    probs : SupportsArray
        A batch of probabilities.
    boxes : SupportsArray
        A batch of boxes.

    Examples
    --------
    Create fake probabilities and boxes and add them to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     probs: SupportsArray
    ...     boxes: SupportsArray
    >>> data = FakeData(np.array([0.5]), np.array([[0, 0, 1, 1]]))

    Validate with a type checker.

    >>> def supports_has_detection_probs(data: HasDetectionProbs) -> None:
    ...     pass
    >>> supports_has_detection_probs(data)  # passes
    """

    ...


@runtime_checkable
class HasDetectionPredictions(HasBoxes, HasScores, Protocol):
    """
    Detection predictions are scores and labels for detection boxes.

    Attributes
    ----------
    scores : SupportsArray
        Scores are predictions for a single class. For example, in binary classification,
        scores are the probability of the positive class.
    labels : SupportsArray
        Labels are predicted label for each score. For example, in binary classification,
        labels are either 0 or 1.
    boxes : SupportsArray
        The predicted bounding boxes.

    Examples
    --------
    Create fake scores, labels, and boxes and add them to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     scores: SupportsArray
    ...     labels: SupportsArray
    ...     boxes: SupportsArray
    >>> data = FakeData(np.array([0.5]), np.array([1]), np.array([[0, 0, 1, 1]]))

    Validate with a type checker.

    >>> def supports_has_detection_predictions(data: HasDetectionPredictions) -> None:
    ...     pass
    >>> supports_has_detection_predictions(data)  # passes
    """

    ...


"""
Models
"""


@runtime_checkable
class Model(Protocol[T_cont, T_co]):
    """
    A protocol for models.

    Methods
    -------
    __call__(data: T_cont) -> T_co
        Run inference on the data.

    get_labels() -> Sequence[str]
        Returns the labels for the model.
    """

    def __call__(self, data: T_cont) -> T_co:
        """Run inference on the data."""
        ...

    def get_labels(self) -> Sequence[str]:
        """Returns the labels for the model."""
        ...


@runtime_checkable
class ImageClassifier(
    Model[SupportsArray, Union[HasLogits, HasProbs, HasScores]],
    Protocol,
):
    """
    An image classifier model that takes in an image and returns logits, probabilities, or scores.

    Methods
    -------
    __call__(data: SupportsArray) -> Union[HasLogits, HasProbs, HasScores]
        Run inference on the data.

    get_labels() -> Sequence[str]
        Returns the labels for the model.

    Examples
    --------
    Create a fake logits and add it to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     logits: SupportsArray
    >>> data = FakeData(np.array([0.5, 0.5]))

    Create a fake model that returns logits.

    >>> from typing import Sequence
    >>> class FakeModel:
    ...     def __call__(self, data: SupportsArray) -> HasLogits:
    ...         return FakeData(np.array([0.5, 0.5]))
    ...     def get_labels(self) -> Sequence[str]:
    ...         return ["cat", "dog"]
    >>> model: ImageClassifier = FakeModel()
    >>> output: HasLogits = model(np.zeros((2, 3, 224, 224)))

    Validate with a type checker.

    >>> def supports_image_classifier(model: ImageClassifier) -> None:
    ...     pass
    >>> supports_image_classifier(FakeModel())  # passes
    """

    ...


@runtime_checkable
class ObjectDetector(
    Model[
        SupportsArray,
        Union[HasDetectionLogits, HasDetectionProbs, HasDetectionPredictions],
    ],
    Protocol,
):
    """
    An object detector model that takes in an image and returns logits, probabilities, or predictions.

    Examples
    --------
    Create a fake logits and add it to a dataclass.

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeData:
    ...     logits: SupportsArray
    ...     boxes: SupportsArray
    >>> data: HasDetectionLogits = FakeData(np.array([0.5]), np.array([[0, 0, 1, 1]]))

    Create a fake model that returns logits.

    >>> from typing import Sequence
    >>> class FakeModel:
    ...     def __call__(self, data: SupportsArray) -> HasDetectionLogits:
    ...         return FakeData(np.array([0.5]), np.array([[0, 0, 1, 1]]))
    ...     def get_labels(self) -> Sequence[str]:
    ...         return ["cat", "dog"]
    >>> model: ObjectDetector = FakeModel()
    >>> output: HasDetectionLogits = model(np.zeros((2, 3, 224, 224)))

    Validate with a type checker.

    >>> def supports_object_detector(model: ObjectDetector) -> None:
    ...     pass
    >>> supports_object_detector(model)  # passes
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
    """
    A protocol for metrics.

    Methods
    -------
    reset() -> None
        Resets the metric.

    update(*args: Any, **kwargs: Any) -> None
        Updates the metric.

    compute() -> Any
        Computes the metric.

    to(*args: Any, **kwargs: Any) -> Self
        Transfers the metric to a device.

    Examples
    --------
    Create a fake metric.

    >>> from typing import Any
    >>> from typing_extensions import Self
    >>> class FakeMetric:
    ...     def reset(self) -> None:
    ...         pass
    ...     def update(self, *args: Any, **kwargs: Any) -> None:
    ...         pass
    ...     def compute(self) -> float:
    ...         return 0.5
    ...     def to(self, *args: Any, **kwargs: Any) -> Self:
    ...         return self
    >>> metric: Metric = FakeMetric()

    Validate with a type checker.

    >>> def supports_metric(metric: Metric) -> None:
    ...     pass
    >>> supports_metric(FakeMetric())  # passes
    """

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
