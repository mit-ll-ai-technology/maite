# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    runtime_checkable,
)

from typing_extensions import ParamSpec, Self, TypeAlias, TypedDict

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
P = ParamSpec("P")


@runtime_checkable
class ArrayLike(Protocol):
    """
    A protocol for an array-like object.

    Examples
    --------

    Arrays like NumPy NDArray objects are `ArrayLike` along
    with PyTorch and JAX tensors.

    >>> import numpy as np
    >>> array_like: ArrayLike = np.ones((3, 224, 224))
    >>> isinstance(array_like, ArrayLike)
    True
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

    For ``TypedDict`` types, validation cannot be done
    with `isinstance`. The simple checks are

    >>> isinstance(data, dict)
    True
    >>> "image" in data
    True

    The toolbox comes with a helper function that can do
    both of these checks:

    >>> from maite.protocols import is_typed_dict
    >>> is_typed_dict(data, HasDataImage)
    True
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

    For ``TypedDict`` types, validation cannot be done
    with `isinstance`. The simple checks are

    >>> isinstance(data, dict)
    True
    >>> "label" in data
    True

    The toolbox comes with a helper function that can do
    both of these checks:

    >>> from maite.protocols import is_typed_dict
    >>> is_typed_dict(data, HasDataLabel)
    True
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

    For ``TypedDict`` types, validation cannot be done
    with `isinstance`. The simple checks are

    >>> isinstance(data, dict)
    True
    >>> "boxes" in data
    True

    The toolbox comes with a helper function that can do
    both of these checks:

    >>> from maite.protocols import is_typed_dict
    >>> is_typed_dict(data, HasDataBoxes)
    True
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

    For ``TypedDict`` types, validation cannot be done
    with `isinstance`. The simple checks are

    >>> isinstance(data, dict)
    True
    >>> "boxes" in data and "labels" in data
    True

    The toolbox comes with a helper function that can do
    both of these checks:

    >>> from maite.protocols import is_typed_dict
    >>> is_typed_dict(data, HasDataBoxesLabels)
    True
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

    For ``TypedDict`` types, validation cannot be done
    with `isinstance`. The simple checks are

    >>> isinstance(data, dict)
    True
    >>> "objects" in data
    True

    The toolbox comes with a helper function that can do
    both of these checks:

    >>> from maite.protocols import is_typed_dict
    >>> is_typed_dict(data, HasDataObjects)
    True
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

    For ``TypedDict`` types, validation cannot be done
    with `isinstance`. The simple checks are

    >>> isinstance(data, dict)
    True
    >>> "image" in data and "label" in data
    True

    The toolbox comes with a helper function that can do
    both of these checks:

    >>> from maite.protocols import is_typed_dict
    >>> is_typed_dict(data, SupportsImageClassification)
    True
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

    For ``TypedDict`` types, validation cannot be done
    with `isinstance`. The simple checks are

    >>> isinstance(data, dict)
    True
    >>> "image" in data and "objects" in data
    True

    The toolbox comes with a helper function that can do
    both of these checks:

    >>> from maite.protocols import is_typed_dict
    >>> is_typed_dict(data, SupportsObjectDetection)
    True
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

DataLoader = Iterable[T]


@runtime_checkable
class VisionDataLoader(DataLoader[SupportsImageClassification], Protocol):
    ...


@runtime_checkable
class ObjectDetectionDataLoader(DataLoader[SupportsObjectDetection], Protocol):
    ...


@runtime_checkable
class Augmentation(Protocol[P, T_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        """
        A generic augmenation protocol.

        Parameters
        ----------
        *args : P.args
            The arguments for the augmentation.
        **kwargs : P.kwargs
            The keyword arguments for the augmentation.

        Returns
        -------
        T_co : Covariant type
            The augmented data output.

        Examples
        --------

        Lets start with a simple augmentation that takes an image and returns an image.

        >>> import numpy as np
        >>> def my_augmentation(image: np.ndarray) -> np.ndarray:
        ...     return image * 0.1

        Now we want to define a workflow that uses this type of augmentation interface
        but allow users to create variations on the augmentation. We can use the
        `Augmentation` protocol to define the expected in inputs and outputs of the expected
        augmentation function.

        >>> def my_workflow(image: np.ndarray, augmentation: Augmentation[[np.ndarray], np.ndarray]) -> np.ndarray:
        ...     return augmentation(image)

        Now we can use the workflow with the augmentation.

        >>> image = np.ones((3, 224, 224))
        >>> my_workflow(image, my_augmentation)
        array([[[0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1],
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
    >>> data: HasLabel = FakeData(np.array([1]))
    >>> isinstance(data, HasLabel)
    True

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "label")
    True
    """

    @property
    def label(self) -> SupportsArray:
        ...


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
    >>> data: HasBoxes = FakeData(np.array([[0, 0, 1, 1]]))
    >>> isinstance(data, HasBoxes)
    True

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "boxes")
    True
    """

    @property
    def boxes(self) -> SupportsArray:
        ...


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
    >>> data: HasLogits = FakeData(np.array([0.5]))
    >>> isinstance(data, HasLogits)
    True

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "logits")
    True
    """

    @property
    def logits(self) -> SupportsArray:
        ...


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
    >>> isinstance(data, HasProbs)
    True

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "probs")
    True
    """

    @property
    def probs(self) -> SupportsArray:
        ...


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
    >>> isinstance(data, HasScores)
    True

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "scores") and hasattr(data, "labels")
    True
    """

    @property
    def scores(self) -> SupportsArray:
        ...

    @property
    def labels(self) -> SupportsArray:
        ...


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
    >>> isinstance(data, HasDetectionLogits)

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "logits") and hasattr(data, "boxes")
    True
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
    >>> isinstance(data, HasDetectionProbs)
    True

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "probs") and hasattr(data, "boxes")
    True
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
    >>> isinstance(data, HasDetectionPredictions)
    True

    The ``isinstance`` check is the same as doing the following:

    >>> hasattr(data, "scores") and hasattr(data, "labels") and hasattr(data, "boxes")
    True
    """

    ...


"""
Models
"""


@runtime_checkable
class Model(Protocol[P, T_co]):
    """
    A protocol for models.

    Methods
    -------
    __call__: Callable[P, T]
        Run inference on the data.  The input data can be anything
        while the output data is covariant.

    get_labels() -> Sequence[str]
        Returns the labels for the model.


    Examples
    --------

    Here we define a model that takes in a numpy NDArray and returns a numpy NDArray.

    >>> from numpy.typing import NDArray
    >>> NDArrayModelType = Model[[NDArray], NDArray]

    Here we define a model that takes in a numpy NDArray and returns a dataclass with logits.

    >>> LogitsModelType = Model[[NDArray], HasLogits]

    We can use different variations on inputs and outputs too. In the following
    model we take in a numpy NDArray and a dictionary and return a tuple of a numpy NDArray
    and a dataclass with probabilities.

    >>> from typing import Dict, Any, Tuple
    >>> TupleModelType = Model[[NDArray, Dict[str, Any]], Tuple[NDArray, HasProbs]]

    This protocol allows users to explicitly define the inputs and outputs of their models
    for a given task.  By using the following approach to defining a given workflow for a task
    we see that protocols help self-document the code.

    >>> def my_task(model: Model[[NDArray], HasProbs], data: NDArray) -> HasProbs:
    ...     return model(data)

    We see that the type hints for the model and data are self-documenting.  We can also
    see that the return type is a dataclass with probabilities.  This is a great way to
    document the code and make it easier for others to understand what is going on.

    For example if we define our model as:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyHasProbs:
    ...     probs: SupportsArray

    >>> from typing import Sequence
    >>> class MyModel:
    ...     def __call__(self, data: SupportsArray) -> MyHasProbs:
    ...         return MyHasProbs(data)
    ...     def get_labels(self) -> Sequence[str]:
    ...         return ["0", "1"]
    >>> my_model = MyModel()
    >>> isinstance(my_model, Model)
    True

    Then we can use it in our task as follows:

    >>> import numpy as np
    >>> my_task(my_model, np.array([0.5]))
    MyHasProbs(probs=array([0.5]))

    A typechecker would validate that `my_model` implements the interface
    defined in ``my_task``.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        """Run inference on the data."""
        ...

    def get_labels(self) -> Sequence[str]:
        """Returns the labels for the model."""
        ...


@runtime_checkable
class ImageClassifier(Model[P, Union[HasLogits, HasProbs, HasScores]], Protocol[P]):
    """
    An image classifier model that takes in an image and returns logits, probabilities, or scores.

    Methods
    -------
    __call__: Callable[[SupportsArray], Union[HasLogits, HasProbs, HasScores]]
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
    >>> model = FakeModel()
    >>> output: HasLogits = model(np.zeros((2, 3, 224, 224)))
    """

    ...


@runtime_checkable
class ObjectDetector(
    Model[
        P,
        Union[HasDetectionLogits, HasDetectionProbs, HasDetectionPredictions],
    ],
    Protocol[P],
):
    """
    An object detector model that takes in an image and returns logits, probabilities, or predictions.

    Methods
    -------
    __call__: Callable[P, Union[HasDetectionLogits, HasDetectionProbs, HasDetectionPredictions]]
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
    ...     boxes: SupportsArray
    >>> data: HasDetectionLogits = FakeData(np.array([0.5]), np.array([[0, 0, 1, 1]]))

    Create a fake model that returns logits.

    >>> from typing import Sequence
    >>> class FakeModel:
    ...     def __call__(self, data: SupportsArray) -> HasDetectionLogits:
    ...         return FakeData(np.array([0.5]), np.array([[0, 0, 1, 1]]))
    ...     def get_labels(self) -> Sequence[str]:
    ...         return ["cat", "dog"]
    >>> model: ObjectDetector[SupportsArray] = FakeModel()
    >>> output: HasDetectionLogits = model(np.zeros((2, 3, 224, 224)))

    One can also be more specific on the types:

    >>> from numpy.typing import NDArray
    >>> class FakeModel2:
    ...     def __call__(self, data: NDArray) -> HasDetectionLogits:
    ...         return FakeData(np.array([0.5]), np.array([[0, 0, 1, 1]]))
    ...     def get_labels(self) -> Sequence[str]:
    ...         return ["cat", "dog"]
    >>> model2: ObjectDetector[NDArray] = FakeModel2()
    >>> output: HasDetectionLogits = model2(np.zeros((2, 3, 224, 224)))
    """

    ...


"""
Metrics
"""


@runtime_checkable
class Metric(Protocol[P, T_co]):
    """
    A generic protocol for metrics.

    This protocol follows the approach outlined for both
    ``torchmetrics`` [1]_ and ``torcheval`` [2]_.  Their
    approach allows metrics to hold state, be transferred
    to a device, and utilize distributed frameworks by
    automatically handling the synchronization of state.

    Methods
    -------
    reset : Callable[[], None]
        Resets the metric.

    update : Callable[[P], None]
        Updates the metric.

    compute : Callable[[], T_co]
        Computes the metric.

    to : Callable[[Any], Self]
        Transfers the metric to a device.
        TODO: Determine if this method is required

    Examples
    --------

    Since we are working with a generic protocol we can define
    different metrics that handle different types of inputs and
    outputs.  For example, we can define a metric that takes in
    a numpy NDArray and returns a numpy NDArray.

    >>> from numpy.typing import NDArray
    >>> MyMetricProtocol = Metric[[NDArray], NDArray]

    An implementation of this metric would be as follows:

    >>> import numpy as np
    >>> from typing import Any
    >>> class MyMetric:
    ...     def __init__(self):
    ...         self._state = np.array([0.0])
    ...     def to(self, device: Any):
    ...         return self
    ...     def reset(self):
    ...         self._state = np.array([0.0])
    ...     def update(self, data: NDArray):
    ...         self._state += data
    ...     def compute(self) -> NDArray:
    ...         return self._state
    >>> my_metric = MyMetric()
    >>> isinstance(my_metric, Metric)
    True

    In general we will have inputs to the metric that are extracted
    from the dataset and the model output:

    >>> AccuracyLikeMetric = Metric[[HasLogits, HasDataLabel], ArrayLike]

    For example,

    >>> class AccuracyLike:
    ...     def __init__(self):
    ...         self._state = np.array([0.0])
    ...     def to(self, device: Any):
    ...         return self
    ...     def reset(self):
    ...         self._state = np.array([0.0])
    ...     def update(self, model_output: HasLogits, batch_data: HasDataLabel):
    ...         self._state += np.mean(model_output.logits == batch_data["label"])
    ...     def compute(self) -> ArrayLike:
    ...         return self._state

    Similar to the :class:``~maite.protocol.Model`` protocol, we can utilize this protocol to self-document
    our interfaces for different tasks.  For example, if we are working on a classification
    task we can define our metric as follows:

    >>> def classification_task(
    ...     model: Model[[SupportsArray], HasProbs],
    ...     data: DataLoader[SupportsImageClassification],
    ...     metric: Metric[[HasProbs, HasDataLabel], ArrayLike],
    ... ) -> ArrayLike:
    ...     metric.reset()
    ...     for batch in data:
    ...         model_output = model(batch["image"])
    ...         metric.update(model_output, batch)
    ...     return metric.compute()

    References
    ----------
    .. [1] https://torchmetrics.readthedocs.io/en/stable/
    .. [2] https://pytorch.org/torcheval/stable/
    """

    def reset(self) -> None:
        """Resets the metric."""
        ...

    def update(self, *args: P.args, **kwargs: P.kwargs) -> None:
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

    def compute(self) -> T_co:
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


#
# Providers
#


TaskName: TypeAlias = Literal["object-detection", "image-classification"]


@runtime_checkable
class ModelProvider(Protocol):
    def help(self, name: str) -> str:
        """
        Get information about the model such as:

            - Instructions for its use
            - Intended purpose
            - Any academic references

        Parameters
        ----------
        name: str
            The key that can be used to retrieve the model, returned by
            ``~ModelProvider.list_models``, same value used to retrieve the
            Model from ``~ModelProvider.load_model``

        Returns
        -------
        output: str
            The informational text.
        """
        ...

    def list_models(
        self,
        *,
        filter_str: str | List[str] | None = None,
        model_name: str | None = None,
        task: TaskName | None = None,
    ) -> Iterable[Any]:
        """
        List models for this provider.

        Parameters
        ----------
        filter_str : str | List[str] | None (default: None)
            A string or list of strings that contain complete or partial names for models.
        model_name : str | None (default: None)
            A string that contain complete or partial names for models.
        task : TaskName | None (default: None)
            A string or list of strings of tasks models were designed for, such as: "image-classification", "object-detection".
        **kwargs : Any
            Any keyword supported by this provider interface.

        Returns
        -------
        Iterable[Any]
            An iterable of model names.

        """
        ...

    def load_model(
        self, model_name: str, task: TaskName | None = None
    ) -> Model[P, T_co]:
        """
        Return a supported model.

        Parameters
        ----------
        model_name : str
            The `model_name` for the model (e.g., "microsoft/resnet-18").
        task : str | None
            The task for the model (e.g., "image-classification"). If None the task will be inferred from the model's interface
        **kwargs : Any
            Any keyword supported by provider interface.

        Returns
        -------
        Model
            A Model object that supports the given task.
        """
        ...


@runtime_checkable
class DatasetProvider(Protocol):
    def help(self, name: str) -> str:
        """
        Get information about the dataset such as:

            - Instructions for its use
            - Intended purpose
            - Any academic references

        Parameters
        ----------
        name: str
            The key that can be used to retrieve the dataset, returned by
            ``~DatasetProvider.list_datasets``, same value used to retrieve the
            Model from ``~DatasetProvider.load_dataset``

        Returns
        -------
        output: str
            The informational text.
        """
        ...

    def list_datasets(self) -> Iterable[str]:
        """
        List datasets for this provider.

        Parameters
        ----------
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Iterable[Any]
            An iterable of dataset names.

        """
        ...

    def load_dataset(
        self,
        *,
        dataset_name: str,
        task: TaskName | None = None,
        split: str | None = None,
    ) -> Dataset[T_co]:
        """
        Load dataset for a given provider.

        Parameters
        ----------
        dataset_name : str
            Name of dataset.
        task : TaskName | None (default: None)
            A string or list of strings of tasks dataset were designed for, such as: "image-classification", "object-detection".
        split : str | None (default: None)
            A string of split to load, such as: "train", "test", "validation".
            If None, the default split will be returned
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Dataset
            A dataset object that supports the given task.

        """
        ...


@runtime_checkable
class MetricProvider(Protocol):
    def help(self, name: str) -> str:
        """
        Get information about the Metric such as:

            - Instructions for its use
            - Intended purpose
            - Any academic references

        Parameters
        ----------
        name: str
            The key that can be used to retrieve the model, returned by
            ``~MetricProvider.list_metric``, same value used to retrieve the
            Model from ``~MetricProvider.load_metric``

        Returns
        -------
        output: str
            The informational text.
        """
        ...

    def list_metrics(self) -> Iterable[Any]:
        """
        List metrics for this provider.

        Parameters
        ----------
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Iterable[Any]
            An iterable of metric names.

        """
        ...

    def load_metric(self, metric_name: str) -> Metric[P, T_co]:
        """
        Return a Metric object.

        Parameters
        ----------
        metric_name : str
            The `metric_name` for the metric (e.g., "accuracy").
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Metric
            A Metric object.
        """
        ...


# Why an alias union, rather than a base protocol?  There are some applications where
# the concept of AnyProvider is a meaningful type hint However there is no useful
# declaration that could be assigned to a base type say with only the common `help`
# method as a requirement
AnyProvider: TypeAlias = Union[ModelProvider, MetricProvider, DatasetProvider]


#
# ArtifactHub
#
@runtime_checkable
class ArtifactHubEndpoint(Protocol):
    def __init__(self, path: Any):
        """Endpoints are initialized with a path argument specifying any information necessary to find the source via the endpoint's target api"""
        ...

    def get_cache_or_reload(self) -> str | os.PathLike[str]:
        """Create a local copy of the resource in the cache (if needed) and return a path suitable to locate the `hubconf.py` file"""
        ...

    def update_options(self) -> Self:
        """Update update any cached state used by the endpoint

        API tokens, validation options, etc.
        If none of these apply, the method may simply be a no-op
        """
        ...


ModelEntrypoint: TypeAlias = Callable[..., Model]
DatasetEntrypoint: TypeAlias = Callable[..., Dataset]
MetricEntrypoint: TypeAlias = Callable[..., Metric]
AnyEntrypoint: TypeAlias = Union[ModelEntrypoint, DatasetEntrypoint, MetricEntrypoint]
