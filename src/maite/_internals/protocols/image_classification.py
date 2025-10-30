# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for image_classification
from __future__ import annotations

from typing import Protocol

from typing_extensions import TypeAlias

from maite._internals.protocols import generic as gen
from maite.protocols import ArrayLike, DatumMetadata

# In below, the dimension names/meanings used are:
#
# N  - image instance
# H  - image height
# W  - image width
# C  - image channel
# Cl - classification label (one-hot for ground-truth label; probabilities or logits for predictions)

InputType: TypeAlias = ArrayLike
"""ArrayLike following (C, H, W) shape semantics"""

TargetType: TypeAlias = ArrayLike  # shape (Cl,)
"""ArrayLike following (Cl,) shape semantics (where 'Cl' refers to number of target classes)"""

DatumMetadataType: TypeAlias = DatumMetadata
"""TypedDict that requires a readonly 'id' field of type `int|str`"""

Datum: TypeAlias = tuple[InputType, TargetType, DatumMetadataType]

# Initialize component classes based on generic and Input/Target/Metadata types


class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
    """
    A dataset protocol for image classification AI problem providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond
    to individual examples (as opposed to batches).

    Indexing into or iterating over an image_classification dataset returns a
    `tuple` of types `ArrayLike`, `ArrayLike`, and `DatumMetadata`.
    These correspond to the model input type, model target type, and datum-level
    metadata, respectively. The `ArrayLike` protocol implementers associated with
    model input and model target types are expected to follow (C, H, W) shape semantics.

    Methods
    -------

    __getitem__(ind: int) -> tuple[ArrayLike, ArrayLike, DatumMetadata]
        Provide map-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata type,
        respectively.

    __len__() -> int
        Return the number of data elements in the dataset.

    Attributes
    ----------

    metadata : DatasetMetadata
        A typed dictionary containing at least an 'id' field of type str

    Examples
    --------

    We create a dummy set of data and use it to create a class that implements
    this lightweight dataset protocol:

    >>> import numpy as np
    >>> from typing import Any
    >>> from typing_extensions import TypedDict
    >>> from maite.protocols import ArrayLike, DatasetMetadata

    Assume we have 5 classes, 10 datapoints, and 10 target labels, and that we want
    to simply have an integer 'id' field in each datapoint's metadata:

    >>> N_CLASSES: int = 5
    >>> N_DATUM: int = 10
    >>> images: list[np.ndarray] = [np.random.rand(3, 32, 16) for _ in range(N_DATUM)]
    >>> targets: np.ndarray = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, N_DATUM)]

    We can type our datum metadata as a maite.protocols DatumMetadata, or define our
    own TypedDict with additional fields

    >>> class MyDatumMetadata(DatumMetadata):
    ...     hour_of_day: float
    >>> datum_metadata = [
    ...     MyDatumMetadata(id=i, hour_of_day=np.random.rand() * 24)
    ...     for i in range(N_DATUM)
    ... ]

    Constructing a compliant dataset just involves a simple wrapper that fetches
    individual datapoints, where a datapoint is a single image, target, metadata 3-tuple.

    >>> class ImageDataset:
    ...     def __init__(
    ...         self,
    ...         dataset_name: str,
    ...         index2label: dict[int, str],
    ...         images: list[np.ndarray],
    ...         targets: np.ndarray,
    ...         datum_metadata: list[MyDatumMetadata],
    ...     ):
    ...         self.images = images
    ...         self.targets = targets
    ...         self.metadata = DatasetMetadata(
    ...             {"id": dataset_name, "index2label": index2label}
    ...         )
    ...         self._datum_metadata = datum_metadata
    ...
    ...     def __len__(self) -> int:
    ...         return len(images)
    ...
    ...     def __getitem__(
    ...         self, ind: int
    ...     ) -> tuple[np.ndarray, np.ndarray, MyDatumMetadata]:
    ...         return self.images[ind], self.targets[ind], self._datum_metadata[ind]

    We can instantiate this class and typehint it as an image_classification.Dataset.
    By using typehinting, we permit a static typechecker to verify protocol compliance.

    >>> from maite.protocols import image_classification as ic
    >>> dataset: ic.Dataset = ImageDataset(
    ...     "a_dataset",
    ...     {i: f"class_name_{i}" for i in range(N_CLASSES)},
    ...     images,
    ...     targets,
    ...     datum_metadata,
    ... )

    Note that when writing a Dataset implementer, return types may be narrower than the
    return types promised by the protocol (np.ndarray is a subtype of ArrayLike), but
    the argument types must be at least as general as the argument types promised by the
    protocol.
    """

    ...


class FieldwiseDataset(
    Dataset, gen.FieldwiseDataset[InputType, TargetType, DatumMetadataType], Protocol
):
    """
    A specialization of Dataset protocol (i.e., a subprotocol) that specifies additional
    accessor methods for getting input, target, and metadata individually.

    Methods
    -------
    __getitem__(ind: int) -> tuple[InputType, TargetType, DatumMetadataType]
        Provide map-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata type,
        respectively.

    __len__() -> int
        Return the number of data elements in the dataset.

    get_input(index: int, /) -> InputType:
        Get input at the given index.

    get_target(index: int, /) -> TargetType:
        Get target at the given index.

    get_metadata(index: int, /) -> DatumMetadataType:
        Get metadata at the given index.

    Examples
    --------
    We create a dummy set of data and use it to create a class that implements
    the dataset protocol:

    >>> import numpy as np
    >>> from typing import Any
    >>> from typing_extensions import TypedDict
    >>> from maite.protocols import ArrayLike, DatasetMetadata

    Assume we have 5 classes, 10 datapoints, and 10 target labels, and that we want
    to simply have an integer 'id' field in each datapoint's metadata:

    >>> N_CLASSES: int = 5
    >>> N_DATUM: int = 10
    >>> images: list[np.ndarray] = [np.random.rand(3, 32, 16) for _ in range(N_DATUM)]
    >>> targets: np.ndarray = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, N_DATUM)]

    We can type our datum metadata as a maite.protocols DatumMetadata, or define our
    own TypedDict with additional fields

    >>> class MyDatumMetadata(DatumMetadata):
    ...     hour_of_day: float
    >>> datum_metadata = [
    ...     MyDatumMetadata(id=i, hour_of_day=np.random.rand() * 24)
    ...     for i in range(N_DATUM)
    ... ]

    Constructing a compliant dataset just involves a simple wrapper that fetches
    individual datapoints, where a datapoint is a single image, target, metadata 3-tuple.

    >>> class ImageDataset:
    ...     def __init__(
    ...         self,
    ...         dataset_name: str,
    ...         index2label: dict[int, str],
    ...         images: list[np.ndarray],
    ...         targets: np.ndarray,
    ...         datum_metadata: list[MyDatumMetadata],
    ...     ):
    ...         self.images = images
    ...         self.targets = targets
    ...         self.metadata = DatasetMetadata(
    ...             {"id": dataset_name, "index2label": index2label}
    ...         )
    ...         self._datum_metadata = datum_metadata
    ...
    ...     def get_input(self, index, /) -> np.ndarray:
    ...         return self.images[index]
    ...
    ...     def get_target(self, index, /) -> np.ndarray:
    ...         return self.targets[index]
    ...
    ...     def get_metadata(self, index, /) -> MyDatumMetadata:
    ...         return self._datum_metadata[index]
    ...
    ...     def __len__(self) -> int:
    ...         return len(images)
    ...
    ...     def __getitem__(
    ...         self, ind: int
    ...     ) -> tuple[np.ndarray, np.ndarray, MyDatumMetadata]:
    ...         return self.images[ind], self.targets[ind], self._datum_metadata[ind]

    We can instantiate this class and typehint it as an image_classification.Dataset.
    By using typehinting, we permit a static typechecker to verify protocol compliance.

    >>> from maite.protocols import image_classification as ic
    >>> dataset: ic.FieldwiseDataset = ImageDataset(
    ...     "a_dataset",
    ...     {i: f"class_name_{i}" for i in range(N_CLASSES)},
    ...     images,
    ...     targets,
    ...     datum_metadata,
    ... )

    """


class DataLoader(gen.DataLoader[InputType, TargetType, DatumMetadataType], Protocol):
    """
    A dataloader protocol for the image classification AI problem providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `Sequence[ArrayLike]` (elements of shape `(C, H, W)`),
    `Sequence[ArrayLike]` (elements shape `(Cl, )`), and `Sequence[DatumMetadata]`,
    which correspond to model input batch, model target type batch, and a datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.

    Methods
    -------

    __iter__ -> Iterator[tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadata]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as `Sequence[ArrayLike]`), model target batch (as
        `Sequence[ArrayLike]`), and batched datum-level metadata
        (as `Sequence[DatumMetadata]`), respectively.

    """


class Model(gen.Model[InputType, TargetType], Protocol):
    """
    A model protocol for the image classification AI problem.

    Implementers must provide a `__call__` method that operates on a batch of model
    inputs (as `Sequence[ArrayLike]`) and returns a batch of model targets (as
    `Sequence[ArrayLike]`)

    Methods
    -------

    __call__(input_batch: Sequence[ArrayLike]) -> Sequence[ArrayLike]
        Make a model prediction for inputs in input batch. Input batch is expected to
        be `Sequence[ArrayLike]` with each element of shape `(C, H, W)`. Target batch
        is expected to be `Sequence[ArrayLike]` with each element of shape `(Cl,)`.

    Attributes
    ----------

    metadata : ModelMetadata
        A typed dictionary containing at least an 'id' field of type str

    Examples
    --------

    We create a multinomial logistic regression classifier for a CIFAR-10-like dataset
    with 10 classes and shape-(3, 32, 32) images.

    >>> import maite.protocols.image_classification as ic
    >>> import numpy as np
    >>> import numpy.typing as npt
    >>> from maite.protocols import ArrayLike, ModelMetadata
    >>> from typing import Sequence

    Creating a MAITE-compliant model involves writing a `__call__` method that takes a
    batch of inputs and returns a batch of predictions (probabilities or logits).

    >>> class LinearClassifier:
    ...     def __init__(self) -> None:
    ...         # Set up required metadata attribute using the default `ModelMetadata` type,
    ...         # using class name for the ID
    ...         self.metadata: ModelMetadata = {"id": self.__class__.__name__}
    ...
    ...         # Initialize weights
    ...         rng = np.random.default_rng(12345678)
    ...         num_classes = 10
    ...         flattened_size = 3 * 32 * 32
    ...         self.weights = -0.2 + 0.4 * rng.random((flattened_size, num_classes))
    ...         self.bias = -0.2 + 0.4 * rng.random((1, num_classes))
    ...
    ...     def __call__(self, batch: Sequence[ArrayLike]) -> Sequence[npt.NDArray]:
    ...         # Convert each element in batch to ndarray, flatten,
    ...         # then combine into 4D array of shape-(N, C, H, W)
    ...         batch_np = np.vstack([np.asarray(x).flatten() for x in batch])
    ...
    ...         # Send input batch through model
    ...         out = batch_np @ self.weights + self.bias
    ...         out = np.exp(out) / np.sum(
    ...             np.exp(out), axis=1, keepdims=True
    ...         )  # softmax
    ...
    ...         # Restructure to sequence of shape-(10,) probabilities
    ...         return [row for row in out]

    We set up a test batch, instantiate the model, and apply it to the batch.

    >>> batch_size = 8
    >>> rng = np.random.default_rng(12345678)
    >>> batch: Sequence[ArrayLike] = [
    ...     -0.2 + 0.4 * rng.random((3, 32, 32)) for _ in range(batch_size)
    ... ]
    >>>
    >>> model: ic.Model = LinearClassifier()
    >>> out = model(batch)

    We can now show the class probabilities returned by the model for each image in the batch.

    >>> np.set_printoptions(
    ...     floatmode="fixed", precision=2
    ... )  # for reproducible output for doctest
    >>> for probs in out:  # doctest: +NORMALIZE_WHITESPACE
    ...     print(np.round(probs, 2))
    [0.16 0.10 0.16 0.14 0.04 0.02 0.06 0.04 0.17 0.10]
    [0.21 0.16 0.04 0.07 0.08 0.05 0.09 0.03 0.18 0.09]
    [0.15 0.11 0.13 0.11 0.09 0.09 0.07 0.04 0.19 0.02]
    [0.04 0.08 0.14 0.07 0.12 0.20 0.11 0.06 0.14 0.04]
    [0.03 0.08 0.06 0.05 0.17 0.18 0.09 0.03 0.12 0.19]
    [0.09 0.04 0.10 0.03 0.32 0.05 0.07 0.04 0.15 0.09]
    [0.15 0.05 0.10 0.05 0.11 0.14 0.04 0.08 0.08 0.20]
    [0.11 0.11 0.08 0.11 0.08 0.05 0.24 0.03 0.08 0.12]

    Note that when writing a Model implementer, return types may be narrower than the
    return types promised by the protocol (npt.NDArray is a subtype of ArrayLike), but
    the argument types must be at least as general as the argument types promised by the
    protocol.
    """


class Metric(gen.Metric[TargetType, DatumMetadata], Protocol):
    """
    A metric protocol for the image classification AI problem.

    A metric in this sense is expected to measure the level of agreement between model
    predictions and ground-truth labels.

    Methods
    -------

    update(pred_batch: Sequence[ArrayLike], target_batch: Sequence[ArrayLike], metadata_batch: Sequence[DatumMetadata]) -> None
        Add predictions and targets (and metadata if applicable) to metric's cache for later calculation. Both
        predictions and targets are expected to be sequences with elements of shape `(Cl,)`.

    compute() -> dict[str, Any]
        Compute metric value(s) for currently cached predictions and targets, returned as
        a dictionary.

    reset() -> None
        Clear contents of current metric's cache of predictions and targets.

    Attributes
    ----------

    metadata : MetricMetadata
        A typed dictionary containing at least an 'id' field of type str

    Examples
    --------

    Create a basic accuracy metric and test it on a small example dataset:

    >>> from typing import Any, Sequence
    >>> import numpy as np
    >>> from maite.protocols import ArrayLike, DatumMetadata
    >>> from maite.protocols import image_classification as ic

    >>> class MyAccuracy:
    ...     metadata: MetricMetadata = {"id": "Example Multiclass Accuracy"}
    ...
    ...     def __init__(self):
    ...         self._total = 0
    ...         self._correct = 0
    ...
    ...     def reset(self) -> None:
    ...         self._total = 0
    ...         self._correct = 0
    ...
    ...     def update(
    ...         self,
    ...         pred_batch: Sequence[ArrayLike],
    ...         target_batch: Sequence[ArrayLike],
    ...         metadata_batch: Sequence[DatumMetadata],
    ...     ) -> None:
    ...         model_preds = [np.array(r) for r in pred_batch]
    ...         true_onehot = [np.array(r) for r in target_batch]
    ...
    ...         # Stack into single array, convert to class indices
    ...         model_classes = np.vstack(model_preds).argmax(axis=1)
    ...         truth_classes = np.vstack(true_onehot).argmax(axis=1)
    ...
    ...         # Compare classes and update running counts
    ...         same = model_classes == truth_classes
    ...         self._total += len(same)
    ...         self._correct += same.sum().item()
    ...
    ...     def compute(self) -> dict[str, Any]:
    ...         if self._total > 0:
    ...             return {"accuracy": self._correct / self._total}
    ...         else:
    ...             raise Exception("No batches processed yet.")

    Instantiate this class and typehint it as an image_classification.Metric.
    By using typehinting, permits a static typechecker to check protocol compliance.

    >>> accuracy: ic.Metric = MyAccuracy()

    To use the metric call update() for each batch of predictions and truth values and call compute() to calculate the final metric values.

    >>> # batch 1
    >>> model_preds = [
    ...     np.array([0.8, 0.1, 0.0, 0.1]),
    ...     np.array([0.1, 0.2, 0.6, 0.1]),
    ... ]  # predicted classes: 0, 2
    >>> true_onehot = [
    ...     np.array([1.0, 0.0, 0.0, 0.0]),
    ...     np.array([0.0, 1.0, 0.0, 0.0]),
    ... ]  # true classes: 0, 1
    >>> metadatas: list[DatumMetadata] = [{"id": 1}, {"id": 2}]
    >>> accuracy.update(model_preds, true_onehot, metadatas)
    >>> print(accuracy.compute())
    {'accuracy': 0.5}
    >>>
    >>> # batch 2
    >>> model_preds = [
    ...     np.array([0.1, 0.1, 0.7, 0.1]),
    ...     np.array([0.0, 0.1, 0.0, 0.9]),
    ... ]  # predicted classes: 2, 3
    >>> true_onehot = [
    ...     np.array([0.0, 0.0, 1.0, 0.0]),
    ...     np.array([0.0, 0.0, 0.0, 1.0]),
    ... ]  # true classes: 2, 3
    >>> metadatas: list[DatumMetadata] = [{"id": 3}, {"id": 4}]
    >>> accuracy.update(model_preds, true_onehot, metadatas)
    >>>
    >>> print(accuracy.compute())
    {'accuracy': 0.75}
    """

    ...


class Augmentation(
    gen.Augmentation[
        InputType,
        TargetType,
        DatumMetadataType,
        InputType,
        TargetType,
        DatumMetadataType,
    ],
    Protocol,
):
    """
    An augmentation protocol for the image classification AI problem.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `Sequence[ArrayLike]` (with elements of shape `(C, H, W)`), `Sequence[ArrayLike]`
    (with elements of shape `(Cl, )`), and `Sequence[DatumMetadata]`. These correspond
    to the model input batch type, model target batch type, and datum-level metadata
    batch type, respectively.

    Methods
    -------

    __call__(datum: tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadata]]) ->\
          tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadata]])
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as `Sequence[ArrayLike]` with elements of shape
        `(C, H, W)`), model target batch (as `Sequence[ArrayLike]` with elements of shape
        `(Cl,)`), and batch metadata (as `Sequence[DatumMetadata]`), respectively.

    Attributes
    ----------

    metadata : AugmentationMetadata
        A typed dictionary containing at least an 'id' field of type str

    Examples
    --------

    We can write an implementer of the augmentation class as either a function or a class.
    The only requirement is that the object provide a __call__ method that takes objects
    at least as general as the types promised in the protocol signature and return types
    at least as specific.

    >>> import copy
    >>> import numpy as np
    >>> from typing import Any
    >>> from collections.abc import Sequence
    >>> from maite.protocols import ArrayLike, DatumMetadata, AugmentationMetadata
    >>>
    >>> class EnrichedDatumMetadata(DatumMetadata):
    ...     new_key: int  # add a field to those already in DatumMetadata
    ...
    >>> class ImageAugmentation:
    ...     def __init__(self, aug_name: str):
    ...         self.metadata: AugmentationMetadata = {'id': aug_name}
    ...     def __call__(
    ...         self,
    ...         data_batch: tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadata]]
    ...     ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[EnrichedDatumMetadata]]:
    ...         inputs, targets, mds = data_batch
    ...         # We copy data passed into the constructor to avoid mutating original inputs
    ...         # By using np.ndarray constructor, the static type-checker will let us treat
    ...         # generic ArrayLike as a more narrow return type
    ...         inputs_aug = [copy.copy(np.array(input)) for input in inputs]
    ...         targets_aug = [copy.copy(np.array(target)) for target in targets]
    ...         # Modify inputs_aug, targets_aug, or mds_aug as needed
    ...         # In this example, we just add a new metadata field
    ...         mds_aug = []
    ...         for i, md in enumerate(mds):
    ...             mds_aug.append(EnrichedDatumMetadata(**md, new_key=i))
    ...         return inputs_aug, targets_aug, mds_aug

    We can typehint an instance of the above class as an Augmentation in the
    image_classification domain:

    >>> from maite.protocols import image_classification as ic
    >>> im_aug: ic.Augmentation = ImageAugmentation(aug_name = 'an_augmentation')

    """
