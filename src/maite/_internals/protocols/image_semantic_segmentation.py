# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Protocol, TypeAlias

import maite._internals.protocols.generic as gen
from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata

InputType: TypeAlias = ArrayLike  # shape (C, H, W)
TargetType: TypeAlias = ArrayLike  # shape (Cl, H, W)
DatumMetadataType: TypeAlias = DatumMetadata

Datum: TypeAlias = tuple[InputType, TargetType, DatumMetadataType]


class Dataset(gen.Dataset[InputType, TargetType, DatasetMetadata], Protocol):
    """
    A dataset protocol for image segmentation AI task providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond
    to individual examples (as opposed to batches).

    Indexing into or iterating over an image_segmentation dataset returns a
    `tuple` of types `ArrayLike`, `ArrayLike`, and `DatumMetadataType`.
    These correspond to the model input type, model target type, and datum-level
    metadata, respectively. The `ArrayLike` protocol implementers associated with
    model input and model target types are expected to follow (C, H, W) and (Cl, H, W)
    shape semantics, respectively. (Note: we use 'Cl' is a placeholder for the array
    dimension corresponding to one-hot classification of a given pixel.)

    Methods
    -------

    __getitem__(ind: int) -> tuple[ArrayLike, ArrayLike, DatumMetadataType]
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

    Assume we have 5 semantic classifications, 10 datapoints, and 10 target labels, and that we want
    to simply have an integer 'id' field in each datapoint's metadata:

    >>> N_CLASSES: int = 5
    >>> N_DATUM: int = 10
    >>> images: list[np.ndarray] = [
    ...     np.random.rand(3, 32, 16) for _ in range(N_DATUM)
    ... ]  # (C, H, W) shape semantics
    >>> targets: np.ndarray = [
    ...     np.random.rand(N_CLASSES, 16, 8)
    ... ]  # (Cl, H, W) shape semantics

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
    ...         targets: list[np.ndarray],
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

    We can instantiate this class and typehint it as an image_semantic_segmentation.Dataset.
    By using typehinting, we permit a static typechecker to verify protocol compliance.

    >>> from maite._internals.protocols import image_semantic_segmentation as iss
    >>> dataset: iss.Dataset = ImageDataset(
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


class DataLoader(gen.DataLoader[InputType, TargetType, DatumMetadataType], Protocol):
    """
    A dataloader protocol for the image segmentatation AI task providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `Sequence[ArrayLike]` (elements of shape `(C, H, W)`),
    `Sequence[ArrayLike]` (elements shape `(Cl, H, W)`), and `Sequence[DatumMetadataType]`,
    which correspond to model input batch, model target type batch, and a datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.

    Methods
    -------

    __iter__ -> Iterator[tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadataType]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as `Sequence[ArrayLike]`), model target batch (as
        `Sequence[ArrayLike]`), and batched datum-level metadata
        (as `Sequence[DatumMetadataType]`), respectively.

    """

    ...


class Model(gen.Model[InputType, TargetType], Protocol):
    """
    A model protocol for the image segmentation AI task.

    Implementers must provide a `__call__` method that operates on a batch of model
    inputs (as `Sequence[ArrayLike]`) and returns a batch of model targets (as
    `Sequence[ArrayLike]`)

    Methods
    -------

    __call__(input_batch: Sequence[ArrayLike]) -> Sequence[ArrayLike]
        Make a model prediction for inputs in input batch. Input batch is expected to
        be `Sequence[ArrayLike]` with each element of shape `(C, H, W)`.

    Channel ordering in model input type is expected to be (C, H, W). Channel ordering
    in model target type is expected to be (Cl, H, W), where 'Cl' is the number of output
    classes.

    Attributes
    ----------

    metadata : ModelMetadata
        A typed dictionary containing at least an 'id' field of type str

    Examples
    --------

    We create a simple fully-connected single layer semantic segmentation classifier
    that takes CIFAR-10-like inputs with shape-(3, 32, 32) (with (C, H, W) shape semantics)
    and returns targets of shape-(5, 32, 32) (with (Cl, H, W) shape semantics). The Model
    output consists of class probabilities assigned to each pixel/class pair.

    >>> import maite._internals.protocols.image_semantic_segmentation as iss
    >>> import numpy as np
    >>> import numpy.typing as npt
    >>> from maite.protocols import ArrayLike, ModelMetadata
    >>> from typing import Sequence

    Creating a MAITE-compliant model involves writing a `__call__` method that takes a
    batch of inputs and returns a batch of predictions (probabilities or logits).

    >>> N_CLASSES: int = 5
    >>> N_CHANNELS: int = 3
    >>> IMG_HEIGHT = 8
    >>> IMG_WIDTH = 8
    >>> class SimpleSemSeg:
    ...     def __init__(self) -> None:
    ...         # Set up required metadata attribute using the default `ModelMetadata` type,
    ...         # using class name for the ID
    ...         self.metadata: ModelMetadata = {"id": self.__class__.__name__}
    ...
    ...         # Initialize weights
    ...         rng = np.random.default_rng(12345678)
    ...         flattened_output_size = N_CLASSES * IMG_HEIGHT * IMG_WIDTH
    ...         flattened_input_size = N_CHANNELS * IMG_HEIGHT * IMG_WIDTH
    ...         self.weights = -0.2 + 0.4 * rng.random(
    ...             (flattened_input_size, flattened_output_size)
    ...         )
    ...         self.bias = -0.2 + 0.4 * rng.random((1, flattened_output_size))
    ...
    ...     def __call__(self, batch: Sequence[ArrayLike]) -> Sequence[npt.NDArray]:
    ...         # Convert each element in batch to ndarray, flatten,
    ...         # then combine into 4D array of shape-(N, C, H, W)
    ...         batch_np = np.vstack([np.asarray(x).flatten() for x in batch])
    ...
    ...         # Send input batch through model
    ...         out = batch_np @ self.weights + self.bias
    ...         out = out.reshape(-1, N_CLASSES, IMG_HEIGHT, IMG_WIDTH)
    ...         out = np.exp(out) / np.sum(
    ...             np.exp(out), axis=1, keepdims=True
    ...         )  # softmax
    ...
    ...         # Restructure to sequence of shape-(Cl, H, W) probabilities
    ...         return [row for row in out]

    We set up a test batch, instantiate the model, and apply it to the batch.

    >>> batch_size = 8
    >>> rng = np.random.default_rng(12345678)
    >>> batch: Sequence[ArrayLike] = [
    ...     -0.2 + 0.4 * rng.random((3, IMG_HEIGHT, IMG_WIDTH))
    ...     for _ in range(batch_size)
    ... ]
    >>> model: iss.Model = SimpleSemSeg()
    >>> out = model(batch)

    We can now show the class probabilities returned by the model for each image in the batch.

    >>> np.set_printoptions(
    ...     floatmode="fixed", precision=2
    ... )  # for reproducible output for doctest
    >>> # print pixel-level probabilities for first image in first semantic class
    >>> print(out[0][0, :, :])  # doctest: +NORMALIZE_WHITESPACE
    [[0.16 0.21 0.22 0.21 0.21 0.17 0.19 0.17]
    [0.22 0.25 0.23 0.19 0.24 0.17 0.23 0.14]
    [0.23 0.17 0.18 0.22 0.18 0.22 0.20 0.22]
    [0.16 0.19 0.18 0.17 0.18 0.15 0.21 0.23]
    [0.19 0.22 0.22 0.26 0.19 0.18 0.20 0.20]
    [0.20 0.24 0.14 0.22 0.18 0.20 0.18 0.26]
    [0.16 0.14 0.21 0.20 0.20 0.30 0.25 0.21]
    [0.19 0.15 0.29 0.26 0.18 0.13 0.14 0.21]]

    Note that when writing a Model implementer, return types may be narrower than the
    return types promised by the protocol (npt.NDArray is a subtype of ArrayLike), but
    the argument types must be at least as general as the argument types promised by the
    protocol.
    """

    ...


class Metric(gen.Metric[TargetType, DatumMetadataType], Protocol):
    """
    A metric protocol for the image segmentation AI task.

    A metric in this sense is expected to measure the level of agreement between model
    predictions and ground-truth labels.

    Methods
    -------

    update(preds: Sequence[ArrayLike], targets: Sequence[ArrayLike], metadatas: Sequence[DatumMetadata]) -> None
        Add predictions and targets to metric's cache for later calculation. Both
        preds and targets are expected to be sequences with elements of shape `(Cl, H, W)`.

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

    Create metric for pixelwise Sensitivity

    >>> from typing import Sequence, Any
    >>> import numpy as np
    >>> from maite.protocols import ArrayLike, MetricMetadata, DatumMetadata
    >>> from maite._internals.protocols import image_semantic_segmentation as iss
    >>> class PixelwiseSensitivity:
    ...     def __init__(
    ...         self, metric_id: str, threshold: float = 0.5, eps: float = 1e-7
    ...     ) -> None:
    ...         self._threshold = threshold
    ...         self._eps = eps
    ...         self._tp = self._fp = self._tn = self._fn = 0
    ...         self.metadata: MetricMetadata = {"id": metric_id}
    ...
    ...     def update(
    ...         self,
    ...         preds: Sequence[ArrayLike],
    ...         tgts: Sequence[ArrayLike],
    ...         mds: Sequence[DatumMetadata],
    ...     ) -> None:
    ...         new_preds = np.stack(preds, axis=0) > self._threshold
    ...         new_tgts = np.stack(tgts, axis=0)
    ...
    ...         self._tp += np.sum((new_preds == new_tgts) & new_tgts, axis=None)
    ...         self._fp += np.sum((new_preds != new_tgts) & ~new_tgts, axis=None)
    ...         self._tn += np.sum((new_preds == new_tgts) & ~new_tgts, axis=None)
    ...         self._fn += np.sum((new_preds != new_tgts) & new_tgts, axis=None)
    ...
    ...     def compute(self) -> dict[str, Any]:
    ...         tp, tn, fp, fn = self._tp, self._tn, self._fp, self._fn
    ...         return {"sensitivity": tp / (tp + fn + self._eps)}
    ...
    ...     def reset(self) -> None:
    ...         self._tp = self._fp = self._tn = self._fn = 0

    >>> psens: iss.Metric = PixelwiseSensitivity(metric_id="thresholded_sensitivity")
    >>> preds = [ np.stack(([[0.8, 0.2],
    ...                      [0.8, 0.2]],
    ...                     [[0.2, 0.8],
    ...                      [0.2, 0.8]])) ]  # Sequence of (Cl, H, W)-shaped preds
    >>> tgts  = [ np.stack(([[1, 1],
    ...                      [0, 0]],
    ...                     [[0, 0],
    ...                      [1, 1]]))]  # Sequence of (Cl, H, W)-shaped targets
    >>> mds = [{"id": 0}, {"id": 1}]
    >>> psens.update(preds, tgts, mds)
    >>> out = psens.compute()
    >>> assert np.isclose(out["sensitivity"], 0.5)
    """  # fmt: skip


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
    An augmentation protocol for the image segmentation AI task.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `Sequence[ArrayLike]` (with elements of shape `(C, H, W)`), `Sequence[ArrayLike]`
    (with elements of shape `(Cl, H, W)`), and `Sequence[DatumMetadataType]`. These correspond
    to the model input batch type, model target batch type, and datum-level metadata
    batch type, respectively.

    Methods
    -------

    __call__(datum: tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadataType]]) ->\
          tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadataType]])
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as `Sequence[ArrayLike]` with elements of shape
        `(C, H, W)`), model target batch (as `Sequence[ArrayLike]` with elements of shape
        `(Cl, H, W)`), and batch metadata (as `Sequence[DatumMetadataType]`), respectively.

    Attributes
    ----------

    metadata : AugmentationMetadata
        A typed dictionary containing at least an 'id' field of type str
    """

    ...
