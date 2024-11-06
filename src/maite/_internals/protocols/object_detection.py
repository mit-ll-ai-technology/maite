# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, Protocol, runtime_checkable

from typing_extensions import TypeAlias

from maite._internals.protocols import generic as gen
from maite.protocols import ArrayLike, DatumMetadata

# We *could* make ArrayLike generic and rely on the subscripts for ArrayLike type
# annotations to hint to the user the appropriate shape. No runtime safety would
# be added by this approach because type subscripts are effectively invisible
# at runtime. No additional static type-checking would occur either, but the
# user would get useful type hints when cursoring over required inputs/targets
# This would also require python 3.11 unless a `from __future__` import were made
# available for earlier versions (which are not available now.)


@runtime_checkable
class ObjectDetectionTarget(Protocol):
    """
    An object-detection target protocol.

    This class is used to encode both predictions and ground-truth labels in the object
    detection problem.

    Implementers must populate the following attributes:

    Attributes
    ----------
    boxes : ArrayLike
        An array representing object detection boxes in a single image with x0, y0, x1, y1
        format and shape `(N_DETECTIONS, 4)`

    labels : ArrayLike
        An array representing the integer labels associated with each detection box of shape
        `(N_DETECTIONS,)`

    scores: ArrayLike
        An array representing the scores associated with each box (of shape `(N_DETECTIONS,)`)
    """

    @property
    def boxes(
        self,
    ) -> ArrayLike:  # shape (N, 4), format X0, Y0, X1, Y1
        ...

    @property
    def labels(self) -> ArrayLike:  # label for each box, shape (N,)
        ...

    @property
    def scores(self) -> ArrayLike:  # shape (N,)
        ...


# TODO: remove typeAlias statements for more user readability (or figure out how to resolve TypeAliases
#       to their targets for end-user.) Knowing a dataset returns a tuple of "InputType, TargetType, DatumMetadataType"
#       isn't helpful to implementers, however the aliasing *is* helpful to developers.
#
#       Perhaps the functionality I want is named TypeVars for generic, so developers can understand that
#       e.g. generic.Dataset typevars are 'InputType', 'TargetType', and 'MetaDataType' and their values in
#       concrete Dataset classes (like object_detection.Dataset) are ArrayLike, ObjectDetectionTarget, DatumMetadataType
#       so users can see an expected return type of Tuple[ArrayLike, ObjectDetectionTarget, DatumMetadataType]

InputType: TypeAlias = ArrayLike  # shape (C, H, W)
TargetType: TypeAlias = ObjectDetectionTarget
DatumMetadataType: TypeAlias = DatumMetadata

InputBatchType: TypeAlias = Sequence[
    ArrayLike
]  # sequence of N ArrayLikes of shape (C, H, W)
TargetBatchType: TypeAlias = Sequence[TargetType]  # sequence of N TargetType instances
DatumMetadataBatchType: TypeAlias = Sequence[DatumMetadataType]

Datum: TypeAlias = tuple[InputType, TargetType, DatumMetadataType]
DatumBatch: TypeAlias = tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]

CollateFn: TypeAlias = Callable[
    [Iterable[Datum]],
    DatumBatch,
]


class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
    """
    A dataset protocol for object detection ML subproblem providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond to
    individual examples (as opposed to batches).

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `ArrayLike`, `ObjectDetectionTarget`, and `DatumMetadataType`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.


    Methods
    -------

    __getitem__(ind: int) -> Tuple[ArrayLike, ObjectDetectionTarget, DatumMetadataType]
        Provide mapping-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata,
        respectively.

    __len__() -> int
        Return the number of data elements in the dataset.

    Attributes
    ----------

    metadata : DatasetMetadata
        A typed dictionary containing at least an 'id' field of type str
    """

    ...


class DataLoader(
    gen.DataLoader[
        InputBatchType,
        TargetBatchType,
        DatumMetadataBatchType,
    ],
    Protocol,
):
    """
    A dataloader protocol for the object detection ML subproblem providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `Sequence[ArrayLike]` (elements of shape `(C, H, W)`),
    `Sequence[ObjectDetectionTarget]`, and `Sequence[DatumMetadataType]`,
    which correspond to model input batch, model target batch, and a datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.


    Methods
    -------

    __iter__ -> Iterator[tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[DatumMetadataType]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as `Sequence[ArrayLike]`), model target batch (as
        `Sequence[ObjectDetectionTarget]`), and batched datum-level metadata
        (as `Sequence[DatumMetadataType]]`), respectively.
    """

    ...


class Model(gen.Model[InputBatchType, TargetBatchType], Protocol):
    """
    A model protocol for the object detection ML subproblem.

    Implementers must provide a `__call__` method that operates on a batch of model inputs
    (as `Sequence[ArrayLike]`s) and returns a batch of model targets (as
    `Sequence[ObjectDetectionTarget]`)

    Methods
    -------

    __call__(input_batch: Sequence[ArrayLike]) -> Sequence[ObjectDetectionTarget]
        Make a model prediction for inputs in input batch. Elements of input batch
        are expected in the shape `(C, H, W)`.

    Attributes
    ----------

    metadata : ModelMetadata
        A typed dictionary containing at least an 'id' field of type str
    """

    ...


class Metric(gen.Metric[TargetBatchType], Protocol):
    """
    A metric protocol for the object detection ML subproblem.

    A metric in this sense is expected to measure the level of agreement between model
    predictions and ground-truth labels.

    Methods
    -------

    update(preds: Sequence[ObjectDetectionTarget], targets: Sequence[ObjectDetectionTarget]) -> None
         Add predictions and targets to metric's cache for later calculation.

     compute() -> dict[str, Any]
         Compute metric value(s) for currently cached predictions and targets, returned as
         a dictionary.

    reset() -> None
        Clear contents of current metric's cache of predictions and targets.

    Attributes
    ----------

    metadata : MetricMetadata
        A typed dictionary containing at least an 'id' field of type str
    """

    ...


class Augmentation(
    gen.Augmentation[
        InputBatchType,
        TargetBatchType,
        DatumMetadataBatchType,
        InputBatchType,
        TargetBatchType,
        DatumMetadataBatchType,
    ],
    Protocol,
):
    """
    An augmentation protocol for the object detection subproblem.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `Sequence[ArrayLike]`, `Sequence[ObjectDetectionTarget]`, and `Sequence[DatumMetadataType]`.
    These correspond to the model input batch type, model target batch type, and datum-level
    metadata batch type, respectively.

    Methods
    -------

    __call__(datum: Tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[DatumMetadataType]]) ->\
          Tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[DatumMetadataType]]
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as `Sequence ArrayLike` with elements of shape
        `(C, H, W)`), model target batch (as `Sequence[ObjectDetectionTarget]`), and
        batch metadata (as `Sequence[DatumMetadataType]`), respectively.

    Attributes
    ----------

    metadata : AugmentationMetadata
        A typed dictionary containing at least an 'id' field of type str
    """

    ...
