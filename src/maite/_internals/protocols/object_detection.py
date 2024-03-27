# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from typing_extensions import Dict, TypeAlias

from maite._internals.protocols import generic as gen
from maite.protocols import ArrayLike

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
#       concrete Dataset classes (like object_detection.Dataset) are ArrayLike, ObjectDetectionTarget, Dict[str,Any]
#       so users can see an expected return type of Tuple[ArrayLike, ObjectDetectionTarget, Dict[str,Any]]

InputType: TypeAlias = ArrayLike  # shape (C, H, W)
TargetType: TypeAlias = ObjectDetectionTarget
DatumMetadataType: TypeAlias = Dict[str, Any]

InputBatchType: TypeAlias = ArrayLike  # shape (N, C, H, W)
TargetBatchType: TypeAlias = Sequence[TargetType]  # length N
DatumMetadataBatchType: TypeAlias = Sequence[DatumMetadataType]

# TODO: Consider what pylance shows on cursoring over: "(type alias) Dataset: type[Dataset[ArrayLike, Dict[Unknown, Unknown], object]]"
# Can these type hints be made more intuitive? Perhaps given a name like type[Dataset[InputType = ArrayLike,...]]
# - This will likely involve eliminating some TypeAlias uses.

# TODO: Determine whether I should/can parameterize on the Datum TypeAlias.
# This could make the pylance messages more intuitive?

# TODO: Consider how we should help type-checker infer method return type when argument type
#       matches more than one method signature. For example: Model.__call__ takes an
#       ArrayLike in two separate method signatures, but the return type differs.
#       In this case, typechecker seems to use the first matching method signature to
#       determine type of output. -> we can handle this problem by considering only
#       batches as the required handled types for model, augmentation, and metric objects
#


class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
    """
    A dataset protocol for object detection ML subproblem providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond to
    individual examples (as opposed to batches).

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `ArrayLike`, `ObjectDetectionTarget`, and `Dict[str,Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.


    Methods
    -------

    __getitem__(ind: int)->Tuple[ArrayLike, ObjectDetectionTarget, Dict[str, Any]]
        Provide mapping-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata,
        respectively.

    __len__()->int
        Return the number of data elements in the dataset.

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
    contain types `ArrayLike` (shape `(N, C, H, W)`), `Sequence[ObjectDetectionTarget]`,
    `Sequence[Dict[str, Any]]`, which correspond to model input batch, model target
    type batch, and datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.

    Methods
    -------

    __iter__->Iterator[tuple[ArrayLike, Sequence[ObjectDetectionTarget], Sequence[Dict[str, Any]]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as an `ArrayLike`), model target batch (as
        `Sequence[ObjectDetectionTarget]`), and batched datum-level metadata
        (as `Sequence[Dict[str,Any]]`), respectively.

    """

    ...


class Model(gen.Model[InputBatchType, TargetBatchType], Protocol):
    """
    A model protocol for the object detection ML subproblem.

    Implementers must provide a `__call__` method that operates on a batch of model inputs
    (as `ArrayLike`s) and returns a batch of model targets (as
    `Sequence[ObjectDetectionTarget]`)

    Methods
    -------

    __call__(input_batch: ArrayLike)->Sequence[ObjectDetectionTarget]
        Make a model prediction for inputs in input batch. Input batch is expected in
        the shape `(N, C, H, W)`.
    """

    ...


class Metric(gen.Metric[TargetBatchType], Protocol):
    """
    A metric protocol for the object detection ML subproblem.

     A metric in this sense is expected to measure the level of agreement between model
     predictions and ground-truth labels.

     Methods
     -------

     update(preds: Sequence[ObjectDetectionTarget], targets: Sequence[ObjectDetectionTarget])->None
         Add predictions and targets to metric's cache for later calculation.

     compute()->Dict[str, Any]
         Compute metric value(s) for currently cached predictions and targets, returned as
         a dictionary.

     reset()->None
         Clear contents of current metric's cache of predictions and targets.
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
    `ArrayLike`, `Sequence[ObjectDetectionTarget]`, and `Sequence[Dict[str,Any]]`. These
    correspond to the model input batch type, model target batch type, and datum-level
    metadata batch type, respectively.

    Methods
    -------

    __call__(datum: Tuple[ArrayLike, Sequence[ObjectDetectionTarget], Sequence[dict[str, Any]]])->
                Tuple[ArrayLike, Sequence[ObjectDetectionTarget], Sequence[dict[str, Any]]]
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as an `ArrayLike` of shape `(N, C, H, W)`),
        model target batch (as `Sequence[ObjectDetectionTarget]`), and batch metadata
        (as `Sequence[Dict[str,Any]]`), respectively.
    """

    ...
