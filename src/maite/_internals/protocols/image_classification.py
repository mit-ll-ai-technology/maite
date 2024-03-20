# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for image_classification
from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence

from typing_extensions import TypeAlias

from . import ArrayLike, generic as gen

# In below, the dimension names/meanings used are:
#
# N  - image instance
# H  - image height
# W  - image width
# C  - image channel
# Cl - classification label (one-hot for ground-truth label, or pseudo-probabilities for predictions)

InputType: TypeAlias = ArrayLike  # shape [H, W, C]
TargetType: TypeAlias = ArrayLike  # shape [Cl]
MetadataType: TypeAlias = Dict[str, Any]

InputBatchType: TypeAlias = ArrayLike  # shape [N, H, W, C]
TargetBatchType: TypeAlias = ArrayLike  # shape [N, Cl]
MetadataBatchType: TypeAlias = Sequence[MetadataType]

# Initialize component classes based on generic and Input/Target/Metadata types


class Dataset(gen.Dataset[InputType, TargetType, MetadataType], Protocol):
    """
    A dataset protocol for image classification ML subproblem providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond to
    individual examples (as opposed to batches).

    Indexing into or iterating over the an image_classification dataset returns a `Tuple` of
    types `ArrayLike`, `ArrayLike`, and `Dict[str,Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.


    Methods
    -------

    __getitem__(ind: int)->Tuple[ArrayLike, ArrayLike, Dict[str, Any]]
        Provide mapping-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata,
        respectively.

    __len__()->int
        Return the number of data elements in the dataset.

    """

    ...


class DataLoader(
    gen.DataLoader[InputBatchType, TargetBatchType, MetadataBatchType], Protocol
):
    """
    A dataloader protocol for the image classification ML subproblem providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `ArrayLike` (shape `(N, C, H, W)`), `ArrayLike` (shape `(N, Cl)`),
    and `Sequence[Dict[str, Any]]`, which correspond to model input batch, model target
    type batch, and a datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.

    Methods
    -------

    __iter__->Iterator[tuple[ArrayLike, ArrayLike, Sequence[Dict[str, Any]]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as an `ArrayLike`), model target batch (as
        `Sequence[ObjectDetectionTarget]`), and batched datum-level metadata
        (as `Sequence[Dict[str,Any]]`), respectively.

    """


class Model(gen.Model[InputBatchType, TargetBatchType], Protocol):
    """
    A model protocol for the image classification ML subproblem.

    Implementers must provide a `__call__` method that operates on a batch of model
    inputs (as `ArrayLike`s) and returns a batch of model targets (implementers of
    `ArrayLike`)

    Methods
    -------

    __call__(input_batch: ArrayLike)->Sequence[ObjectDetectionTarget]
        Make a model prediction for inputs in input batch. Input batch is expected in
        the shape [N, C, H, W].
    """


class Metric(gen.Metric[TargetBatchType], Protocol):
    """
    A metric protocol for the image classification ML subproblem.

    A metric in this sense is expected to measure the level of agreement between model
    predictions and ground-truth labels.

    Methods
    -------

    update(preds: ArrayLike, targets: Sequence[ObjectDetectionTarget])->None
        Add predictions and targets to metric's cache for later calculation. Both
        preds and targets are expected to be of shape `(N, Cl)`.

    compute()->Dict[str, Any]
        Compute metric value(s) for currently cached predictions and targets, returned as
        a dictionary.

    clear()->None
        Clear contents of current metric's cache of predictions and targets.
    """

    ...


class Augmentation(
    gen.Augmentation[
        InputBatchType,
        TargetBatchType,
        MetadataBatchType,
        InputBatchType,
        TargetBatchType,
        MetadataBatchType,
    ],
    Protocol,
):
    """
    An augmentation protocol for the image classification subproblem.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `ArrayLike` (of shape `(N, C, H, W)`), `ArrayLike` (of shape `(N, Cl)`), and
    `Dict[str,Any]`. These correspond to the model input type, model target type, and
    datum-specific metadata, respectively.

    Methods
    -------

    __call__(datum: Tuple[ArrayLike, ArrayLike, dict[str, Any]])->
                Tuple[ArrayLike, ArrayLike, dict[str, Any]]
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as an `ArrayLike` of shape `(N, C, H, W)`),
        model target batch (as an `ArrayLike` of shape `(N, Cl)`), and batch metadata (as
        `Sequence[Dict[str,Any]]`), respectively.
    """
