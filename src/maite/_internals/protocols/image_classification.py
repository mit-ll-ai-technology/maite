# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for image_classification
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Protocol, Sequence, Tuple

from typing_extensions import TypeAlias

from maite._internals.protocols import generic as gen
from maite.protocols import ArrayLike

# In below, the dimension names/meanings used are:
#
# N  - image instance
# H  - image height
# W  - image width
# C  - image channel
# Cl - classification label (one-hot for ground-truth label, or pseudo-probabilities for predictions)

InputType: TypeAlias = ArrayLike  # shape (C, H, W)
TargetType: TypeAlias = ArrayLike  # shape (Cl,)
DatumMetadataType: TypeAlias = Dict[str, Any]

InputBatchType: TypeAlias = Sequence[
    InputType
]  # sequence of N ArrayLikes of shape (C, H, W)
TargetBatchType: TypeAlias = Sequence[TargetType]  # sequence of N TargetType instances
DatumMetadataBatchType: TypeAlias = Sequence[DatumMetadataType]

Datum: TypeAlias = Tuple[InputType, TargetType, DatumMetadataType]
DatumBatch: TypeAlias = Tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]

CollateFn: TypeAlias = Callable[
    [Iterable[Datum]],
    DatumBatch,
]

# Initialize component classes based on generic and Input/Target/Metadata types


class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
    """
    A dataset protocol for image classification ML subproblem providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond
    to individual examples (as opposed to batches).

    Indexing into or iterating over the an image_classification dataset returns a
    `Tuple` of types `ArrayLike`, `ArrayLike`, and `Dict[str,Any]`.
    These correspond to the model input type, model target type, and datum-level
    metadata, respectively.

    Methods
    -------

    __getitem__(ind: int) -> Tuple[ArrayLike, ArrayLike, Dict[str, Any]]
        Provide map-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata type,
        respectively.

    __len__() -> int
        Return the number of data elements in the dataset.

    Examples
    --------

    We create a dummy set of data and use it to create a class that implements
    this lightweight dataset protocol:

    >>> import numpy as np
    >>> from typing import List, Dict, Any, Tuple
    >>> from maite.protocols import ArrayLike

    Assume we have 5 classes, 10 datapoints, and 10 target labels, and that we want
    to simply have an integer 'id' field in each datapoint's metadata:

    >>> N_CLASSES: int = 5
    >>> N_DATUM: int = 10
    >>> images: List[np.ndarray] = [np.random.rand(3, 32, 16) for _ in range(N_DATUM)]
    >>> targets: np.ndarray = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, N_DATUM)]
    >>> metadata: List[Dict] = [{'id': i} for i in range(N_DATUM)]

    Constructing a compliant dataset just involves a simple wrapper that fetches
    individual datapoints, where a datapoint is a single image, target, metadata 3-tuple.

    >>> class ImageDataset:
    ...     def __init__(self,
    ...                  images: List[np.ndarray],
    ...                  targets: np.ndarray,
    ...                  metadata: List[Dict[str, Any]]):
    ...         self.images = images
    ...         self.targets = targets
    ...         self.metadata = metadata
    ...     def __len__(self) -> int:
    ...         return len(images)
    ...     def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    ...         return self.images[ind], self.targets[ind], self.metadata[ind]

    We can instantiate this class and typehint it as an image_classification.Dataset.
    By using typehinting, we permit a static typechecker to verify protocol compliance.

    >>> from maite.protocols import image_classification as ic
    >>> dataset: ic.Dataset = ImageDataset(images, targets, metadata)

    Note that when writing a Dataset implementer, return types may be narrower than the
    return types promised by the protocol (np.ndarray is a subtype of ArrayLike), but
    the argument types must be at least as general as the argument types promised by the
    protocol.
    """

    ...


class DataLoader(
    gen.DataLoader[InputBatchType, TargetBatchType, DatumMetadataBatchType], Protocol
):
    """
    A dataloader protocol for the image classification ML subproblem providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `Sequence[ArrayLike]` (elements of shape `(C, H, W)`),
    `Sequence[ArrayLike]` (elements shape `(Cl, )`), and `Sequence[Dict[str, Any]]`,
    which correspond to model input batch, model target type batch, and a datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.

    Methods
    -------

    __iter__ -> Iterator[tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[Dict[str, Any]]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as `Sequence[ArrayLike]`), model target batch (as
        `Sequence[ArrayLike]`), and batched datum-level metadata
        (as `Sequence[Dict[str,Any]]`), respectively.

    """


class Model(gen.Model[InputBatchType, TargetBatchType], Protocol):
    """
    A model protocol for the image classification ML subproblem.

    Implementers must provide a `__call__` method that operates on a batch of model
    inputs (as `Sequence[ArrayLike]) and returns a batch of model targets (as
    `Sequence[ArrayLike]`)

    Methods
    -------

    __call__(input_batch: Sequence[ArrayLike]) -> Sequence[ArrayLike]
        Make a model prediction for inputs in input batch. Input batch is expected to
        be `Sequence[ArrayLike]` with each element of shape `(C, H, W)`.
    """


class Metric(gen.Metric[TargetBatchType], Protocol):
    """
    A metric protocol for the image classification ML subproblem.

    A metric in this sense is expected to measure the level of agreement between model
    predictions and ground-truth labels.

    Methods
    -------

    update(preds: Sequence[ArrayLike], targets: Sequence[ArrayLike]) -> None
        Add predictions and targets to metric's cache for later calculation. Both
        preds and targets are expected to be sequences with elements of shape `(Cl,)`.

    compute() -> Dict[str, Any]
        Compute metric value(s) for currently cached predictions and targets, returned as
        a dictionary.

    reset() -> None
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
    An augmentation protocol for the image classification subproblem.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `Sequence[ArrayLike]` (with elements of shape `(C, H, W)`), `Sequence[ArrayLike]`
    (with elements of shape `(Cl, )`), and `Sequence[Dict[str,Any]]`. These correspond
    to the model input batch type, model target batch type, and datum-level metadata
    batch type, respectively.

    Methods
    -------

    __call__(datum: Tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[dict[str, Any]]]) ->\
          Tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[dict[str, Any]]])
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as `Sequence[ArrayLike]` with elements of shape
        `(C, H, W)`), model target batch (as `Sequence[ArrayLike]` with elements of shape
        `(Cl,)`), and batch metadata (as `Sequence[Dict[str, Any]]`), respectively.

    Examples
    --------

    We can write an implementer of the augmentation class as either a function or a class.
    The only requirement is that the object provide a __call__ method that takes objects
    at least as general as the types promised in the protocol signature and return types
    at least as specific.

    >>> import copy
    >>> import numpy as np
    >>> from typing import Dict, Any, Tuple, Sequence
    >>> from maite.protocols import ArrayLike
    >>>
    >>> class ImageAugmentation:
    ...     def __call__(
    ...         self,
    ...         data_batch: Tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[Dict[str, Any]]]
    ...     ) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[Dict[str, Any]]]:
    ...         inputs, targets, mds = data_batch
    ...         # We copy data passed into the constructor to avoid mutating original inputs
    ...         # By using np.ndarray constructor, the static type-checker will let us treat
    ...         # generic ArrayLike as a more narrow return type
    ...         inputs_aug = [copy.copy(np.array(input)) for input in inputs]
    ...         targets_aug = [copy.copy(np.array(target)) for target in targets]
    ...         mds_aug = copy.deepcopy(mds)  # deepcopy in case of nested structure
    ...         # Modify inputs_aug, targets_aug, or mds_aug as needed
    ...         # In this example, we just add a new metadata field
    ...         for i, md in enumerate(mds_aug):
    ...             md['new_key'] = i
    ...         return inputs_aug, targets_aug, mds_aug

    We can typehint an instance of the above class as an Augmentation in the
    image_classification domain:

    >>> from maite.protocols import image_classification as ic
    >>> im_aug: ic.Augmentation = ImageAugmentation()

    """
