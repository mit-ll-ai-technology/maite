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
    A dataset protocol for object detection ML subproblem providing datum-level data
    access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond
    to individual examples (as opposed to batches).

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `ArrayLike`, `ObjectDetectionTarget`, and `DatumMetadataType`. These
    correspond to the model input type, model target type, and datum-level metadata,
    respectively.

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

    Examples
    --------

    We create a dummy set of a data and use it to create a class the implements this
    lightweight dataset protocol:

    >>> import numpy as np
    >>> from dataclasses import dataclass
    >>> from maite.protocols import (
    ...     DatumMetadata,
    ...     DatasetMetadata,
    ...     object_detection as od,
    ... )

    Specify the parameters that will be used to create the dummy dataset.

    >>> N_DATUM = 5  # data points in dataset
    >>> N_CLASSES = 2  # possible classes that can be detected
    >>> C = 3  # number of color channels
    >>> H = 10  # image height
    >>> W = 10  # image width

    The dummy dataset will consist of a set of images, their associated annotations (detections),
    and some image-specific metadata (hour of day each image was taken).

    We define helper functions to create the dummy annotations. Each image datum will be
    randomly assigned zero, one, or two detections. Annotations for each image consist
    of randomly generated bounding boxes and class labels.

    >>> def generate_random_bbox(
    ...     n_classes: int, min_size: int = 2, max_size: int = 4
    ... ) -> np.ndarray:
    ...     # Generate random coordinates for top-left corner of bbox
    ...     x1 = np.random.randint(0, W - min_size)
    ...     y1 = np.random.randint(0, H - min_size)
    ...     # Generate random width and height, ensuring bounding box stays within image boundaries
    ...     bbox_width = np.random.randint(min_size, min(max_size, W - x1))
    ...     bbox_height = np.random.randint(min_size, min(max_size, H - y1))
    ...     # Set coordinates for bottom-right corner of bbox
    ...     x2 = x1 + bbox_width
    ...     y2 = y1 + bbox_height
    ...     # Pick random class label
    ...     label = np.random.choice(n_classes)
    ...     return np.array([x1, y1, x2, y2, label])

    >>> def generate_random_annotation(max_num_detections: int = 2) -> np.ndarray:
    ...     num_detections = np.random.choice(max_num_detections + 1)
    ...     annotation = [generate_random_bbox(N_CLASSES) for _ in range(num_detections)]
    ...     return np.vstack(annotation) if num_detections > 0 else np.empty(0)

    We now create the dummy dataset of images, corresponding annotations, and metadata.

    >>> images: list[np.ndarray] = list(np.random.rand(N_DATUM, C, H, W))
    >>> annotations: list[np.ndarray] = [
    ...     generate_random_annotation() for _ in range(N_DATUM)
    ... ]
    >>> hour_of_day: list[int] = [np.random.choice(24) for _ in range(N_DATUM)]
    >>> dataset: list[tuple] = list(zip(images, annotations, hour_of_day))

    To support our MAITE dataset, we create an object detection target class that defines
    the boxes, labels, and scores for each detection in an image.

    >>> @dataclass
    ... class MyObjectDetectionTarget:
    ...     boxes: np.ndarray
    ...     labels: np.ndarray
    ...     scores: np.ndarray

    Lastly, we extend `maite.protocols.DatumMetadata` to hold datum-specifc metadata to
    add the notional hour of day field (in addition to the required unique id).

    >>> class MyDatumMetadata(DatumMetadata):
    ...     hour_of_day: int

    Constructing a compliant dataset now just involves a simple wrapper that fetches
    individual data points, where a data point is a single image, ground truth detection(s),
    and metadata.

    >>> class ImageDataset:
    ...     # Set up required dataset-level metadata
    ...     metadata: DatasetMetadata = {
    ...         "id": "Dummy Dataset",
    ...         "index2label": {i: f"class_name_{i}" for i in range(N_CLASSES)}
    ...     }
    ...     def __init__(self, dataset: list[tuple[np.ndarray, np.ndarray, int]]):
    ...         self.dataset = dataset
    ...     def __len__(self) -> int:
    ...         return len(self.dataset)
    ...     def __getitem__(
    ...         self, index: int
    ...     ) -> tuple[np.ndarray, od.ObjectDetectionTarget, od.DatumMetadataType]:
    ...         if index < 0 or index >= len(self):
    ...             raise IndexError(f"Index {index} is out of range for the dataset, which has length {len(self)}.")
    ...         image, annotations, hour_of_day = self.dataset[index]
    ...         # Structure ground truth target
    ...         boxes, labels = [], []
    ...         for _, ann in enumerate(annotations):
    ...             bbox = ann[:-1]
    ...             label = ann[-1:]
    ...             if len(bbox) != 0:
    ...                 boxes.append(bbox)
    ...                 labels.append(label)
    ...         od_target = MyObjectDetectionTarget(
    ...             boxes=np.array(boxes), labels=np.array(labels), scores=np.ones(len(boxes))
    ...         )
    ...         # Structure datum-level metadata
    ...         datum_metadata: MyDatumMetadata = {"id": str(index), "hour_of_day": hour_of_day}
    ...         return image, od_target, datum_metadata

    We can instantiate this class and type hint it as an object_detection.Dataset. By
    using type hinting, we permit a static typechecker to verify protocol compliance.

    >>> maite_od_dataset: od.Dataset = ImageDataset(dataset)

    Note that when writing a Dataset implementer, return types may be narrower than the
    return types promised by the protocol (np.ndarray is a subtype of ArrayLike), but
    the argument types must be at least as general as the argument types promised by the
    protocol.
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
