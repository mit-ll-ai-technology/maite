# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
        An array representing the scores associated with each box (of shape `(N_DETECTIONS,)`
        or `(N_DETECTIONS, N_CLASSES)`.
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
    def scores(self) -> ArrayLike:  # shape (N,) or (N, CLASSES)
        ...


# TODO: remove typeAlias statements for more user readability (or figure out how to resolve TypeAliases
#       to their targets for end-user.) Knowing a dataset returns a tuple of "InputType, TargetType, DatumMetadataType"
#       isn't helpful to implementers, however the aliasing *is* helpful to developers.
#
#       Perhaps the functionality I want is named TypeVars for generic, so developers can understand that
#       e.g. generic.Dataset typevars are 'InputType', 'TargetType', and 'MetaDataType' and their values in
#       concrete Dataset classes (like object_detection.Dataset) are ArrayLike, ObjectDetectionTarget, DatumMetadataType
#       so users can see an expected return type of Tuple[ArrayLike, ObjectDetectionTarget, DatumMetadata]

InputType: TypeAlias = ArrayLike
"""ArrayLike following (C, H, W) shape semantics"""

TargetType: TypeAlias = ObjectDetectionTarget

DatumMetadataType: TypeAlias = DatumMetadata
"""TypedDict that requires a readonly 'id' field of type `int|str`"""

Datum: TypeAlias = tuple[InputType, TargetType, DatumMetadataType]


class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
    """
    A dataset protocol for object detection AI problem providing datum-level data
    access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond
    to individual examples (as opposed to batches).

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `ArrayLike`, `ObjectDetectionTarget`, and `DatumMetadata`. These
    correspond to the model input type, model target type, and datum-level metadata,
    respectively. The `ArrayLike` protocol implementers associated with model input and
    model target types are expected to follow (C, H, W) shape semantics.

    Methods
    -------

    __getitem__(ind: int) -> Tuple[ArrayLike, ObjectDetectionTarget, DatumMetadata]
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
    ...     annotation = [
    ...         generate_random_bbox(N_CLASSES) for _ in range(num_detections)
    ...     ]
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
    ...         "index2label": {i: f"class_name_{i}" for i in range(N_CLASSES)},
    ...     }
    ...
    ...     def __init__(self, dataset: list[tuple[np.ndarray, np.ndarray, int]]):
    ...         self.dataset = dataset
    ...
    ...     def __len__(self) -> int:
    ...         return len(self.dataset)
    ...
    ...     def __getitem__(
    ...         self, index: int
    ...     ) -> tuple[np.ndarray, od.ObjectDetectionTarget, DatumMetadata]:
    ...         if index < 0 or index >= len(self):
    ...             raise IndexError(
    ...                 f"Index {index} is out of range for the dataset, which has length {len(self)}."
    ...             )
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
    ...             boxes=np.array(boxes),
    ...             labels=np.array(labels),
    ...             scores=np.ones(len(boxes)),
    ...         )
    ...         # Structure datum-level metadata
    ...         datum_metadata: MyDatumMetadata = {
    ...             "id": str(index),
    ...             "hour_of_day": hour_of_day,
    ...         }
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

    get_target(index: int, /) -> ObjectDetectionTarget:
        Get target at the given index.

    get_metadata(index: int, /) -> DatumMetadataType:
        Get metadata at the given index.

    Examples
    --------
    We create a dummy set of data and use it to create a class that implements
    the dataset protocol:

    >>> import numpy as np
    >>> from maite.protocols import (
    ...     DatumMetadata,
    ...     DatasetMetadata,
    ...     object_detection as od,
    ... )

    Constructing a compliant dataset now just involves a simple wrapper that fetches
    individual data points, where a data point is a single image, ground truth detection(s),
    and metadata.

    >>> class ExampleDataset:
    ...     # Set up required dataset-level metadata
    ...     metadata: DatasetMetadata = {
    ...         "id": "Dummy Dataset",
    ...         "index2label": {i: f"class_{i}" for i in range(5)},
    ...     }
    ...
    ...     def __init__(
    ...         self,
    ...         inputs: list[np.ndarray],
    ...         targets: list[TargetType],
    ...         metadatas: list[DatumMetadataType],
    ...     ):
    ...         self.inputs = inputs
    ...         self.targets = targets
    ...         self.metadatas = metadatas
    ...
    ...     def __len__(self) -> int:
    ...         return len(self.inputs)
    ...
    ...     def __getitem__(
    ...         self, index: int
    ...     ) -> tuple[np.ndarray, od.ObjectDetectionTarget, od.DatumMetadataType]:
    ...         image = self.inputs[index]
    ...         target = self.targets[index]
    ...         metadata = self.metadatas[index]
    ...         return image, target, metadata
    ...
    ...     def get_input(self, index: int, /) -> np.ndarray:
    ...         return self.inputs[index]
    ...
    ...     def get_target(self, index: int, /) -> ObjectDetectionTarget:
    ...         return self.targets[index]
    ...
    ...     def get_metadata(self, index: int, /) -> DatumMetadataType:
    ...         return self.metadatas[index]

    We can instantiate this class and type hint it as an object_detection.Dataset. By
    using type hinting, we permit a static typechecker to verify protocol compliance.

    >>> maite_od_dataset: od.FieldwiseDataset = ExampleDataset([], [], [])
    """

    ...


class DataLoader(
    gen.DataLoader[
        InputType,
        TargetType,
        DatumMetadataType,
    ],
    Protocol,
):
    """
    A dataloader protocol for the object detection AI problem providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `Sequence[ArrayLike]` (elements of shape `(C, H, W)`),
    `Sequence[ObjectDetectionTarget]`, and `Sequence[DatumMetadata]`,
    which correspond to model input batch, model target batch, and a datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.


    Methods
    -------

    __iter__ -> Iterator[tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[DatumMetadata]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as `Sequence[ArrayLike]`), model target batch (as
        `Sequence[ObjectDetectionTarget]`), and batched datum-level metadata
        (as `Sequence[DatumMetadata]]`), respectively.
    """

    ...


class Model(gen.Model[InputType, TargetType], Protocol):
    """
    A model protocol for the object detection AI problem.

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

    Examples
    --------

    We create a simple MAITE-compliant object detection model, note it is a dummy model.

    >>> from dataclasses import dataclass
    >>> from typing import Sequence
    >>> import numpy as np
    >>> import maite.protocols.object_detection as od
    >>> from maite.protocols import ModelMetadata

    We define an object detection target dataclass as the output of the object detection model

    >>> @dataclass
    ... class MyObjectDetectionTarget:
    ...     boxes: np.ndarray
    ...     labels: np.ndarray
    ...     scores: np.ndarray

    Specify parameters that will be used to create a dummy dataset.

    >>> N_DATAPOINTS = 2  # datapoints in dataset
    >>> N_CLASSES = 5  # possible classes that can be detected
    >>> C = 3  # number of color channels
    >>> H = 32  # img height
    >>> W = 32  # img width

    Now create a batch of data to form the inputs of the MAITE's object detection model.

    >>> simple_batch: list[np.ndarray] = [
    ...     np.random.rand(C, H, W) for _ in range(N_DATAPOINTS)
    ... ]

    We define a simple object detection model, note here there is not an actual object detection
    model. In the __call__ method, it just outputs the MyObjectDetectionTarget.

    >>> class ObjectDetectionDummyModel:
    ...     metadata: ModelMetadata = {"id": "ObjectDetectionDummyModel"}
    ...
    ...     def __call__(
    ...         self, batch: Sequence[od.InputType]
    ...     ) -> Sequence[MyObjectDetectionTarget]:
    ...         # For the simplicity, we don't provide an object detection model here, but the output from a model.
    ...         DETECTIONS_PER_IMG = (
    ...             2  # number of bounding boxes detections per image/datapoints
    ...         )
    ...         all_boxes = np.array(
    ...             [[1, 3, 5, 9], [2, 5, 8, 12], [4, 10, 8, 20], [3, 5, 6, 15]]
    ...         )  # all detection boxes for N_DATAPOINTS
    ...         all_predictions = list()
    ...         for datum_idx in range(N_DATAPOINTS):
    ...             boxes = all_boxes[datum_idx : datum_idx + DETECTIONS_PER_IMG]
    ...             labels = np.random.randint(N_CLASSES, size=DETECTIONS_PER_IMG)
    ...             scores = np.random.rand(DETECTIONS_PER_IMG)
    ...             predictions = MyObjectDetectionTarget(boxes, labels, scores)
    ...             all_predictions.append(predictions)
    ...         return all_predictions

    We can instantiate this class and typehint it as a maite object detection model.
    By using typehinting, we permit a static typechecker to verify protocol compliance.

    >>> od_dummy_model: od.Model = ObjectDetectionDummyModel()
    >>> od_dummy_model.metadata
    {'id': 'ObjectDetectionDummyModel'}
    >>> predictions = od_dummy_model(simple_batch)
    >>> predictions  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [MyObjectDetectionTarget(boxes=array([[ 1,  3,  5,  9], [ 2,  5,  8, 12]]), labels=array([..., ...]), scores=array([..., ...])),
    MyObjectDetectionTarget(boxes=array([[ 2,  5,  8, 12], [ 4, 10,  8, 20]]), labels=array([..., ...]), scores=array([..., ...]))]
    """

    ...


class Metric(gen.Metric[TargetType, DatumMetadataType], Protocol):
    """
    A metric protocol for the object detection AI problem.

    A metric in this sense is expected to measure the level of agreement between model
    predictions and ground-truth labels.

    Methods
    -------

    update(pred_batch: Sequence[ObjectDetectionTarget], target_batch: Sequence[ObjectDetectionTarget], metadata_batch: Sequence[DatumMetadata]) -> None
         Add predictions and targets (and metadata if applicable) to metric's cache for later calculation.

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

    Below, we write and test a class that implements the Metric protocol for object detection.
    For simplicity, the metric will compute the intersection over union (IoU) averaged over
    all predicted and associated ground truth boxes for a single image, and then take the mean
    over the per-image means.

    Note that when writing a `Metric` implementer, return types may be narrower than the
    return types promised by the protocol, but the argument types must be at least as
    general as the argument types promised by the protocol.

    >>> from dataclasses import dataclass
    >>> from maite.protocols import ArrayLike, MetricMetadata, DatumMetadata
    >>> from typing import Any, Sequence
    >>> import maite.protocols.object_detection as od
    >>> import numpy as np

    >>> class MyIoUMetric:
    ...     def __init__(self, id: str):
    ...         self.pred_boxes = []  # elements correspond to predicted boxes in single image
    ...         self.target_boxes = []  # elements correspond to ground truth boxes in single image
    ...         # Store provided id for this metric instance
    ...         self.metadata = MetricMetadata(id=id)
    ...
    ...     def reset(self) -> None:
    ...         self.pred_boxes = []
    ...         self.target_boxes = []
    ...
    ...     def update(
    ...         self,
    ...         pred_batch: Sequence[od.ObjectDetectionTarget],
    ...         target_batch: Sequence[od.ObjectDetectionTarget],
    ...         metadata_batch: Sequence[DatumMetadata],
    ...     ) -> None:
    ...         self.pred_boxes.extend(pred_batch)
    ...         self.target_boxes.extend(target_batch)
    ...
    ...     @staticmethod
    ...     def iou_vec(boxes_a: ArrayLike, boxes_b: ArrayLike) -> np.ndarray:
    ...         # Break up points into separate columns
    ...         x0a, y0a, x1a, y1a = np.split(boxes_a, 4, axis=1)
    ...         x0b, y0b, x1b, y1b = np.split(boxes_b, 4, axis=1)
    ...         # Calculate intersections
    ...         xi_0, yi_0 = np.split(
    ...             np.maximum(
    ...                 np.append(x0a, y0a, axis=1), np.append(x0b, y0b, axis=1)
    ...             ),
    ...             2,
    ...             axis=1,
    ...         )
    ...         xi_1, yi_1 = np.split(
    ...             np.minimum(
    ...                 np.append(x1a, y1a, axis=1), np.append(x1b, y1b, axis=1)
    ...             ),
    ...             2,
    ...             axis=1,
    ...         )
    ...         ints: np.ndarray = np.maximum(0, xi_1 - xi_0) * np.maximum(
    ...             0, yi_1 - yi_0
    ...         )
    ...         # Calculate unions (as sum of areas minus their intersection)
    ...         unions: np.ndarray = (
    ...             (x1a - x0a) * (y1a - y0a)
    ...             + (x1b - x0b) * (y1b - y0b)
    ...             - (xi_1 - xi_0) * (yi_1 - yi_0)
    ...         )
    ...         return ints / unions
    ...
    ...     def compute(self) -> dict[str, Any]:
    ...         mean_iou_by_img: list[float] = []
    ...         for pred_box, tgt_box in zip(self.pred_boxes, self.target_boxes):
    ...             single_img_ious = self.iou_vec(pred_box.boxes, tgt_box.boxes)
    ...             mean_iou_by_img.append(float(np.mean(single_img_ious)))
    ...         return {"mean_iou": np.mean(np.array(mean_iou_by_img)).item()}

    Now we can instantiate our IoU Metric class:

    >>> iou_metric: od.Metric = MyIoUMetric(id="IoUMetric")

    To use the metric, we populate two lists that encode predicted object detections
    and ground truth object detections for a single image. (Ordinarily, predictions
    would be produced by a model.)

    >>> prediction_boxes: list[tuple[int, int, int, int]] = [
    ...     (1, 1, 12, 12),
    ...     (100, 100, 120, 120),
    ...     (180, 180, 270, 270),
    ... ]

    >>> target_boxes: list[tuple[int, int, int, int]] = [
    ...     (1, 1, 10, 10),
    ...     (100, 100, 120, 120),
    ...     (200, 200, 300, 300),
    ... ]

    The MAITE Metric protocol requires the `pred_batch` and `target_batch` arguments to the
    `update` method to be assignable to Sequence[ObjectDetectionTarget] (where ObjectDetectionTarget
    encodes detections in a single image). We define an implementation of ObjectDetectionTarget and use it
    to pass ground truth and predicted detections.

    >>> @dataclass
    ... class ObjectDetectionTargetImpl:
    ...     boxes: np.ndarray
    ...     labels: np.ndarray
    ...     scores: np.ndarray

    >>> num_boxes = len(target_boxes)
    >>> fake_labels = np.random.randint(0, 9, num_boxes)
    >>> fake_scores = np.zeros(num_boxes)
    >>> pred_batch = [
    ...     ObjectDetectionTargetImpl(
    ...         boxes=np.array(prediction_boxes), labels=fake_labels, scores=fake_scores
    ...     )
    ... ]
    >>> target_batch: Sequence[ObjectDetectionTargetImpl] = [
    ...     ObjectDetectionTargetImpl(
    ...         boxes=np.array(target_boxes), labels=fake_labels, scores=fake_scores
    ...     )
    ... ]
    >>> metadata_batch: Sequence[DatumMetadata] = [{"id": 1}]

    Finally, we call `update` using this one-element batch, compute the metric value, and print it:

    >>> iou_metric.update(pred_batch, target_batch, metadata_batch)
    >>> print(iou_metric.compute())
    {'mean_iou': 0.6802112029384757}
    """


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
    An augmentation protocol for the object detection AI problem.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `Sequence[ArrayLike]`, `Sequence[ObjectDetectionTarget]`, and `Sequence[DatumMetadata]`.
    These correspond to the model input batch type, model target batch type, and datum-level
    metadata batch type, respectively.

    Methods
    -------

    __call__(datum: Tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[DatumMetadata]]) ->\
          Tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[DatumMetadata]]
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as `Sequence ArrayLike` with elements of shape
        `(C, H, W)`), model target batch (as `Sequence[ObjectDetectionTarget]`), and
        batch metadata (as `Sequence[DatumMetadata]`), respectively.

    Attributes
    ----------

    metadata : AugmentationMetadata
        A typed dictionary containing at least an 'id' field of type str

    Examples
    --------

    We can write an implementer of the augmentation class as either a function or a class.
    The only requirement is that the object provide a `__call__` method that takes objects
    at least as general as the types promised in the protocol signature and return types
    at least as specific.

    We create a dummy set of data and use it to create a class the implements this
    lightweight protocol and augments the data:

    >>> import numpy as np
    >>> np.random.seed(1)
    >>> import copy
    >>> from dataclasses import dataclass
    >>> from typing import Any, Sequence
    >>> from maite.protocols import AugmentationMetadata, DatumMetadata, object_detection as od

    First, we specify parameters that will be used to create the dummy dataset.

    >>> N_DATAPOINTS = 3  # datapoints in dataset
    >>> N_CLASSES = 2  # possible classes that can be detected
    >>> C = 3  # number of color channels
    >>> H = 10  # img height
    >>> W = 10  # img width

    Next, we create the input data to be used by the Augmentation. In this example, we
    create the following batch of object detection data:

    • `xb` is the input batch data. Our batch will include `N_DATAPOINTS` number of samples. Note
      that we initialize all of the data to zeros in this example to demonstrate the augmentations
      better.

    • `yb` is the object detection target data, which in this example represents zero object
      detections for each input datum (by having empty bounding boxes and class labels and scores).

    • `mdb` is the associated metadata for each input datum.

    >>> @dataclass
    ... class MyObjectDetectionTarget:
    ...     boxes: np.ndarray
    ...     labels: np.ndarray
    ...     scores: np.ndarray
    >>> xb: Sequence[od.InputType] = list(np.zeros((N_DATAPOINTS, C, H, W)))
    >>> yb: Sequence[od.TargetType] = list(
    ...     MyObjectDetectionTarget(boxes=np.empty(0), labels=np.empty(0), scores=np.empty(0))
    ...     for _ in range(N_DATAPOINTS)
    ... )
    >>> mdb: Sequence[DatumMetadata] = list({"id": i} for i in range(N_DATAPOINTS))
    >>> # Display the first datum in batch, first color channel, and only first 5 rows and cols
    >>> np.set_printoptions(floatmode='fixed', precision=3)  # for reproducible output for doctest
    >>> np.array(xb[0])[0][:5, :5]  # doctest: +NORMALIZE_WHITESPACE
    array([[0.000, 0.000, 0.000, 0.000, 0.000],
           [0.000, 0.000, 0.000, 0.000, 0.000],
           [0.000, 0.000, 0.000, 0.000, 0.000],
           [0.000, 0.000, 0.000, 0.000, 0.000],
           [0.000, 0.000, 0.000, 0.000, 0.000]])

    Now we create the Augmentation, which will apply random noise (rounded to 3 decimal
    places) to the input data using the `numpy.random.random` and `np.round` functions.

    >>> np_noise = lambda shape: np.round(np.random.random(shape), 3)
    >>> class ImageAugmentation:
    ...     def __init__(self, aug_func: Any, metadata: AugmentationMetadata):
    ...         self.aug_func = aug_func
    ...         self.metadata = metadata
    ...     def __call__(
    ...         self,
    ...         batch: tuple[Sequence[od.InputType], Sequence[od.TargetType], Sequence[DatumMetadata]],
    ...     ) -> tuple[Sequence[od.InputType], Sequence[od.TargetType], Sequence[DatumMetadata]]:
    ...         xb, yb, mdb = batch
    ...         # Copy data passed into the constructor to avoid mutating original inputs
    ...         xb_aug = [copy.copy(input) for input in xb]
    ...         # Add random noise to the input batch data, xb
    ...         # (Note that all batch data dimensions (shapes) are the same in this example)
    ...         shape = np.array(xb[0]).shape
    ...         xb_aug = [x + self.aug_func(shape) for x in xb]
    ...         # Note that this example augmentation only affects inputs--not targets
    ...         return xb_aug, yb, mdb

    We can typehint an instance of the above class as an Augmentation in the object
    detection domain:

    >>> noise: od.Augmentation = ImageAugmentation(np_noise, metadata={"id": "np_rand_noise"})

    Now we can apply the `noise` augmentation to our 3-tuple batch of data. Recall that
    our data was initialized to all zeros, so any non-zero values in the augmented data
    is a result of the augmentation.

    >>> xb_aug, yb_aug, mdb_aug = noise((xb, yb, mdb))
    >>> # Display the first datum in batch, first color channel, and only first 5 rows and cols
    >>> np.array(xb_aug[0])[0][:5, :5]  # doctest: +NORMALIZE_WHITESPACE
    array([[0.417, 0.720, 0.000, 0.302, 0.147],
           [0.419, 0.685, 0.204, 0.878, 0.027],
           [0.801, 0.968, 0.313, 0.692, 0.876],
           [0.098, 0.421, 0.958, 0.533, 0.692],
           [0.989, 0.748, 0.280, 0.789, 0.103]])
    """

    ...
