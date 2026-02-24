# import component generics from generic.py and specialize them for
# multi-object tracking domain

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Protocol, Sequence, TypeAlias

from typing_extensions import ReadOnly, Required

from maite import protocols
from maite._internals.protocols import generic as gen
from maite.protocols import ArrayLike


class VideoFrame(Protocol):
    """
    Contents of a single decoded video frame.

    Attributes
    ----------
    pixels : ArrayLike
        An array representing pixel values in a single frame with (C, H, W) shape semantics

    time_s : float
        Time associated with video frame (relative to 0 seconds for first frame of source)

    pts : int
        Presentation time stamp associated with video frame (relative to 0 for first frame of source)

    frame_index : int
        Zero-based index of this frame within the yielded `VideoStream`
        (i.e., decode/sampling output order). This is not necessarily the
        absolute frame number in the source video.
    """

    @property
    def pixels(self) -> ArrayLike: ...

    @property
    def time_s(self) -> float: ...

    @property
    def pts(self) -> int: ...

    @property
    def frame_index(self) -> int: ...


VideoStream: TypeAlias = Iterable[VideoFrame]


class SingleFrameObjectTrackingTarget(Protocol):
    """
    Single-frame object-tracking target.

    This class is used to encode frame-level detections/tracks (both predictions and ground-truth labels) in the
    multi-object tracking AI problem.

    Implementers must populate the following attributes:

    Attributes
    ----------
    boxes : ArrayLike
        An array representing object detection boxes in a single frame with x0, y0, x1, y1
        format and shape `(N_DETECTIONS, 4)`

    labels : ArrayLike
        An array representing the integer labels associated with each detection box of shape
        `(N_DETECTIONS,)`

    scores: ArrayLike
        An array representing the scores associated with each box (of shape `(N_DETECTIONS,)`
        or `(N_DETECTIONS, N_CLASSES)`.

    track_ids: ArrayLike
        An array of per-detection track identifiers with shape `(N_DETECTIONS,)`.
        Use non-negative integer-like values for tracked detections, and `-1` for
        detections that are not assigned to any track. `NaN` values are not supported.
    """

    @property
    def boxes(
        self,
    ) -> ArrayLike:  # shape (N_DETECTIONS, 4), format X0, Y0, X1, Y1
        ...

    @property
    def labels(self) -> ArrayLike:  # label for each box, shape (N_DETECTIONS,)
        ...

    @property
    def scores(self) -> ArrayLike:  # shape (N_DETECTIONS,) or (N_DETECTIONS, CLASSES)
        ...

    @property
    def track_ids(
        self,
    ) -> ArrayLike:  # shape (N_DETECTIONS,); track IDs are >= 0, with -1 reserved for untracked detections (no NaNs)
        ...


class MultiobjectTrackingTarget(Protocol):
    """Set of tracked objects over a sequence of frames.

    Attributes
    ----------
    frame_tracks : Sequence[SingleFrameObjectTrackingTarget]
        A sequence whose elements represent frame-level detections in sequential frames of a video

    """

    @property
    def frame_tracks(self) -> Sequence[SingleFrameObjectTrackingTarget]: ...


## Alternative (that wouldn't allow for additional fields)
# MultiobjectTrackingTarget: TypeAlias = Sequence[SingleFrameObjectTrackingTarget]


class DatumMetadata(protocols.DatumMetadata):
    """
    Typed dictionary containing salient characteristics of video.

    Attributes
    ----------
    id : ReadOnly[str | int]
        Unique datum identifier
    height : ReadOnly[int]
        Video height in pixels
    width : ReadOnly[int]
        Video width in pixels
    time_base : ReadOnly[Fraction]
        Video time base in fractions of a second
    size : ReadOnly[int]
        Video size in bytes
    """

    id: Required[ReadOnly[str | int]]
    height: ReadOnly[int]
    width: ReadOnly[int]
    time_base: ReadOnly[Fraction]
    size: ReadOnly[int]


class DatasetMetadata(protocols.DatasetMetadata):
    """
    Multi-object tracking dataset-level metadata

    Dictionaries satisfying this type require an 'id' for the
    dataset and may provide an index2label field mapping integer
    identifiers for class labels to their string descriptions

    Attributes
    ----------
    id : str
        Identifier for a single Dataset instance
    index2label : NotRequired[ReadOnly[dict[int, str]]]
        Mapping from integer labels to corresponding string descriptions
    """

    ...


# Type aliases for convenience.

InputType: TypeAlias = VideoStream

TargetType: TypeAlias = MultiobjectTrackingTarget

DatumMetadataType: TypeAlias = DatumMetadata

Datum: TypeAlias = tuple[InputType, TargetType, DatumMetadataType]


class Dataset(
    gen.Dataset[InputType, TargetType, DatumMetadataType],
    Protocol,
):
    """
    A dataset protocol for multi-object tracking AI problem providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond
    to individual examples (as opposed to batches).

    Indexing into or iterating over an multi-object tracking dataset returns a
    `tuple` of types `VideoStream`, `MultiobjectTrackingTarget`, and `DatumMetadata`.
    These correspond to the model input type, model target type, and datum-level
    metadata, respectively.

    Methods
    -------

    __getitem__(ind: int) -> tuple[VideoStream, MultiobjectTrackingTarget, DatumMetadata]
        Provide map-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata type,
        respectively.

    __len__() -> int
        Return the number of data elements in the dataset.

    Attributes
    ----------

    metadata : DatasetMetadata
        A read-only typed dictionary containing at least an 'id' key of type str
        that may also contain an 'index2label' key of type dict[str,int] mapping integer
        indices to str labels.
    """

    ...


class DataLoader(
    gen.DataLoader[InputType, TargetType, DatumMetadataType],
    Protocol,
):
    """
    A dataloader protocol for the multi-object tracking AI problem providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `Sequence[VideoStream]`, `Sequence[MultiobjectTrackingTarget]`,
    and `Sequence[DatumMetadata]`, which correspond to model input batch, model
    target type batch, and a datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.

    Methods
    -------

    __iter__ -> Iterator[tuple[Sequence[VideoStream], Sequence[MultiobjectTrackingTarget], Sequence[DatumMetadata]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as `Sequence[VideoStream]`), model target batch (as
        `Sequence[MultiobjectTrackingTarget]`), and batched datum-level metadata
        (as `Sequence[DatumMetadata]`), respectively.

    """

    ...


class Model(gen.Model[InputType, TargetType], Protocol):
    """
    A model protocol for the multi-object tracking AI problem.

    Implementers must provide a `__call__` method that operates on a batch of model
    inputs (as `Sequence[VideoStream]`) and returns a batch of model targets (as
    `Sequence[MultiobjectTrackingTarget]`)

    Methods
    -------

    __call__(input_batch: Sequence[VideoStream]) -> Sequence[MultiobjectTrackingTarget]
        Make a model prediction for inputs in input batch. Input batch is expected to
        be `Sequence[VideoStream]`. Target batch is expected to be
        `Sequence[MultiobjectTrackingTarget]`.

    Attributes
    ----------

    metadata : ModelMetadata
        A typed dictionary containing at least an 'id' field of type str

    """

    ...


class Metric(gen.Metric[TargetType, DatumMetadataType], Protocol):
    """
    A metric protocol for the multi-object tracking AI problem.

    A metric in this sense is expected to measure the level of agreement between model
    predictions and ground-truth labels.

    Methods
    -------

    update(pred_batch: Sequence[MultiobjectTrackingTarget], target_batch: Sequence[MultiobjectTrackingTarget], metadata_batch: Sequence[DatumMetadata]) -> None
        Add predictions and targets (and metadata if applicable) to metric's cache for later calculation.

    compute() -> Mapping[str, Any]
        Compute metric value(s) for currently cached predictions and targets, returned as
        a read-only mapping.

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
        InputType,
        TargetType,
        DatumMetadataType,
        InputType,
        TargetType,
        DatumMetadataType,
    ]
):
    """
    An augmentation protocol for the multi-object tracking AI problem.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `Sequence[VideoStream]`, `Sequence[MultiobjectTrackingTarget]`, and 
    `Sequence[DatumMetadata]`. These correspond to the model input batch type, 
    model target batch type, and datum-level metadata batch type, respectively.

    Methods
    -------

    __call__(datum: tuple[Sequence[VideoStream], Sequence[MultiobjectTrackingTarget], Sequence[DatumMetadata]]) ->\
          tuple[Sequence[VideoStream], Sequence[MultiobjectTrackingTarget], Sequence[DatumMetadata]])
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as `Sequence[VideoStream]`, model target batch
        (as `Sequence[MultiobjectTrackingTarget]`, and batch metadata 
        (as `Sequence[DatumMetadata]`), respectively.

    Attributes
    ----------

    metadata : AugmentationMetadata
        A typed dictionary containing at least an 'id' field of type str
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
    """
