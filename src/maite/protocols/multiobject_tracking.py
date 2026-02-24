from typing import TypeAlias

from maite._internals.protocols.multiobject_tracking import (  # isort:skip
    InputType as _InputType,
    TargetType as _TargetType,
    DatumMetadataType as _DatumMetadataType,
    VideoFrame,
    VideoStream as _VideoStream,
    SingleFrameObjectTrackingTarget,
    MultiobjectTrackingTarget,
    DatumMetadata,
    Augmentation,
    DataLoader,
    Dataset,
    FieldwiseDataset,
    Metric,
    Model,
)

# Note: we need to put triple-quoted strings in public-facing module for sphinx to pick them up
# (This, and not wanting to use two unlinked aliases is why we use _-prefixed values below)

VideoStream: TypeAlias = _VideoStream
"""Semantic alias for a video input stream as `Iterable[VideoFrame]`.

Use `VideoStream` when emphasizing that a value is a temporally ordered sequence
of frame-level inputs for a single video.
"""

InputType: TypeAlias = _InputType
"""Role alias for model/dataset inputs in multi-object-tracking protocol signatures.

Use `InputType` in generic protocol positions (`Dataset`, `DataLoader`, `Model`,
`Augmentation`) where the type parameter represents "input position".
Currently equivalent to :py:type:`~maite.protocols.multiobject_tracking.VideoStream`.
"""

TargetType: TypeAlias = _TargetType
"""Role alias for model/dataset targets in multi-object-tracking protocol signatures.

Use `TargetType` in generic protocol positions where the type parameter represents
"target position". Currently equivalent to
:py:type:`~maite.protocols.multiobject_tracking.MultiobjectTrackingTarget`.
"""

DatumMetadataType: TypeAlias = _DatumMetadataType
"""Role alias for per-datum metadata in multi-object-tracking protocol signatures.

Use `DatumMetadataType` in generic protocol positions where metadata appears as a
type argument. Currently equivalent to
:py:type:`~maite.protocols.multiobject_tracking.DatumMetadata`.
"""

__all__ = [
    "InputType",
    "TargetType",
    "DatumMetadataType",
    "VideoFrame",
    "VideoStream",
    "SingleFrameObjectTrackingTarget",
    "MultiobjectTrackingTarget",
    "DatumMetadata",
    "Augmentation",
    "DataLoader",
    "Dataset",
    "FieldwiseDataset",
    "Metric",
    "Model",
]
