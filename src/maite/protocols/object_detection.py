# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TypeAlias

from maite._internals.protocols.object_detection import (  # isort:skip
    Image as _Image,
    InputType as _InputType,
    TargetType as _TargetType,
    DatumMetadataType as _DatumMetadataType,
    DatumMetadata as _DatumMetadata,
    ObjectDetectionTarget,
    Augmentation,
    DataLoader,
    Dataset,
    FieldwiseDataset,
    Metric,
    Model,
)

# Note: we need to put triple-quoted strings in public-facing module for sphinx to pick them up
# (This, and not wanting to use two unlinked aliases is why we use _-prefixed values below)

Image: TypeAlias = _Image
"""Semantic alias for a single image datum.

Use `Image` when emphasizing domain meaning ("this value is an image"), rather
than protocol position. Expected shape semantics are `(C, H, W)`.
"""

DatumMetadata = _DatumMetadata
"""Semantic alias for per-datum metadata in object-detection tasks.

Use `DatumMetadata` when defining or discussing metadata fields for one example
(e.g., required `id` plus optional task-specific fields).
"""

InputType: TypeAlias = _InputType
"""Role alias for model/dataset input in the object-detection protocol family.

Use `InputType` in generic protocol contexts (`Dataset`, `DataLoader`, `Model`,
`Augmentation`) where the type parameter represents "input position". Currently
equivalent to :py:type:`~maite.protocols.object_detection.Image`.
"""

TargetType: TypeAlias = _TargetType
"""Role alias for model/dataset target in the object-detection protocol family.

Use `TargetType` in generic protocol contexts where the type parameter represents
"target position". Currently equivalent to
:py:type:`~maite.protocols.object_detection.ObjectDetectionTarget`.
"""

DatumMetadataType: TypeAlias = _DatumMetadataType
"""Role alias for datum-level metadata in object-detection protocol signatures.

Use `DatumMetadataType` in generic protocol contexts where metadata appears as a
type argument. Currently equivalent to
:py:type:`~maite.protocols.object_detection.DatumMetadata`.
"""

## TODO: consider exporting _Datum
# Datum: TypeAlias = _Datum
# """Alias of tuple[:py:type:`~maite.protocols.object_detection.InputType`, :py:type:`~maite.protocols.object_detection.TargetType`, :py:type:`~maite.protocols.object_detection.DatumMetadataType`]"""


__all__ = [
    "Image",
    "InputType",
    "TargetType",
    "DatumMetadataType",
    "DatumMetadata",
    "Augmentation",
    "DataLoader",
    "Dataset",
    "FieldwiseDataset",
    "Metric",
    "Model",
    "ObjectDetectionTarget",
]
