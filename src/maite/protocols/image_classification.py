# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TypeAlias

from maite._internals.protocols.image_classification import (  # isort:skip
    Image as _Image,
    InputType as _InputType,
    TargetType as _TargetType,
    DatumMetadataType as _DatumMetadataType,
    ImgClassification as _ImgClassification,
    DatumMetadata as _DatumMetadata,
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

Use `Image` when you want to emphasize domain meaning ("this value is an image"),
rather than protocol position. Expected shape semantics are `(C, H, W)`.
"""

ImgClassification: TypeAlias = _ImgClassification
"""Semantic alias for a single classification target/prediction vector.

Use `ImgClassification` when referring to class-label vectors directly
(e.g., labels, probabilities, or logits). Expected shape semantics are `(Cl,)`,
where 'Cl' refers to number of target classes.
"""

DatumMetadata = _DatumMetadata
"""Semantic alias for per-datum metadata in image-classification tasks.

Use `DatumMetadata` when defining or discussing metadata fields for one example
(e.g., required `id` plus any task-specific extensions).
"""

InputType: TypeAlias = _InputType
"""Role alias for model/dataset input in the image-classification protocol family.

Use `InputType` in generic protocol contexts (`Dataset`, `DataLoader`, `Model`,
`Augmentation`) where the type parameter represents "input position". Currently
equivalent to :py:type:`~maite.protocols.image_classification.Image`.
"""

TargetType: TypeAlias = _TargetType
"""Role alias for model/dataset target in the image-classification protocol family.

Use `TargetType` in generic protocol contexts where the type parameter represents
"target position". Currently equivalent to
:py:type:`~maite.protocols.image_classification.ImgClassification`.
"""

DatumMetadataType: TypeAlias = _DatumMetadataType
"""Role alias for datum-level metadata in image-classification protocol signatures.

Use `DatumMetadataType` in generic protocol contexts where metadata appears as a
type argument. Currently equivalent to
:py:type:`~maite.protocols.image_classification.DatumMetadata`.
"""

## TODO: consider exporting _Datum
# Datum: TypeAlias = _Datum
# """Alias of tuple[:py:type:`~maite.protocols.object_detection.InputType`, :py:type:`~maite.protocols.object_detection.TargetType`, :py:type:`~maite.protocols.object_detection.DatumMetadataType`]"""

__all__ = [
    "Image",
    "ImgClassification",
    "DatumMetadata",
    "InputType",
    "TargetType",
    "DatumMetadataType",
    "Augmentation",
    "DataLoader",
    "Dataset",
    "FieldwiseDataset",
    "Metric",
    "Model",
]
