# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for image_classification
from __future__ import annotations

from typing import Any, Dict, Sequence

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

Dataset = gen.Dataset[InputType, TargetType, MetadataType]

DataLoader = gen.DataLoader[
    InputBatchType,
    TargetBatchType,
    MetadataBatchType,
]
Model = gen.Model[InputBatchType, TargetBatchType]
Metric = gen.Metric[TargetBatchType]

Augmentation = gen.Augmentation[
    InputBatchType,
    TargetBatchType,
    MetadataBatchType,
    InputBatchType,
    TargetBatchType,
    MetadataBatchType,
]
