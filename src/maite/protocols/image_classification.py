# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for image_classification

from . import DatumMetadata

from typing import (
    Protocol,
    Sequence,
    Any,
    runtime_checkable,
    Hashable,
)

from typing_extensions import TypeAlias

from . import generic as gen


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any: ...


# In below, the dimension names/meanings used are:
#
# N  - image instance
# H  - image height
# W  - image width
# C  - image channel
# Cl - classification label (one-hot for ground-truth label, or pseudo-probabilities for predictions)

InputType: TypeAlias = ArrayLike  # shape [H, W, C]
OutputType: TypeAlias = ArrayLike  # shape [Cl]
MetadataType: TypeAlias = DatumMetadata

InputBatchType: TypeAlias = ArrayLike  # shape [N, H, W, C]
OutputBatchType: TypeAlias = ArrayLike  # shape [N, Cl]
MetadataBatchType: TypeAlias = Sequence[MetadataType]

# Initialize component classes based on generic and Input/Output/Metadata types

Dataset = gen.Dataset[InputType, OutputType, MetadataType]
DataLoader = gen.DataLoader[
    InputType,
    OutputType,
    MetadataType,
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
]
Model = gen.Model[InputType, OutputType, InputBatchType, OutputBatchType]
Metric = gen.Metric[OutputType, OutputBatchType]

Augmentation = gen.Augmentation[
    InputType,
    OutputType,
    MetadataType,
    InputType,
    OutputType,
    MetadataType,
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
]