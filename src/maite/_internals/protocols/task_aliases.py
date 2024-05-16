# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Define aliases over all tasks"""

# We need a module beyond protocols/__init__.py for 'union aliases' because aliases depend on all
# task-specific modules (e.g. 'image_classification.py') and task-specific modules depend
# on __init__.py (through ArrayLike). Thus, introducing these aliases in __init__.py
# results in a circular import). These task-agnostic aliases are meant to be imported
# only in code where such task-specific aliases are not defined.

# We don't define aliases over all *components* here (e.g. 'AnyProvider: TypeAlias = ...', 'AnyEntrypoint: ...')
# because we don't plan on defining component-specific provider modules defined separately from
# consumers of these 'AnyProvider'. The list of component types should be far more static than
# the list of tasks.

from typing import Literal, Union

from typing_extensions import TypeAlias

from maite._internals.protocols import (
    image_classification as ic,
    object_detection as od,
)

# define "Some<ComponentType>" and "Some<PrimitiveType>" TypeAliases
SomeModel: TypeAlias = Union[ic.Model, od.Model]
SomeMetric: TypeAlias = Union[ic.Metric, od.Metric]
SomeDataset: TypeAlias = Union[ic.Dataset, od.Dataset]
SomeDataLoader: TypeAlias = Union[ic.DataLoader, od.DataLoader]
SomeAugmentation: TypeAlias = Union[ic.Augmentation, od.Augmentation]

SomeInputBatchType: TypeAlias = Union[ic.InputBatchType, od.InputBatchType]
SomeTargetBatchType: TypeAlias = Union[ic.TargetBatchType, od.TargetBatchType]
SomeMetadataBatchType: TypeAlias = Union[
    ic.DatumMetadataBatchType, od.DatumMetadataBatchType
]

SomeInputType: TypeAlias = Union[ic.InputType, od.InputType]
SomeTargetType: TypeAlias = Union[ic.TargetType, od.TargetType]
SomeMetadataType: TypeAlias = Union[ic.DatumMetadataType, od.DatumMetadataType]

TaskName: TypeAlias = Literal["object-detection", "image-classification"]
