# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


from maite._internals.protocols import ArrayLike as _ArrayLike

# This assignment is necessary to get sphinx/VSCode language server to display docstring around our ArrayLike TypeAlias
# (can't be done in ArrayLike first definition _internals or sphinx doesn't pick it up)
ArrayLike = _ArrayLike
"""
Object coercible into a NumPy ndarray (alias of `numpy.typing.ArrayLike`)
"""

# E402 because ruff complains assignments above imports, but one example in project.py relies on ArrayLike being first imported
from maite._internals.protocols.generic import (  # noqa: E402
    Augmentation,
    AugmentationMetadata,
    DataLoader,
    Dataset,
    DatasetMetadata,
    DatumMetadata,
    Metric,
    MetricMetadata,
    Model,
    ModelMetadata,
)

__all__ = [
    "ArrayLike",
    "DatasetMetadata",
    "ModelMetadata",
    "MetricMetadata",
    "AugmentationMetadata",
    "DatumMetadata",
    "Model",
    "Metric",
    "DataLoader",
    "Dataset",
    "Augmentation",
]
