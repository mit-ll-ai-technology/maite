# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike as NumpyArrayLike
from typing_extensions import TypeAlias

ArrayLike: TypeAlias = NumpyArrayLike


# define minimal DatumMetadata protocol class
@runtime_checkable
class DatumMetadata(Protocol):
    @property
    def uuid(self) -> int:
        ...
