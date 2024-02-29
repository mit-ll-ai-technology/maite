# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Hashable, Protocol, runtime_checkable, Any

# define ArrayLike protocol
@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
        ...


# define minimal DatumMetadata protocol class
@runtime_checkable
class DatumMetadata(Protocol):
    @property
    def uuid(self) -> Hashable:
        ...