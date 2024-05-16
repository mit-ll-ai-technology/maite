# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Protocol, runtime_checkable


# define ArrayLike protocol
@runtime_checkable
class ArrayLike(Protocol):
    """
    A protocol for an array-like object.

    Examples
    --------

    Arrays like NumPy NDArray objects are `ArrayLike` along
    with PyTorch and JAX tensors.

    >>> import numpy as np
    >>> from maite.protocols import ArrayLike
    >>> array_like: ArrayLike = np.ones((3, 224, 224))
    >>> isinstance(array_like, ArrayLike)
    True
    """

    def __array__(self) -> Any:
        ...


# define minimal DatumMetadata protocol class
@runtime_checkable
class DatumMetadata(Protocol):
    @property
    def uuid(self) -> int:
        ...
