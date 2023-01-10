from typing import Protocol

from numpy.typing import NDArray

__all__ = ["ArrayLike"]


class ArrayLike(Protocol):
    "A protocol for array like objects that can be coecerd into NumPy arrays."

    def __array__(self) -> NDArray:
        ...
