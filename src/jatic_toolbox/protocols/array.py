from typing import Any, Generic, NewType, Sequence, SupportsIndex, Tuple, Union

from numpy.typing import NDArray as _NDArray
from typing_extensions import Protocol, TypeVarTuple, Unpack, runtime_checkable

__all__ = ["ArrayLike", "ShapedArray"]


Shape = TypeVarTuple("Shape")
Height = NewType("Height", int)
Width = NewType("Width", int)


@runtime_checkable
class ArrayLike(Protocol):
    "A protocol for array like objects that can be coecerd into a given type of array."

    def __array__(self) -> Any:
        ...


@runtime_checkable
class ShapedArray(ArrayLike, Protocol[Unpack[Shape]]):
    # Carries variadic shape information and supports array-like
    # see: https://peps.python.org/pep-0646/
    @property
    def shape(self) -> Tuple[Unpack[Shape]]:
        ...


class NDArray(Generic[Unpack[Shape]], _NDArray[Any]):
    @property
    def shape(self) -> Tuple[Unpack[Shape]]:
        ...

    @shape.setter
    def shape(self, value: Union[SupportsIndex, Sequence[SupportsIndex]]) -> None:
        ...
