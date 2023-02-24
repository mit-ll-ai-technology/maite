from typing import (
    Any,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from PIL.Image import Image
from typing_extensions import Protocol, Self, TypeAlias, runtime_checkable

__all__ = [
    "TypedCollection",
    "ImageType",
    "ArrayLike",
    "Augmentation",
    "ModelOutput",
    "HasProbs",
    "HasLogits",
    "HasObjectDetections",
    "Model",
    "Classifier",
    "ObjectDetector",
    "ShapedArray",
]


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_cont = TypeVar("T_cont", contravariant=True)


"""
ArrayLike is any method that implements
the `__array__` method. 

ShapedArray is an ArrayLike that also implements
the `shape` method.

ImageType is an ArrayLike or a Image object.  
  - This supports numpy and Image objects 
"""


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
        ...


@runtime_checkable
class ShapedArray(Protocol):
    def __array__(self) -> Any:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...


ImageType: TypeAlias = Union[ArrayLike, Image]
RandomStates: TypeAlias = Any
A = TypeVar("A", bound=ArrayLike)

"""
A `TypedCollection` is a homogeneous collection of Python objects that can be used
to define a consistent type in an interface. For example, if the inputs and outputs
of the interface are expected to be a NumPy `NDArray` or PyTorch `Tensor`, a
`TypedCollection` type can be used to define this consistency.
(e.g., tuple, list, dict, OrderedDict, NamedTuple, etc.).

Typed collections support the very powerful "PyTree" functionalities currently
implemented here:

- `PyTorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/_TypedCollection.py>`_
- `JAX TypedCollections <https://jax.readthedocs.io/en/latest/TypedCollections.html>`_
- `Optimized TypedCollections <https://github.com/metaopt/optree>`_
"""
TypedCollection: TypeAlias = Union[
    T, Sequence[T], Mapping[Any, T], Mapping[Any, Sequence[T]]
]


class Augmentation(Protocol[T]):
    def __call__(
        self,
        *inputs: TypedCollection[T],
        rng: Optional[RandomStates],
    ) -> Union[TypedCollection[T], Tuple[TypedCollection[T], ...]]:
        """
        Applies an agumentation to each item in the input and returns a corresponding container of augmented items.

        Inputs can be arrays or nested data structures of data collections (e.g., list, tuple, dict).

        Parameters
        ----------
        *inputs : TypedCollection[T]
            Any arbitrary structure of nested Python containers, e.g., list of image arrays.
            All types comprising the tree must be the same.

        rng : RandomNumberGenerator | None (default: None)
            An optional random number generator for reproducibility.

        Returns
        -------
        Union[TypedCollection[T], Tuple[TypedCollection[T], ...]]
            A corresponding collection of transformed objects.

        Notes
        -----
        Assumes all data values on inputs and outputs will be the same
        type, e.g., all NumPy NDArrays or all PyTorch Tensors.
        """
        ...


class DataClass(Protocol):
    def __getattribute__(self, __name: str) -> Any:
        ...

    def __setattr__(self, __name: str, __value: Any) -> None:
        ...


ModelOutput: TypeAlias = Union[
    DataClass,
    NamedTuple,
]


@runtime_checkable
class HasLogits(Protocol[A]):
    logits: A


@runtime_checkable
class HasProbs(Protocol[A]):
    probs: A


# TODO: Determine best "required" attributes
@runtime_checkable
class HasObjectDetections(Protocol[A]):
    scores: Sequence[Sequence[Dict[Any, float]]]
    boxes: A


class Model(Protocol[T_cont]):
    def __call__(self, data: TypedCollection[T_cont]) -> ModelOutput:
        """
        A Model applies a function on the data and returns
        a mapping of the data to a given set of outputs.

        Parameters
        ----------
        data : TypedCollection[ArrayLike]
            Data in the form of an array, sequence of arrays, or mapping of arrays..

        Returns
        -------
        ModelOutput
            The output of the Model defined as `dataclasses.dataclass` or
            a `NamedTuple`.
        """
        ...


class Classifier(Protocol[T_cont]):
    def __call__(self, data: TypedCollection[T_cont]) -> Union[HasLogits, HasProbs]:
        """Classifier protocol."""
        ...


class ObjectDetector(Protocol[T_cont]):
    def __call__(self, data: TypedCollection[T_cont]) -> HasObjectDetections:
        """Object detector protocol."""
        ...
