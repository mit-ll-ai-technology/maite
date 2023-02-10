from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from numpy.typing import ArrayLike as NumpyArrayLike, NDArray
from torch import Tensor as TorchTensor
from typing_extensions import Protocol, TypeAlias

try:
    from PIL.Image import Image
except ImportError:
    Image = TypeVar("Image")

__all__ = [
    "NestedCollection",
    "ImageType",
    "ArrayLike",
    "NDArray",
    "Tensor",
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

"""
ArrayLike is any object that can be manipulated into a Numpy array.

ShapedArray provides the method "shape".
"""
Tensor = TypeVar("Tensor", bound=TorchTensor)
ArrayLike: TypeAlias = Union[Sequence, NumpyArrayLike, NDArray, Tensor]


class ShapedArray(Protocol):
    def __array__(self) -> NDArray:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...


T = TypeVar("T")
T_co = TypeVar("T_co", contravariant=True)
S = TypeVar("S", bound=ArrayLike)
RandomStates: TypeAlias = Any
ImageType: TypeAlias = Union[ArrayLike, Image, NDArray, Tensor]


"""
`NestedCollection` is is a recursive structure of arbitrarily nested Python containers
(e.g., tuple, list, dict, OrderedDict, NamedTuple, etc.).

Nested collections support the very powerful "NestedCollection" functionalities currently
implemented here:

- `PyTorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/_NestedCollection.py>`_
- `JAX NestedCollections <https://jax.readthedocs.io/en/latest/NestedCollections.html>`_
- `Optimized NestedCollections <https://github.com/metaopt/optree>`_

Examples
--------
>>> NestedCollection[int]
typing.Union[int, typing.Sequence[int], typing.Mapping[typing.Any, int]]
"""
NestedCollection: TypeAlias = Union[T, Sequence[T], Mapping[Any, T]]


@runtime_checkable
class Augmentation(Protocol[T]):
    def __call__(
        self, *inputs: NestedCollection[T], rng: Optional[RandomStates] = None
    ) -> NestedCollection[T]:
        """
        Applies an agumentation to each item in the input and returns a corresponding container of augmented items.

        Inputs can be arrays or nested data structures of data collections (e.g., list, tuple, dict).

        Parameters
        ----------
        *inputs : NestedCollection[T]
            Any arbitrary structure of nested Python containers, e.g., list of image arrays.
            All types comprising the tree must be the same.

        rng : RandomNumberGenerator | None (default: None)
            An optional random number generator for reproducibility.

        Returns
        -------
        NestedCollection[T]
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
class HasLogits(Protocol[S]):
    logits: S


@runtime_checkable
class HasProbs(Protocol[S]):
    probs: S


# TODO: Determine best "required" attributes
@runtime_checkable
class HasObjectDetections(Protocol[S]):
    scores: Sequence[Sequence[Dict[Any, S]]]
    boxes: Sequence[S]


class Model(Protocol):
    def __call__(self, *data: NestedCollection[T]) -> ModelOutput:
        """
        A Model applies a function on the data and returns
        a mapping of the data to a given set of outputs.

        Parameters
        ----------
        *data : NestedCollection[SupportsArray]
            A nest of array types to process through a function.

        Returns
        -------
        ModelOutput
            The output of the Model defined as `dataclasses.dataclass` or
            a dictionary.
        """
        ...


class Classifier(Protocol[T_co]):
    def __call__(self, data: Iterable[T_co]) -> Union[HasLogits, HasProbs]:
        """Classifier protocol."""
        ...


class ObjectDetector(Protocol[T_co]):
    def __call__(self, data: Iterable[T_co]) -> HasObjectDetections:
        """Object detector protocol."""
        ...
