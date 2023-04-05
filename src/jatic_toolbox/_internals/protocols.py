from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import (
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

from .import_utils import is_numpy_available, is_torch_available

__all__ = [
    "ArrayLike",
    "Augmentation",
    "Classifier",
    "ClassifierDataLoader",
    "ClassifierDataset",
    "ClassifierWithParameters",
    "DataLoader",
    "Dataset",
    "DatasetDict",
    "HasLogits",
    "HasObjectDetections",
    "HasProbs",
    "HasTarget",
    "Metric",
    "MetricCollection",
    "Model",
    "ModelOutput",
    "ObjectDetector",
    "ShapedArray",
    "SupportsClassification",
    "TypedCollection",
]


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_cont = TypeVar("T_cont", contravariant=True)
P = ParamSpec("P")

"""
ArrayLike is any method that implements
the `__array__` method.

ShapedArray is an ArrayLike that also implements
the `shape` method.
"""


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...


@runtime_checkable
class ShapedArray(Protocol):
    # TODO: Deprecate
    def __array__(self) -> Any:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...


RandomStates: TypeAlias = Any
A = TypeVar("A", bound=ArrayLike)

"""
Dataset and Loaders
"""


@runtime_checkable
class HasTarget(Protocol[A]):
    target: A


@runtime_checkable
class SupportsClassification(Protocol[A]):
    data: A
    target: A


class SupportsImageClassification(TypedDict):
    image: ArrayLike
    label: ArrayLike


class ObjectDetection(TypedDict):
    bbox: ArrayLike
    label: ArrayLike


class SupportsObjectDetection(TypedDict):
    image: Union[ArrayLike, Sequence[ArrayLike]]
    objects: Sequence[ObjectDetection]


@runtime_checkable
class Dataset(Protocol[T]):
    features: Mapping[str, Any]

    def set_transform(self, transform: Callable[[T], T]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: Any) -> T:
        ...


DatasetDict: TypeAlias = Dataset[Dict[str, Dataset[T]]]
ClassifierDataset: TypeAlias = Dataset[SupportsClassification[A]]
VisionDataset: TypeAlias = Dataset[SupportsImageClassification]
ObjectDetectionDataset: TypeAlias = Dataset[SupportsObjectDetection]


"""
Not sure if this is the general solution but
it is required to support PyTorch DataLoaders.

The reason I think it will generalize is that
it allows an implementation to return custom
iterators from the DataLoader.  We just require
parametrizing the iterator so we can know the
output.
"""


class _DataLoaderIterator(Protocol[T_co]):
    def __next__(self) -> T_co:
        ...


class DataLoader(Protocol[T_co]):
    def __iter__(self) -> _DataLoaderIterator[T_co]:
        ...


ClassifierDataLoader: TypeAlias = DataLoader[SupportsClassification[A]]


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
        rng: Optional[RandomStates] = None,
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


@runtime_checkable
class HasDetectionLogits(Protocol[A]):
    pred_boxes: A
    logits: A


@runtime_checkable
class HasDetectionProbs(Protocol[A]):
    pred_boxes: A
    probs: A


@runtime_checkable
class HasObjectDetections(Protocol[A]):
    boxes: Union[A, Sequence[A]]
    scores: Union[A, Sequence[A]]
    labels: Optional[Union[A, Sequence[Any]]]


class Model(Protocol):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self:
        ...


class Classifier(Model, Protocol[A]):
    def __call__(self, data: TypedCollection[A]) -> Union[HasLogits[A], HasProbs[A]]:
        """
        A Model applies a function on the data and returns
        a mapping of the data to a given set of outputs.

        Parameters
        ----------
        data : TypedCollection[ArrayLike]
            Data in the form of an array, sequence of arrays, or mapping of arrays..

        Returns
        -------
        HasLogits[ArrayLike] | HasProbs[ArrayLike]
            The output of the Model defined as `dataclasses.dataclass` or
            a `NamedTuple` with either a "logits" or "probs" attribute.
        """
        ...


# classifier with parameters (whitebox)
class ClassifierWithParameters(Classifier[A], Protocol[A]):
    def parameters(self) -> Iterable[A]:
        ...


class ObjectDetector(Model, Protocol[A]):
    def __call__(self, data: TypedCollection[A]) -> HasObjectDetections[A]:
        """Object detector protocol."""
        ...


"""
Metric protocol is based off of:
  - `torchmetrics`
  - `torcheval`
"""


class Metric(Protocol[T_co]):
    def reset(self) -> None:
        ...

    def update(self, *args: Any, **kwargs: Any) -> None:
        ...

    def compute(self) -> T_co:
        ...

    def to(self, *args: Any, **kwargs: Any) -> Self:
        ...


MetricCollection: TypeAlias = Metric[Dict[str, A]]


if is_numpy_available():
    from numpy.typing import NDArray

    NumPyDataset: TypeAlias = ClassifierDataset[NDArray[Any]]
    NumPyDataLoader: TypeAlias = ClassifierDataLoader[NDArray[Any]]
    NumPyClassifier: TypeAlias = Classifier[NDArray[Any]]
    NumPyClassifierWithParameters = ClassifierWithParameters[NDArray[Any]]
    NumPyObjectDetector: TypeAlias = ObjectDetector[NDArray[Any]]
    NumPyMetric: TypeAlias = Metric[NDArray[Any]]
    NumPyMetricCollection: TypeAlias = MetricCollection[NDArray[Any]]

    __all__.extend(
        [
            "NumPyClassifier",
            "NumPyClassifierWithParameters",
            "NumPyDataLoader",
            "NumPyDataset",
            "NumPyMetric",
            "NumPyMetricCollection",
        ]
    )


if is_torch_available():
    from torch import Tensor

    TorchDataset: TypeAlias = ClassifierDataset[Tensor]
    TorchDataLoader: TypeAlias = ClassifierDataLoader[Tensor]
    TorchClassifier: TypeAlias = Classifier[Tensor]
    TorchClassifierWithParameters = ClassifierWithParameters[Tensor]
    TorchObjectDetector: TypeAlias = ObjectDetector[Tensor]
    TorchMetric: TypeAlias = Metric[Tensor]
    TorchMetricCollection: TypeAlias = MetricCollection[Tensor]

    __all__.extend(
        [
            "TorchClassifier",
            "TorchClassifierWithParameters",
            "TorchDataLoader",
            "TorchDataset",
            "TorchMetric",
            "TorchMetricCollection",
        ]
    )
