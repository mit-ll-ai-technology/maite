from typing import Any, Optional, TypeVar

from typing_extensions import Protocol

from .tree import PyTree

T = TypeVar("T")
RandomStates = Any


class RandomNumberGenerator(Protocol):
    """
    Generic type for random number generators.

    TODO: Unfortunately the PyTorch Generator does not have the methods below.

    Common generators are:

    - Python: `random.Random` with default of `random.Random()`
    - NumPy: `numpy.random.Generator` with default of `numpy.random.default_rng()`
    - PyTorch: `torch.Generator` with default of `torch.default_generator`
    """

    # def __getstate__(self) -> Dict[str, Any]:
    #     """Return internal state."""
    #     ...

    # def __setstate__(self, state: Dict[str, Any]) -> None:
    #     """Set internal state."""
    #     ...


class Augmentation(Protocol[T]):
    def __call__(
        self, *inputs: PyTree[T], rng: Optional[RandomStates] = None
    ) -> PyTree[T]:
        """
        Applies an agumentation to each item in the input and returns a corresponding container of augmented items.

        Inputs can be arrays or nested data structures of data collections (e.g., list, tuple, dict).

        Parameters
        ----------
        *inputs : PyTree[T]
            Any arbitrary structure of nested Python containers, e.g., list of image arrays.
            All types comprising the tree must be the same.

        rng : RandomNumberGenerator | None (default: None)
            An optional random number generator for reproducibility.

        Returns
        -------
        PyTree[T]
            A corresponding collection of transformed objects.
        """
        ...
