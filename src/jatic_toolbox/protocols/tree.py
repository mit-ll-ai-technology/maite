from typing import Any, Mapping, Sequence, TypeVar, Union

from typing_extensions import TypeAlias

T = TypeVar("T")

"""
Generic PyTree Type.

A PyTree is a recursive structure of arbitrarily nested Python containers
(e.g., tuple, list, dict, OrderedDict, NamedTuple, etc.).

Some implementations:

- `PyTorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/_pytree.py>`_
- `JAX PyTrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_
- `Optimized PyTrees <https://github.com/metaopt/optree>`_

Examples
--------
>>> PyTree[int]
typing.Union[int, typing.List[int], typing.Tuple[int, ...], typing.Dict[typing.Any, int]]
"""
PyTree: TypeAlias = Union[T, Sequence[T], Mapping[Any, T]]
