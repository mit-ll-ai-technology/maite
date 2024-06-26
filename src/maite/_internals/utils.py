# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Iterable, Tuple, TypeVar, cast
from weakref import WeakSet

from maite._internals.import_utils import is_torch_available, is_tqdm_available
from maite._internals.protocols.generic import (
    DataLoader,
    DatumMetadataType_co,
    InputType_co,
    TargetType_co,
)

T = TypeVar("T", bound=Callable)


def is_typed_dict(obj: Any) -> bool:
    if not isinstance(obj, type):
        return False

    return all(
        hasattr(obj, attr)
        for attr in ("__required_keys__", "__optional_keys__", "__optional_keys__")
    )


class ContextDecorator(metaclass=ABCMeta):
    @abstractmethod
    def __enter__(self):  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, type, value, traceback):  # pragma: no cover
        raise NotImplementedError()

    def __call__(self, func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return cast(T, wrapper)


class evaluating(ContextDecorator):
    """A context manager / decorator that temporarily places one
    or more modules in eval mode during the context."""

    def __init__(self, *modules: Callable) -> None:
        """
        Parameters
        ----------
        *modules: Module

        Notes
        -----
        A module's state is restored faithfully; e.g., a module that
        was already in eval mode will not be placed in train mode upon
        leaving the `evaluating` context.

        Examples
        --------
        >>> from torch.nn import Linear
        >>> from maite.utils import evaluating

        Using `evaluating` as a context manager.

        >>> module = Linear(1, 1)
        >>> module.training
        True
        >>> with evaluating(module):
        ...     print(module.training)
        False
        >>> module.training
        True

        Using `evaluating` as a decorator.

        >>> def f():
        ...     print("hello world")
        ...     return module.training
        >>> f = evaluating(module)(f)
        >>> module.training
        True
        >>> f()
        hello world
        False
        >>> module.training
        True
        """
        self._states: Dict[bool, WeakSet[Callable]] = {
            True: WeakSet(),
            False: WeakSet(),
        }

        for m in modules:
            if is_torch_available():
                from torch.nn import Module

                if isinstance(m, Module):
                    self._states[m.training].add(m)

    def __enter__(self) -> None:
        for train_status in self._states:
            for m in self._states[train_status]:
                if is_torch_available():
                    from torch.nn import Module

                    if isinstance(m, Module):
                        m.eval()

    def __exit__(self, type, value, traceback) -> None:
        for train_status in self._states:
            for m in self._states[train_status]:
                if is_torch_available():
                    from torch.nn import Module

                    if isinstance(m, Module):
                        m.train(train_status)


def add_progress_bar(
    dataloader: DataLoader[InputType_co, TargetType_co, DatumMetadataType_co],
) -> Iterable[Tuple[InputType_co, TargetType_co, DatumMetadataType_co]]:
    """Wrap a dataloader with tqdm to display progress bars.

    Note tqdm output can be disabled as of entirely as of version 4.66.0 by setting the
    environment variable TQDM_DISABLE=1.

    Parameters
    ----------
    dataloader : DataLoader[InputType_co, TargetType_co, DatumMetadataType_co]
        The dataloader to wrap.

    Returns
    -------
    Iterable[Tuple[InputType_co, TargetType_co, DatumMetadataType_co]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input, model target , and datum-level metadata.
    """
    if is_tqdm_available():
        # tqdm.auto will resolve to tqdm.autonotebook, tqdm.asyncio or tqdm.std
        # depending on the environment
        from tqdm.auto import tqdm

        return tqdm(dataloader)
    else:
        return dataloader
