# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from maite._internals.import_utils import is_tqdm_available
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


def add_progress_bar(
    dataloader: DataLoader[InputType_co, TargetType_co, DatumMetadataType_co],
) -> Iterable[tuple[InputType_co, TargetType_co, DatumMetadataType_co]]:
    """Wrap a dataloader with tqdm to display progress bars.

    Note tqdm output can be disabled as of entirely as of version 4.66.0 by setting the
    environment variable TQDM_DISABLE=1.

    Parameters
    ----------
    dataloader : DataLoader[InputType_co, TargetType_co, DatumMetadataType_co]
        The dataloader to wrap.

    Returns
    -------
    Iterable[tuple[InputType_co, TargetType_co, DatumMetadataType_co]]
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
