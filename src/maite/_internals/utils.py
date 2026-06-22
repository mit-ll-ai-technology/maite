# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Any, TypeVar, cast

from maite._internals.import_utils import is_tqdm_available
from maite._internals.protocols.generic import (
    DataLoader,
    DatumMetadataType_co,
    InputType_co,
    TargetType_co,
)

T = TypeVar("T", bound=Callable)


def is_typed_dict(obj: Any) -> bool:  # noqa: ANN401, deliberate use of 'Any' type
    if not isinstance(obj, type):
        return False

    return all(hasattr(obj, attr) for attr in ("__required_keys__", "__optional_keys__", "__optional_keys__"))


class ContextDecorator(metaclass=ABCMeta):
    @abstractmethod
    def __enter__(self):  # pragma: no cover  # noqa: ANN204
        raise NotImplementedError()  # noqa: RSE102

    @abstractmethod
    def __exit__(self, _type, _value, _traceback):  # pragma: no cover  # noqa: ANN001, ANN204
        raise NotImplementedError()  # noqa: RSE102

    def __call__(self, func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            with self:
                return func(*args, **kwargs)

        return cast(T, wrapper)


def add_progress_bar(
    dataloader: DataLoader[InputType_co, TargetType_co, DatumMetadataType_co],
) -> Iterable[
    tuple[
        Sequence[InputType_co],
        Sequence[TargetType_co],
        Sequence[DatumMetadataType_co],
    ]
]:
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

    return dataloader
