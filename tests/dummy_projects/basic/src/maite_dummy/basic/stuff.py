# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypedDict

from typing_extensions import TypeAlias

__all__ = [
    "CONSTANT",
    "Alias",
    "AClass",
    "ADataClass",
    "ATypedDict",
    "AProtocol",
    "a_func",
]

CONSTANT: Final = 22

Alias: TypeAlias = Literal["a", "b"]


def a_func(x: int) -> None:
    ...


@dataclass
class ADataClass:
    x: int


class AClass:
    @staticmethod
    def static_meth() -> None:
        ...

    def meth(self) -> None:
        ...


class ATypedDict(TypedDict):
    x: int


class AProtocol(Protocol):
    def meth(self) -> None:
        ...
