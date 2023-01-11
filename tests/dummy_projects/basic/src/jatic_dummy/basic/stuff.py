from dataclasses import dataclass

from typing_extensions import Final, Literal, Protocol, TypeAlias, TypedDict

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
