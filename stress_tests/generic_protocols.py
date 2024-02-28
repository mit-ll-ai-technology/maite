from maite.protocols.generic import (
    Dataset,
    InputType_co,
    InputType_cn,
    InputType_in,
    OutputType_co,
    OutputType_cn,
    OutputType_in,
    DatumMetadataType_co,
    DatumMetadataBatchType_cn,
    DatumMetadataBatchType_in,
    DatumMetadataBatchType_co,
    DatumMetadataBatchType_cn,
    DatumMetadataBatchType_in,
)
from typing import Hashable, Tuple, Protocol, Generic, TypeAlias, TypeVar


# check we can make a DataSet implementor
class DatasetImpl:
    def __getitem__(self, _ind: Hashable) -> Tuple[int, int, int]:
        return 0, 0, 0

    def __len__(self) -> int:
        return 0


dataset_inst: Dataset[int, int, int] = DatasetImpl()

a = dataset_inst.__getitem__(1)


# check whether composing generics permits correct type variance relationship
# from perspective of type-checker

T_co = TypeVar("T_co", covariant=True)
T_cn = TypeVar("T_cn", contravariant=True)


# Demonstrate no static type error if we don't use protocol types
# If we use nominal inheritance, it seems like static type-checker
# is less concerned about whether class acts covariant/contravariant
# wrt typevar. Deep dive into rules of of nominal typechecking wrt
# typevars would be a useful excursion, but is probably less relevant
# for our current position. Makes me question the use of 'contravariant'
# or 'covariant' descriptors in nominal subtyping of non-protocol Generic classes.
class Thing_nonprot(Generic[T_co]):
    def meth(self, _t: Tuple[T_co]) -> None: ...


# Show correct flagging when using covariant typevar as an argument input in
# protocol generic; this makes sense. The way the TypeVar is being used is consistent with
# contravariant class variance wrt the TypeVar, while; the TypeVar is declared as covariant.
class Thing(Protocol, Generic[T_co]):
    def meth(self, _t: Tuple[T_co]) -> None: ...


class Thing2(Protocol, Generic[T_cn]):
    def meth(self, _t: Tuple[T_cn]) -> None: ...


# background on generic type aliases:
#   https://mypy.readthedocs.io/en/latest/generics.html#generic-type-aliases
#   https://peps.python.org/pep-0484/#type-aliases
TypedTupleAlias_co: TypeAlias = Tuple[T_cn]
TypedTupleAlias_cn: TypeAlias = Tuple[T_co]


class TakeATupleAlias(Protocol, Generic[T_cn, T_co]):
    def meth(self, _t: TypedTupleAlias_co[T_cn]) -> TypedTupleAlias_cn[T_co]: ...


## Show how we expect augmentation to wrok


class MiniAug(Protocol, Generic[InputType_co, InputType_cn, DatumMetadataType_co]):
    def __call__(
        self, _i: InputType_cn
    ) -> Tuple[InputType_co, DatumMetadataType_co]: ...

    @property
    def metadata(self) -> DatumMetadataType_co: ...


from dataclasses import dataclass


@dataclass
class Ma_Impl:
    metadata: bytes

    def __call__(self, _i: float) -> Tuple[str, bytes]: ...


from typing import Sequence

mi: MiniAug[Sequence, int, bytes] = Ma_Impl(metadata=b"asdf")


## Show Tuple generic is covariant

TupType: TypeAlias = Tuple[float, Sequence]
tup_valid: TupType = (10, "asdf")
tup_invalid: TupType = (object, object)
