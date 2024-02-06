# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

__all__ = []


from typing import Tuple, TypeAlias, TypeVar, Protocol, Hashable, Generic, Any, overload

# Note
# (1) the use of each generic variable can differ in generic components
# (2) the use of a generic affects the statically-correct way to declare thta generic's type variance
# Because of (1) and (2), I can't use the same TypeVars in all places where an input/output/metadata object
# is expected. This sacrifices some clarity in this file, but hopefully the classes can appear intuitive
# from the perspective of the end-users that only interact with parameterized generics.

# create instance-level versions of all generic type vars
InputType_co = TypeVar("InputType_co", covariant=True)
OutputType_co = TypeVar("OutputType_co", covariant=True)
DatumMetadataType_co = TypeVar("DatumMetadataType_co", covariant=True)

InputType_cn = TypeVar("InputType_cn", contravariant=True)
OutputType_cn = TypeVar("OutputType_cn", contravariant=True)
DatumMetadataType_cn = TypeVar("DatumMetadataType_cn", contravariant=True)

InputType_in = TypeVar("InputType_in", contravariant=False, covariant=False)
OutputType_in = TypeVar("OutputType_in", contravariant=False, covariant=False)
DatumMetadataType_in = TypeVar(
    "DatumMetadataType_in", contravariant=False, covariant=False
)


# create batch "versions" of all type vars
InputBatchType_co = TypeVar("InputBatchType_co", covariant=True)
OutputBatchType_co = TypeVar("OutputBatchType_co", covariant=True)
DatumMetadataBatchType_co = TypeVar("DatumMetadataBatchType_co", covariant=True)

InputBatchType_cn = TypeVar("InputBatchType_cn", contravariant=True)
OutputBatchType_cn = TypeVar("OutputBatchType_cn", contravariant=True)
DatumMetadataBatchType_cn = TypeVar("DatumMetadataBatchType_cn", contravariant=True)

InputBatchType_in = TypeVar("InputBatchType_in", contravariant=False, covariant=False)
OutputBatchType_in = TypeVar("OutputBatchType_in", covariant=False, contravariant=False)
DatumMetadataBatchType_in = TypeVar(
    "DatumMetadataBatchType_in", covariant=False, contravariant=False
)

# Datum and DatumBatch TypeAlias objects still need subscripts when used. The fact
# that we are using "{InputType,OutputType,DatumMetadataType}_co" instead of the
# contravariant or invariant counterparts is irrelvant because Datum and
# DatumBatch need 3 types when they are used, otherwise the types are inferred
# as 'Any'
Datum: TypeAlias = Tuple[InputType_co, OutputType_co, DatumMetadataType_co]
DatumBatch: TypeAlias = Tuple[
    InputBatchType_co, OutputBatchType_co, DatumMetadataBatchType_co
]
# TODO 0: add docstrings
#
# TODO 1: check type expected by pytorch getitem
#
# TODO 2: Decide if I really need different TypeVars based on context?
#           - in the context of Dataset, class is covariant to all types
#           - in the context of Augmentation, class is invariant to all types
# TODO 3: Consider whether 'to' method should be a part of Metric/Model/Dataset protocol
#           (intuitively, this seems more like framework-specific implementation detail)
#
# TODO 4: Consider whether using Datum as a TypeAlias is more confusing than helpful
#         It seems we just need 3 typevars in the TypeAlias assignment and the TypeVar
#         type variance is completely inconsequential (because they are substituted for
#         when the TypeAlias is used (as 'Any' or as the provided bracketed types.) We
#         could also define a 4th version of Input/Output/Metadata types, but this also
#         seems confusing.
# TODO 5: Consider how easily and usefully variadic generics (which sound a bit scary)
#         could be used to helpfully represent the shape of an expected array.
#         So, for example, instead of having type hints that specified that 'InputType=ArrayLike'
#         we could say 'InputType=ArrayLike[H,W,C]'.
#
# TODO 6: Add AugmentationMetadata
#
# TODO 7: Verify use of 'overload' decorator in protocol definition
#       Methods/signatures advertised by a protocol class *must* be mirrored by 
#       compatible types in implementation. If overload decorator is present,
#       only the overloaded methods are the source of these "promised" signatures.
#       If more than one signature is advertised by a protocol, then implementors
#       must use overload-decorator to advertise compatible signatures.




# Generic versions of all protocols
class Dataset(Protocol, Generic[InputType_co, OutputType_co, DatumMetadataType_co]):
    def __getitem__(
        self, _ind: Hashable
    ) -> Tuple[InputType_co, OutputType_co, DatumMetadataType_co]:
        ...

    def __len__(self) -> int:
        ...


class Model(
    Protocol,
    Generic[InputType_cn, OutputType_co, InputBatchType_cn, OutputBatchType_co],
):
    
    @overload
    def __call__(
        self, _single_input: InputType_cn
    ) -> OutputType_co:
        ...

    @overload
    def __call__(
        self, _batch_input: InputBatchType_cn
    ) -> OutputBatchType_co:
        ...

    @overload
    def __call__(
        self, _single_input_or_batch: InputType_cn | InputBatchType_cn
    ) -> OutputType_co | OutputBatchType_co:
        ...

    def __call__(
        self, _single_input_or_batch: InputType_cn | InputBatchType_cn
    ) -> OutputType_co | OutputBatchType_co:
        ...


class Metric(Protocol, Generic[OutputType_cn, OutputBatchType_cn]):
    def reset(self) -> None:
        ...

    @overload
    def update(
        self,
        _preds: OutputType_cn,
        _targets: OutputType_cn,
    ) -> None:
        ...

    @overload
    def update(
        self,
        _preds_batch: OutputBatchType_cn,
        _targets_batch: OutputBatchType_cn,
    ) -> None:
        ...
    
    def update(
        self,
        _preds_or_preds_batch: OutputType_cn | OutputBatchType_cn,
        _targets_or_targets_batch: OutputType_cn | OutputBatchType_cn,
    ) -> None:
        ...

    def compute(self) -> dict[str, Any]:
        ...

    # don't believe Metric needs to guarantee a 'to' method exists


class Augmentation(
    Protocol,
    Generic[
        InputType_co,
        OutputType_co,
        DatumMetadataType_co,
        InputType_cn,
        OutputType_cn,
        DatumMetadataType_cn,
        InputBatchType_co,
        OutputBatchType_co,
        DatumMetadataBatchType_co,
        InputBatchType_cn,
        OutputBatchType_cn,
        DatumMetadataBatchType_cn,
    ],
):
    @overload
    def __call__(
        self, _datum: Datum[InputType_cn, OutputType_cn, DatumMetadataType_cn]
    ) -> Datum[InputType_co, OutputType_co, DatumMetadataType_co]:
        ...

    @overload
    def __call__(
        self,
        _batch: DatumBatch[InputBatchType_cn, OutputBatchType_cn, DatumMetadataBatchType_cn],
    ) -> DatumBatch[InputBatchType_co, OutputBatchType_co, DatumMetadataBatchType_co]:
        ...

    def __call__(
        self,
        _datum_or_datum_batch: Datum[InputType_cn, OutputType_cn, DatumMetadataType_cn]
        | DatumBatch[InputBatchType_cn, OutputBatchType_cn, DatumMetadataBatchType_cn],
    ) -> (
        Datum[InputType_co, OutputType_co, DatumMetadataType_co]
        | DatumBatch[InputBatchType_co, OutputBatchType_co, DatumMetadataBatchType_co]
    ):
        ...


# We need the __iter__ method to return an instance of the same type as the object
# on which we are calling the method. Instead of using typing.Self, which was
# introduced in Python 3.11 (PEP 673), we can use a bound TypeVar as clunkier method
# toward the same ends (see https://peps.python.org/pep-0673/)
T_DataLoader = TypeVar("T_DataLoader", bound="DataLoader")


class DataLoader(
    Protocol,
    Generic[
        InputType_co,
        OutputType_co,
        DatumMetadataType_co,
        InputBatchType_co,
        OutputBatchType_co,
        DatumMetadataBatchType_co,
    ],
):
    def __next__(
        self,
    ) -> DatumBatch[InputBatchType_co, OutputBatchType_co, DatumMetadataBatchType_co]:
        ...

    def __iter__(self: T_DataLoader) -> T_DataLoader:
        ...

    @property
    def _dataset(
        self,
    ) -> Dataset[InputBatchType_co, OutputType_co, DatumMetadataType_co]:
        ...


# Sanity checks


# check we can make a DataSet implementor
class DatasetImpl:
    def __getitem__(self, _ind: Hashable) -> Datum[int, int, int]:
        return 0, 0, 0

    def __len__(self) -> int:
        return 0


dataset_inst: Dataset[int, int, int] = DatasetImpl()

a = dataset_inst.__getitem__(1)
reveal_type(a)


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
    def meth(self, _t: Tuple[T_co]) -> None:
        ...


# Show correct flagging when using covariant typevar as an argument input in
# protocol generic; this makes sense. The way the TypeVar is being used is consistent with
# contravariant class variance wrt the TypeVar, while; the TypeVar is declared as covariant.
class Thing(Protocol, Generic[T_co]):
    def meth(self, _t: Tuple[T_co]) -> None:
        ...


class Thing2(Protocol, Generic[T_cn]):
    def meth(self, _t: Tuple[T_cn]) -> None:
        ...


# background on generic type aliases:
#   https://mypy.readthedocs.io/en/latest/generics.html#generic-type-aliases
#   https://peps.python.org/pep-0484/#type-aliases
TypedTupleAlias_co: TypeAlias = Tuple[T_cn]
TypedTupleAlias_cn: TypeAlias = Tuple[T_co]


class TakeATupleAlias(Protocol, Generic[T_cn, T_co]):
    def meth(self, _t: TypedTupleAlias_co[T_cn]) -> TypedTupleAlias_cn[T_co]:
        ...


## Show how we expect augmentation to wrok


class MiniAug(Protocol, Generic[InputType_co, InputType_cn, DatumMetadataType_co]):
    def __call__(self, _i: InputType_cn) -> Tuple[InputType_co, DatumMetadataType_co]:
        ...

    @property
    def metadata(self) -> DatumMetadataType_co:
        ...


from dataclasses import dataclass


@dataclass
class Ma_Impl:
    metadata: bytes

    def __call__(self, _i: float) -> Tuple[str, bytes]:
        ...


from typing import Sequence

mi: MiniAug[Sequence, int, bytes] = Ma_Impl(metadata=b"asdf")


## Show Tuple generic is covariant

TupType: TypeAlias = Tuple[float, Sequence]
tup_valid: TupType = (10, "asdf")
tup_invalid: TupType = (object, object)
