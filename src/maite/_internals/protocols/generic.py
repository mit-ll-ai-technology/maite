# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import NotRequired, ReadOnly, Required, TypeAlias

# Note
# (1) the use of each generic variable can differ in generic components
# (2) the use of a generic affects the statically-correct way to declare that generic's type variance
# Because of (1) and (2), I can't use the same TypeVars in all places where an input/target/metadata object
# is expected. This sacrifices some clarity in this file, but hopefully the classes can appear intuitive
# from the perspective of the end-users that only interact with parameterized generics.

# (3) Methods/signatures advertised by a protocol class *must* be mirrored by
# compatible types in implementation. If overload decorator is present,
# only the 'overload'-decorated methods are the source of these "promised" signatures
# for the type checker. If more than one signature is advertised by a protocol,
# then implementers must use 'overload'-decorator to advertise compatible signatures.

# In light of these rules, what should we do?
#   - Including overload in protocol signatures is clear, but does require
#     protocol implementers to leverage 'overload' decorator, which isn't quite 'beginner' level
#   - Not including overload in protocol signatures makes it possible to implement
#     protocols without helping type checker determine specific return types.
#     this is more flexible for implementer, but will require more type narrowing
#     on non-specific ("union") return types. (If typechecker can't determine return
#     type from Model('BatchInputType') is a 'BatchTargetType' and not an TargetType,
#     user has to type narrow.
#   - Seems like overload decorator is the cost of handling multiple types of input arguments


# create instance-level versions of all generic type vars
InputType_co = TypeVar("InputType_co", covariant=True)
TargetType_co = TypeVar("TargetType_co", covariant=True)
DatumMetadataType_co = TypeVar("DatumMetadataType_co", covariant=True)

InputType_cn = TypeVar("InputType_cn", contravariant=True)
TargetType_cn = TypeVar("TargetType_cn", contravariant=True)
DatumMetadataType_cn = TypeVar("DatumMetadataType_cn", contravariant=True)

InputType_in = TypeVar("InputType_in", contravariant=False, covariant=False)
TargetType_in = TypeVar("TargetType_in", contravariant=False, covariant=False)
DatumMetadataType_in = TypeVar(
    "DatumMetadataType_in", contravariant=False, covariant=False
)


# create batch "versions" of all type vars
InputBatchType_co = TypeVar("InputBatchType_co", covariant=True)
TargetBatchType_co = TypeVar("TargetBatchType_co", covariant=True)
DatumMetadataBatchType_co = TypeVar("DatumMetadataBatchType_co", covariant=True)

InputBatchType_cn = TypeVar("InputBatchType_cn", contravariant=True)
TargetBatchType_cn = TypeVar("TargetBatchType_cn", contravariant=True)
DatumMetadataBatchType_cn = TypeVar("DatumMetadataBatchType_cn", contravariant=True)

InputBatchType_in = TypeVar("InputBatchType_in", contravariant=False, covariant=False)
TargetBatchType_in = TypeVar("TargetBatchType_in", covariant=False, contravariant=False)
DatumMetadataBatchType_in = TypeVar(
    "DatumMetadataBatchType_in", covariant=False, contravariant=False
)

MetricComputeReturnType = dict[str, Any]

# TODO: Consider whether using Datum as a TypeAlias is more confusing than helpful
#         It seems we just need 3 typevars in the TypeAlias assignment and the TypeVar
#         type variance is completely inconsequential (because they are substituted for
#         when the TypeAlias is used (as 'Any' or as the provided bracketed types.) We
#         could also define a 4th version of Input/Target/Metadata types, but this also
#         seems confusing.
# TODO: Consider how easily and usefully variadic generics (which sound a bit scary)
#         could be used to helpfully represent the shape of an expected array.
#         So, for example, instead of having type hints that specified that 'InputType=ArrayLike'
#         we could say 'InputType=ArrayLike[H,W,C]'. -- This would require python > 3.11, which is
#         not an option in short to mid term.


# Generic versions of all protocols

# myee: this version of Dataset was identical to DataLoader so both redundant
# - and made it hard to distinguish between the two classes

# class Dataset(Protocol, Generic[InputType_co, TargetType_co, DatumMetadataType_co]):
#     def __iter__(
#             self
#     ) -> Iterator[tuple[InputType_co, TargetType_co, DatumMetadataType_co]]:
#         ...


# Define component metadata types
# (currently these are completely ML subproblem agnostic, but it is
# possible to make each generic to support specializing by subproblem)


# If we created some 'standard' set of required component fields, we could
# use inheritance to reduce redundant text, but I'm resisting urge to
# prematurely optimize.
class DatumMetadata(TypedDict):
    # doc-ignore: PR01, EX01
    """
    Metadata associated with a single datum.

    Attributes
    ----------
    id : int|str
        Identifier for a single datum
    """

    id: Required[ReadOnly[int | str]]


class DatasetMetadata(TypedDict):
    # doc-ignore: PR01, EX01
    """
    Metadata associated with a Dataset object.

    Attributes
    ----------
    id : str
        Identifier for a single Dataset instance
    index2label : NotRequired[ReadOnly[dict[int, str]]]
        Mapping from integer labels to corresponding string descriptions
    """

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


class ModelMetadata(TypedDict):
    # doc-ignore: PR01, EX01
    """
    Metadata associated with a Model object.

    Attributes
    ----------
    id : str
        Identifier for a single Dataset instance
    index2label : NotRequired[ReadOnly[dict[int, str]]]
        Mapping from integer labels to corresponding string descriptions
    """

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


class MetricMetadata(TypedDict):
    # doc-ignore: PR01, EX01
    """
    Metadata associated with a Metric object.

    Attributes
    ----------
    id : str
        Identifier for a single Metric instance
    """
    id: Required[ReadOnly[str]]


class AugmentationMetadata(TypedDict):
    # doc-ignore: PR01, EX01
    """
    Metadata associated with an Augmentation object.

    Attributes
    ----------
    id : str
        Identifier for a single Augmentation instance
    """
    id: Required[ReadOnly[str]]


@runtime_checkable
class Dataset(Protocol, Generic[InputType_co, TargetType_co, DatumMetadataType_co]):
    metadata: DatasetMetadata

    def __getitem__(
        self, __ind: int
    ) -> tuple[InputType_co, TargetType_co, DatumMetadataType_co]:
        ...

    def __len__(self) -> int:
        ...


@runtime_checkable
class DataLoader(Protocol, Generic[InputType_co, TargetType_co, DatumMetadataType_co]):
    def __iter__(
        self,
    ) -> Iterator[tuple[InputType_co, TargetType_co, DatumMetadataType_co]]:
        ...

    # no longer having Dataloader operate as the Iterator, just using it as an iterable
    # so it doesn't need to return itself from the __iter__ method nor have a __next__ method


@runtime_checkable
class Model(
    Protocol,
    Generic[InputBatchType_cn, TargetBatchType_co],
):
    metadata: ModelMetadata

    def __call__(self, __batch_input: InputBatchType_cn) -> TargetBatchType_co:
        ...


@runtime_checkable
class Metric(Protocol, Generic[TargetBatchType_cn]):
    metadata: MetricMetadata

    def reset(self) -> None:
        ...

    def update(
        self,
        __preds_batch: TargetBatchType_cn,
        __targets_batch: TargetBatchType_cn,
    ) -> None:
        ...

    def compute(self) -> MetricComputeReturnType:
        ...

    # don't believe Metric needs to guarantee a 'to' method exists
    # if all inputs to update are on GPU, operation should happen in framework specific way


@runtime_checkable
class Augmentation(
    Protocol,
    Generic[
        InputBatchType_co,
        TargetBatchType_co,
        DatumMetadataBatchType_co,
        InputBatchType_cn,
        TargetBatchType_cn,
        DatumMetadataBatchType_cn,
    ],
):
    metadata: AugmentationMetadata

    def __call__(
        self,
        __batch: tuple[
            InputBatchType_cn, TargetBatchType_cn, DatumMetadataBatchType_cn
        ],
    ) -> tuple[InputBatchType_co, TargetBatchType_co, DatumMetadataBatchType_co]:
        ...


T_in = TypeVar("T_in")
T_tgt = TypeVar("T_tgt")
T_md = TypeVar("T_md")

CollateFn: TypeAlias = Callable[
    [Iterable[tuple[T_in, T_tgt, T_md]]],
    tuple[Sequence[T_in], Sequence[T_tgt], Sequence[T_md]],
]
