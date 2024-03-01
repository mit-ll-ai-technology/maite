# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any, Generic, Hashable, Iterator, Protocol, Tuple, TypeVar, overload

# Note
# (1) the use of each generic variable can differ in generic components
# (2) the use of a generic affects the statically-correct way to declare that generic's type variance
# Because of (1) and (2), I can't use the same TypeVars in all places where an input/output/metadata object
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
#     type from Model('BatchInputType') is a 'BatchOutputType' and not an OutputType,
#     user has to type narrow.
#   - Seems like overload decorator is the cost of handling multiple types of input arguments


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
#       If more than one signature is advertised by a protocol, then implementers
#       must use overload-decorator to advertise compatible signatures.


# Generic versions of all protocols

class Dataset(Protocol, Generic[InputType_co, OutputType_co, DatumMetadataType_co]):
    def __iter__(
            self
    )-> Iterator[Tuple[InputType_co, OutputType_co, DatumMetadataType_co]]:
        ...


class DataLoader(Protocol, Generic[InputType_co, OutputType_co, DatumMetadataType_co]):
    def __iter__(
        self,
    ) -> Iterator[Tuple[InputType_co, OutputType_co, DatumMetadataType_co]]:
        ...

    # no longer having Dataloader operate as the Iterator, just using it as an iterable
    # so it doesn't need to return itself from the __iter__ method nor have a __next__ method


class Model(
    Protocol,
    Generic[InputBatchType_cn, OutputBatchType_co],
):
    def __call__(self, __batch_input: InputBatchType_cn) -> OutputBatchType_co:
        ...


class Metric(Protocol, Generic[OutputBatchType_cn]):
    def reset(self) -> None:
        ...

    def update(
        self,
        __preds_batch: OutputBatchType_cn,
        __targets_batch: OutputBatchType_cn,
    ) -> None:
        ...

    def compute(self) -> dict[str, Any]:
        ...

    # don't believe Metric needs to guarantee a 'to' method exists
    # if all inputs to update are on GPU, operation should happen in framework specific way


class Augmentation(
    Protocol,
    Generic[
        InputBatchType_co,
        OutputBatchType_co,
        DatumMetadataBatchType_co,
        InputBatchType_cn,
        OutputBatchType_cn,
        DatumMetadataBatchType_cn,
    ],
):
    def __call__(
        self,
        __batch: Tuple[
            InputBatchType_cn, OutputBatchType_cn, DatumMetadataBatchType_cn
        ],
    ) -> Tuple[InputBatchType_co, OutputBatchType_co, DatumMetadataBatchType_co]:
        ...
