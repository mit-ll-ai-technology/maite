# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, runtime_checkable

from typing_extensions import NotRequired, ReadOnly, Required, TypeAlias

from ..compat import TypedDict

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

MetricComputeReturnType = Mapping[str, Any]

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
# (currently these are completely AI problem agnostic, but it is
# possible to make each generic to support specializing by AI problem)


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
    """
    Generic version of a protocol that specifies datum-level random access to a data source.

    Implementers must provide index lookup (via `__getitem__(ind: int, /)` method) and
    support `len` (via `__len__()` method). Data elements returned via `__getitem__`
    correspond to tuples of `InputType`, `TargetType`, and `DatumMetadataType`. The
    shape/value semantics of these three types are dictated by the concrete types used
    to specialize this generic.

    Additionally, Datasets are expected to contain a metadata attribute of type
    `DatasetMetadata` with general information about the data source.

    Note: In practice, this class is specialized within AI-problem specific submodules
    using structural types for `InputType`, `TargetType`, and `DatumMetadataType`.
    Implementing this class directly (i.e., without specializing on concrete types) is not
    recommended. Static type checkers will effectively consider all non-specified type parameters
    as `Any`-type, effectively masking potential type incompatibilities.
    """

    metadata: DatasetMetadata

    def __getitem__(
        self, __ind: int
    ) -> tuple[InputType_co, TargetType_co, DatumMetadataType_co]: ...

    def __len__(self) -> int: ...


@runtime_checkable
class FieldwiseDataset(
    Dataset[InputType_co, TargetType_co, DatumMetadataType_co], Protocol
):
    def get_input(self, __ind: int, /) -> InputType_co: ...

    def get_target(self, __ind: int, /) -> TargetType_co: ...

    def get_metadata(self, __ind: int, /) -> DatumMetadataType_co: ...


@runtime_checkable
class DataLoader(Protocol, Generic[InputType_co, TargetType_co, DatumMetadataType_co]):
    """
    Generic version of a protocol that specifies batch-level access to a data source.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `Sequence[InputType]`, `Sequence[TargetType]`, and `Sequence[DatumMetadataType]`,
    which correspond to model input batch, model target type batch, and a datum metadata batch.

    Note: In practice, this class is specialized within AI-problem specific submodules
    using structural types for `InputType`, `TargetType`, and `DatumMetadataType`.
    Implementing this class directly (i.e., without specializing on concrete types) is not
    recommended. Static type checkers will effectively consider all non-specified type parameters
    as `Any`-type, effectively masking potential type incompatibilities.
    """

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            Sequence[InputType_co],
            Sequence[TargetType_co],
            Sequence[DatumMetadataType_co],
        ]
    ]: ...

    # no longer having Dataloader operate as the Iterator, just using it as an iterable
    # so it doesn't need to return itself from the __iter__ method nor have a __next__ method


@runtime_checkable
class Model(
    Protocol,
    Generic[InputType_cn, TargetType_co],
):
    """
    Generic version of a protocol that specifies inference behavior on data batches.

    Implementers must provide a `__call__` method that operates on a batch of model
    inputs (as `Sequence[InputType]`) and returns a batch of model targets (as
    `Sequence[TargetType]`).

    Note: In practice, this class is specialized within AI-problem specific submodules
    using structural types for `InputType` and `TargetType`.
    Implementing this class directly (i.e., without specializing on concrete types) is not
    recommended. Static type checkers will effectively consider all non-specified type parameters
    as `Any`-type, effectively masking potential type incompatibilities.
    """

    metadata: ModelMetadata

    def __call__(
        self, __batch_input: Sequence[InputType_cn]
    ) -> Sequence[TargetType_co]: ...


@runtime_checkable
class Metric(Protocol, Generic[TargetType_cn, DatumMetadataType_cn]):
    """
    Generic version of a protocol that specifies a model/data alignment calculation behavior on data batches.

    Implementers must provide `update`, `compute`, and `reset` methods as specified in below "Methods" section.
    Briefly, these methods are designed to update a cache based on a batch of model inference predictions and their
    intended targets, compute a metric based on current cache contents, and clear the current cache, respectively.

    Note: In practice, this class is specialized within AI-problem specific submodules
    using structural types for `TargetType` and `DatumMetadataType`.
    Implementing this class directly (i.e., without specializing on concrete types) is not
    recommended. Static type checkers will effectively consider all non-specified type parameters
    as `Any`-type, effectively masking potential type incompatibilities.

    Methods
    -------

    update(pred_batch: Sequence[InputType], target_batch: Sequence[TargetType], metadata_batch: Sequence[DatumMetadataType]) -> None
        Add predictions and targets (and metadata if applicable) to metric's cache for later calculation.

    compute() -> dict[str, Any]
        Compute metric value(s) for currently cached predictions and targets, returned as
        a dictionary.

    reset() -> None
        Clear contents of current metric's cache of predictions and targets.

    Attributes
    ----------

    metadata : MetricMetadata
        A typed dictionary containing at least an 'id' field of type str.
    """

    metadata: MetricMetadata

    def reset(self) -> None: ...

    def update(
        self,
        __pred_batch: Sequence[TargetType_cn],
        __target_batch: Sequence[TargetType_cn],
        __metadata_batch: Sequence[DatumMetadataType_cn],
    ) -> None: ...

    def compute(self) -> MetricComputeReturnType: ...

    # don't believe Metric needs to guarantee a 'to' method exists
    # if all inputs to update are on GPU, operation should happen in framework specific way


@runtime_checkable
class Augmentation(
    Protocol,
    Generic[
        InputType_co,
        TargetType_co,
        DatumMetadataType_co,
        InputType_cn,
        TargetType_cn,
        DatumMetadataType_cn,
    ],
):
    """
    Generic version of a protocol that specifies a batch-level perturbation to data.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `Sequence[InputType]`, `Sequence[TargetType]`, and `Sequence[DatumMetadataType]`.
    Elements of this tuple correspond to the model input batch, model target batch,
    and datum-level metadata batch, respectively.

    Additionally, `Augmentation` protocol implementers are expected to contain a metadata attribute of type
    `AugmentationMetadata` with general information about the augmentation.

    Note: In practice, this class is specialized within AI-problem specific submodules
    using structural types for `TargetType` and `DatumMetadataType`.
    Implementing this class directly (i.e., without specializing on concrete types) is not
    recommended. Static type checkers will effectively consider all non-specified type parameters
    as `Any`-type, effectively masking potential type incompatibilities.
    """

    metadata: AugmentationMetadata

    def __call__(
        self,
        __batch: tuple[
            Sequence[InputType_cn],
            Sequence[TargetType_cn],
            Sequence[DatumMetadataType_cn],
        ],
    ) -> tuple[
        Sequence[InputType_co], Sequence[TargetType_co], Sequence[DatumMetadataType_co]
    ]: ...


T_in = TypeVar("T_in")
T_tgt = TypeVar("T_tgt")
T_md = TypeVar("T_md")

CollateFn: TypeAlias = Callable[
    [Iterable[tuple[T_in, T_tgt, T_md]]],
    tuple[Sequence[T_in], Sequence[T_tgt], Sequence[T_md]],
]
