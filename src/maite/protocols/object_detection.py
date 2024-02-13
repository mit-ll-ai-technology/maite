# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from typing import (
    Protocol,
    Sequence,
    Any,
    runtime_checkable,
    Hashable,
)

import generic as gen


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
        ...


# We *could* make ArrayLike generic and rely on the subscripts for ArrayLike type
# annotations to hint to the user the appropriate shape. No runtime safety would
# be added by this approach because type subscripts are effectively invisible
# at runtime. No additional static type-checking would occur either, but the
# user would get useful type hints when cursoring over required inputs/outputs
# This would also require python 3.11 unless a `from __future__` import were made
# available for earlier versions (which are not available now.)
@runtime_checkable
class ObjDetectionOutput(Protocol):
    @property
    def boxes(
        self,
    ) -> ArrayLike:  # shape [N, 4], format X0, Y0, X1, Y1 (document this somewhere?)
        ...

    @property
    def labels(self) -> ArrayLike:  # label for each shape [N]
        ...

    @property
    def scores(self) -> ArrayLike:  # shape [N]
        ...


InputType = ArrayLike
OutputType = ObjDetectionOutput
MetadataType = object

InputBatchType = ArrayLike
OutputBatchType = Sequence[OutputType]
MetadataBatchType = Sequence[object]

# TODO: Consider what pylance shows on cursoring over: "(type alias) Dataset: type[Dataset[ArrayLike, dict[Unknown, Unknown], object]]"
# Can these type hints be made more intuitive? Perhaps given a name like type[Dataset[InputType = ArrayLike,...]]
# - This will likely involve eliminating some TypeAlias uses.

# TODO: Determine whether I should/can parameterize on the Datum TypeAlias.
# This could make the pylance messages more intuitive?

# TODO: Consider how we should help type-checker infer method return type when argument type
#       matches more than one method signature. For example: Model.__call__ takes an
#       ArrayLike in two separate method signatures, but the return type differs.
#       In this case, typechecker seems to use the first matching method signature to
#       determine type of output.

# TODO: Consider potential strategies for defining OutputType
#
# Some potential solutions:
## 0) A protocol class with named/typed fields like the following:
##
##      class ObjDetectionOutput(Protocol):
##          x0: float
##          y0: float
##          x1: float
##          x2: float
##          label: int
##          score: float
##
##    This has the following advantages:
##       - Follows structural subtype class variance rules (ObjDetectionOutput covariant wrt read-only
##         attributes, classes using ObjDetectionOutput as return type covariant wrt read-only attributes,
##         and classes using ObjDetectionOutput as method argument type contravariant wrt
##         read-only attributes.) (Similar to TypedDict)
##       - Permits additional fields to be added by component implementors
##         and application developers (unlike TypedDicts)
##       - Permits implementor use in covariant contexts without either explicit importing
##         or redefining protocols locally (i.e. in Dataset/Dataloader/Model components can
##         return structural subtype of ObjDetectionOutput, but not in Augmentation or Metric
##         components) (like TypedDicts, but doesn't cap additional fields)
##       - Application developers could import implementor classes and workflows
##         and be assured that all protocol-compliant workflows would interoperate.

##
## 1) A Typed Dict -- This type is self-documenting and would permit users to simply populate
##    regular dictionaries in their implementations. The dictionary type is also
##    familiar to users of TorchMetrics and TorchVision as an output type for object
##    detection. Containing classes returning this object would be covariant
##    in the types of the dictionary keys. Containing classes taking this type as an
##    input argument would be contravariant wrt the types of the dictionary keys.
##
##    Using TypedDicts in this style ("as protocols", so to speak) does admit some challenges.
##    Users would be unable to add any fields to their objects, and
#
# class ObjDetectionTypedDict(TypedDict)):
#     x0: float
#     x1: float
#     y0: float
#     y1: float
#     label: int
#     score: float
#
#     - For a user to pass static type checking in their own component or
#       workflow implementations, we need to permit them to use this type
#       as both an input (in Metric and Augmentation components) and an
#       output (for Dataset, DataLoader, and Model components).
#       Using a TypedDict as though it were a protocol has drawbacks
#
#       1) Import this particular TypedDict from MAITE
#       2) Redefine a 'compatible' TypedDict
#       3) "Under-annotate" to simply 'dict[str, Any]' and rely on MAITE utilities
#          to complain about incompatible types
#       4) Not annotate -- this is not a good answer
#
## 2) A typed 6-tuple -- this type clear about what the fields correspond to, but
##    leverages a ubiquitous python class.
#
#     ObjDetectionOutput: TypeAlias = Tuple[float, float, float, float, int, float]
#
## 3) A named tuple -- This is very clear and using built-in Python, but this
##    puts onus on implementor to return a named tuple since regular tuples with
##    compatible sizes/types don't seem to support assignment to this named tuple type.
#
# class ObjDetectionNamedOutput(NamedTuple):
#     x0: float
#     x1: float
#     y0: float
#     y1: float
#     label: int
#     score: float
#
## 4) Variadic generics -- this type hint could specifies the purpose of
##    each entry in a returned tuple, but using the type of each entry
##    to denote its meaning masks potentially useful information about its
##    real type. From the user's perspective, if a dimension is of type 'x0', what type is it?
#
##    The more obvious use case would be to denote size of Tensors/Arrays as below.
##
##    #An example of using type annotations to communicate shape information:
##
##    ```
##    H: TypeAlias = int
##    W: TypeAlias = int
##    C: TypeAlias = int
##
##    def get_some_data() -> Tensor[H,W,C]:
##    ...
##    ```
##
##    This is also a relatively new language feature which was (only introduced in python 3.11 with PEP 646)
##    See https://mit-ll-ai-technology.github.io/maite/explanation/type_hints_for_API_design.html#on-using-annotations-to-write-legible-documentation
##    or https://peps.python.org/pep-0646/ for more information.

Dataset = gen.Dataset[InputType, OutputType, MetadataType]
DataLoader = gen.DataLoader[
    InputType, OutputType, MetadataType, InputBatchType, OutputBatchType, MetadataType
]
Model = gen.Model[InputType, OutputType, InputBatchType, OutputBatchType]
Metric = gen.Metric[OutputType, OutputBatchType]

Augmentation = gen.Augmentation[
    InputType,
    OutputType,
    MetadataType,
    InputType,
    OutputType,
    MetadataType,
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
]


# test lightweight implementations
#
# pretend we have a model such that:
# InputType = np.array
# OutputType = ObjDetectionOutput
# MetadataType is an ordinary Python Class with integer-formatted 'id' field


from typing import Hashable, Tuple, overload
import numpy as np

from dataclasses import dataclass


@dataclass
class ObjDetectionOutput_impl:
    boxes: np.ndarray = np.array(
        [[0, 0, 1, 1], [1, 1, 2, 2]]
    )  # shape [N, 4], format X0, Y0, X1, Y1 (document this somewhere?)
    labels: np.ndarray = np.array([2, 5])  # shape [N]
    scores: np.ndarray = np.array([0, 1])  # shape [N]

    # This is a prescribed method to go from type required by protocol
    # to the narrower implementor type
    @classmethod
    def from_arraylike(cls, odo: ObjDetectionOutput) -> "ObjDetectionOutput_impl":
        return ObjDetectionOutput_impl(
            boxes=np.array(odo.boxes),
            labels = np.array(odo.labels),
            scores=np.array(odo.scores),
        )


@dataclass
class DatumMetadata_impl:
    id: int

    # This is a prescribed method to go from type required by protocol
    # to the narrower implementor type
    @classmethod
    def from_obj(cls, o: object) -> "DatumMetadata_impl":
        if isinstance(o, cls):
            return o
        else:
            return DatumMetadata_impl(id=-1)  # just assign an id=1


class DataSet_impl:
    def __init__(self):
        ...

    def __len__(self) -> int:
        return 10

    def __getitem__(
        self, h: Hashable
    ) -> Tuple[np.ndarray, ObjDetectionOutput_impl, DatumMetadata_impl]:
        input = np.arange(5 * 4 * 3).reshape(5, 4, 3)
        output = ObjDetectionOutput_impl()
        metadata = DatumMetadata_impl(id=1)

        return (input, output, metadata)


class DataLoaderImpl:
    def __init__(self, d: Dataset):
        self._dataset = d

    def __next__(
        self,
    ) -> Tuple[ArrayLike, list[ObjDetectionOutput_impl], list[DatumMetadata_impl]]:
        input_batch = np.array([self._dataset[i] for i in range(6)])
        output_batch = [ObjDetectionOutput_impl() for _ in range(6)]
        metadata_batch = [DatumMetadata_impl(i) for i in range(6)]

        return (input_batch, output_batch, metadata_batch)

    def __iter__(self) -> "DataLoaderImpl":
        return self


class AugmentationImpl:
    def __init__(self):
        ...

    @overload
    def __call__(
        self, _datum: Tuple[InputType, OutputType, MetadataType]
    ) -> Tuple[np.ndarray, OutputType, DatumMetadata_impl]:
        ...

    @overload
    def __call__(
        self, _datum_batch: Tuple[InputBatchType, OutputBatchType, MetadataBatchType]
    ) -> Tuple[np.ndarray, OutputBatchType, list[DatumMetadata_impl]]:
        ...

    def __call__(
        self,
        _datum_or_datum_batch: Tuple[InputType, OutputType, MetadataType]
        | Tuple[InputBatchType, OutputBatchType, MetadataBatchType],
    ) -> (
        Tuple[np.ndarray, OutputType, DatumMetadata_impl]
        | Tuple[np.ndarray, OutputBatchType, list[DatumMetadata_impl]]
    ):
        if isinstance(
            _datum_or_datum_batch[2], Sequence
        ):  # use second element's type to determine what input is
            # -- assume we are processing batch --

            # type narrow for static typechecker
            # (For this need to use functions like `isinstance`, `issubclass`, `type`, or user-defined typeguards)
            # We convert from broad types with guaranteed fields into specific types

            # Note: I'm not using parametrized information about generics because isinstance
            # checks don't apply to generics. But using the unparametrized generic is
            # good enough to type narrow for type-checker
            assert (
                isinstance(_datum_or_datum_batch[0], InputBatchType)
                and isinstance(
                    _datum_or_datum_batch[1], Sequence
                )  # Cant "isinstance check" against OutputBatchType directly
                and isinstance(
                    _datum_or_datum_batch[2], Sequence
                )  # Cant "isinstance check" against MetadataBatchType directly
            )

            input_batch_aug = np.array(_datum_or_datum_batch[0])
            output_batch_aug = [i for i in _datum_or_datum_batch[1]]
            metadata_batch_aug = [
                DatumMetadata_impl.from_obj(i) for i in _datum_or_datum_batch[2]
            ]

            # manipulate input_batch, output_batch, and metadata_batch

            return (input_batch_aug, output_batch_aug, metadata_batch_aug)

        else:
            # -- assume we are processing instance --

            assert (
                isinstance(_datum_or_datum_batch[0], InputType)
                and isinstance(_datum_or_datum_batch[1], OutputType)
                and isinstance(_datum_or_datum_batch[2], MetadataType)
            )

            input_aug = np.array(_datum_or_datum_batch[0])
            output_batch_aug = _datum_or_datum_batch[1]
            metadata_aug = DatumMetadata_impl.from_obj(_datum_or_datum_batch[2])

            return (input_aug, output_batch_aug, metadata_aug)


class Model_impl:

    @overload
    def __call__(
        self, _input: InputType | InputBatchType
    ) -> OutputType | OutputBatchType:
        ...

    @overload
    def __call__(self, _input: InputType) -> OutputType:
        ...

    @overload
    def __call__(self, _input: InputBatchType) -> OutputBatchType:
        ...

    def __call__(
        self,
        _input_or_input_batch: InputType | InputBatchType,
    ) -> OutputType | OutputBatchType:
        ...

        arr_input = np.array(_input_or_input_batch)
        if arr_input.ndim == 4:
            # process batch

            return [ObjDetectionOutput_impl() for _ in range(10)]

        else:
            # process instance
            return ObjDetectionOutput_impl()


class Metric_impl:
    def __init__(self):
        ...

    def reset(self) -> None:
        ...

    @overload
    def update(self, _pred: OutputType, _target: OutputType) -> None:
        ...

    @overload
    def update(
        self, _pred_batch: OutputBatchType, _target_batch: OutputBatchType
    ) -> None:
        ...

    def update(
        self,
        _preds: OutputType | OutputBatchType,
        _targets: OutputType | OutputBatchType,
    ) -> None:
        return None

    def compute(self) -> dict[str, Any]:
        return {"metric1": "val1", "metric2": "val2"}


# try to run through "evaluate" workflow

aug: Augmentation = AugmentationImpl()
metric: Metric = Metric_impl()
dataset: Dataset = DataSet_impl()
dataloader: DataLoader = DataLoaderImpl(d=dataset)
model: Model = Model_impl()

preds: list[OutputBatchType] = []
for input_batch, output_batch, metadata_batch in dataloader:
    input_batch_aug, output_batch_aug, metadata_batch_aug = aug(
        (input_batch, output_batch, metadata_batch)
    )
    assert not isinstance(output_batch_aug, OutputType)
    # This is onerous type-narrowing, because I can't run an isinstance check
    # directly on parametrized generic types (e.g. 'list[OutputType]', which is 
    # OutputBatchType). I have to use 'not isinstance' to rule out preds_batch
    # being an OutputType instead.

    preds_batch = model(input_batch_aug)
    assert not isinstance(preds_batch, OutputType)
    #preds_batch = cast(OutputBatchType, preds_batch) # could do this instead

    # Annoyingly, still need this type narrowing because type-checker can't
    # predict the output type of Model.__call__ based on input type. (batch 
    # input and singular inputs are both ArrayLike.) Perhaps we should explicitly
    # require that the InputType be different than a InputBatchType

    # The fact that InputType and InputBatchType are the same in this file 
    # shows an interesting corner case for pyright. The static typechecker
    # seems to take the first matching signature from Model_impl to determine
    # the type of the returned variable. Thus, if multiple method overloads list
    # the same input type and differing output types, only the first listed
    # method will be considered. (Thus method ordering affects static type-correctness)
    # This might be a reason to enforce InputType/InputBatchType to be different.

    # Tracing TypeAliases is problematic for usability, but it can be more convenient
    # for the developer to use them. The problem is that cursoring over an object
    # with a TypeAliased type doesn't show the underlying type that would be 
    # more meaningful to the user. "OutputBatchType" might a TypeAlias of
    # "list[ObjDetectionOutput]" but the user can't figure that out without
    # looking at source code. 
    preds.append(preds_batch)

    metric.update(preds_batch, output_batch_aug)

metric_scores = metric.compute()
