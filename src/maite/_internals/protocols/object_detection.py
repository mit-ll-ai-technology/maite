# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from typing_extensions import TypeAlias, Dict

from . import ArrayLike, DatumMetadata, generic as gen

# We *could* make ArrayLike generic and rely on the subscripts for ArrayLike type
# annotations to hint to the user the appropriate shape. No runtime safety would
# be added by this approach because type subscripts are effectively invisible
# at runtime. No additional static type-checking would occur either, but the
# user would get useful type hints when cursoring over required inputs/outputs
# This would also require python 3.11 unless a `from __future__` import were made
# available for earlier versions (which are not available now.)


@runtime_checkable
class ObjectDetectionOutput(Protocol):
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


# TODO: remove typeAlias statements for more user readability (or figure out how to resolve TypeAliases
#       to their targets for end-user.) Knowing a dataset returns a tuple of "InputType, OutputType, MetadataType"
#       isn't helpful to implementers, however the aliasing *is* helpful to developers.
#
#       Perhaps the functionality I want is named parameters for generic, so developers can understand that
#       e.g. generic.Dataset typevars are 'InputType', 'OutputType', and 'MetaDataType' and their values in
#       concrete Dataset classes (like object_detection.Dataset) are ArrayLike, ObjDetectionOutput, class
#       closer to named parameters for a generic, so cursoring over image

InputType: TypeAlias = ArrayLike  # shape [H, W, C]
OutputType: TypeAlias = ObjectDetectionOutput  # shape [Cl]
MetadataType: TypeAlias = DatumMetadata

InputBatchType: TypeAlias = ArrayLike  # shape [N, H, W, C]
OutputBatchType: TypeAlias = Sequence[OutputType]  # length N
MetadataBatchType: TypeAlias = Sequence[DatumMetadata]

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
# 0) A protocol class with named/typed fields like the following:
#
#      class ObjDetectionOutput(Protocol):
#          x0: float
#          y0: float
#          x1: float
#          x2: float
#          label: int
#          score: float
#
#    This has the following advantages:
#       - Follows structural subtype class variance rules (ObjDetectionOutput covariant wrt read-only
#         attributes, classes using ObjDetectionOutput as return type covariant wrt read-only attributes,
#         and classes using ObjDetectionOutput as method argument type contravariant wrt
#         read-only attributes.) (Similar to TypedDict)
#       - Permits additional fields to be added by component implementers
#         and application developers (unlike TypedDicts)
#       - Permits implementer use in covariant contexts without either explicit importing
#         or redefining protocols locally (i.e. in Dataset/Dataloader/Model components can
#         return structural subtype of ObjDetectionOutput, but not in Augmentation or Metric
#         components) (like TypedDicts, but doesn't cap additional fields)
#       - Application developers could import implementer classes and workflows
#         and be assured that all protocol-compliant workflows would interoperate.

#
# 1) A Typed Dict -- This type is self-documenting and would permit users to simply populate
#    regular dictionaries in their implementations. The dictionary type is also
#    familiar to users of TorchMetrics and TorchVision as an output type for object
#    detection. Containing classes returning this object would be covariant
#    in the types of the dictionary keys. Containing classes taking this type as an
#    input argument would be contravariant wrt the types of the dictionary keys.
#
#    Using TypedDicts in this style ("as protocols", so to speak) does admit some challenges.
#    Users would be unable to add any fields to their objects, and
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
# 2) A typed 6-tuple -- this type clear about what the fields correspond to, but
#    leverages a ubiquitous python class.
#
#     ObjDetectionOutput: TypeAlias = Tuple[float, float, float, float, int, float]
#
# 3) A named tuple -- This is very clear and using built-in Python, but this
#    puts onus on implementer to return a named tuple since regular tuples with
#    compatible sizes/types don't seem to support assignment to this named tuple type.
#
# class ObjDetectionNamedOutput(NamedTuple):
#     x0: float
#     x1: float
#     y0: float
#     y1: float
#     label: int
#     score: float
#
# 4) Variadic generics -- this type hint could specifies the purpose of
#    each entry in a returned tuple, but using the type of each entry
#    to denote its meaning masks potentially useful information about its
#    real type. From the user's perspective, if a dimension is of type 'x0', what type is it?
#
#    The more obvious use case would be to denote size of Tensors/Arrays as below.
#
#    #An example of using type annotations to communicate shape information:
#
#    ```
#    H: TypeAlias = int
#    W: TypeAlias = int
#    C: TypeAlias = int
#
#    def get_some_data() -> Tensor[H,W,C]:
#    ...
#    ```
#
#    This is also a relatively new language feature which was (only introduced in python 3.11 with PEP 646)
#    See https://mit-ll-ai-technology.github.io/maite/explanation/type_hints_for_API_design.html#on-using-annotations-to-write-legible-documentation
#    or https://peps.python.org/pep-0646/ for more information.

DataLoader = gen.DataLoader[
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
]
Model = gen.Model[InputBatchType, OutputBatchType]
Metric = gen.Metric[OutputBatchType]

Augmentation = gen.Augmentation[
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
]
