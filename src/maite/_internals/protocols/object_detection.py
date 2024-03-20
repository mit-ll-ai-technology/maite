# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from typing_extensions import Dict, TypeAlias

from . import ArrayLike, DatumMetadata, generic as gen

# We *could* make ArrayLike generic and rely on the subscripts for ArrayLike type
# annotations to hint to the user the appropriate shape. No runtime safety would
# be added by this approach because type subscripts are effectively invisible
# at runtime. No additional static type-checking would occur either, but the
# user would get useful type hints when cursoring over required inputs/targets
# This would also require python 3.11 unless a `from __future__` import were made
# available for earlier versions (which are not available now.)


@runtime_checkable
class ObjectDetectionTarget(Protocol):
    """
    An object-detection target protocol.

    This class is used to encode both predictions and ground-truth labels in the object
    detection problem.

    Implementers must populate the following attributes:

    Attributes
    ----------
    boxes : ArrayLike
        An array representing object detection boxes in a single image with x0, y0, x1, y1
        format and shape `(N_DETECTIONS, 4)`

    labels : ArrayLike
        An array representing the integer labels associated with each detection box of shape
        `(N_DETECTIONS,)`

    scores: ArrayLike
        An array representing the scores associated with each box (of shape `(N_DETECTIONS,)`)
    """

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
#       to their targets for end-user.) Knowing a dataset returns a tuple of "InputType, TargetType, MetadataType"
#       isn't helpful to implementers, however the aliasing *is* helpful to developers.
#
#       Perhaps the functionality I want is named TypeVars for generic, so developers can understand that
#       e.g. generic.Dataset typevars are 'InputType', 'TargetType', and 'MetaDataType' and their values in
#       concrete Dataset classes (like object_detection.Dataset) are ArrayLike, ObjectDetectionTarget, Dict[str,Any]
#       so users can see an expected return type of Tuple[ArrayLike, ObjectDetectionTarget, Dict[str,Any]]

InputType: TypeAlias = ArrayLike  # shape [H, W, C]
TargetType: TypeAlias = ObjectDetectionTarget
MetadataType: TypeAlias = Dict[str, Any]

InputBatchType: TypeAlias = ArrayLike  # shape [N, H, W, C]
TargetBatchType: TypeAlias = Sequence[TargetType]  # length N
MetadataBatchType: TypeAlias = Sequence[DatumMetadata]

# TODO: Consider what pylance shows on cursoring over: "(type alias) Dataset: type[Dataset[ArrayLike, Dict[Unknown, Unknown], object]]"
# Can these type hints be made more intuitive? Perhaps given a name like type[Dataset[InputType = ArrayLike,...]]
# - This will likely involve eliminating some TypeAlias uses.

# TODO: Determine whether I should/can parameterize on the Datum TypeAlias.
# This could make the pylance messages more intuitive?

# TODO: Consider how we should help type-checker infer method return type when argument type
#       matches more than one method signature. For example: Model.__call__ takes an
#       ArrayLike in two separate method signatures, but the return type differs.
#       In this case, typechecker seems to use the first matching method signature to
#       determine type of output. -> we can handle this problem by considering only
#       batches as the required handled types for model, augmentation, and metric objects
#
# TODO: Consider other potential strategies for defining TargetType

# Some potential solutions:
# 0) A protocol class with named/typed fields like the following:
#
#      class ObjectDetectionTarget(Protocol):
#          x0: float
#          y0: float
#          x1: float
#          x2: float
#          label: int
#          score: float
#
#    This has the following advantages:
#       - Follows structural subtype class variance rules (ObjectDetectionTarget covariant wrt read-only
#         attributes, classes using ObjectDetectionTarget as return type covariant wrt read-only attributes,
#         and classes using ObjectDetectionTarget as method argument type contravariant wrt
#         read-only attributes.) (Similar to TypedDict)
#       - Permits additional fields to be added by component implementers
#         and application developers (unlike TypedDicts)
#       - Permits implementer use in covariant contexts without either explicit importing
#         or redefining protocols locally (i.e. in Dataset/Dataloader/Model components can
#         return structural subtype of ObjectDetectionTarget, but not in Augmentation or Metric
#         components) (like TypedDicts, but doesn't cap additional fields)
#       - Application developers could import implementer classes and workflows
#         and be assured that all protocol-compliant workflows would interoperate.
#
#
# 1) A Typed Dict -- This type is self-documenting and would permit users to simply populate
#    regular dictionaries in their implementations. The dictionary type is also
#    familiar to users of TorchMetrics and TorchVision as an output type for object
#    detection. Containing classes returning this object would be covariant
#    in the types of the dictionary keys. Containing classes taking this type as an
#    input argument would be contravariant wrt the types of the dictionary keys.
#
#    Using TypedDicts in this style ("as protocols", so to speak) does admit some challenges.
#    Most notably, users would be unable to add any application-specific data-attributes or methods
#    to their objects.
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
#       3) "Under-annotate" to simply 'Dict[str, Any]' and rely on MAITE utilities
#          to complain about incompatible types
#       4) Not annotate -- this is not a good answer
#
# 2) A typed 6-tuple -- this type clear about what the fields correspond to, but
#    leverages a ubiquitous python class.
#
#     ObjectDetectionTarget: TypeAlias = Tuple[float, float, float, float, int, float]
#
# 3) A named tuple -- This is very clear and using built-in Python, but this
#    puts onus on implementer to return a named tuple since regular tuples with
#    compatible sizes/types don't seem to support assignment to this named tuple type.
#
# class ObjDetectionNamedTarget(NamedTuple):
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
#    ```
#    #An example of using type annotations to communicate shape information:
#
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
#
# 5) Generic dict (e.g. Dict[str, Union[ArrayLike, str]]--this is expandable by the user,
#    (unlike TypedDicts), but does not prescribe key names (which would need to be checked
#    dynamically. This requirement for dynamic checking is a substantial disadvantage when
#    writing a protocol library because users could only see incompatibilities in their
#    implementations after running. The purpose of protocols would be to permit
#    statically valid implementers to provide some assurance of performance.
#    (Note: perfect assurance isn't currently possible because e.g. shape of an array is
#    not checkable statically, and could create runtime exceptions.)


class Dataset(gen.Dataset[InputType, TargetType, MetadataType], Protocol):
    """
    A dataset protocol for object detection ML subproblem providing datum-level
    data access.

    Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
    support `len` (via `__len__()` method). Data elements looked up this way correspond to
    individual examples (as opposed to batches).

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `ArrayLike`, `ObjectDetectionTarget`, and `Dict[str,Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.


    Methods
    -------

    __getitem__(ind: int)->Tuple[ArrayLike, ObjectDetectionTarget, Dict[str, Any]]
        Provide mapping-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata,
        respectively.

    __len__()->int
        Return the number of data elements in the dataset.

    """

    ...


class DataLoader(
    gen.DataLoader[
        InputBatchType,
        TargetBatchType,
        MetadataBatchType,
    ],
    Protocol,
):
    """
    A dataloader protocol for the object detection ML subproblem providing
    batch-level data access.

    Implementers must provide an iterable object (returning an iterator via the
    `__iter__` method) that yields tuples containing batches of data. These tuples
    contain types `ArrayLike` (shape `(N, C, H, W)`), `Sequence[ObjectDetectionTarget]`,
    `Sequence[Dict[str, Any]]`, which correspond to model input batch, model target
    type batch, and datum metadata batch.

    Note: Unlike Dataset, this protocol does not require indexing support, only iterating.

    Methods
    -------

    __iter__->Iterator[tuple[ArrayLike, Sequence[ObjectDetectionTarget], Sequence[Dict[str, Any]]]]
        Return an iterator over batches of data, where each batch contains a tuple of
        of model input batch (as an `ArrayLike`), model target batch (as
        `Sequence[ObjectDetectionTarget]`), and batched datum-level metadata
        (as `Sequence[Dict[str,Any]]`), respectively.

    """

    ...


class Model(gen.Model[InputBatchType, TargetBatchType], Protocol):
    """
    A model protocol for the image classification ML subproblem.

    Implementers must provide a `__call__` method that operates on a batch of model inputs
    (as ArrayLikes) and returns a batch of model targets (implementers of
    Sequence[ObjectDetectionTarget])

    Methods
    -------

    __call__(input_batch: ArrayLike)->Sequence[ObjectDetectionTarget]
        Make a model prediction for inputs in input batch. Input batch is expected in
        the shape [N, C, H, W].
    """

    ...


class Metric(gen.Metric[TargetBatchType], Protocol):
    """
    A metric protocol for the object detection ML subproblem.

     A metric in this sense is expected to measure the level of agreement between model
     predictions and ground-truth labels.

     Methods
     -------

     update(preds: Sequence[ObjectDetectionTarget], targets: Sequence[ObjectDetectionTarget])->None
         Add predictions and targets to metric's cache for later calculation.

     compute()->Dict[str, Any]
         Compute metric value(s) for currently cached predictions and targets, returned as
         a dictionary.

     clear()->None
         Clear contents of current metric's cache of predictions and targets.
    """

    ...


class Augmentation(
    gen.Augmentation[
        InputBatchType,
        TargetBatchType,
        MetadataBatchType,
        InputBatchType,
        TargetBatchType,
        MetadataBatchType,
    ],
    Protocol,
):
    """
    An augmentation protocol for the object detection subproblem.

    An augmentation is expected to take a batch of data and return a modified version of
    that batch. Implementers must provide a single method that takes and returns a
    labeled data batch, where a labeled data batch is represented by a tuple of types
    `ArrayLike`, `ObjectDetectionTarget`, and `Dict[str,Any]`. These correspond to the model
    input type, model target type, and datum-specific metadata, respectively.

    Methods
    -------

    __call__(datum: Tuple[ArrayLike, ObjectDetectionTarget, dict[str, Any]])->
                Tuple[ArrayLike, ObjectDetectionTarget, dict[str, Any]]
        Return a modified version of original data batch. A data batch is represented
        by a tuple of model input batch (as an `ArrayLike` of shape `(N, C, H, W)`),
        model target batch (as `Sequence[ObjectDetectionTarget]`), and batch metadata
        (as `Sequence[Dict[str,Any]]`), respectively.
    """

    ...
