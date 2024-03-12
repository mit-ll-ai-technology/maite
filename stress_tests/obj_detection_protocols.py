from dataclasses import dataclass
from typing import Any, Hashable, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt

from maite.protocols.object_detection import (
    ArrayLike,
    Augmentation,
    DataLoader,
    Dataset,
    InputBatchType,
    InputType,
    MetadataBatchType,
    MetadataType,
    Metric,
    Model,
    TargetBatchType,
    TargetType,
)

# test lightweight implementations
#
# pretend we have a model such that:
# InputType = np.array
# TargetType = ObjectDetectionTarget
# MetadataType is an ordinary Python Class with integer-formatted 'id' field


@dataclass
class ObjectDetectionTarget_impl:
    boxes: npt.NDArray = np.array(
        [[0, 0, 1, 1], [1, 1, 2, 2]]
    )  # shape [N, 4], format X0, Y0, X1, Y1 (document this somewhere?)
    labels: npt.NDArray = np.array([2, 5])  # shape [N]
    scores: npt.NDArray = np.array([0, 1])  # shape [N]


@dataclass
class DatumMetadata_impl:
    uuid: Hashable


class DataSet_impl:
    def __init__(self):
        ...

    def __len__(self) -> int:
        return 10

    def __getitem__(
        self, h: Hashable
    ) -> Tuple[npt.NDArray, ObjectDetectionTarget_impl, DatumMetadata_impl]:
        input = np.arange(5 * 4 * 3).reshape(5, 4, 3)
        target = ObjectDetectionTarget_impl()
        metadata = DatumMetadata_impl(uuid=1)

        return (input, target, metadata)


class DataLoaderImpl:
    def __init__(self, d: Dataset):
        self._dataset = d

    def __next__(
        self,
    ) -> Tuple[ArrayLike, list[ObjectDetectionTarget_impl], list[DatumMetadata_impl]]:
        input_batch = np.array([self._dataset[i] for i in range(6)])
        target_batch = [ObjectDetectionTarget_impl() for _ in range(6)]
        metadata_batch = [DatumMetadata_impl(uuid=i) for i in range(6)]

        return (input_batch, target_batch, metadata_batch)

    def __iter__(self) -> "DataLoaderImpl":
        return self


class AugmentationImpl:
    def __init__(self):
        ...

    @overload
    def __call__(
        self, __datum: Tuple[InputType, TargetType, MetadataType]
    ) -> Tuple[npt.NDArray, TargetType, DatumMetadata_impl]:
        ...

    @overload
    def __call__(
        self,
        __datum_batch: Tuple[InputBatchType, TargetBatchType, MetadataBatchType],
    ) -> Tuple[npt.NDArray, TargetBatchType, list[DatumMetadata_impl]]:
        ...

    def __call__(
        self,
        _datum_or_datum_batch: (
            Tuple[InputType, TargetType, MetadataType]
            | Tuple[InputBatchType, TargetBatchType, MetadataBatchType]
        ),
    ) -> (
        Tuple[npt.NDArray, TargetType, DatumMetadata_impl]
        | Tuple[npt.NDArray, TargetBatchType, list[DatumMetadata_impl]]
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
                )  # Cant "isinstance check" against TargetBatchType directly
                and isinstance(
                    _datum_or_datum_batch[2], Sequence
                )  # Cant "isinstance check" against MetadataBatchType directly
            )

            input_batch_aug = np.array(_datum_or_datum_batch[0])
            target_batch_aug = [i for i in _datum_or_datum_batch[1]]
            metadata_batch_aug = [
                DatumMetadata_impl(uuid=i) for i in _datum_or_datum_batch[2]
            ]

            # manipulate input_batch, target_batch, and metadata_batch

            return (input_batch_aug, target_batch_aug, metadata_batch_aug)

        else:
            # -- assume we are processing instance --

            assert (
                isinstance(_datum_or_datum_batch[0], InputType)
                and isinstance(_datum_or_datum_batch[1], TargetType)
                and isinstance(_datum_or_datum_batch[2], MetadataType)
            )

            input_aug = np.array(_datum_or_datum_batch[0])
            target_batch_aug = _datum_or_datum_batch[1]
            metadata_aug = DatumMetadata_impl(uuid=_datum_or_datum_batch[2].uuid)

            return (input_aug, target_batch_aug, metadata_aug)


class Model_impl:
    @overload
    def __call__(
        self, __input: InputType | InputBatchType
    ) -> TargetType | TargetBatchType:
        ...

    @overload
    def __call__(self, __input: InputType) -> TargetType:
        ...

    @overload
    def __call__(self, __input: InputBatchType) -> TargetBatchType:
        ...

    def __call__(
        self,
        _input_or_input_batch: InputType | InputBatchType,
    ) -> TargetType | TargetBatchType:
        ...

        arr_input = np.array(_input_or_input_batch)
        if arr_input.ndim == 4:
            # process batch

            return [ObjectDetectionTarget_impl() for _ in range(10)]

        else:
            # process instance
            return ObjectDetectionTarget_impl()


class Metric_impl:
    def __init__(self):
        ...

    def reset(self) -> None:
        ...

    @overload
    def update(self, __pred: TargetType, __target: TargetType) -> None:
        ...

    @overload
    def update(
        self, __pred_batch: TargetBatchType, __target_batch: TargetBatchType
    ) -> None:
        ...

    def update(
        self,
        _preds: TargetType | TargetBatchType,
        _targets: TargetType | TargetBatchType,
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

preds: list[TargetBatchType] = []
for input_batch, target_batch, metadata_batch in dataloader:
    input_batch_aug, target_batch_aug, metadata_batch_aug = aug(
        (input_batch, target_batch, metadata_batch)
    )
    assert not isinstance(target_batch_aug, TargetType)
    # This is onerous type-narrowing, because I can't run an isinstance check
    # directly on parametrized generic types (e.g. 'list[TargetType]', which is
    # TargetBatchType). I have to use 'not isinstance' to rule out preds_batch
    # being an TargetType instead.

    preds_batch = model(input_batch_aug)
    assert not isinstance(preds_batch, TargetType)
    # preds_batch = cast(TargetBatchType, preds_batch) # could do this instead

    # Annoyingly, still need this type narrowing because type-checker can't
    # predict the target type of Model.__call__ based on input type. (batch
    # input and singular inputs are both ArrayLike.) Perhaps we should explicitly
    # require that the InputType be different than a InputBatchType

    # The fact that InputType and InputBatchType are the same in this file
    # shows an interesting corner case for pyright. The static typechecker
    # seems to take the first matching signature from Model_impl to determine
    # the type of the returned variable. Thus, if multiple method overloads list
    # the same input type and differing target types, only the first listed
    # method will be considered. (Thus method ordering affects static type-correctness)
    # This might be a reason to enforce InputType/InputBatchType to be different.

    # Tracing TypeAliases is problematic for usability, but it can be more convenient
    # for the developer to use them. The problem is that cursoring over an object
    # with a TypeAliased type doesn't show the underlying type that would be
    # more meaningful to the user. "TargetBatchType" might a TypeAlias of
    # "list[ObjectDetectionTarget]" but the user can't figure that out without
    # looking at source code.
    preds.append(preds_batch)

    metric.update(preds_batch, target_batch_aug)

metric_scores = metric.compute()
