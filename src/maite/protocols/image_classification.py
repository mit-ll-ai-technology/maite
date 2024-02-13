# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for image_classification

from typing import (
    Protocol,
    Sequence,
    Any,
    runtime_checkable,
    TypedDict,
    NamedTuple,
    Hashable,
    TypeAlias,
    NewType,
)

import generic as gen


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
        ...


InputType = ArrayLike  # shape [H, W, C]
OutputType = ArrayLike  # shape [Cl], where Cl is "number of classes"
MetadataType = object

InputBatchType = ArrayLike  # shape [N, H, W, C]
OutputBatchType = ArrayLike  # shape [N, Cl]
MetadataBatchType = Sequence[object]

# Initialize component classes based on generic and Input/Output/Metadata types

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
# pretend we have a set of components such that:
#
# InputType = np.array of shape [H, W, C]
# OutputType = np.array of shape [Cl]
# MetadataType is an ordinary Python Class with integer-formatted 'id' field
#
# InputBatchType = np.array of shape [N, H, W, C]
# OutputBatchType = np.array of shape [N, Cl]
# MetadataBatchType is an ordinary Python Class with integer-formatted 'id' field


from typing import Hashable, Tuple, overload, cast, Iterable
import numpy as np
import random

from dataclasses import dataclass

N_CLASSES = 5  # how many distinct classes are we predicting between?


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
    ) -> Tuple[np.ndarray, np.ndarray, DatumMetadata_impl]:
        input = np.arange(5 * 4 * 3).reshape(5, 4, 3)
        output = np.arange(5 * 4 * 3).reshape(5, 4, 3)
        metadata = DatumMetadata_impl(id=1)

        return (input, output, metadata)


class DataLoaderImpl:
    def __init__(self, d: Dataset):
        self._dataset = d

    def __next__(
        self,
    ) -> Tuple[ArrayLike, ArrayLike, list[DatumMetadata_impl]]:
        input_batch = np.array([self._dataset[i] for i in range(6)])
        output_batch = np.array([self._dataset[i] for i in range(6)])
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
    ) -> Tuple[np.ndarray, np.ndarray, DatumMetadata_impl]:
        ...

    @overload
    def __call__(
        self, _datum_batch: Tuple[InputBatchType, OutputBatchType, MetadataBatchType]
    ) -> Tuple[np.ndarray, np.ndarray, list[DatumMetadata_impl]]:
        ...

    def __call__(
        self,
        _datum_or_datum_batch: Tuple[InputType, OutputType, MetadataType]
        | Tuple[InputBatchType, OutputBatchType, MetadataBatchType],
    ) -> (
        Tuple[np.ndarray, np.ndarray, DatumMetadata_impl]
        | Tuple[np.ndarray, np.ndarray, list[DatumMetadata_impl]]
    ):
        if isinstance(
            _datum_or_datum_batch[-1], Sequence
        ):  # use last element's type to type-narrow between batch or instance
            # -- proceed with handling batch --

            # type narrow for static typechecker
            # (For this need to use functions like `isinstance`, `issubclass`, `type`, or user-defined typeguards)
            # We convert from broad types with guaranteed fields into specific types

            # Note: I'm not using parametrized information about generics because isinstance
            # checks can't be applied to generics. But using the unparametrized generic is
            # good enough to type narrow between batch/individual
            assert (
                isinstance(_datum_or_datum_batch[0], InputBatchType)
                and isinstance(_datum_or_datum_batch[1], OutputBatchType)
                and isinstance(
                    _datum_or_datum_batch[2], Sequence
                )  # Cant "isinstance check" against MetadataBatchType directly
            )

            input_batch_aug = np.array(_datum_or_datum_batch[0])
            output_batch_aug = np.array(_datum_or_datum_batch[1])
            metadata_batch_aug = [
                DatumMetadata_impl.from_obj(i) for i in _datum_or_datum_batch[2]
            ]

            # manipulate input_batch, output_batch, and metadata_batch

            return (input_batch_aug, output_batch_aug, metadata_batch_aug)

        else:
            # -- proceed with handling instance --

            assert (
                isinstance(_datum_or_datum_batch[0], InputType)
                and isinstance(_datum_or_datum_batch[1], OutputType)
                and isinstance(_datum_or_datum_batch[2], MetadataType)
            )

            input_aug = np.array(_datum_or_datum_batch[0])
            output_batch_aug = np.array(_datum_or_datum_batch[1])
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
            N, H, W, C = arr_input.shape
            batch_output = np.zeros((N, N_CLASSES))
            batch_output[np.arange(N, random.randint(1, N_CLASSES))] = 1
            return batch_output

        else:
            # process instance
            H, W, C = arr_input.shape
            single_output = np.zeros((N_CLASSES,))
            single_output[random.randint(1, N_CLASSES)] = 1
            return single_output


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

    preds_batch = model(input_batch_aug)

    preds.append(preds_batch)

    metric.update(preds_batch, output_batch_aug)

metric_scores = metric.compute()
