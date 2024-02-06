# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from typing import Protocol, Sequence, Any, runtime_checkable

import generic as gen

@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any:
        ...


InputType = ArrayLike
OutputType = dict  # this should be some specialized dict, but for now make it trivial
MetadataType = object

InputBatchType = ArrayLike
OutputBatchType = Sequence[OutputType]
MetadataBatchType = Sequence[object]

# TODO: Consider what pylance shows on cursoring over: "(type alias) Dataset: type[Dataset[ArrayLike, dict[Unknown, Unknown], object]]"
# Can these type hints be made more intuitive? Perhaps given a name like type[Dataset[InputType = ArrayLike,...]]

# TODO: Determine whether I should/can parameterize on the Datum TypeAlias.
# This could make the pylance messages more intuitive?

# TODO: Consider how we should help type-checker infer method return type when argument type
#       matches more than one method signature. For example: Model.__call__ takes an
#       ArrayLike in two separate method signatures, but the return type differs.
#       In this case, typechecker seems to use the first matching method signature to
#       determine type of output.

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
# OutputType = dict[str, Any] # should narrow this to make comparisons possible
# MetadataType is an ordinary Python Class with integer-formatted 'id' field


from typing import Hashable, Tuple, overload, cast, Iterable
import numpy as np

from dataclasses import dataclass


@dataclass
class DatumMetadata_impl:
    id: int

    @classmethod
    def from_obj(cls, o: object) -> "DatumMetadata_impl":
        if isinstance(o, cls):
            return o
        else:
            return DatumMetadata_impl(id=-1)  # just assign a random id


class DataSet_impl:
    def __init__(self):
        ...

    def __len__(self) -> int:
        return 10

    def __getitem__(self, h: Hashable) -> Tuple[np.ndarray, dict, DatumMetadata_impl]:
        input = np.arange(5 * 4 * 3).reshape(5, 4, 3)
        output = dict(score=10)
        metadata = DatumMetadata_impl(id=1)

        return (input, output, metadata)


class DataLoaderImpl:
    def __init__(self, d: Dataset):
        self._dataset = d

    def __next__(self) -> Tuple[ArrayLike, list[dict], list[DatumMetadata_impl]]:
        input_batch = np.array([self._dataset[i] for i in range(6)])
        output_batch = [{"key": f"val{i}"} for i in range(6)]
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
    ) -> Tuple[np.ndarray, dict, DatumMetadata_impl]:
        ...

    @overload
    def __call__(
        self, _datum_batch: Tuple[InputBatchType, OutputBatchType, MetadataBatchType]
    ) -> Tuple[np.ndarray, list[dict], list[DatumMetadata_impl]]:
        ...

    def __call__(
        self,
        _datum_or_datum_batch: Tuple[InputType, OutputType, MetadataType]
        | Tuple[InputBatchType, OutputBatchType, MetadataBatchType],
    ) -> (
        Tuple[np.ndarray, dict, DatumMetadata_impl]
        | Tuple[np.ndarray, list[dict], list[DatumMetadata_impl]]
    ):
        if isinstance(
            _datum_or_datum_batch[2], Sequence
        ):  # use second element's type to determine what input is
            # process batch

            # type narrow (need to use functions like `isinstance`, `issubclass`, `type`, or user-defined typeguards)
            # convert from broad types with guaranteed fields into specific types

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

            reveal_type(
                _datum_or_datum_batch[1]
            )  # Note: reveal_type with _datum_or_datum_batch object directly doesn't type-narrow

            input_batch_aug = np.array(_datum_or_datum_batch[0])
            output_batch_aug = [i for i in _datum_or_datum_batch[1]]
            metadata_batch_aug = [
                DatumMetadata_impl.from_obj(i) for i in _datum_or_datum_batch[2]
            ]

            # manipulate input_batch, output_batch, and metadata_batch

            return (input_batch_aug, output_batch_aug, metadata_batch_aug)

        else:
            # process instance

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

            return [dict(output=i) for i in range(arr_input.shape[0])]

        else:
            # process instance
            return dict(output=0)


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

preds:list[OutputBatchType] = []
for input_batch, output_batch, metadata_batch in dataloader:
    input_batch_aug, output_batch_aug, metadata_batch_aug = aug(
        (input_batch, output_batch, metadata_batch)
    )

    preds_batch = model(input_batch_aug)
    
    assert not isinstance(preds_batch, OutputType)
    preds.append(preds_batch)
    # This is onerous type-narrowing, because I can't run an isinstance check
    # directly on generic types (e.g. 'Sequence[dict]', which is OutputBatchType)
    # I have to use 'not isinstance' to rule out preds_batch being a 
    # dictionary instead of a sequence of dictionaries.
    #
    # Perhaps we should always make singular input/output/metadata types
    # non-generic (parameterized or otherwise)?

    metric.update(preds_batch, output_batch_aug)
    # problem: typechecker seems to assume that first matching signature
    # in a set of 'overloads' happens to give the right return type. In
    # the case of the Model component in Object detection problems, we
    # want instances of Model component protocol to take ArrayLike objects
    # for both batch and single inputs.

    # another problem: output_batch_aug is a `list[dict]` not *literally*
    # an OutputBatchType (a Sequence[dict]), only viewable after you trace)
    # the TypeAliasing used in this file. FOr some reason, output_batch_aug
    # is not interpreted as a valid implementor of Sequence[dict].

metric_scores = metric.compute()