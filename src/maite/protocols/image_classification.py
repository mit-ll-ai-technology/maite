# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for image_classification

from . import DatumMetadata

from typing import (
    Protocol,
    Sequence,
    Any,
    runtime_checkable,
    Hashable,
)

from typing_extensions import TypeAlias

from . import generic as gen


@runtime_checkable
class ArrayLike(Protocol):
    def __array__(self) -> Any: ...


# In below, the dimension names/meanings used are:
#
# N  - image instance
# H  - image height
# W  - image width
# C  - image channel
# Cl - classification label (one-hot for ground-truth label, or pseudo-probabilities for predictions)

InputType: TypeAlias = ArrayLike  # shape [H, W, C]
OutputType: TypeAlias = ArrayLike  # shape [Cl]
MetadataType: TypeAlias = DatumMetadata

InputBatchType: TypeAlias = ArrayLike  # shape [N, H, W, C]
OutputBatchType: TypeAlias = ArrayLike  # shape [N, Cl]
MetadataBatchType: TypeAlias = Sequence[MetadataType]

# Initialize component classes based on generic and Input/Output/Metadata types

Dataset = gen.Dataset[InputType, OutputType, MetadataType]
DataLoader = gen.DataLoader[
    InputType,
    OutputType,
    MetadataType,
    InputBatchType,
    OutputBatchType,
    MetadataBatchType,
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

if __name__ == "__main__":
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

    from typing import Hashable, Tuple, overload
    import numpy as np
    import numpy.typing as npt
    import random

    from dataclasses import dataclass

    N_CLASSES = 5  # how many distinct classes are we predicting between?

    @dataclass
    class DatumMetadata_impl:
        uuid: Hashable

        # This is a prescribed method to go from type required by protocol
        # to the narrower implementor type. Whenever this protocol is used
        # as an input to some method (e.g. Augmentation protocol) this method
        # can typenarrow from protocol type to implementor type.
        # (This is what `__array__` does in any ArrayLike implementor to
        # convert ArrayLike -> <specific type>.)

        # TODO: consider whether we should take/return protocol types and
        # how to best permit metadata updating without undermining promised
        # types in augmentation.
        #
        # Motivating example:
        # Something interesting happens if implementors want to require a
        # subtype of Hashable (like int) instead. Any method of an implementor
        # component taking DatumMetadata needs to take object of the
        # protocol type or broader(because of contravariance wrt input arguments),
        # but that broader type might to be narrowed to an implementation
        # class within the component. If that narrowing requires a loss in
        # information (e.g. an augmentation received DatumMetadata of type 'str'
        # and wanted to narrow to a type like 'int' for implementation, there is a
        # problem).
        @classmethod
        def from_protocol(cls, o: DatumMetadata) -> "DatumMetadata_impl":
            if isinstance(o, cls):
                return o
            else:
                return DatumMetadata_impl(uuid=-1)  # just assign an uuid=-1
                # This is actually a real

    class Dataset_impl:
        def __init__(self): ...

        def __len__(self) -> int:
            return 10

        def __getitem__(
            self, h: Hashable
        ) -> Tuple[npt.NDArray, npt.NDArray, DatumMetadata_impl]:
            input = np.arange(5 * 4 * 3).reshape(5, 4, 3)
            output = np.arange(5 * 4 * 3).reshape(5, 4, 3)
            metadata = DatumMetadata_impl(uuid=1)

            return (input, output, metadata)

    class DataLoaderImpl:
        def __init__(self, d: Dataset):
            self._dataset = d

        def __next__(
            self,
        ) -> Tuple[ArrayLike, ArrayLike, list[DatumMetadata_impl]]:
            input_batch = np.array([self._dataset[i] for i in range(6)])
            output_batch = np.array([self._dataset[i] for i in range(6)])
            metadata_batch = [DatumMetadata_impl(uuid=i) for i in range(6)]

            return (input_batch, output_batch, metadata_batch)

        def __iter__(self) -> "DataLoaderImpl":
            return self

    class AugmentationImpl:
        def __init__(self): ...

        @overload
        def __call__(
            self, __datum: Tuple[InputType, OutputType, MetadataType]
        ) -> Tuple[npt.NDArray, npt.NDArray, DatumMetadata_impl]: ...

        @overload
        def __call__(
            self,
            __datum_batch: Tuple[InputBatchType, OutputBatchType, MetadataBatchType],
        ) -> Tuple[npt.NDArray, npt.NDArray, list[DatumMetadata_impl]]: ...

        def __call__(
            self,
            _datum_or_datum_batch: (
                Tuple[InputType, OutputType, MetadataType]
                | Tuple[InputBatchType, OutputBatchType, MetadataBatchType]
            ),
        ) -> (
            Tuple[npt.NDArray, npt.NDArray, DatumMetadata_impl]
            | Tuple[npt.NDArray, npt.NDArray, list[DatumMetadata_impl]]
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
                    DatumMetadata_impl.from_protocol(i)
                    for i in _datum_or_datum_batch[2]
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
                metadata_aug = DatumMetadata_impl(uuid=_datum_or_datum_batch[2].uuid)

                return (input_aug, output_batch_aug, metadata_aug)

    class Model_impl:
        @overload
        def __call__(
            self, __input: InputType | InputBatchType
        ) -> OutputType | OutputBatchType: ...

        @overload
        def __call__(self, __input: InputType) -> OutputType: ...

        @overload
        def __call__(self, __input: InputBatchType) -> OutputBatchType: ...

        def __call__(
            self,
            __input_or_input_batch: InputType | InputBatchType,
        ) -> OutputType | OutputBatchType:
            ...

            arr_input = np.array(__input_or_input_batch)
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
        def __init__(self): ...

        def reset(self) -> None: ...

        @overload
        def update(self, __pred: OutputType, __target: OutputType) -> None: ...

        @overload
        def update(
            self, __pred_batch: OutputBatchType, __target_batch: OutputBatchType
        ) -> None: ...

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
    dataset: Dataset = Dataset_impl()
    dataloader: DataLoader = DataLoaderImpl(d=dataset)
    model: Model = Model_impl()

    preds: list[OutputBatchType] = []
    for input_batch, output_batch, metadata_batch in dataloader:
        input_batch_aug, output_batch_aug, metadata_batch_aug = aug(
            (input_batch, output_batch, metadata_batch)
        )

        preds_batch = model(input_batch_aug)
        assert isinstance(preds_batch, OutputBatchType)

        # appending predictions here could take into account their being numpy arrays
        preds.append(preds_batch)

        metric.update(preds_batch, output_batch_aug)

    metric_scores = metric.compute()

    # Interesting "failure mode" for static type checking:
    # If you cursor over the type of metadata_batch that is returned from
    # "aug" function in evaluate workflow, you'll see it isn't a
    # list[DatumMetadata_impl] as you might expect. The Augmentation
    # implementation class has two methods that each take a tuple:
    # The first method signature is:
    #
    #  (Tuple[ArrayLike, ArrayLike, object]) -> Tuple[np.array, np.array, Augmentation_impl]
    #
    # and the second is:
    #
    #  (Tuple[ArrayLike, ArrayLike, list[object]]) -> Tuple[np.array, np.array, list[Augmentation_impl]]
    #
    # So, given an input tuple with 3rd-element type 'list[object]' one might expect
    # the third element of the output tuple to be typed list[Augmentation_impl],
    # but this is not the case. The reason is because everything in python is an
    # instance of type object (including list[object]). This is another example where
    # two compatible 'overload'ed type signatures cause the type-checker to use the
    # first and (quietly) ignore the second.
    #
    # What should we do? Probably alter the type of DatumMetadata to be less broad
    # (in practice, this means make it ANYTHING else besides object.)
