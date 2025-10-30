# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


import copy
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from maite.protocols import (
    ArrayLike,
    AugmentationMetadata,
    DatasetMetadata,
    DatumMetadata,
    MetricMetadata,
    ModelMetadata,
)
from maite.protocols.image_classification import (
    DatumMetadataType,
    InputType,
    TargetType,
)

# define lightweight component implementations
#
# pretend we have a set of components such that:
#
# InputType is np.array of shape (C, H, W)
# TargetType is np.array of shape (Cl,)
# DatumMetadataType is DatumMetadata (i.e. a specialized TypedDict)

N_CLASSES = 5  # how many classes
N_DATAPOINTS = 10  # datapoints in dataset
C = 3  # number of color channels
H = 5  # img height
W = 4  # img width


class DatasetImpl:
    def __init__(self):
        self._data = np.random.rand(N_DATAPOINTS, C, H, W)

        self._targets = np.zeros((N_DATAPOINTS, N_CLASSES))
        for data_index in range(self._targets.shape[0]):
            self._targets[data_index, data_index % N_CLASSES] = 1

        self._data_metadata: list[DatumMetadata] = [
            {"id": i} for i in range(self._data.shape[0])
        ]

        self.metadata = DatasetMetadata(
            id="simple_dataset", index2label={i: f"class_{i}" for i in range(N_CLASSES)}
        )

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, ind: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        return (self._data[ind], self._targets[ind], self._data_metadata[ind])

    def get_input(self, ind, /) -> np.ndarray:
        return self._data[ind]

    def get_target(self, ind, /) -> np.ndarray:
        return self._targets[ind]

    def get_metadata(self, ind, /) -> DatumMetadata:
        return self._data_metadata[ind]


class DataLoaderImpl:
    def __init__(self, dataset: DatasetImpl, batch_size=2):
        self._dataset = dataset
        self._batch_size = batch_size

    def __iter__(
        self,
    ) -> Iterator[
        tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[DatumMetadata]]
    ]:
        # calculate number of batches
        n_batches = len(self._dataset) // self._batch_size

        # handle last batch if dataset length not divisible by batch size
        if len(self._dataset) / self._batch_size > n_batches:
            n_batches += 1

        for i_batch in range(n_batches):
            # i_batch goes from 0->n_batches

            batch_data = [
                self._dataset[i]
                for i in range(
                    i_batch * self._batch_size,
                    np.min([(i_batch + 1) * self._batch_size, len(self._dataset)]),
                )
            ]
            batch_inputs = []
            batch_targets = []
            batch_mds: list[DatumMetadata] = []
            for batch_tup in batch_data:
                batch_inputs.append(np.array(batch_tup[0]))
                batch_targets.append(np.array(batch_tup[1]))
                batch_mds.append(batch_tup[2])

            yield (batch_inputs, batch_targets, batch_mds)


class EnrichedDatumMetadata(DatumMetadata):
    new_key: str


class AugmentationImpl:
    def __init__(self):
        self.metadata = AugmentationMetadata({"id": "simple_augmentation"})

    def __call__(
        self,
        __datum_batch: tuple[
            Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]
        ],
    ) -> tuple[list[np.ndarray], list[np.ndarray], Sequence[EnrichedDatumMetadata]]:
        input_batch_aug = copy.deepcopy([np.array(elem) for elem in __datum_batch[0]])
        target_batch_aug = copy.deepcopy([np.array(elem) for elem in __datum_batch[1]])
        # metadata_batch_aug = copy.deepcopy(__datum_batch[2])

        # -- manipulate input_batch, target_batch, and metadata_batch --
        metadata_batch_aug: list[EnrichedDatumMetadata] = []

        # add new value to metadata_batch_aug
        for md in __datum_batch[2]:
            metadata_batch_aug.append(
                EnrichedDatumMetadata(**copy.deepcopy(md), new_key="new_val")
            )

        # modify input batch
        for inp_batch_elem in input_batch_aug:
            inp_batch_elem += 1

        # modify target batch
        for tgt_batch_elem in target_batch_aug:
            tgt_batch_elem = np.mod(tgt_batch_elem + 1, N_CLASSES)

        return (input_batch_aug, target_batch_aug, metadata_batch_aug)


class ModelImpl:
    def __init__(self):
        self.metadata = ModelMetadata({"id": "simple_model"})

    def __call__(self, __input_batch: Sequence[InputType]) -> list[np.ndarray]:
        target_batch = np.zeros((N_DATAPOINTS, N_CLASSES))
        for i, target_instance in enumerate(target_batch):
            target_instance[i % N_CLASSES] = 1

        return [i for i in target_batch]


class MetricImpl:
    def __init__(self):
        self.metadata = MetricMetadata({"id": "simple_metric"})

    def reset(self) -> None:
        return None

    def update(
        self,
        __pred_batch: Sequence[TargetType],
        __target_batch: Sequence[TargetType],
        __metadata_batch: Sequence[DatumMetadataType],
    ) -> None:
        return None

    def compute(self) -> dict[str, Any]:
        return {"metric1": "val1", "metric2": "val2"}


class MockDataset:
    """CIFAR-10 shaped dummy dataset with image_i[:,:,:] = i and class_i = (i % 10)."""

    def __init__(self, size: int = 8):
        # Require dataset size to be even so expected accuracy easier to predict when modifying odd instances
        assert size % 2 == 0, (
            "size of mock dataset must be even to support tests more easily"
        )
        self.size = size
        self.metadata = DatasetMetadata(id="MockDataset")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, DatumMetadataType]:
        if not (0 <= i < self.size):
            raise IndexError

        input = i * np.ones((3, 32, 32))
        target = np.zeros(10)
        target[i % 10] = 1
        metadata = DatumMetadataType(id=i)

        return input, target, metadata


class MockModel:
    """Predicts class(x) as int(x[0,0,0]) % 10."""

    def __init__(self):
        self.metadata = ModelMetadata(id="MockModel")

    def __call__(self, batch: Sequence[ArrayLike]) -> list[np.ndarray]:
        targets = []
        for x in batch:
            x_np = np.asarray(x)
            y = int(x_np[0, 0, 0]) % 10
            target = np.zeros(10)
            target[y] = 1
            targets.append(target)
        return targets


class MockAugmentation:
    """Changes x[0,0,0] to x[0,0,0] + 1 for each input x in batch where int(x[0,0,0]) % 2 is 1."""

    def __init__(self):
        self.metadata = AugmentationMetadata(id="MockAugmentation")

    def __call__(
        self,
        batch: tuple[
            Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadataType]
        ],
    ) -> tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadataType]]:
        xb, yb, mdb = batch
        xb_aug = []
        for x in xb:
            x_aug = np.array(x)  # copy
            x_aug[0, 0, 0] += 1 if int(x_aug[0, 0, 0]) % 2 == 1 else 0
            xb_aug.append(x_aug)
        return xb_aug, yb, mdb


class SimpleAccuracyMetric:
    metadata: MetricMetadata = {"id": "A simple accuracy metric"}

    def __init__(self) -> None:
        self._total = 0
        self._correct = 0

    def reset(self) -> None:
        self._total = 0
        self._correct = 0

    def update(
        self,
        pred_batch: Sequence[ArrayLike],
        target_batch: Sequence[ArrayLike],
        metadata_batch: Sequence[DatumMetadataType],
    ) -> None:
        model_probs = [np.array(r) for r in pred_batch]
        true_onehot = [np.array(r) for r in target_batch]

        # Stack into single array, convert to class indices
        model_classes = np.vstack(model_probs).argmax(axis=1)
        truth_classes = np.vstack(true_onehot).argmax(axis=1)

        # Compare classes and update running counts
        same = model_classes == truth_classes
        self._total += len(same)
        self._correct += same.sum()

    def compute(self) -> dict[str, Any]:
        if self._total > 0:
            return {"accuracy": self._correct / self._total}
        else:
            raise RuntimeError("No batches processed yet.")
