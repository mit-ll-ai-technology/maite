# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


import copy
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

from maite.protocols.image_classification import (
    DatumMetadataBatchType,
    InputBatchType,
    TargetBatchType,
)

# define lightweight component implementations
#
# pretend we have a set of components such that:
#
# InputType = np.array of shape [C, H, W]
# TargetType = np.array of shape [Cl]
# DatumMetadataType is an Dict[str, Any]
#
# InputBatchType = np.array of shape [N, C, H, W]
# TargetBatchType = np.array of shape [N, Cl]
# DatumMetadataBatchType = list[Dict[str, Any]]

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

        self._data_metadata = [{"some_metadata": i} for i in range(self._data.shape[0])]

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        return (self._data[ind], self._targets[ind], self._data_metadata[ind])


class DataLoaderImpl:
    def __init__(self, dataset: DatasetImpl, batch_size=2):
        self._dataset = dataset
        self._batch_size = batch_size

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[Sequence[np.ndarray], Sequence[np.ndarray], List[Dict[str, Any]]]
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
            batch_mds = []
            for batch_tup in batch_data:
                batch_inputs.append(np.array(batch_tup[0]))
                batch_targets.append(np.array(batch_tup[1]))
                batch_mds.append(batch_tup[2])

            yield (batch_inputs, batch_targets, batch_mds)


class AugmentationImpl:
    def __init__(self):
        ...

    def __call__(
        self,
        __datum_batch: Tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Sequence[Dict[str, Any]]]:
        input_batch_aug = copy.deepcopy([np.array(elem) for elem in __datum_batch[0]])
        target_batch_aug = copy.deepcopy([np.array(elem) for elem in __datum_batch[1]])
        metadata_batch_aug = copy.deepcopy(__datum_batch[2])

        # -- manipulate input_batch, target_batch, and metadata_batch --

        # add new value to metadata_batch_aug
        for md in metadata_batch_aug:
            assert (
                "new_key" not in md.keys()
            ), "bad practice to write over keys in metadata"
            md["new_key"] = "new_val"

        # modify input batch
        for inp_batch_elem in input_batch_aug:
            inp_batch_elem += 1

        # modify target batch
        for tgt_batch_elem in target_batch_aug:
            tgt_batch_elem = np.mod(tgt_batch_elem + 1, N_CLASSES)

        return (input_batch_aug, target_batch_aug, metadata_batch_aug)


class ModelImpl:
    def __call__(self, __input_batch: InputBatchType) -> List[np.ndarray]:
        target_batch = np.zeros((N_DATAPOINTS, N_CLASSES))
        for i, target_instance in enumerate(target_batch):
            target_instance[i % N_CLASSES] = 1

        return [i for i in target_batch]


class MetricImpl:
    def __init__(self):
        return None

    def reset(self) -> None:
        return None

    def update(
        self, __pred_batch: TargetBatchType, __target_batch: TargetBatchType
    ) -> None:
        return None

    def compute(self) -> Dict[str, Any]:
        return {"metric1": "val1", "metric2": "val2"}
