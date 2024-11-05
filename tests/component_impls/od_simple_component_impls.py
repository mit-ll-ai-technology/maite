# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


import copy
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from maite.protocols import (
    AugmentationMetadata,
    DatasetMetadata,
    DatumMetadata,
    MetricMetadata,
    ModelMetadata,
)
from maite.protocols.object_detection import (
    DatumMetadataBatchType,
    InputBatchType,
    ObjectDetectionTarget,
    TargetBatchType,
)

# test lightweight implementations
#
# mock up a set of components such that:
#
# InputType is np.array of shape (C, H, W)
# TargetType is ObjectDetectionTarget
# DatumMetadataType is DatumMetadata (i.e. a specialized TypedDict)
#
# InputBatchType is Sequence[np.array] with elements of shape (C, H, W)
# TargetBatchType is Sequence[ObjectDetectionTarget]
# DatumMetadataBatchType is Sequence[DatumMetadata]


N_CLASSES = 5  # how many classes
N_DATAPOINTS = 10  # datapoints in dataset
C = 3  # number of color channels
H = 5  # img height
W = 4  # img width
OBJ_PER_IMG = 3  # number of (planned) objects per image
OBJ_PER_MODEL_PRED = 2


@dataclass
class ObjectDetectionTargetImpl:
    boxes: np.ndarray  # shape: (N_DETECTIONS, 4), X0,Y0,X1,Y1
    scores: np.ndarray  # shape: (N_DETECTIONS,)
    labels: np.ndarray  # shape: (N_DETECTIONS,)

    @classmethod
    def from_protocol(
        cls, prot_obj: ObjectDetectionTarget
    ) -> "ObjectDetectionTargetImpl":
        return cls(
            boxes=np.array(prot_obj.boxes),
            scores=np.array(prot_obj.scores),
            labels=np.array(prot_obj.labels),
        )


class DatasetImpl:
    def __init__(self):
        self._data: np.ndarray = np.random.rand(N_DATAPOINTS, C, H, W)

        self._targets: list[ObjectDetectionTargetImpl] = [
            ObjectDetectionTargetImpl(
                boxes=np.array(
                    [
                        [
                            i / (OBJ_PER_IMG) * (W - 1),
                            i / (OBJ_PER_IMG) * (H - 1),
                            i / (OBJ_PER_IMG) * (W - 1) + 1,
                            i / (OBJ_PER_IMG) * (H - 1) + 1,
                        ]
                        for i in range(OBJ_PER_IMG)
                    ]
                ),
                scores=np.linspace(0, 1, OBJ_PER_IMG),
                labels=np.array([i % N_CLASSES for i in range(OBJ_PER_IMG)]),
            )
            for _ in range(N_DATAPOINTS)
        ]

        self._data_metadata: Sequence[DatumMetadata] = [
            {"id": i} for i in range(self._data.shape[0])
        ]

        self.metadata = DatasetMetadata(
            id="simple_dataset", index2label={i: f"class_{i}" for i in range(N_CLASSES)}
        )

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(
        self, ind: int
    ) -> tuple[np.ndarray, ObjectDetectionTargetImpl, DatumMetadata]:
        return (self._data[ind], self._targets[ind], self._data_metadata[ind])


class DataLoaderImpl:
    def __init__(self, dataset: DatasetImpl, batch_size: int = 2):
        self._dataset = dataset
        self._batch_size = batch_size

    def __iter__(
        self,
    ) -> Iterator[
        tuple[list[np.ndarray], list[ObjectDetectionTargetImpl], list[DatumMetadata]]
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
            batch_inputs: list[np.ndarray] = []
            batch_targets: list[ObjectDetectionTargetImpl] = []
            batch_mds: list[DatumMetadata] = []
            for batch_tup in batch_data:
                batch_inputs.append(np.array(batch_tup[0]))
                batch_targets.append(batch_tup[1])
                batch_mds.append(batch_tup[2])

            yield (batch_inputs, batch_targets, batch_mds)


class EnrichedDatumMetadata(DatumMetadata):
    new_key: str


class AugmentationImpl:
    def __init__(self):
        self.metadata = AugmentationMetadata(id="simple_augmentation")

    def __call__(
        self,
        __datum_batch: tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType],
    ) -> tuple[
        list[np.ndarray],
        list[ObjectDetectionTargetImpl],
        list[EnrichedDatumMetadata],
    ]:
        input_batch_aug = copy.deepcopy([np.array(elem) for elem in __datum_batch[0]])
        target_batch_aug = copy.deepcopy(
            [ObjectDetectionTargetImpl.from_protocol(i) for i in __datum_batch[1]]
        )
        metadata_batch_aug: list[EnrichedDatumMetadata] = []

        # -- manipulate input_batch, target_batch, and metadata_batch --

        # add new value to metadata_batch_aug
        for md in __datum_batch[2]:
            metadata_batch_aug.append(
                EnrichedDatumMetadata(**copy.deepcopy(md), new_key="new_val")
            )

        # modify input batch
        for in_batch_elem in input_batch_aug:
            in_batch_elem += 1

        # modify target batch
        for target in target_batch_aug:
            target.scores = target.scores / 2

        return (input_batch_aug, target_batch_aug, metadata_batch_aug)


class ModelImpl:
    def __init__(self):
        self.metadata = ModelMetadata(id="simple_model")

    def __call__(
        self, __input_batch: InputBatchType
    ) -> Sequence[ObjectDetectionTargetImpl]:
        pass

        target_batch = [
            ObjectDetectionTargetImpl(
                boxes=np.array(
                    [
                        [
                            i / (OBJ_PER_MODEL_PRED) * (W - 1),
                            i / (OBJ_PER_MODEL_PRED) * (H - 1),
                            i / (OBJ_PER_MODEL_PRED) * (W - 1) + 1,
                            i / (OBJ_PER_MODEL_PRED) * (H - 1) + 1,
                        ]
                        for i in range(OBJ_PER_MODEL_PRED)
                    ]
                ),
                scores=np.linspace(0, 1, OBJ_PER_MODEL_PRED),
                labels=np.array([i % N_CLASSES for i in range(OBJ_PER_MODEL_PRED)]),
            )
            for _ in range(np.array(__input_batch).shape[0])
        ]

        return target_batch


class MetricImpl:
    def __init__(self):
        self.metadata = MetricMetadata(id="simple_metric")

    def reset(self) -> None:
        return None

    def update(
        self, __pred_batch: TargetBatchType, __target_batch: TargetBatchType
    ) -> None:
        return None

    def compute(self) -> dict[str, Any]:
        return {"metric1": "val1", "metric2": "val2"}
