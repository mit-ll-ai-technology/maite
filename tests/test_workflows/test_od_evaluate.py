# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

from maite._internals.protocols import generic as gen
from maite.workflows import evaluate

from maite.protocols.object_detection import (  # isort: skip
    Model,
    Metric,
    Augmentation,
    DataLoader,
    Dataset,
    InputBatchType,
    TargetBatchType,
    DatumMetadataBatchType,
    ObjectDetectionTarget,
)

# test lightweight implementations
#
# mock up a set of components such that:
#
# InputType = np.array of shape [C, H, W]
# TargetType = ObjectDetectionTarget
# DatumMetadataType = Dict[str, Any]
#
# InputBatchType = np.array of shape [N, C, H, W]
# TargetBatchType = Sequence[ObjectDetectionTarget]
# DatumMetadataBatchType = list[Dict[str, Any]]


N_CLASSES = 5  # how many classes
N_DATAPOINTS = 10  # datapoints in dataset
C = 3  # number of color channels
H = 5  # img height
W = 4  # img width
OBJ_PER_IMG = 3  # number of (planned) objects per image
OBJ_PER_MODEL_PRED = 2


def test_evaluate_od():
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

            self._targets: List[ObjectDetectionTargetImpl] = [
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

            self._data_metadata: Sequence[Dict[str, Any]] = [
                {"some_metadata": i} for i in range(self._data.shape[0])
            ]

        def __len__(self) -> int:
            return self._data.shape[0]

        def __getitem__(
            self, ind: int
        ) -> Tuple[np.ndarray, ObjectDetectionTargetImpl, Dict[str, Any]]:
            return (self._data[ind], self._targets[ind], self._data_metadata[ind])

    class DataLoaderImpl:
        def __init__(self, dataset: DatasetImpl, batch_size: int = 2):
            self._dataset = dataset
            self._batch_size = batch_size

        def __iter__(
            self,
        ) -> Iterator[
            Tuple[np.ndarray, List[ObjectDetectionTargetImpl], List[Dict[str, Any]]]
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
                batch_inputs: List[np.ndarray] = []
                batch_targets: List[ObjectDetectionTargetImpl] = []
                batch_mds: List[Dict[str, Any]] = []
                for batch_tup in batch_data:
                    batch_inputs.append(np.array(batch_tup[0]))
                    batch_targets.append(batch_tup[1])
                    batch_mds.append(batch_tup[2])

                input_batch = np.concatenate(batch_inputs, 0)

                yield (input_batch, batch_targets, batch_mds)

    class AugmentationImpl:
        def __init__(self):
            ...

        def __call__(
            self,
            __datum_batch: Tuple[
                InputBatchType, TargetBatchType, DatumMetadataBatchType
            ],
        ) -> Tuple[
            np.ndarray, List[ObjectDetectionTargetImpl], Sequence[Dict[str, Any]]
        ]:
            input_batch_aug = np.array(__datum_batch[0])
            target_batch_aug = copy.deepcopy(
                [ObjectDetectionTargetImpl.from_protocol(i) for i in __datum_batch[1]]
            )
            metadata_batch_aug = copy.deepcopy(__datum_batch[2])

            # -- manipulate input_batch, target_batch, and metadata_batch --

            # add new value to metadata_batch_aug
            for md in metadata_batch_aug:
                assert (
                    "new_key" not in md.keys()
                ), "bad practice to write over keys in metadata"
                md["new_key"] = "new_val"

            # modify input batch
            input_batch_aug += 1

            # modify target batch
            for target in target_batch_aug:
                target.scores = target.scores / 2

            return (input_batch_aug, target_batch_aug, metadata_batch_aug)

    class Model_impl:
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

    class Metric_impl:
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

    # Run types through "evaluate" workflow

    aug: Augmentation = AugmentationImpl()
    metric: Metric = Metric_impl()
    dataset: Dataset = DatasetImpl()
    dataloader: DataLoader = DataLoaderImpl(dataset=dataset)
    model: Model = Model_impl()

    assert isinstance(aug, gen.Augmentation)
    assert isinstance(metric, gen.Metric)
    assert isinstance(dataset, gen.Dataset)
    assert isinstance(dataloader, gen.DataLoader)
    assert isinstance(model, gen.Model)

    evaluate(model=model, dataloader=dataloader, metric=metric, augmentation=aug)
    evaluate(model=model, dataset=dataset, metric=metric, augmentation=aug)
