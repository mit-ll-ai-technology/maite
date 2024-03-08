from __future__ import annotations

from typing import Optional, Sequence, Tuple

from maite._internals.protocols.generic import MetricComputeReturnType
from maite._internals.protocols.object_detection import (
    Augmentation,
    DataLoader,
    InputBatchType,
    MetadataBatchType,
    Metric,
    Model,
    OutputBatchType,
)
from maite._internals.workflows import generic as gen

# TODO: populate evaluate overloads to expose simpler API
# @overload
# def evaluate():
#     ...

# TODO: The evaluate implementation will need to take a superset of all
# possible arguments and short-circuits where needed
# Take:
# - preds (as an iterable[TargetBatchType] or iterable[TargetType]
# - (if not preds not provided) Dataset & batch_size or DataLoader


def evaluate(
    model: Model,
    metric: Metric,
    dataloader: DataLoader,
    augmentation: Optional[Augmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[OutputBatchType],
    Sequence[Tuple[InputBatchType, OutputBatchType, MetadataBatchType]],
]:
    metric_results, preds, aug_data = gen._evaluate(
        model=model,
        metric=metric,
        dataloader=dataloader,
        augmentation=augmentation,
        return_augmented_data=return_augmented_data,
        return_preds=return_preds,
    )

    return metric_results, preds, aug_data


# TODO: populate predict with overloads
# def predict(
#     model: Model,
#     metric: Metric,
#     dataloader: Optional[DataLoader],
#     dataset: Optional[Dataset],
# ):
#     ...
