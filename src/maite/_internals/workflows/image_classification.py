from __future__ import annotations

from typing import Optional, Sequence, Tuple

import maite._internals.protocols.image_classification as ic
from maite._internals.protocols.generic import MetricComputeReturnType
from maite._internals.protocols.image_classification import (
    Augmentation,
    DataLoader,
    InputBatchType,
    MetadataBatchType,
    Metric,
    OutputBatchType,
)
from maite._internals.workflows import generic as gen

# TODO: populate evaluate overloads to expose simpler API
# @overload
# def evaluate(...):
#     ...

# (above todo has 2 subtask TODOs:)
# TODO: Write evaluate implementation to take a superset of all
# possible arguments in overloads and perform short-circuits where needed
#
# Function implementation should take:
# - preds (as an iterable[TargetBatchType] or iterable[TargetType]
#    - The latter requires that we can use iterable[TargetType] to create
#      iterable[TargetBatchType] (this is what pytorch calls collation)
# - either (dataset and batch_size) or a dataloader
#   - Prior requires a procedure to convert Input/Target/DatumMetadata from instance->batch
#     (In the pytorch world, this would be accomplished via a collation function.
#      collation is necessary to concatenate ArrayLikes before being fed to the model
#      rather than just feeding each input sequentially to the model.)

# TODO: Permit returned predictions to be of type tuple[InputDataType, TargetDataType, MetadataType].
#       This seems much more natural as it is independent of batch_size.
#   - This would require we use some method to iterate over "Batch-types" to get individual types
#     and package individual predictions into an iterable. batch objects are iterable and then
#     checking that the type of object returned from the iterable is ArrayLike.)


def evaluate(
    *,
    model: ic.Model,
    metric: Metric,
    dataloader: DataLoader,
    augmentation: Optional[Augmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[OutputBatchType],
    Sequence[tuple[InputBatchType, OutputBatchType, MetadataBatchType]],
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


# TODO: populate predict and use overloads for broader api
# def predict(
#     model: Model,
#     metric: Metric,
#     dataloader: Optional[DataLoader],
#     dataset: Optional[Dataset],
# ):
#     ...
