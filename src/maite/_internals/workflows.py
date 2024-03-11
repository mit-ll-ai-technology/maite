from typing import Any, Optional, Sequence, Tuple, overload

from maite._internals.protocols.generic import (
    Augmentation,
    DataLoader,
    Metric,
    MetricComputeReturnType,
    Model,
)
from maite.protocols import image_classification as ic, object_detection as od

# TODO: populate evaluate overloads to expose simpler API
# @overload
# def evaluate(...):
#     ...

# (above todo has 2 subtask TODOs:)
# TODO: Write evaluate implementation to take a superset of all
# possible arguments in overloads and perform short-circuits where needed
#
# Function implementation should take:
# - preds (as iterable[TargetBatchType] or iterable[TargetType])
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


# TODO: add overload-decorated versions that preserve typing information for
# each problem subtype. Contents of implementation should be the same, but
# static checker will then know how to type outputs of this function without
# casts


@overload
def evaluate(
    *,
    model: ic.Model,
    metric: ic.Metric,
    dataloader: ic.DataLoader,
    augmentation: Optional[ic.Augmentation] = None,
    return_augmented_data=False,
    return_preds=False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[ic.OutputBatchType],
    Sequence[Tuple[ic.InputBatchType, ic.OutputBatchType, ic.MetadataBatchType]],
]:
    ...


@overload
def evaluate(
    *,
    model: od.Model,
    metric: od.Metric,
    dataloader: od.DataLoader,
    augmentation: Optional[od.Augmentation] = None,
    return_augmented_data=False,
    return_preds=False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[od.OutputBatchType],
    Sequence[Tuple[od.InputBatchType, od.OutputBatchType, od.MetadataBatchType]],
]:
    ...


# TODO: Narrow return types to be a union of accepted return types,
# 'Any' is too liberal and forgoes any potentially useful type-checking
# in this implementation
def evaluate(
    model: Model,
    metric: Metric,
    dataloader: DataLoader,
    augmentation: Optional[Augmentation] = None,
    return_augmented_data=False,
    return_preds=False,
) -> Tuple[MetricComputeReturnType, Sequence[Any], Sequence[Tuple[Any, Any, Any]],]:
    # Validate potential input combinations
    # user needs to provide ONE of either (a) dataloader (iterable over
    # tuples of batch-types) or (b) a dataset (an iterable over tuples
    # of individual types) and a batch_size. (We could take both and warn,
    # but we're opting to be more conservative since providing both might
    # be an indicator of misunderstanding.)

    preds_batches = []
    metric.reset()
    augmented_data_batches = []

    for input_datum_batch, target_datum_batch, metadata_batch in dataloader:
        if augmentation is not None:
            (
                input_datum_batch_aug,
                target_datum_batch_aug,
                metadata_batch_aug,
            ) = augmentation((input_datum_batch, target_datum_batch, metadata_batch))
        else:
            input_datum_batch_aug, target_datum_batch_aug, metadata_batch_aug = (
                input_datum_batch,
                target_datum_batch,
                metadata_batch,
            )

        preds_batch = model(input_datum_batch_aug)
        metric.update(preds_batch, target_datum_batch_aug)

        # store any requested intermediate data for returning to user
        if return_augmented_data:
            augmented_data_batches.append(
                (input_datum_batch_aug, target_datum_batch_aug, metadata_batch_aug)
            )

        if return_preds:
            preds_batches.append(preds_batch)

    # TODO: investigate better munging/recollating iterable/batches
    # aug_data is currently a sequence of tuples of batch-types
    # this is a little artificial as batch-size is typically less
    # more a detail of the forward pass.
    #
    # There should be problem-specific way to go from a
    # sequence of tuples of InputBatchType/TargetBatchType/MetadataBatchType
    # to a sequence of InputType/TargetType/MetadataType.

    # We don't currently guarantee that by iterating over a
    # <Input|Output|Metadata>BatchType that one will get individual instances
    # of <Input|Output|Metadata> type
    # ArrayLike protocol does not guarantee any iteration support, (although
    # we could convert to a numpy array and iterate on it.)

    preds = preds_batches
    aug_data = augmented_data_batches
    metric_results = metric.compute()

    return metric_results, preds, aug_data


# TODO: populate predict and use overloads for broader api
# def predict(
#     model: Model,
#     metric: Metric,
#     dataloader: Optional[DataLoader],
#     dataset: Optional[Dataset],
# ):
#     ...
