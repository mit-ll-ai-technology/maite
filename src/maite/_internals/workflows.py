from typing import Any, Optional, Sequence, Tuple, overload

import torch.utils.data

from maite._internals.protocols.generic import (
    Augmentation,
    DataLoader,
    Dataset,
    Metric,
    MetricComputeReturnType,
    Model,
)
from maite.errors import InvalidArgument
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


# TODO: Consider the implications of having a flexible implementation that only relies on membership
#       in generic class types. Can we guarantee (or test) that satisfying one of our overload
#       type signatures is always sufficient to return the types promised by that signature?


class DummyMetric(Metric):
    """
    Metric that does nothing and returns an empty dictionary from compute
    """

    def reset(self) -> None:
        ...

    def update(
        self,
        __preds_batch: Any,
        __targets_batch: Any,
    ) -> None:
        return None

    def compute(self) -> MetricComputeReturnType:
        return dict()


def basic_collate_fn(batch):
    from torch.utils.data import default_collate

    return (
        default_collate(
            [t[0] for t in batch]
        ),  # collate sequence of inputs (into single tensor)
        default_collate(
            [t[1] for t in batch]
        ),  # collate sequence of outputs (into single tensor)
        [t[2] for t in batch],  # leave as sequence of dicts
    )


class SimpleTorchDataLoader:
    """
    Simple pytorch dataloader to create by default should a user not provide their own
    dataloader but does provide a Dataset.
    """

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

        # reason for type ignore is that MAITE `Dataset` doesn't completely
        # match PyTorch `Dataset` or `IterableDataset` - which has `__add__()` method.
        # This method doesn't seem to be needed by DataLoader
        self.dataloader = torch.utils.data.DataLoader(dataset, collate_fn=basic_collate_fn, batch_size=batch_size)  # type: ignore

    def __iter__(self):
        return self.dataloader.__iter__()


# begin all overloads for evaluate function
# (one per ML domain and call signature)


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
    model: ic.Model,
    metric: ic.Metric,
    dataset: ic.Dataset,
    batch_size: int,
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
    model: ic.Model,
    dataloader: ic.DataLoader,
    metric: Optional[ic.Metric] = None,
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
    model: ic.Model,
    dataset: ic.Dataset,
    metric: Optional[ic.Metric] = None,
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


@overload
def evaluate(
    *,
    model: od.Model,
    metric: od.Metric,
    dataset: od.Dataset,
    batch_size: int,
    augmentation: Optional[od.Augmentation] = None,
    return_augmented_data=False,
    return_preds=False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[od.OutputBatchType],
    Sequence[Tuple[od.InputBatchType, od.OutputBatchType, od.MetadataBatchType]],
]:
    ...


@overload
def evaluate(
    *,
    model: od.Model,
    dataloader: od.DataLoader,
    metric: Optional[od.Metric] = None,
    augmentation: Optional[od.Augmentation] = None,
    return_augmented_data=False,
    return_preds=False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[od.OutputBatchType],
    Sequence[Tuple[od.InputBatchType, od.OutputBatchType, od.MetadataBatchType]],
]:
    ...


@overload
def evaluate(
    *,
    model: od.Model,
    dataset: od.Dataset,
    metric: Optional[od.Metric] = None,
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
# in this implementation.


def evaluate(
    *,
    model: Model,
    metric: Optional[Metric] = None,
    dataloader: Optional[DataLoader] = None,
    dataset: Optional[Dataset] = None,
    batch_size: Optional[int] = None,
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

    # if metric is None, make it a DummyMetric that does nothing

    # if dataloader is None, check that we have Dataset and batch_size,
    # construct a dataloader, and assign to dataloader

    if metric is None:
        # user provided no metric
        metric = DummyMetric()

    if dataloader is None and dataset is None:
        # user provided neither a dataloader nor a dataset
        raise InvalidArgument("One of dataloader and dataset must be provided")

    if dataset is not None and batch_size is None:
        # user provided a dataset and not a batch_size
        raise InvalidArgument("If dataset is provided, batch_size is also required")

    if dataloader is None:
        assert dataset is not None  # should never trigger due to previous checks
        assert batch_size is not None  # should never trigger due to previous checks

        dataloader = SimpleTorchDataLoader(dataset=dataset, batch_size=batch_size)

    # dataloader, metric, and model are populated by this point
    assert dataloader is not None
    assert metric is not None
    assert model is not None

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


@overload
def predict(
    *,
    model: ic.Model,
    dataset: ic.Dataset,
) -> ic.OutputBatchType:
    ...


@overload
def predict(
    *,
    model: ic.Model,
    dataloader: ic.DataLoader,
) -> ic.OutputBatchType:
    ...


@overload
def predict(
    *,
    model: od.Model,
    dataset: od.Dataset,
) -> od.OutputBatchType:
    ...


@overload
def predict(
    *,
    model: od.Model,
    dataloader: od.DataLoader,
) -> od.OutputBatchType:
    ...


def predict(
    *,
    model: Model,
    dataloader: Optional[DataLoader] = None,
    dataset: Optional[Dataset] = None,
) -> Any:
    metric_results, preds, aug_data = evaluate(
        model=model, dataloader=dataloader, dataset=dataset
    )

    return preds
