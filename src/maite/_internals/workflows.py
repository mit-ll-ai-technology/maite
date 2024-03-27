from typing import Any, Optional, Sequence, Tuple, Union, overload

import torch.utils.data
from typing_extensions import TypeAlias

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

SomeModel: TypeAlias = Union[ic.Model, od.Model]
SomeMetric: TypeAlias = Union[ic.Metric, od.Metric]
SomeDataset: TypeAlias = Union[ic.Dataset, od.Dataset]
SomeDataLoader: TypeAlias = Union[ic.DataLoader, od.DataLoader]
SomeAugmentation: TypeAlias = Union[ic.Augmentation, od.Augmentation]

SomeTargetBatchType: TypeAlias = Union[ic.TargetBatchType, od.TargetBatchType]
SomeInputBatchType: TypeAlias = Union[ic.InputBatchType, od.InputBatchType]
SomeMetadataBatchType: TypeAlias = Union[
    ic.DatumMetadataBatchType, od.DatumMetadataBatchType
]

SomeInputType: TypeAlias = Union[ic.InputType, od.InputType]
SomeTargetType: TypeAlias = Union[ic.TargetType, od.TargetType]
SomeMetadataType: TypeAlias = Union[ic.DatumMetadataType, od.DatumMetadataType]

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

# TODO: Permit returned predictions to be of type tuple[InputDataType, TargetDataType, DatumMetadataType].
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


def basic_collate_fn_image_classification(batch):
    from torch.utils.data import default_collate

    return (
        default_collate(
            [t[0] for t in batch]
        ),  # collate sequence of inputs (into single tensor)
        default_collate(
            [t[1] for t in batch]
        ),  # collate sequence of targets (into single tensor)
        [t[2] for t in batch],  # leave as sequence of dicts
    )


def basic_collate_fn(batch):
    from torch.utils.data import default_collate

    # if batch types for image classification (ic) or object detection (od)
    # are sequences (vs something like ArrayLike's with batch dimension)
    # then need to override default collate behavior
    # each original item in batch is 3-tuple of (input, target, metadata)
    first_tuple = batch[0]
    assert len(first_tuple) == 3

    # inputs: collate sequence of inputs (into single tensor)
    inputs = default_collate([t[0] for t in batch])

    # targets
    # - image classification: collate sequence of targets (into single tensor)
    # - object detection: leave as sequence of ObjectDetectionTarget
    assert isinstance(first_tuple[1], ic.TargetType) or isinstance(
        first_tuple[1], od.TargetType
    )
    if isinstance(first_tuple[1], ic.TargetType):
        targets = default_collate([t[1] for t in batch])
    else:
        targets = [t[1] for t in batch]

    # metadata: leave as sequence of dicts
    metadata = [t[2] for t in batch]

    return inputs, targets, metadata


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
# (one per ML domain/supported call signature)


# make predictions and calculate metrics over a dataloader (image_classification)
@overload
def evaluate(
    *,
    model: ic.Model,
    dataloader: ic.DataLoader,
    metric: Optional[ic.Metric] = None,
    augmentation: Optional[ic.Augmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[ic.TargetBatchType],
    Sequence[Tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]],
]:
    ...


# make predictions and calculate metrics over a dataset (img. classification)
@overload
def evaluate(
    *,
    model: ic.Model,
    dataset: ic.Dataset,
    batch_size: int = 1,
    metric: Optional[ic.Metric] = None,
    augmentation: Optional[ic.Augmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[ic.TargetBatchType],
    Sequence[Tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]],
]:
    ...


# make predictions and calculate metrics over a dataloader (obj. detection)
@overload
def evaluate(
    *,
    model: od.Model,
    dataloader: od.DataLoader,
    metric: Optional[od.Metric] = None,
    augmentation: Optional[od.Augmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[od.TargetBatchType],
    Sequence[Tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]],
]:
    ...


# make predictions and calculate metrics over a dataset (obj. detection)
@overload
def evaluate(
    *,
    model: od.Model,
    dataset: od.Dataset,
    batch_size: int = 1,
    metric: Optional[od.Metric] = None,
    augmentation: Optional[od.Augmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[od.TargetBatchType],
    Sequence[Tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]],
]:
    ...


# TODO: Narrow return types to be a union of accepted return types,
# 'Any' is too liberal and forgoes any potentially useful type-checking
# in this implementation.


def _evaluate(
    *,
    model: Model,
    metric: Optional[Metric] = None,
    dataloader: Optional[DataLoader] = None,
    dataset: Optional[Dataset] = None,
    batch_size: Optional[int] = None,
    augmentation: Optional[Augmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
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
        batch_size = 1

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
    # sequence of tuples of InputBatchType/TargetBatchType/DatumMetadataBatchType
    # to a sequence of InputType/TargetType/DatumMetadataType.

    # We don't currently guarantee that by iterating over a
    # <Input|Target|DatumMetadata>BatchType that one will get individual instances
    # of <Input|Target|DatumMetadata> type
    # ArrayLike protocol does not guarantee any iteration support, (although
    # we could convert to a numpy array and iterate on it.)

    preds = preds_batches
    aug_data = augmented_data_batches
    metric_results = metric.compute()

    return metric_results, preds, aug_data


def evaluate(
    *,
    model: SomeModel,
    metric: Optional[SomeMetric] = None,
    dataset: Optional[SomeDataset] = None,
    batch_size: int = 1,
    dataloader: Optional[SomeDataLoader] = None,
    augmentation: Optional[SomeAugmentation] = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> Tuple[
    MetricComputeReturnType,
    Sequence[SomeTargetBatchType],
    Sequence[Tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
]:
    # doc-ignore: EX01
    """
    Evaluate a model's performance on data according to some metric with optional augmentation.

    Some data source (either a dataloader or a dataset) must be provided
    or an InvalidArgument exception is raised.

    Parameters
    ----------
    model : SomeModel
        Maite Model object.

    metric : Optional[SomeMetric], (default=None)
        Compatible maite Metric.

    dataset : Optional[SomeDataset], (default=None)
        Compatible maite dataset.

    batch_size : int, (default=1)
        Batch size for use with dataset (ignored if dataset=None).

    dataloader : Optional[SomeDataloader], (default=None)
        Compatible maite dataloader.

    augmentation : Optional[SomeAugmentation], (default=None)
        Compatible maite augmentation.

    return_augmented_data : bool, (default=False)
        Set to True to return post-augmentation data as a function output.

    return_preds : bool, (default=False)
        Set to True to return raw predictions as a function output.

    Returns
    -------
    Tuple[Dict[str, Any], Sequence[TargetType], Sequence[Tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]]]
        Tuple of returned metric value, sequence of model predictions, and
        sequence of data batch tuples fed to the model during inference. The actual
        types represented by InputBatchType, TargetBatchType, and DatumMetadataBatchType will vary
        by the domain of the components provided as input arguments (e.g. image
        classification or object detection.)
        Note that the second and third return arguments will be empty if
        return_augmented_data is False or return_preds is False, respectively.
    """

    # Pass to untyped internal _evaluate
    # (We are relying on overload type signatures to statically validate
    # input arguments and relying on individual components to ensure outputs
    # are of the types promised by compatible overload-decorated evaluate signature

    return _evaluate(
        model=model,
        metric=metric,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=return_augmented_data,
        return_preds=return_preds,
    )


@overload
def predict(
    *,
    model: ic.Model,
    dataset: ic.Dataset,
    batch_size: int = 1,
    augmentation: Optional[ic.Augmentation] = None,
) -> Tuple[
    Sequence[ic.TargetBatchType],
    Sequence[Tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]],
]:
    ...


@overload
def predict(
    *,
    model: ic.Model,
    dataloader: ic.DataLoader,
    augmentation: Optional[ic.Augmentation] = None,
) -> Tuple[
    Sequence[ic.TargetBatchType],
    Sequence[Tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]],
]:
    ...


@overload
def predict(
    *,
    model: od.Model,
    dataset: od.Dataset,
    batch_size: int = 1,
    augmentation: Optional[od.Augmentation] = None,
) -> Tuple[
    Sequence[od.TargetBatchType],
    Sequence[Tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]],
]:
    ...


@overload
def predict(
    *,
    model: od.Model,
    dataloader: od.DataLoader,
    augmentation: Optional[od.Augmentation] = None,
) -> Tuple[
    Sequence[od.TargetBatchType],
    Sequence[Tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]],
]:
    ...


def predict(
    *,
    model: SomeModel,
    dataloader: Optional[SomeDataLoader] = None,
    dataset: Optional[SomeDataset] = None,
    batch_size: int = 1,
    augmentation: Optional[SomeAugmentation] = None,
) -> Tuple[
    Sequence[SomeTargetBatchType],
    Sequence[Tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
]:
    # doc-ignore: EX01
    """
    Make predictions for a given model & data source with optional augmentation.

    Some data source (either a dataloader or a dataset) must be provided
    or an InvalidArgument exception is raised.

    Parameters
    ----------
    model : SomeModel
        Maite Model object.

    dataloader : Optional[SomeDataloader], (default=None)
        Compatible maite dataloader.

    dataset : Optional[SomeDataset], (default=None)
        Compatible maite dataset.

    batch_size : int, (default=1)
        Batch size for use with dataset (ignored if dataset=None).

    augmentation : Optional[SomeAugmentation], (default=None)
        Compatible maite augmentation.

    Returns
    -------
    Tuple[Sequence[SomeTargetBatchType], Sequence[Tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
        A tuple of the predictions (as a sequence of batches) and a sequence
        of tuples containing the information associated with each batch.
    """

    metric_results, preds, aug_data = _evaluate(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        augmentation=augmentation,
        batch_size=batch_size,
        return_augmented_data=True,
        return_preds=True,
    )

    return preds, aug_data
