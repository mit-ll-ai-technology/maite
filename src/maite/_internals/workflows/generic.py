# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Generic, TypeVar, overload

from maite._internals.protocols import (
    image_classification as ic,
    object_detection as od,
)
from maite._internals.protocols.generic import (
    Augmentation,
    CollateFn,
    DataLoader,
    Dataset,
    Metric,
    MetricComputeReturnType,
    Model,
)
from maite._internals.protocols.task_aliases import (  # SomeInputType,; SomeMetadataType,; SomeTargetType,
    OpenAugmentationType,
    OpenDataLoaderType,
    OpenDatasetType,
    OpenMetricType,
    OpenModelType,
    SomeInputBatchType,
    SomeInputType,
    SomeMetadataBatchType,
    SomeMetadataType,
    SomeTargetBatchType,
    SomeTargetType,
)
from maite._internals.utils import add_progress_bar
from maite.errors import InvalidArgument
from maite.protocols import MetricMetadata

# TODO: Permit returned predictions to be of type tuple[InputDataType, TargetDataType, DatumMetadataType].
#       This seems much more natural as it is independent of batch_size.
#   - This would require we use some method to iterate over "Batch-types" to get individual types
#     and package individual predictions into an iterable. batch objects are iterable and then
#     checking that the type of object returned from the iterable is ArrayLike.)


# TODO: Consider the implications of having a flexible implementation that only relies on membership
#       in generic class types. Can we guarantee (or test) that satisfying one of our overload
#       type signatures is always sufficient to return the types promised by that signature?

#       A: Yes -- This can be done by using static type checker to determine whether evaluate
#          and predict implementations returns promised types from each overload when inputs
#          are typed as promised by call signatures. See gitlab issue 425 and discussion on MR129.


class _DummyMetric(Metric):
    """
    Metric that does nothing and returns an empty dictionary from compute
    """

    def __init__(self):
        self.metadata = MetricMetadata({"id": "dummy_metric"})

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


# Note: These 3 TypeVars don't capture coupling constraints between
# types of each (i.e. all should be drawn from same subproblem).

T_in = TypeVar("T_in", bound=SomeInputType)
T_tgt = TypeVar("T_tgt", bound=SomeTargetType)
T_md = TypeVar("T_md", bound=SomeMetadataType)


def default_collate_fn(
    batch_data_as_singles: Iterable[tuple[T_in, T_tgt, T_md]]
) -> tuple[Sequence[T_in], Sequence[T_tgt], Sequence[T_md]]:
    """
    Describe how to create a tuple of
    (InputBatchType, TargetBatchType, DatumMetadataBatchType)
    from an iterator of tuples of
    (InputType, TargetType, DatumMetadataType)
    """

    # first just unpack inputs/targets/metadata from iterator into separate lists
    input_batch: list[T_in] = []
    target_batch: list[T_tgt] = []
    metadata_batch: list[T_md] = []
    for input_datum, target_datum, metadata_datum in batch_data_as_singles:
        input_batch.append(input_datum)
        target_batch.append(target_datum)
        metadata_batch.append(metadata_datum)

    return input_batch, target_batch, metadata_batch


class _SimpleDataLoader(Generic[T_in, T_tgt, T_md]):
    """
    Simple dataloader to create by default should a user not provide their own
    dataloader but does provide a Dataset.

    Preconditions: dataset and collate_fn come from same ml subproblem domain
    """

    def __init__(
        self,
        dataset: Dataset[T_in, T_tgt, T_md],
        batch_size: int,
        collate_fn: CollateFn = default_collate_fn,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(self):
        # iterate over first batch_size examples from dataset, collate them, and yield result
        n_batches = (
            len(self.dataset) // self.batch_size
            if len(self.dataset) % self.batch_size == 0
            else len(self.dataset) // self.batch_size + 1
        )

        for batch_no in range(n_batches):
            batch_data_as_singles = [
                self.dataset[i]
                for i in range(
                    batch_no * self.batch_size, (batch_no + 1) * self.batch_size
                )
                if i < len(self.dataset)
            ]

            batch_inputs, batch_targets, batch_metadata = self.collate_fn(
                batch_data_as_singles
            )

            yield (batch_inputs, batch_targets, batch_metadata)

    def __len__(self) -> int:
        """Length of `SimpleDataLoader`.

        Returns
        -------
        int
            Length of the `dataset` if `batch_size` is <= 0. Otherwise, the total number of
            times the `dataset` can be iterated over given the `batch_size`, before it is exhausted.
        """
        if self.batch_size > 0:
            from math import ceil

            return ceil(len(self.dataset) / self.batch_size)
        return len(self.dataset)


# begin all overloads for evaluate function
# (There are currently no task-specific default arguments, which means
# that all runtime-behavior of evaluate is common and no task-specific
# procedures need to exist, at least not yet.)


@overload
def evaluate(
    *,
    model: ic.Model,
    metric: ic.Metric | None = None,
    dataloader: ic.DataLoader | None = None,
    dataset: ic.Dataset | None = None,
    batch_size: int = 1,
    augmentation: ic.Augmentation | None = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> tuple[
    MetricComputeReturnType,
    Sequence[ic.TargetBatchType],
    Sequence[tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]],
]:
    ...


@overload
def evaluate(
    *,
    model: od.Model,
    metric: od.Metric | None = None,
    dataloader: od.DataLoader | None = None,
    dataset: od.Dataset | None = None,
    batch_size: int = 1,
    augmentation: od.Augmentation | None = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> tuple[
    MetricComputeReturnType,
    Sequence[od.TargetBatchType],
    Sequence[tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]],
]:
    ...


def evaluate(
    *,
    model: OpenModelType,
    metric: OpenMetricType | None = None,
    dataloader: OpenDataLoaderType | None = None,
    dataset: OpenDatasetType | None = None,
    batch_size: int = 1,
    augmentation: OpenAugmentationType | None = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> tuple[
    MetricComputeReturnType,
    Sequence[SomeTargetBatchType],
    Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
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

    metric : SomeMetric | None, (default=None)
        Compatible maite Metric.

    dataloader : SomeDataloader | None, (default=None)
        Compatible maite dataloader.

    dataset : SomeDataset | None, (default=None)
        Compatible maite dataset.

    batch_size : int, (default=1)
        Batch size for use with dataset (ignored if dataset=None).

    augmentation : SomeAugmentation | None, (default=None)
        Compatible maite augmentation.

    return_augmented_data : bool, (default=False)
        Set to True to return post-augmentation data as a function output.

    return_preds : bool, (default=False)
        Set to True to return raw predictions as a function output.

    Returns
    -------
    tuple[dict[str, Any], Sequence[TargetType], Sequence[tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]]]
        Tuple of returned metric value, sequence of model predictions, and
        sequence of data batch tuples fed to the model during inference. The actual
        types represented by InputBatchType, TargetBatchType, and DatumMetadataBatchType will vary
        by the domain of the components provided as input arguments (e.g. image
        classification or object detection.)
        Note that the second and third return arguments will be empty if
        return_augmented_data is False or return_preds is False, respectively.
    """

    return _evaluate(
        model=model,
        metric=metric,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        collate_fn=default_collate_fn,
        return_augmented_data=return_augmented_data,
        return_preds=return_preds,
    )


def _evaluate(
    *,
    model: Model,
    metric: Metric | None = None,
    dataloader: DataLoader | None = None,
    dataset: Dataset | None = None,
    batch_size: int = 1,
    augmentation: Augmentation | None = None,
    collate_fn: CollateFn | None = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
) -> tuple[
    MetricComputeReturnType,
    Sequence[SomeTargetBatchType],
    Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
]:
    """
    Task-agnostically typed version of evaluate for use by any workflows that
    wish to leverage evaluate internally in a task-agnostic way.
    """

    # Validate potential input combinations.
    # User needs to provide ONE of either (a) dataloader (iterable over
    # tuples of batch-types) or (b) a dataset (an iterable over tuples
    # of individual types) and a collate_fn. (batch_size is assumed
    # to be 1 if not provided)

    # if metric is None, make it a _DummyMetric that does nothing

    # if dataloader is None, check that we have Dataset and collate_fn.
    # If so, construct a dataloader, and assign to dataloader variable

    if metric is None:
        # user provided no metric
        metric = _DummyMetric()

    if dataloader is None and dataset is None:
        # user provided neither a dataloader nor a dataset
        raise InvalidArgument("One of dataloader and dataset must be provided")

    if dataloader is None and dataset is not None:
        if collate_fn is None:
            raise InvalidArgument(
                "If dataset is provided, then collate_fn is required"
                + "to permit building a dataloader"
            )

    if dataloader is None:
        assert dataset is not None  # should never trigger due to previous checks
        assert batch_size is not None  # shouldn't trigger due to default value
        assert collate_fn is not None  # should never trigger due to previous checks

        dataloader = _SimpleDataLoader(
            dataset=dataset, batch_size=batch_size, collate_fn=collate_fn
        )

    # dataloader, metric, and model are populated by this point
    assert dataloader is not None
    assert metric is not None
    assert model is not None

    preds_batches = []
    metric.reset()
    augmented_data_batches = []

    for input_datum_batch, target_datum_batch, metadata_batch in add_progress_bar(
        dataloader
    ):
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
    # this is a little artificial as batch-size is typically less relevant
    # to test ande evaluation and more an implementation detail of the forward
    # pass.

    preds = preds_batches
    aug_data = augmented_data_batches
    metric_results = metric.compute()

    return metric_results, preds, aug_data


@overload
def predict(
    *,
    model: ic.Model,
    dataloader: ic.DataLoader | None = None,
    dataset: ic.Dataset | None = None,
    batch_size: int = 1,
    augmentation: ic.Augmentation | None = None,
) -> tuple[
    Sequence[ic.TargetBatchType],
    Sequence[tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]],
]:
    ...


@overload
def predict(
    *,
    model: od.Model,
    dataloader: od.DataLoader | None = None,
    dataset: od.Dataset | None = None,
    batch_size: int = 1,
    augmentation: od.Augmentation | None = None,
) -> tuple[
    Sequence[od.TargetBatchType],
    Sequence[tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]],
]:
    ...


def predict(
    *,
    model: OpenModelType,
    dataloader: OpenDataLoaderType | None = None,
    dataset: OpenDatasetType | None = None,
    batch_size: int = 1,
    augmentation: OpenAugmentationType | None = None,
) -> tuple[
    Sequence[SomeTargetBatchType],
    Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
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

    dataloader : SomeDataloader | None, (default=None)
        Compatible maite dataloader.

    dataset : SomeDataset | None, (default=None)
        Compatible maite dataset.

    batch_size : int, (default=1)
        Batch size for use with dataset (ignored if dataset=None).

    augmentation : SomeAugmentation | None, (default=None)
        Compatible maite augmentation.

    Returns
    -------
    tuple[Sequence[SomeTargetBatchType], Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
        A tuple of the predictions (as a sequence of batches) and a sequence
        of tuples containing the information associated with each batch.
    """

    _, preds, aug_data = _evaluate(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        augmentation=augmentation,
        batch_size=batch_size,
        collate_fn=default_collate_fn,
        return_augmented_data=True,
        return_preds=True,
    )

    return preds, aug_data
