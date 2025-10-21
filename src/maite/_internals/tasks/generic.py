# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Generic, TypeVar

from maite._internals.protocols.generic import (
    Augmentation,
    CollateFn,
    DataLoader,
    Dataset,
    Metric,
    MetricComputeReturnType,
    Model,
)
from maite._internals.utils import add_progress_bar
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

    def reset(self) -> None: ...

    def update(
        self,
        __pred_batch: Any,
        __target_batch: Any,
        __metadata_batch: Any,
    ) -> None:
        return None

    def compute(self) -> MetricComputeReturnType:
        return dict()


# Note: These 3 TypeVars don't capture coupling constraints between
# types of each (i.e. all should be drawn from same AI problem).

T_input = TypeVar("T_input")
T_target = TypeVar("T_target")
T_metadata = TypeVar("T_metadata")
T_metadata_aug = TypeVar("T_metadata_aug")


def default_collate_fn(
    batch_data_as_singles: Iterable[tuple[T_input, T_target, T_metadata]],
) -> tuple[Sequence[T_input], Sequence[T_target], Sequence[T_metadata]]:
    """
    Describe how to create a tuple of batches (i.e.,
    (Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]) )
    from an iterator of tuples of
    (InputType, TargetType, DatumMetadataType)
    """

    # first just unpack inputs/targets/metadata from iterator into separate lists
    input_batch: list[T_input] = []
    target_batch: list[T_target] = []
    metadata_batch: list[T_metadata] = []
    for input_datum, target_datum, metadata_datum in batch_data_as_singles:
        input_batch.append(input_datum)
        target_batch.append(target_datum)
        metadata_batch.append(metadata_datum)

    return input_batch, target_batch, metadata_batch


class _SimpleDataLoader(Generic[T_input, T_target, T_metadata]):
    """
    Simple dataloader to create by default should a user not provide their own
    dataloader but does provide a Dataset.

    Preconditions: dataset and collate_fn come from same AI problem domain
    """

    def __init__(
        self,
        dataset: Dataset[T_input, T_target, T_metadata],
        batch_size: int,
        collate_fn: CollateFn[T_input, T_target, T_metadata] = default_collate_fn,
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


def augment_dataloader(
    *,
    augmentation: Augmentation[
        T_input,
        T_target,
        T_metadata_aug,
        T_input,
        T_target,
        T_metadata,
    ],
    dataloader: DataLoader[T_input, T_target, T_metadata],
) -> DataLoader[T_input, T_target, T_metadata_aug]:
    # doc-ignore: EX01, YD01
    # YD01 We prefer to document the function as returning a DataLoader, rather than yielding augmented batch elements.
    """
    Create an `DataLoader` of augmented inputs from a `Dataset` or `DataLoader`.

    Parameters
    ----------
    augmentation : Augmentation
        Compatible maite augmentation.

    dataloader : Dataloader
        Compatible maite dataloader.

    Returns
    -------
    DataLoader[InputType, TargetType, DatumMetadataType]
        DataLoader for augmented inputs.
    """
    for input_batch in add_progress_bar(dataloader):
        augmented_batch = augmentation(input_batch)
        yield augmented_batch


# begin all overloads for evaluate function
# (There are currently no task-specific default arguments, which means
# that all runtime-behavior of evaluate is common and no task-specific
# procedures need to exist, at least not yet.)


def evaluate(
    *,
    model: Model[T_input, T_target],
    metric: Metric[T_target, T_metadata] | None = None,
    dataloader: DataLoader[T_input, T_target, T_metadata] | None = None,
    dataset: Dataset[T_input, T_target, T_metadata] | None = None,
    batch_size: int = 1,
    augmentation: (
        Augmentation[
            T_input,
            T_target,
            T_metadata,
            T_input,
            T_target,
            T_metadata,
        ]
        | None
    ) = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
    collate_fn: CollateFn[T_input, T_target, T_metadata] = default_collate_fn,
) -> tuple[
    MetricComputeReturnType,
    Sequence[Sequence[T_target]],
    Sequence[tuple[Sequence[T_input], Sequence[T_target], Sequence[T_metadata]]],
]:
    # doc-ignore: EX01
    """
    Evaluate a model's performance on data according to some metric with optional augmentation.

    The types handled by all passed components must be compatible to avoid static type checking
    errors. For example, if the __getitem__ method of a passed dataset returns some `InputType`
    in the first element of the return tuple then the `model.__call__` argument must be type hinted
    such that `Sequence[InputType]` can be assigned to it.

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

    collate_fn : Callable[[Iterable[tuple[T_input, T_target, T_metadata]]], tuple[Sequence[T_input], Sequence[T_target], Sequence[T_metadata]] ], (default=None)
        Callable responsible for transforming an iterable of 3-tuples where each encodes a single
        datapoint in some batch into a tuple of 3 sequences that each represent a batch of collated
        inputs, collated targets, and collated metadata, respectively. Defaults to naively push
        elements from input iterable onto sequences in their order of iteration.

    Returns
    -------
    tuple[dict[str, Any], Sequence[TargetType], Sequence[tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]]]
        Tuple of returned metric value, sequence of model predictions, and
        sequence of data batch tuples fed to the model during inference. The actual
        types represented by InputType, TargetType, and DatumMetadataType will vary
        by the AI problem of the components provided as input arguments (e.g., image
        classification or object detection.)
        Note that the second and third return arguments will be empty if
        return_augmented_data is False or return_preds is False, respectively.

    Raises
    ------
    ValueError
        If neither a dataloader nor a dataset is provided
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

    if dataloader is not None and dataset is not None:
        raise ValueError("Do not provide both a dataloader and a dataset")

    if dataloader is None and dataset is None:
        # user provided neither a dataloader nor a dataset
        raise ValueError("One of dataloader and dataset must be provided")

    if dataloader is None and dataset is not None:
        if collate_fn is None:
            raise ValueError(
                "If dataset is provided, then collate_fn is required"
                + "to permit building a dataloader"
            )

    if dataloader is None:
        assert dataset is not None  # should never trigger due to previous checks
        assert batch_size is not None  # shouldn't trigger due to default value
        assert collate_fn is not None  # should never trigger due to previous checks

        dataloader = _SimpleDataLoader[T_input, T_target, T_metadata](
            dataset=dataset, batch_size=batch_size, collate_fn=collate_fn
        )

    # dataloader, metric, and model are populated by this point
    assert dataloader is not None
    assert metric is not None
    assert model is not None

    pred_batches = []
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

        pred_batch = model(input_datum_batch_aug)
        metric.update(pred_batch, target_datum_batch_aug, metadata_batch_aug)

        # store any requested intermediate data for returning to user
        if return_augmented_data:
            augmented_data_batches.append(
                (input_datum_batch_aug, target_datum_batch_aug, metadata_batch_aug)
            )

        if return_preds:
            pred_batches.append(pred_batch)

    # TODO: investigate better munging/recollating iterable/batches
    # aug_data is currently a sequence of tuples of batch-types
    # this is a little artificial as batch-size is typically less relevant
    # to test ande evaluation and more an implementation detail of the forward
    # pass.

    preds = pred_batches
    aug_data = augmented_data_batches
    metric_results = metric.compute()

    return metric_results, preds, aug_data


def evaluate_from_predictions(
    *,
    metric: Metric[T_target, T_metadata],
    pred_batches: Sequence[Sequence[T_target]],
    target_batches: Sequence[Sequence[T_target]],
    metadata_batches: Sequence[Sequence[T_metadata]],
) -> MetricComputeReturnType:
    # doc-ignore: EX01
    """
    Evaluate pre-calculated predictions against target (truth) data for some specified metric.

    Parameters
    ----------

    metric : SomeMetric
        Compatible MAITE Metric.

    pred_batches : Sequence[Sequence[SomeTargetType]]
        Sequence of batches of predictions generated by running inference on some model.

    target_batches : Sequence[Sequence[SomeTargetType]]
        Sequence of batches of ground-truth targets that correspond to provided predictions argument.

    metadata_batches : Sequence[Sequence[SomeDatumMetadataType]]
        Sequence of batches of datum metadata type.

    Returns
    -------
    metric_calculation: MetricComputeReturnType
        The resulting metric value.

    Raises
    ------
    ValueError
        If predictions or targets arguments have zero length (i.e. no batches), differing lengths,
        or corresponding elements (batches) have differing lengths.
    """
    # Checks for argument compliance.

    if len(pred_batches) != len(target_batches):
        raise ValueError(
            "Arguments predictions and truth_datum are expected to have the same number of elements (batches)"
        )

    if len(pred_batches) < 1 or len(target_batches) < 1:
        raise ValueError("Predictions and targets must have at least one element.")

    metric.reset()
    for pred_batch, target_batch, metadata_batch in zip(
        pred_batches, target_batches, metadata_batches
    ):
        if len(pred_batch) != len(target_batch):
            raise ValueError(
                "Corresponding prediction and target batches must have the same length."
            )
        metric.update(pred_batch, target_batch, metadata_batch)
    metric_results = metric.compute()
    return metric_results


def predict(
    *,
    model: Model[T_input, T_target],
    dataloader: (DataLoader[T_input, T_target, T_metadata] | None) = None,
    dataset: Dataset[T_input, T_target, T_metadata] | None = None,
    batch_size: int = 1,
    augmentation: Augmentation[
        T_input,
        T_target,
        T_metadata,
        T_input,
        T_target,
        T_metadata,
    ]
    | None = None,
    return_augmented_data: bool = False,
) -> tuple[
    Sequence[Sequence[T_target]],
    Sequence[tuple[Sequence[T_input], Sequence[T_target], Sequence[T_metadata]]],
]:
    # doc-ignore: EX01
    """
    Make predictions for a given model & data source with optional augmentation.

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

    return_augmented_data : bool, (default=False)
        Set to True to return post-augmentation data as a function output.

    Returns
    -------
    tuple[Sequence[Sequence[SomeTargetType], Sequence[tuple[Sequence[SomeInputType], Sequence[SomeTargetType], Sequence[SomeMetadataType]]],
        A tuple of the predictions (as a sequence of batches) and a sequence
        of tuples containing the information associated with each batch.
        Note that the second return argument will be empty if
        return_augmented_data is False.

    Raises
    ------
    ValueError
        If neither a dataloader nor a dataset is provided.
    """

    _, preds, aug_data = evaluate(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        augmentation=augmentation,
        batch_size=batch_size,
        collate_fn=default_collate_fn,
        return_augmented_data=return_augmented_data,
        return_preds=True,
    )

    return preds, aug_data
