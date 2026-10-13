# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
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


class _DummyMetric(Metric):
    """Metric that does nothing and returns an empty dictionary from compute"""

    def __init__(self) -> None:
        self.metadata = MetricMetadata({"id": "dummy_metric"})

    def reset(self) -> None: ...

    def update(
        self,
        _pred_batch: Any,  # noqa: ANN401, deliberate use of 'Any' type
        _target_batch: Any,  # noqa: ANN401, deliberate use of 'Any' type
        _metadata_batch: Any,  # noqa: ANN401, deliberate use of 'Any' type
        /,
    ) -> None:
        return None

    def compute(self) -> MetricComputeReturnType:
        return {}


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
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            Sequence[T_input],
            Sequence[T_target],
            Sequence[T_metadata],
        ]
    ]:
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
                    batch_no * self.batch_size,
                    (batch_no + 1) * self.batch_size,
                )
                if i < len(self.dataset)
            ]

            batch_inputs, batch_targets, batch_metadata = self.collate_fn(
                batch_data_as_singles,
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
    # doc-ignore: YD01
    # YD01 We prefer to document the function as returning a DataLoader, rather than yielding augmented
    # batch elements.
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

    Examples
    --------
    We illustrate with an image classification problem, but the same pattern
    applies to any MAITE task. See the `maite.protocols.image_classification`
    documentation for details on writing compliant components.

    >>> import numpy as np
    >>> import numpy.typing as npt
    >>> from collections.abc import Sequence
    >>> from maite.protocols import ArrayLike, AugmentationMetadata, DatumMetadata
    >>> from maite.protocols import image_classification as ic

    An `Augmentation` takes a data batch (a tuple of input, target, and metadata
    batches) and returns a modified batch. This one zeros out the first channel
    of every image.

    >>> class ZeroFirstChannel:
    ...     metadata: AugmentationMetadata = {"id": "zero_first_channel"}
    ...
    ...     def __call__(
    ...         self,
    ...         batch: tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[DatumMetadata]],
    ...     ) -> tuple[Sequence[npt.NDArray], Sequence[ArrayLike], Sequence[DatumMetadata]]:
    ...         inputs, targets, metadata = batch
    ...         inputs_aug = []
    ...         for x in inputs:
    ...             arr = np.asarray(x).copy()
    ...             arr[0] = 0.0
    ...             inputs_aug.append(arr)
    ...         return inputs_aug, targets, metadata

    A `DataLoader` is any iterable of such batches. Here we build one by hand from
    two shape-`(3, 4, 4)` images.

    >>> images = [np.ones((3, 4, 4)) for _ in range(2)]
    >>> targets = [np.eye(3)[c] for c in [0, 1]]
    >>> metadata: list[DatumMetadata] = [{"id": 0}, {"id": 1}]
    >>> dataloader = [(images, targets, metadata)]

    Wrapping the dataloader with the augmentation yields a new dataloader whose
    batches have been augmented.

    >>> from maite.tasks import augment_dataloader
    >>> augmentation: ic.Augmentation = ZeroFirstChannel()
    >>> augmented = augment_dataloader(augmentation=augmentation, dataloader=dataloader)
    >>> aug_inputs, _, _ = next(iter(augmented))
    >>> float(np.asarray(aug_inputs[0])[0].sum())  # first channel is now all zeros
    0.0
    """
    for input_batch in add_progress_bar(dataloader):
        augmented_batch = augmentation(input_batch)
        yield augmented_batch


def _resolve_dataloader(
    dataloader: DataLoader[T_input, T_target, T_metadata] | None,
    dataset: Dataset[T_input, T_target, T_metadata] | None,
    *,
    batch_size: int,
    collate_fn: CollateFn[T_input, T_target, T_metadata],
) -> DataLoader[T_input, T_target, T_metadata]:
    """Return the provided `dataloader`, or build a `_SimpleDataLoader` from `dataset`.

    Exactly one of `dataloader` or `dataset` must be provided. The branch structure
    both validates this and narrows the types for static analysis.

    Raises
    ------
    ValueError
        If both `dataloader` and `dataset` are provided, or if neither is.
    """
    if dataloader is not None:
        if dataset is not None:
            raise ValueError("Do not provide both a dataloader and a dataset")
        return dataloader

    if dataset is None:
        raise ValueError("One of dataloader and dataset must be provided")

    return _SimpleDataLoader[T_input, T_target, T_metadata](
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


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

    collate_fn : SomeCollateFn, (default=default_collate_fn)
        Callable responsible for transforming an iterable of 3-tuples where each encodes a single
        datapoint in some batch into a tuple of 3 sequences that each represent a batch of collated
        inputs, collated targets, and collated metadata, respectively. Defaults to naively push
        elements from input iterable onto sequences in their order of iteration.

    Returns
    -------
    tuple[dict[str, Any],
          Sequence[TargetType],
          Sequence[tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]]]
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

    Examples
    --------
    We illustrate with an image classification problem, but the same pattern
    applies to any MAITE task. See the `maite.protocols.image_classification`
    documentation for details on writing compliant components.

    >>> import numpy as np
    >>> import numpy.typing as npt
    >>> from collections.abc import Sequence
    >>> from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata, MetricMetadata, ModelMetadata
    >>> from maite.protocols import image_classification as ic

    We set up a tiny three-class problem in which the class of an image is the
    index of its brightest channel.

    >>> def make_image(bright_channel: int) -> npt.NDArray:
    ...     img = np.full((3, 4, 4), 0.1)
    ...     img[bright_channel] = 0.9
    ...     return img
    >>> images = [make_image(c) for c in [0, 1, 2, 0]]
    >>> targets = [np.eye(3)[t] for t in [0, 1, 2, 1]]  # note: last image is mislabeled

    A `Model` predicts the brightest-channel class for each image in a batch,
    returning a one-hot score vector per image.

    >>> class ChannelClassifier:
    ...     metadata: ModelMetadata = {"id": "ChannelClassifier"}
    ...
    ...     def __call__(self, batch: Sequence[ArrayLike]) -> Sequence[npt.NDArray]:
    ...         preds = []
    ...         for x in batch:
    ...             cls = int(np.asarray(x).mean(axis=(1, 2)).argmax())
    ...             preds.append(np.eye(3)[cls])
    ...         return preds

    A `Metric` that accumulates predictions and targets across batches and
    computes accuracy on `compute`.

    >>> class Accuracy:
    ...     metadata: MetricMetadata = {"id": "accuracy"}
    ...
    ...     def __init__(self) -> None:
    ...         self.reset()
    ...
    ...     def reset(self) -> None:
    ...         self._correct = 0
    ...         self._total = 0
    ...
    ...     def update(
    ...         self,
    ...         preds: Sequence[ArrayLike],
    ...         targets: Sequence[ArrayLike],
    ...         metadata: Sequence[DatumMetadata],
    ...     ) -> None:
    ...         pred_classes = np.vstack([np.asarray(p) for p in preds]).argmax(axis=1)
    ...         true_classes = np.vstack([np.asarray(t) for t in targets]).argmax(axis=1)
    ...         self._correct += int((pred_classes == true_classes).sum())
    ...         self._total += len(pred_classes)
    ...
    ...     def compute(self) -> dict[str, float]:
    ...         return {"accuracy": self._correct / self._total}

    A `Dataset` that returns an (input, target, metadata) 3-tuple per index.

    >>> class ImageDataset:
    ...     metadata: DatasetMetadata = {"id": "example_dataset"}
    ...
    ...     def __init__(self, images: list[npt.NDArray], targets: list[npt.NDArray]) -> None:
    ...         self.images = images
    ...         self.targets = targets
    ...
    ...     def __len__(self) -> int:
    ...         return len(self.images)
    ...
    ...     def __getitem__(self, i: int) -> tuple[npt.NDArray, npt.NDArray, DatumMetadata]:
    ...         md: DatumMetadata = {"id": i}
    ...         return self.images[i], self.targets[i], md

    We type-hint each instance against its protocol so a static type checker can
    verify compliance, then run `evaluate`. Passing a `dataset` (rather than a
    `dataloader`) lets `evaluate` build batches of `batch_size` internally.

    >>> from maite.tasks import evaluate
    >>> model: ic.Model = ChannelClassifier()
    >>> metric: ic.Metric = Accuracy()
    >>> dataset: ic.Dataset = ImageDataset(images, targets)
    >>> metric_results, _, _ = evaluate(
    ...     model=model,
    ...     metric=metric,
    ...     dataset=dataset,
    ...     batch_size=2,
    ... )

    Three of the four images are classified in agreement with their (possibly
    noisy) targets.

    >>> metric_results
    {'accuracy': 0.75}

    Set `return_preds=True` and/or `return_augmented_data=True` to additionally
    receive the per-batch predictions and the (post-augmentation) data batches
    fed to the model; both are empty sequences otherwise.
    """

    # User needs to provide ONE of either (a) dataloader (iterable over
    # tuples of batch-types) or (b) a dataset (an iterable over tuples
    # of individual types); a collate_fn and batch_size default to sensible
    # values used only when building a dataloader from a dataset.

    if model is None:
        raise ValueError("Model must be provided")

    if metric is None:
        # user provided no metric
        metric = _DummyMetric()

    dataloader = _resolve_dataloader(dataloader, dataset, batch_size=batch_size, collate_fn=collate_fn)

    pred_batches = []
    metric.reset()
    augmented_data_batches = []

    for input_datum_batch, target_datum_batch, metadata_batch in add_progress_bar(
        dataloader,
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
                (input_datum_batch_aug, target_datum_batch_aug, metadata_batch_aug),
            )

        if return_preds:
            pred_batches.append(pred_batch)

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
    """
    Evaluate pre-calculated predictions against target (truth) data for some specified metric.

    This is useful when predictions have already been computed (e.g., by a prior
    call to `predict`, or by a model outside of MAITE) and only the metric value
    is needed, avoiding a second inference pass.

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

    Examples
    --------
    We illustrate with an image classification problem, but the same pattern
    applies to any MAITE task. See the `maite.protocols.image_classification`
    documentation for details on writing compliant components.

    >>> import numpy as np
    >>> from collections.abc import Sequence
    >>> from maite.protocols import ArrayLike, DatumMetadata, MetricMetadata
    >>> from maite.protocols import image_classification as ic

    A `Metric` that accumulates predictions and targets across batches and
    computes accuracy on `compute`.

    >>> class Accuracy:
    ...     metadata: MetricMetadata = {"id": "accuracy"}
    ...
    ...     def __init__(self) -> None:
    ...         self.reset()
    ...
    ...     def reset(self) -> None:
    ...         self._correct = 0
    ...         self._total = 0
    ...
    ...     def update(
    ...         self,
    ...         preds: Sequence[ArrayLike],
    ...         targets: Sequence[ArrayLike],
    ...         metadata: Sequence[DatumMetadata],
    ...     ) -> None:
    ...         pred_classes = np.vstack([np.asarray(p) for p in preds]).argmax(axis=1)
    ...         true_classes = np.vstack([np.asarray(t) for t in targets]).argmax(axis=1)
    ...         self._correct += int((pred_classes == true_classes).sum())
    ...         self._total += len(pred_classes)
    ...
    ...     def compute(self) -> dict[str, float]:
    ...         return {"accuracy": self._correct / self._total}

    Predictions and targets are each supplied as a sequence of batches, where a
    batch is a sequence of per-datum one-hot score vectors. Here we have two
    batches of two datapoints, with one of the four predictions disagreeing with
    its target.

    >>> pred_batches = [[np.eye(3)[c] for c in [0, 1]], [np.eye(3)[c] for c in [2, 0]]]
    >>> target_batches = [[np.eye(3)[c] for c in [0, 1]], [np.eye(3)[c] for c in [2, 1]]]
    >>> metadata_batches: list[list[DatumMetadata]] = [
    ...     [{"id": 0}, {"id": 1}],
    ...     [{"id": 2}, {"id": 3}],
    ... ]

    >>> from maite.tasks import evaluate_from_predictions
    >>> metric: ic.Metric = Accuracy()
    >>> evaluate_from_predictions(
    ...     metric=metric,
    ...     pred_batches=pred_batches,
    ...     target_batches=target_batches,
    ...     metadata_batches=metadata_batches,
    ... )
    {'accuracy': 0.75}
    """
    # Checks for argument compliance.

    if len(pred_batches) != len(target_batches):
        raise ValueError(
            "Arguments predictions and truth_datum are expected to have the same number of elements (batches)",
        )

    if len(pred_batches) < 1 or len(target_batches) < 1:
        raise ValueError("Predictions and targets must have at least one element.")

    metric.reset()
    for pred_batch, target_batch, metadata_batch in zip(
        pred_batches,
        target_batches,
        metadata_batches,
        strict=True,
    ):
        if len(pred_batch) != len(target_batch):
            raise ValueError(
                "Corresponding prediction and target batches must have the same length.",
            )
        metric.update(pred_batch, target_batch, metadata_batch)
    return metric.compute()


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
    tuple[Sequence[Sequence[SomeTargetType]],
          Sequence[tuple[Sequence[SomeInputType], Sequence[SomeTargetType], Sequence[SomeMetadataType]]]],
        A tuple of the predictions (as a sequence of batches) and a sequence
        of tuples containing the information associated with each batch.
        Note that the second return argument will be empty if
        return_augmented_data is False.

    Raises
    ------
    ValueError
        If neither a dataloader nor a dataset is provided.

    Examples
    --------
    We illustrate with an image classification problem, but the same pattern
    applies to any MAITE task. See the `maite.protocols.image_classification`
    documentation for details on writing compliant components.

    >>> import numpy as np
    >>> import numpy.typing as npt
    >>> from collections.abc import Sequence
    >>> from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata, ModelMetadata
    >>> from maite.protocols import image_classification as ic

    We set up a tiny three-class problem in which the class of an image is the
    index of its brightest channel.

    >>> def make_image(bright_channel: int) -> npt.NDArray:
    ...     img = np.full((3, 4, 4), 0.1)
    ...     img[bright_channel] = 0.9
    ...     return img
    >>> images = [make_image(c) for c in [0, 1, 2, 0]]
    >>> targets = [np.eye(3)[t] for t in [0, 1, 2, 0]]

    A `Model` predicts the brightest-channel class for each image in a batch,
    returning a one-hot score vector per image.

    >>> class ChannelClassifier:
    ...     metadata: ModelMetadata = {"id": "ChannelClassifier"}
    ...
    ...     def __call__(self, batch: Sequence[ArrayLike]) -> Sequence[npt.NDArray]:
    ...         preds = []
    ...         for x in batch:
    ...             cls = int(np.asarray(x).mean(axis=(1, 2)).argmax())
    ...             preds.append(np.eye(3)[cls])
    ...         return preds

    A `Dataset` that returns an (input, target, metadata) 3-tuple per index.
    `predict` ignores targets, so any placeholder target of the correct type
    will do.

    >>> class ImageDataset:
    ...     metadata: DatasetMetadata = {"id": "example_dataset"}
    ...
    ...     def __init__(self, images: list[npt.NDArray], targets: list[npt.NDArray]) -> None:
    ...         self.images = images
    ...         self.targets = targets
    ...
    ...     def __len__(self) -> int:
    ...         return len(self.images)
    ...
    ...     def __getitem__(self, i: int) -> tuple[npt.NDArray, npt.NDArray, DatumMetadata]:
    ...         md: DatumMetadata = {"id": i}
    ...         return self.images[i], self.targets[i], md

    Running `predict` returns one sequence of prediction batches (plus a sequence
    of augmented-data batches, empty here since `return_augmented_data` defaults
    to False).

    >>> from maite.tasks import predict
    >>> model: ic.Model = ChannelClassifier()
    >>> dataset: ic.Dataset = ImageDataset(images, targets)
    >>> preds, _ = predict(model=model, dataset=dataset, batch_size=2)

    We flatten the batched predictions and take the argmax of each score vector
    to recover the predicted class per image.

    >>> [int(np.asarray(p).argmax()) for batch in preds for p in batch]
    [0, 1, 2, 0]
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
