# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import torch as tr
from torch.utils.data import DataLoader
from typing_extensions import Self

import maite
import maite.protocols as pr
from maite._internals.interop.api import (
    DATASET_PROVIDERS,
    METRIC_PROVIDERS,
    MODEL_PROVIDERS,
)
from maite.errors import InvalidArgument

from ..import_utils import is_torch_available, is_tqdm_available
from ..utils import evaluating
from .utils import is_pil_image

ArrayLike = pr.ArrayLike
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class HasToDevice(Protocol):
    def to(self, device: str | int) -> Self:
        ...


@contextmanager
def set_device(device: Optional[str | int]):
    _device = None
    if is_torch_available():
        import torch as tr

        _device = tr.device("cpu")
        if device is not None and tr.cuda.is_available():
            if isinstance(device, str):
                device = f"cuda:{device}"

            tr.cuda.set_device(device)
            _device = tr.device(device)

    yield _device


@contextmanager
def transfer_to_device(*modules: T, device) -> Iterator[tuple[T]]:
    """
    Transfers a list of modules to a device.

    Parameters
    ----------
    *modules : HasToDevice | Dict[str, HasToDevice] | List[HasToDevice]
        A list of modules to transfer to a device.

    device : str | int
        The device to transfer the modules to.

    Yields
    ------
    HasToDevice | Dict[str, HasToDevice] | List[HasToDevice]
        The modules transferred to the device.

    Examples
    --------
    >>> import torch as tr
    >>> from maite._internals.interop.utils import transfer_to_device
    >>> model_1 = tr.Linear(1, 1)
    >>> model_2 = tr.Conv1d(1, 1, 1)
    >>> tensor = [dict(val=tr.rand(1, 1))]
    >>> with transfer_to_device(model_1, model_2, tensor, device="cuda:0") as (model_1, model_2, tensor):
    ...     print(model_1.weight.device)
    ...     print(model_2.weight.device)
    ...     print(tensor[0]["val"].device)
    cuda:0
    cuda:0
    cuda:0
    >>> print(model_1.weight.device)
    cpu
    >>> print(model_2.weight.device)
    cpu
    >>> print(tensor[0]["val"].device)
    cpu
    """
    if is_torch_available():
        from torch import Tensor, nn
        from torch.utils._pytree import tree_flatten, tree_unflatten

        flatten_modules, tree = tree_flatten(modules)
        device_modules = [
            m.to(device) if isinstance(m, (nn.Module, Tensor, HasToDevice)) else m
            for m in flatten_modules
        ]

        try:
            unflattened_modules = tree_unflatten(device_modules, tree)
            yield unflattened_modules
        finally:
            # this should only matter for Modules since tensors make a copy
            [m.to("cpu") for m in flatten_modules if isinstance(m, nn.Module)]

    else:
        yield modules


def collate_and_pad(
    preprocessor: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]
    | None = None,
) -> Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]:
    """
    Collates and pads a batch of examples.

    Parameters
    ----------
    preprocessor : Optional[Preprocessor], optional
        A callable that takes a batch of examples and returns a batch of examples, by default None
    """

    def collator(batch: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """
        Collates a batch of examples.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            A list of dictionaries, where each dictionary represents an example.

        Returns
        -------
        collated_batch : Dict[str, Any]
            A dictionary where each key corresponds to a feature or label and the value is a batch of data for that feature or label.
        """

        if len(batch) == 0 or batch is None:
            return dict()

        assert isinstance(batch[0], dict), "Batch must be a list of dictionaries."

        keys = set(batch[0].keys())
        for example in batch:
            assert (
                set(example.keys()) == keys
            ), "All examples in the batch must have the same keys."

        if preprocessor is not None:
            # process data in the batch, e.g, PIL to numpy
            batch = preprocessor(batch)

        collated_batch = {}
        for key in batch[0].keys():
            batch_item = [example[key] for example in batch]

            if is_pil_image(batch_item[0]):
                collated_batch[key] = batch_item

            elif isinstance(batch_item[0], tr.Tensor):
                shape_first = batch_item[0].shape
                if all([shape_first == item.shape for item in batch_item]):
                    collated_batch[key] = tr.stack(batch_item)
                else:
                    collated_batch[key] = batch_item

            elif isinstance(batch_item[0], (int, float)):
                collated_batch[key] = tr.as_tensor(batch_item)

            elif isinstance(batch_item[0], str):
                collated_batch[key] = batch_item

            elif isinstance(batch_item[0], (tuple, list)):
                if len(batch_item[0]) > 0:
                    if isinstance(batch_item[0][0], tr.Tensor):
                        collated_batch[key] = tr.stack(batch_item)

                    elif isinstance(batch_item[0][0], (int, float)):
                        collated_batch[key] = tr.as_tensor(batch_item)

                    else:
                        collated_batch[key] = batch_item
                else:
                    collated_batch[key] = batch_item

            elif isinstance(batch_item[0], dict):
                collated_batch[key] = []
                for i, d in enumerate(batch_item):
                    if TYPE_CHECKING:
                        assert isinstance(d, dict)

                    collated_batch[key].append({})
                    for subkey in d.keys():
                        if isinstance(
                            batch_item[0][subkey], (tr.Tensor, list, tuple, int, float)
                        ):
                            collated_batch[key][i].update(
                                {subkey: tr.as_tensor(d[subkey])}
                            )
                        else:
                            collated_batch[key][i].update({subkey: d[subkey]})

            elif batch_item[0] is None:
                collated_batch[key] = None

            else:
                try:
                    collated_batch[key] = tr.as_tensor(batch_item)
                except TypeError:
                    collated_batch[key] = batch_item

        return collated_batch

    return collator


def get_dataloader(
    dataset: pr.VisionDataset | pr.ObjectDetectionDataset,
    batch_size: int = 32,
    split: Literal["train", "test"] = "test",
    shuffle: bool | None = None,
    collate_fn: Callable[[Any], Any] | None = None,
    preprocessor: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]
    | None = None,
    **kwargs: Any,
) -> pr.DataLoader:
    """
    Returns a data loader for a MAITE dataset.

    Parameters
    ----------
    dataset : VisionDataset | ObjectDetectionDataset
        The dataset to load.
    batch_size : int (default: 32)
        The batch size to use.
    split : str (default: "test")
        The split to use.
    shuffle : bool | None (default: None)
        Whether to shuffle the data. If None, the data is shuffled for the train split and not shuffled for the test split.
    collate_fn : Callable[[Any], Any] | None (default: None)
        The collate function to use for the data loader.
    preprocessor : Preprocessor | None (default: None)
        A callable that takes a batch of examples and returns a batch of examples, by default None
    **kwargs : Any
        Keyword arguments for `torch.utils.data.DataLoader`.

    Returns
    -------
    DataLoader
        A data loader object.
    """
    if shuffle is None:
        shuffle = True if split == "train" else False

    return DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=collate_and_pad(preprocessor) if collate_fn is None else collate_fn,
        shuffle=shuffle,
        **kwargs,
    )


class EvaluationTask(ABC):
    """
    Base evaluation task functionality.

    Parameters
    ----------
    task : str
        The task to evaluate on.
    """

    task: str

    def _infer_device(self):
        if is_torch_available():
            import torch

            if torch.cuda.is_available():
                device = 0  # first GPU
            else:
                device = -1  # CPU

            return device

        raise RuntimeError("No supported backend found.")

    @abstractmethod
    def _evaluate_on_dataset(self, *args: Any, **kwds: Any) -> dict[str, Any]:
        raise NotImplementedError()

    def __call__(
        self,
        model: str | pr.ImageClassifier | pr.ObjectDetector,
        data: str | pr.VisionDataset | pr.ObjectDetectionDataset,
        metric: str | Mapping[str, pr.Metric[[Any, Any], Any]] = "default_metric",
        augmentation: None
        | pr.Augmentation[
            [pr.SupportsImageClassification | pr.SupportsObjectDetection],
            pr.SupportsImageClassification | pr.SupportsObjectDetection,
        ] = None,
        batch_size: int = 1,
        device: None | str | int = None,
        collate_fn: None | Callable = None,
        use_progress_bar: bool = True,
        dataset_kwargs: Mapping[str, Any] = {},
        model_kwargs: Mapping[str, Any] = {},
        metric_kwargs: Mapping[str, Any] = {},
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Evaluate an image classification model for a given dataset.

        Parameters
        ----------
        task : str
            The task to evaluate on.
        model : str | ImageClassifier | ObjectDetector
            The model to evaluate.
        data : str | VisionDataset | ObjectDetectionDataset
            The dataset to evaluate on.
        metric : str | Mapping[str, pr.Metric]
            The metric to evaluate the model on.
        augmentation : Augmentation | None (default: None)
            The augmentation to apply to the dataset.
        batch_size : int
            The batch size to use for evaluation.
        device : str | int | None (default: None)
            The device to use for evaluation. If None, the device is automatically selected.
        collate_fn : Callable | None (default: None)
            The collate function to use for the data loader.
        use_progress_bar : bool (default: True)
            Whether to use a progress bar.
        **kwargs : Any
            Keyword arguments for `torch.utils.data.DataLoader`.

        Returns
        -------
        Dict[str, Any]
            The evaluation results.

        Examples
        --------
        >>> from maite import evaluate, load_model, load_dataset, load_metric
        >>> from torchvision.transforms.functional import to_tensor

        Load a model and dataset and evaluate it on the CIFAR10 test set.

        >>> model = load_model(provider="huggingface", model_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", task="image-classification")
        >>> data = load_dataset(provider="torchvision", dataset_name="CIFAR10", task="image-classification", split="test", root="~/.cache/torchvision/datasets", download=True)
        >>> assert not isinstance(data, dict)
        Load the accuracy metric and evaluate the model.

        >>> acc_metric = load_metric(provider="torchmetrics", metric_name="Accuracy", task="multiclass", num_classes=10)

        Evaluate the model on the given dataset.

        >>> evaluator = evaluate("image-classification")
        >>> evaluator(model, data, metric=dict(accuracy=acc_metric), batch_size=32, device=0)
        {'accuracy': tensor(0.9788, device='cuda:0')}
        """

        if isinstance(data, str):
            if data in maite.DATASET_REGISTRY:
                dataset_kwargs = {
                    **maite.DATASET_REGISTRY[data],
                    **dataset_kwargs,
                }
                data = maite.load_dataset(**dataset_kwargs)
                assert isinstance(data, pr.Dataset)

            elif "::" in data:
                provider, dataset_name = data.split("::", 1)
                if TYPE_CHECKING:
                    provider = cast(DATASET_PROVIDERS, provider)

                data = maite.load_dataset(
                    provider=provider, dataset_name=dataset_name, **dataset_kwargs
                )
                assert isinstance(data, pr.Dataset)
            else:
                raise ValueError(f"Unknown dataset: {data}")

            if TYPE_CHECKING:
                assert not isinstance(data, str)

        if isinstance(model, str):
            if model in maite.MODEL_REGISTRY:
                model_kwargs = {**maite.MODEL_REGISTRY[model], **model_kwargs}
                model = maite.load_model(**model_kwargs)
                assert isinstance(model, (pr.ImageClassifier, pr.ObjectDetector))

            elif "::" in model:
                provider, model_name = model.split("::", 1)

                if TYPE_CHECKING:
                    provider = cast(MODEL_PROVIDERS, provider)

                model = maite.load_model(
                    provider=provider, model_name=model_name, **model_kwargs
                )

            else:
                raise ValueError(f"Unknown model: {model}")

            if TYPE_CHECKING:
                assert not isinstance(model, str)

        if isinstance(metric, str):
            metric_str = metric

            if metric in maite.METRIC_REGISTRY:
                metric_kwargs = {
                    **maite.METRIC_REGISTRY[metric],
                    **metric_kwargs,
                }
                metric_out = maite.load_metric(**metric_kwargs)

            elif metric == "default_metric":
                if self.task == "image-classification":
                    label_list = model.get_labels()
                    num_labels = len(label_list)

                    if num_labels == 0:
                        raise InvalidArgument(
                            "Length of label list from model's get_labels() function was 0."
                        )

                    default_acc_dict = maite.METRIC_REGISTRY["multiclass_accuracy"]
                    default_acc_dict["num_classes"] = num_labels
                    default_acc = maite.load_metric(**default_acc_dict)
                    metric_out = {"default_multiclass_accuracy": default_acc}

                elif self.task == "object-detection":
                    default_map_dict = maite.METRIC_REGISTRY["mean_average_precision"]
                    default_map = maite.load_metric(**default_map_dict)
                    metric_out = {"default_map": default_map}

                else:
                    raise InvalidArgument(
                        "Task must either be 'image-classification' or 'object-detection'."
                    )

            elif "::" in metric:
                provider, metric_name = metric.split("::", 1)

                if TYPE_CHECKING:
                    provider = cast(METRIC_PROVIDERS, provider)

                metric_out = maite.load_metric(
                    provider=provider, metric_name=metric_name, **metric_kwargs
                )

            else:
                raise ValueError(f"Unknown metric: {metric}")

            if TYPE_CHECKING:
                assert not isinstance(metric, str)

            if not isinstance(metric_out, dict):
                metric = {metric_str: metric_out}
            else:
                metric = metric_out

        assert isinstance(metric, Mapping)
        dl = get_dataloader(
            data,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **kwargs,
        )

        return self._evaluate_on_dataset(
            data=dl,
            model=model,
            metric=metric,
            augmentation=augmentation,
            device=device,
            use_progress_bar=use_progress_bar,
        )


class ImageClassificationEvaluator(EvaluationTask):
    """Evaluator for image classification tasks."""

    def __init__(
        self,
        task: Literal["image-classification"] = "image-classification",
    ) -> None:
        """
        Initialize the evaluator.

        Parameters
        ----------
        task : str (default: "image-classification")
            The task to evaluate on.
        """
        super().__init__()
        self.task = task

    def _evaluate_on_dataset(
        self,
        data: pr.VisionDataLoader,
        model: pr.ImageClassifier[pr.SupportsArray],
        metric: Mapping[str, pr.Metric[[Any, Any], Any]],
        augmentation: pr.Augmentation[
            [pr.SupportsImageClassification], pr.SupportsImageClassification
        ]
        | None = None,
        device: str | int | None = None,
        use_progress_bar: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate a model on a dataset.

        Parameters
        ----------
        data : DataLoader[SupportsImageClassification]
            The data to evaluate on.
        model : Classifier[ArrayLike]
            The model to evaluate.
        metric : Mapping[str, Metric]
            The metric to use.
        augmentation : Augmentation | None (default: None)
            The augmentation to use.
        device : str | int | None (default: None)
            The device to use.
        use_progress_bar : bool (default: True)
            Whether to use a progress bar.

        Returns
        -------
        Dict[str, Any]
            The evaluation results.
        """
        if device is None:
            device = self._infer_device()

        # Reset metrics
        [v.reset() for v in metric.values()]

        def get_iterator(data) -> Iterator[pr.SupportsImageClassification]:
            if use_progress_bar and is_tqdm_available():
                from tqdm.auto import tqdm

                iterator = tqdm(data)

                if TYPE_CHECKING:
                    iterator = cast(Iterator[pr.SupportsImageClassification], iterator)

                return iterator
            else:
                return data

        with set_device(device) as _device:
            with evaluating(model), transfer_to_device(model, metric, device=_device):
                for batch in get_iterator(data):
                    if augmentation is not None:
                        batch = augmentation(batch)

                    with tr.inference_mode(), transfer_to_device(
                        batch, device=_device
                    ) as (batch_device,):
                        assert pr.is_typed_dict(batch_device, pr.HasDataImage)
                        output = model(batch_device["image"])

                    if isinstance(output, pr.HasLogits):
                        assert pr.is_typed_dict(batch_device, pr.HasDataLabel)

                        [
                            v.update(output.logits, batch_device["label"])
                            for k, v in metric.items()
                        ]
                    elif isinstance(output, pr.HasProbs):
                        assert pr.is_typed_dict(batch_device, pr.HasDataLabel)

                        [
                            v.update(output.probs, batch_device["label"])
                            for k, v in metric.items()
                        ]
                    else:
                        raise ValueError(
                            "Model output does not contain `logits` or `probs` attribute."
                        )

        computed_metrics = {k: v.compute() for k, v in metric.items()}
        return computed_metrics


class ObjectDetectionEvaluator(EvaluationTask):
    """Evaluator for object detection tasks."""

    def __init__(
        self,
        task: Literal["object-detection"] = "object-detection",
    ) -> None:
        """
        Initialize the evaluator.

        Parameters
        ----------
        task : str (default: "object-detection")
            The task to evaluate on.
        """
        super().__init__()
        self.task = task

    def _evaluate_on_dataset(
        self,
        data: pr.DataLoader[pr.SupportsObjectDetection],
        model: pr.ObjectDetector[pr.SupportsArray],
        metric: Mapping[str, pr.Metric[[Any, Any], Any]],
        augmentation: pr.Augmentation[
            [pr.SupportsObjectDetection], pr.SupportsObjectDetection
        ]
        | None = None,
        device: str | int | None = None,
        use_progress_bar: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate a model on a dataset.

        Parameters
        ----------
        data : DataLoader[SupportsObjectDetection]
            The data to evaluate on.
        model : ObjectDetector[ArrayLike]
            The model to evaluate.
        metric : Mapping[str, Metric]
            The metric to use.
        augmentation : Augmentation | None (default: None)
            The augmentation to use.
        device : str | int | None (default: None)
            The device to use.
        use_progress_bar : bool (default: True)
            Whether to use a progress bar.

        Returns
        -------
        Dict[str, Any]
            The evaluation results.

        Examples
        --------
        >>> from maite import evaluate
        >>> evaluator = evaluate("object-detection")
        >>> evaluator._evaluate_on_dataset(...)
        """
        if device is None:
            device = self._infer_device()

        # Reset metrics
        [v.reset() for v in metric.values()]

        def get_iterator(data) -> Iterator[pr.SupportsObjectDetection]:
            if use_progress_bar and is_tqdm_available():
                from tqdm.auto import tqdm

                iterator = tqdm(data)

                if TYPE_CHECKING:
                    iterator = cast(Iterator[pr.SupportsObjectDetection], iterator)

                return iterator
            else:
                return data

        with set_device(device) as _device:
            with evaluating(model), transfer_to_device(model, metric, device=_device):
                for batch in get_iterator(data):
                    if augmentation is not None:
                        batch = augmentation(batch)

                    with tr.inference_mode(), transfer_to_device(
                        batch, device=_device
                    ) as (batch_device,):
                        assert pr.is_typed_dict(batch_device, pr.HasDataImage)
                        output = model(batch_device["image"])

                    if isinstance(output, pr.HasDetectionPredictions):
                        assert pr.is_typed_dict(batch_device, pr.HasDataObjects)

                        [
                            v.update(output, batch_device["objects"])
                            for k, v in metric.items()
                        ]
                    else:
                        raise ValueError(
                            "Model output does not support the `maite.protocols.HasObjectDetection` protocol."
                        )

        computed_metrics = {k: v.compute() for k, v in metric.items()}
        return computed_metrics


@overload
def evaluate(task: Literal["image-classification"]) -> ImageClassificationEvaluator:
    ...


@overload
def evaluate(task: Literal["object-detection"]) -> ObjectDetectionEvaluator:
    ...


def evaluate(
    task: Literal["image-classification", "object-detection"]
) -> ImageClassificationEvaluator | ObjectDetectionEvaluator:
    """
    Provide an evaluator for a given task and provider.

    Parameters
    ----------
    task : str
        The task to evaluate on.

    Returns
    -------
    ImageClassificationEvaluator | ObjectDetectionEvaluator
        The evaluation task.

    Examples
    --------
    An example using `image-classification` task:

    >>> from maite import evaluate, load_model, load_dataset, load_metric

    Load a model and dataset and evaluate it on the CIFAR10 test set.

    >>> model = load_model(provider="huggingface", model_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", task="image-classification")
    >>> data = load_dataset(provider="torchvision", dataset_name="CIFAR10", task="image-classification", split="test", root="~/.cache/torchvision/datasets", download=True)
    >>> assert not isinstance(data, dict)

    Load the accuracy metric and evaluate the model.

    >>> acc_metric = load_metric(provider="torchmetrics", metric_name="Accuracy", task="multiclass", num_classes=10)

    Evaluate the model on the given dataset.

    >>> evaluator = evaluate("image-classification")
    >>> evaluator(model, data, metric=dict(accuracy=acc_metric), batch_size=32, device=0)
    {'accuracy': tensor(0.9788, device='cuda:0')}

    An example using `object-detection` task:

    >>> from maite._internals.interop.utils import collate_as_lists

    Load a model and dataset and evaluate it on the Fashionpedia validation set.

    >>> model = load_model(provider="torchvision", model_name="fasterrcnn_resnet50_fpn", task="object-detection")
    >>> data = load_dataset(provider="huggingface", dataset_name="detection-datasets/fashionpedia", task="object-detection", split="val")
    >>> assert not isinstance(data, dict)

    Load the mean average precision metric and evaluate the model.

    >>> metric = load_metric(provider="torchmetrics", metric_name="MeanAveragePrecision")

    Now we can evaluate the model on the given dataset.

    >>> evaluator = evaluate("object-detection")
    >>> evaluator(model, data, metric=dict(mAP=metric), batch_size=4, device=0, collate_fn=collate_as_lists)
    """
    if task == "image-classification":
        return ImageClassificationEvaluator()
    elif task == "object-detection":
        return ObjectDetectionEvaluator()

    raise ValueError(f"Task {task} is not supported.")
