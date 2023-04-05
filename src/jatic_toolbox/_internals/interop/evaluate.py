from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
)

import torch as tr
from torch.utils.data import DataLoader
from typing_extensions import Literal, Protocol, Self, TypeAlias, runtime_checkable

from jatic_toolbox import protocols as pr

from ..import_utils import is_hydra_zen_available, is_torch_available, is_tqdm_available
from ..utils import evaluating

if is_hydra_zen_available():
    from hydra_zen import instantiate  # type: ignore
    from hydra_zen.typing import Builds  # type: ignore

else:
    from dataclasses import dataclass
    from typing import Generic

    T = TypeVar("T")

    @dataclass
    class Builds(Generic[T]):
        __target__: str

    def instantiate(obj: Builds[Callable[..., T]], *args, **kwargs) -> T:
        raise RuntimeError(
            "hydra_zen is not installed. Please install it using `pip install hydra-zen`."
        )


__all__ = ["ImageClassificationEvaluator", "evaluate"]

ArrayLike = pr.ArrayLike
T_co = TypeVar("T_co", covariant=True)
Model: TypeAlias = Union[
    pr.Classifier[ArrayLike],
    pr.ObjectDetector[ArrayLike],
]

Metric: TypeAlias = Mapping[str, pr.Metric[ArrayLike]]


@runtime_checkable
class HasToDevice(Protocol):
    def to(self, device: Union[str, int]) -> Self:
        ...


@contextmanager
def set_device(device: Optional[Union[str, int]]):
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
def transfer_to_device(*modules, device):
    # TODO: Maybe transfer back to cpu upon exit?
    for m in modules:
        if is_torch_available():
            from torch import Tensor, nn

            if isinstance(m, (nn.Module, Tensor, HasToDevice)):
                m.to(device)

            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (nn.Module, Tensor, HasToDevice)):
                        m[k] = v.to(device)

    yield


def get_dataloader(
    dataset: pr.Dataset[Dict[str, Any]],
    batch_size: int = 32,
    split: Literal["train", "test"] = "test",
    shuffle: Optional[bool] = None,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    **kwargs: Any,
) -> pr.DataLoader[Any]:
    """
    Returns a data loader for a JATIC dataset.

    Parameters
    ----------
    dataset : Dataset[Dict[str, Any]]
        The dataset to load.

    batch_size : int (default: 32)
        The batch size to use.

    device : Optional[Union[str, tr.device]] = None
        The device to transfer data to.

    **kwargs : Any
        Keyword arguments for `torch.utils.data.DataLoader`.

    Returns
    -------
    DataLoader
        A JATIC PyTorch data loader object.
    """
    if shuffle is None:
        shuffle = True if split == "train" else False

    return DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        **kwargs,
    )


class EvaluationTask(ABC):
    """
    Base evalulation task functionality.

    Parameters
    ----------
    task : str
        The task to evaluate on.
    """

    task: Literal["image-classification", "object-detection"]

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
    def _evaluate_on_dataset(self, *args: Any, **kwds: Any) -> Dict[str, Any]:
        raise NotImplementedError()

    def __call__(
        self,
        model: Union[Builds[Callable[..., Model]], Model],
        data: Union[Builds[Callable[..., pr.Dataset[Any]]], pr.Dataset[Any]],
        metric: Union[
            Builds[Callable[..., Mapping[str, pr.Metric[ArrayLike]]]],
            Mapping[str, pr.Metric[ArrayLike]],
        ],
        augmentation: Optional[
            Union[
                Builds[Callable[..., pr.Augmentation[ArrayLike]]],
                pr.Augmentation[ArrayLike],
            ]
        ] = None,
        batch_size: int = 1,
        device: Optional[Union[str, int]] = None,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        use_progress_bar: bool = True,
        input_key: str = "image",
        label_key: str = "label",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evalulate an image classification model for a given dataset.

        Parameters
        ----------
        task : str
            The task to evaluate on.
        model : Classifier[ArrayLike] | ObjectDetector[ArrayLike]
            The model to evaluate.
        data : Dataset
            The dataset to evaluate on.
        metric : Mapping[str, Metric]]
            The metric to evaluate the model on.
        augmentation : Optional[Augmentation]
            The augmentation to apply to the dataset.
        batch_size : int
            The batch size to use for evaluation.
        device : Optional[Union[str, int]]
            The device to use for evaluation. If None, the device is automatically selected.
        collate_fn : Callable | None (default: None)
            The collate function to use for the data loader.
        use_progress_bar : bool (default: True)
            Whether to use a progress bar.
        input_key : str (default: "image")
            The key to use for the input data.
        label_key : str (default: "label")
            The key to use for the labels.
        **kwargs : Any
            Keyword arguments for `torch.utils.data.DataLoader`.

        Returns
        -------
        Dict[str, Any]
            The evaluation results.

        Examples
        --------
        >>> from jatic_toolbox import evaluate, load_model, load_dataset, load_metric
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
        if isinstance(model, Builds):
            model = instantiate(model)

        if isinstance(data, Builds):
            data = instantiate(data)

        if isinstance(augmentation, Builds):
            augmentation = instantiate(augmentation)

        if TYPE_CHECKING:
            if augmentation is not None:
                assert isinstance(augmentation, pr.Augmentation)

        if isinstance(metric, Builds):
            metric = instantiate(metric)

        assert isinstance(metric, Mapping)

        dl = get_dataloader(
            data, batch_size=batch_size, collate_fn=collate_fn, **kwargs
        )

        return self._evaluate_on_dataset(
            data=dl,
            model=model,
            metric=metric,
            augmentation=augmentation,
            device=device,
            use_progress_bar=use_progress_bar,
            input_key=input_key,
            label_key=label_key,
        )


class ImageClassificationEvaluator(EvaluationTask):
    """Evalulator for image classification tasks."""

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
        data: pr.DataLoader[pr.SupportsImageClassification],
        model: pr.Classifier[ArrayLike],
        metric: Mapping[str, pr.Metric[ArrayLike]],
        augmentation: Optional[pr.Augmentation[ArrayLike]] = None,
        device: Optional[Union[str, int]] = None,
        use_progress_bar: bool = True,
        input_key: str = "image",
        label_key: str = "label",
    ) -> Dict[str, Any]:
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
        augmentation : Optional[Augmentation] (default: None)
            The augmentation to use.
        device : Optional[Union[str, int]] (default: None)
            The device to use.
        use_progress_bar : bool (default: True)
            Whether to use a progress bar.

        Returns
        -------
        Dict[str, Any]
            The evaluation results.
        """
        if use_progress_bar and is_tqdm_available():
            from tqdm.auto import tqdm

        if device is None:
            device = self._infer_device()

        with set_device(device) as _device:
            with evaluating(model), transfer_to_device(model, metric, device=_device):
                if use_progress_bar:
                    iterator: pr.DataLoader[pr.SupportsImageClassification] = tqdm(data)  # type: ignore
                else:
                    iterator = data

                for batch in iterator:
                    assert isinstance(batch, dict), "Batch is not a dictionary."
                    assert input_key in batch, f"Batch does not contain an {input_key}."
                    assert label_key in batch, f"Batch does not contain a {label_key}."

                    if augmentation is not None:
                        batch = augmentation(batch)

                    if TYPE_CHECKING:
                        batch = cast(pr.SupportsImageClassification, batch)

                    with tr.inference_mode(), transfer_to_device(batch, device=_device):
                        output = model(batch[input_key])

                    if isinstance(output, pr.HasLogits):
                        [
                            v.update(output.logits, batch[label_key])
                            for k, v in metric.items()
                        ]
                    elif isinstance(output, pr.HasProbs):
                        [
                            v.update(output.probs, batch[label_key])
                            for k, v in metric.items()
                        ]
                    else:
                        raise ValueError(
                            "Model output does not contain `logits` or `probs` attribute."
                        )

        return {k: v.compute() for k, v in metric.items()}


class ObjectDetectionEvaluator(EvaluationTask):
    """Evalulator for object detection tasks."""

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

    def __call__(
        self,
        *args: Any,
        input_key: str = "image",
        label_key: str = "objects",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.

        Parameters
        ----------
        task : str
            The task to evaluate on.
        model : Classifier[ArrayLike] | ObjectDetector[ArrayLike]
            The model to evaluate.
        data : Iterable
            The dataset to evaluate on.
        metric : Mapping[str, Metric]]
            The metric to evaluate the model on.
        augmentation : Optional[Augmentation]
            The augmentation to apply to the dataset.
        batch_size : int
            The batch size to use for evaluation.
        device : Optional[Union[str, int]]
            The device to use for evaluation. If None, the device is automatically selected.
        collate_fn : Callable | None (default: None)
            The collate function to use for the data loader.
        use_progress_bar : bool (default: True)
            Whether to use a progress bar.
        input_key : str (default: "image")
            The key to use for the input data.
        label_key : str (default: "objects")
            The key to use for the labels.
        **kwargs : Any
            Keyword arguments for `torch.utils.data.DataLoader`.

        Returns
        -------
        Dict[str, Any]
            The evaluation results.
        """
        return super().__call__(
            *args, input_key=input_key, label_key=label_key, **kwargs
        )

    def _evaluate_on_dataset(
        self,
        data: pr.DataLoader[pr.SupportsObjectDetection],
        model: pr.ObjectDetector[ArrayLike],
        metric: Mapping[str, pr.Metric[ArrayLike]],
        augmentation: Optional[pr.Augmentation[ArrayLike]] = None,
        device: Optional[Union[str, int]] = None,
        use_progress_bar: bool = True,
        input_key: str = "image",
        label_key: str = "objects",
    ) -> Dict[str, Any]:
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
        augmentation : Optional[Augmentation] (default: None)
            The augmentation to use.
        device : Optional[Union[str, int]] (default: None)
            The device to use.
        use_progress_bar : bool (default: True)
            Whether to use a progress bar.

        Returns
        -------
        Dict[str, Any]
            The evaluation results.

        Examples
        --------
        >>> from jatic_toolbox import evaluate
        >>> evaluator = evaluate("object-detection")
        >>> evaluator._evaluate_on_dataset(...)
        """
        if use_progress_bar and is_tqdm_available():
            from tqdm.auto import tqdm

        if device is None:
            device = self._infer_device()

        with set_device(device) as _device:
            with evaluating(model), transfer_to_device(model, metric, device=_device):
                if use_progress_bar:
                    iterator: pr.DataLoader[pr.SupportsObjectDetection] = tqdm(data)  # type: ignore
                else:
                    iterator = data

                for batch in iterator:
                    assert isinstance(batch, dict), "Batch is not a dictionary."
                    assert input_key in batch, f"Batch does not contain an {input_key}."
                    assert label_key in batch, f"Batch does not contain a {label_key}."

                    if augmentation is not None:
                        batch = augmentation(batch)

                    if TYPE_CHECKING:
                        batch = cast(pr.SupportsObjectDetection, batch)

                    with tr.inference_mode(), transfer_to_device(batch, device=_device):
                        output = model(batch[input_key])

                    if isinstance(output, pr.HasObjectDetections):
                        [v.update(output, batch[label_key]) for k, v in metric.items()]
                    else:
                        raise ValueError(
                            "Model output does not support the `jatic_toolbox.protocols.HasObjectDetection` protocol."
                        )

        return {k: v.compute() for k, v in metric.items()}


def evaluate(task: str) -> EvaluationTask:
    """
    Provide an evaluator for a given task and provider.

    Parameters
    ----------
    task : str
        The task to evaluate on.

    Returns
    -------
    EvaluationTask
        The evaluation task.

    Examples
    --------
    An example using `image-classification` task:

    >>> from jatic_toolbox import evaluate, load_model, load_dataset, load_metric
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

    An example using `object-detection` task:

    >>> from jatic_toolbox._internals.interop.utils import collate_as_lists

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
