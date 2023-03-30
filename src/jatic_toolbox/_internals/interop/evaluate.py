from abc import abstractmethod
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch as tr
from torch.utils.data import DataLoader
from typing_extensions import Literal, Protocol, Self, runtime_checkable

from jatic_toolbox import protocols

from ..import_utils import is_torch_available
from ..utils import evaluating
from .api import load_dataset, load_metric, load_model

__all__ = ["ImageClassificationEvaluator", "evaluate"]


T_co = TypeVar("T_co", covariant=True)


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
            tr.cuda.set_device(device)
            _device = tr.device("cuda")

    yield _device


@contextmanager
def transfer_to_device(*modules, device):
    # TODO: Maybe change to transfer back to cpu upon exit?
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
    dataset: protocols.Dataset[Dict[str, Any]],
    batch_size: int = 32,
    split: Literal["train", "test"] = "test",
    shuffle: Optional[bool] = None,
    collate_fn: Optional[Callable] = None,
    **kwargs: Any,
) -> protocols.DataLoader:
    """
    Returns a data loader for a JATIC image dataset.

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


class EvaluationTask:
    """
    Base evalulation task functionality.

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
    def _evaluate_on_dataset(self, *args: Any, **kwds: Any) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()


class ImageClassificationEvaluator(EvaluationTask):
    """Evalulator for image classification tasks."""

    def __init__(
        self,
        task: str = "image-classification",
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
        data: protocols.DataLoader[protocols.SupportsImageClassification],
        model: protocols.Classifier[protocols.ArrayLike],
        metric: Mapping[str, protocols.Metric],
        augmentation: Optional[protocols.Augmentation] = None,
        device: Optional[Union[str, int]] = None,
        use_progress_bar: bool = True,
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

        Examples
        --------
        >>> from jatic_toolbox import evaluate
        >>> evaluator = evaluate("image-classification")
        >>> evaluator._evaluate_on_dataset(...)
        """
        from tqdm.auto import tqdm

        if device is None:
            device = self._infer_device()

        with set_device(device) as _device:
            with evaluating(model), transfer_to_device(model, metric, device=_device):
                if use_progress_bar:
                    iterator: protocols.DataLoader[protocols.SupportsImageClassification] = tqdm(data)  # type: ignore
                else:
                    iterator = data

                for batch in iterator:
                    assert isinstance(batch, dict), "Batch is not a dictionary."
                    assert "image" in batch, "Batch does not contain an image."
                    assert "label" in batch, "Batch does not contain a label."

                    if augmentation is not None:
                        batch = augmentation(batch)

                    if TYPE_CHECKING:
                        batch = cast(protocols.SupportsImageClassification, batch)

                    with tr.inference_mode(), transfer_to_device(batch, device=_device):
                        output = model(batch["image"])

                    if isinstance(output, protocols.HasLogits):
                        [
                            v.update(output.logits, batch["label"])
                            for k, v in metric.items()
                        ]
                    else:
                        [
                            v.update(output.probs, batch["label"])
                            for k, v in metric.items()
                        ]

        return {k: v.compute() for k, v in metric.items()}

    def __call__(
        self,
        model: Union[Tuple[str, str], Callable],
        data: Union[Tuple[str, str], protocols.Dataset],
        metric: Union[
            Tuple[str, Union[str, Sequence[str]]], Mapping[str, protocols.Metric]
        ],
        augmentation: Optional[Union[Tuple[str, str], protocols.Augmentation]] = None,
        split: str = "test",
        batch_size: int = 1,
        device: Optional[Union[str, int]] = None,
        collate_fn: Optional[Callable] = None,
        use_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evalulate an image classification model for a given dataset.

        Parameters
        ----------
        model : Union[Tuple[str, str], Callable]
            The model to evaluate. Can be a tuple of provider and model name or a callable.
        data : Union[Tuple[str, str], Iterable]
            The dataset to evaluate on. Can be a tuple of provider and dataset name or an iterable.
        metric : Union[Tuple[str, str], Mapping[str, Metric]]
            The metric to evaluate the model on. Can be a tuple of provider and metric name or a mapping of metric names to metric objects.
        augmentation : Optional[Union[Tuple[str, str], Augmentation]]
            The augmentation to apply to the dataset. Can be a tuple of provider and augmentation name or an augmentation object.
        split : str
            The split of the dataset to evaluate on.
        batch_size : int
            The batch size to use for evaluation.
        device : Optional[Union[str, int]]
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
        >>> from jatic_toolbox import evaluate
        >>> evaluator = evaluate("image-classification")
        >>> evaluator(("huggingface", "resnet18"), ("huggingface", "cifar10"), ("torchmetrics", "Accuracy"))
        {'Accuracy': 0.1}
        """
        if isinstance(model, (list, tuple)):
            provider, name = model
            model = load_model(
                provider=provider, task="image-classification", model_name=name
            )

        if TYPE_CHECKING:
            assert isinstance(model, protocols.Classifier)

        if isinstance(data, (list, tuple)):
            provider, name = data
            data = load_dataset(provider=provider, dataset_name=name, split=split)

        if isinstance(augmentation, (list, tuple)):
            raise NotImplementedError(
                "Augmentations from providers are not supported yet."
            )

        if TYPE_CHECKING:
            if augmentation is not None:
                assert isinstance(augmentation, protocols.Augmentation)

        if isinstance(metric, (list, tuple)):
            num_classes = -1
            if hasattr(data, "num_classes"):
                num_classes = data.num_classes  # type: ignore

            elif hasattr(data, "features"):
                assert isinstance(data.features, dict)
                if "label" in data.features:
                    num_classes = data.features["label"].num_classes

            provider, metric_name = metric
            if isinstance(metric_name, str):
                metric = {
                    metric_name: load_metric(
                        provider=provider,
                        metric_name=metric_name,
                        task="multiclass",
                        num_classes=num_classes,
                    )
                }
            elif isinstance(metric_name, (list, tuple)):
                metric = {}
                for n in metric_name:
                    metric[n] = load_metric(
                        provider=provider,
                        metric_name=n,
                        task="multiclass",
                        num_classes=num_classes,
                    )

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
        )


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
    >>> from jatic_toolbox import evaluate
    >>> evaluator = evaluate("image-classification")
    >>> evaluator(("huggingface", "resnet18"), ("huggingface", "cifar10"), ("torchmetrics", "Accuracy"))
    {'Accuracy': 0.1}
    """
    if task == "image-classification":
        return ImageClassificationEvaluator()

    raise ValueError(f"Task {task} is not supported.")
