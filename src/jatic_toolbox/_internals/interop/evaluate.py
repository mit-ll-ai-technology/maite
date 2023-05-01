from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch as tr
from torch.utils.data import DataLoader
from typing_extensions import Literal, Protocol, Self, TypeAlias, runtime_checkable

from jatic_toolbox import protocols as pr

from ..import_utils import (
    is_hydra_zen_available,
    is_pil_available,
    is_torch_available,
    is_tqdm_available,
)
from ..utils import evaluating

if is_hydra_zen_available():
    from hydra_zen import instantiate  # type: ignore
    from hydra_zen.typing import Builds  # type: ignore

else:  # pragma: no cover
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
    """
    Transfers a list of modules to a device.

    Parameters
    ----------
    *modules : HasToDevice | Dict[str, HasToDevice] | List[HasToDevice]
        A list of modules to transfer to a device.

    device : Union[str, int]
        The device to transfer the modules to.

    Yields
    ------
    Union[HasToDevice, Dict[str, HasToDevice], List[HasToDevice]]
        The modules transferred to the device.

    Examples
    --------
    >>> import torch as tr
    >>> from jatic_toolbox._internals.interop.utils import transfer_to_device
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
            if len(modules) == 1:
                yield unflattened_modules[0]
            else:
                yield unflattened_modules
        finally:
            # this should only matter for Modules since tensors make a copy
            [m.to("cpu") for m in flatten_modules if isinstance(m, nn.Module)]

    else:
        yield


def collate_and_pad(
    preprocessor: Optional[Callable[[Any], Any]] = None, processor_key: str = "image"
) -> Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]:
    """
    Collates and pads a batch of examples.

    Parameters
    ----------
    preprocessor : Optional[Callable[[Any], Any]], optional
        A callable that takes a batch of examples and returns a batch of examples, by default None

    processor_key : str, optional
        The key of the batch that will be passed to the preprocessor, by default "image"

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

            if is_pil_available():
                from PIL.Image import Image

                if isinstance(batch_item[0], Image):
                    # handle PIL Image: taken from `torchvision.transforms.functional.to_tensor`
                    def to_array(pic: Image):
                        mode_to_nptype = {
                            "I": np.int32,
                            "I;16": np.int16,
                            "F": np.float32,
                        }
                        img = tr.from_numpy(
                            np.array(
                                pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True
                            )
                        )

                        if pic.mode == "1":
                            img = 255 * img

                        channels = len(pic.getbands())
                        img = img.view(pic.size[1], pic.size[0], channels)
                        # put it from HWC to CHW format
                        img = img.permute((2, 0, 1))

                        if isinstance(img, tr.ByteTensor):
                            return img.to(dtype=tr.get_default_dtype()).div(255)
                        else:
                            return img

                    batch_item = [to_array(x) for x in batch_item]

            if isinstance(batch_item[0], tr.Tensor):
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
    dataset: pr.Dataset[Dict[str, Any]],
    batch_size: int = 32,
    split: Literal["train", "test"] = "test",
    shuffle: Optional[bool] = None,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    preprocessor: Optional[Callable[[Any], Any]] = None,
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
        collate_fn=collate_and_pad(preprocessor) if collate_fn is None else collate_fn,
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
        preprocessor: Optional[Callable[[Any], Any]] = None,
        post_processor: Optional[Callable[[Any], Any]] = None,
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

        if isinstance(preprocessor, Builds):
            preprocessor = instantiate(preprocessor)
        elif preprocessor is None:
            if hasattr(model, "preprocessor"):
                preprocessor = model.preprocessor

        if isinstance(post_processor, Builds):
            post_processor = instantiate(post_processor)
        elif post_processor is None:
            if hasattr(model, "post_processor"):
                post_processor = model.post_processor

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
            data,
            batch_size=batch_size,
            collate_fn=collate_fn,
            preprocessor=preprocessor,
            **kwargs,
        )

        return self._evaluate_on_dataset(
            data=dl,
            model=model,
            metric=metric,
            augmentation=augmentation,
            post_processor=post_processor,
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
        post_processor: Optional[Callable[[Any], pr.HasProbs]] = None,
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
        else:
            tqdm = lambda x: x  # noqa: E731

        if device is None:
            device = self._infer_device()

        # Reset metrics
        [v.reset() for v in metric.values()]

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

                    with tr.inference_mode(), transfer_to_device(
                        batch, device=_device
                    ) as batch_device:
                        output = model(batch_device)
                        if post_processor is not None:
                            output = post_processor(output)

                    if isinstance(output, pr.HasLogits):
                        [
                            v.update(output.logits, batch_device[label_key])
                            for k, v in metric.items()
                        ]
                    elif isinstance(output, pr.HasProbs):
                        [
                            v.update(output.probs, batch_device[label_key])
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
        post_processor: Optional[Callable[[Any], pr.HasObjectDetections]] = None,
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
        else:
            tqdm = lambda x: x  # noqa: E731

        if device is None:
            device = self._infer_device()

        # Reset metrics
        [v.reset() for v in metric.values()]

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

                    with tr.inference_mode(), transfer_to_device(
                        batch, device=_device
                    ) as batch_device:
                        output = model(batch_device)
                        if post_processor is not None:
                            output = post_processor(output)

                    if isinstance(output, pr.HasObjectDetections):
                        [
                            v.update(output, batch_device[label_key])
                            for k, v in metric.items()
                        ]
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
