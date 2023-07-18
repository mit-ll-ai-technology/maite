from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import TypeAlias

from jatic_toolbox._internals.interop.huggingface.api import HuggingFaceAPI
from jatic_toolbox._internals.interop.torcheval.api import TorchEvalAPI
from jatic_toolbox._internals.interop.torchmetrics.api import TorchMetricsAPI
from jatic_toolbox._internals.interop.torchvision.api import TorchVisionAPI
from jatic_toolbox.protocols import (
    ArrayLike,
    Dataset,
    ImageClassifier,
    Metric,
    ObjectDetector,
)

from .registry import DATASET_REGISTRY, METRIC_REGISTRY, MODEL_REGISTRY

A = TypeVar("A", bound=ArrayLike)
T = TypeVar("T")
SUPPORTED_TASKS: TypeAlias = Literal["image-classification", "object-detection"]

DATASET_PROVIDERS: TypeAlias = Literal["huggingface", "torchvision"]
MODEL_PROVIDERS: TypeAlias = Literal["huggingface", "torchvision"]
METRIC_PROVIDERS: TypeAlias = Literal["torchmetrics", "torcheval"]


def list_datasets(
    *,
    provider: DATASET_PROVIDERS,
    **kwargs: Any,
) -> Iterable[Any]:
    """
    List datasets for a given provider.

    Parameters
    ----------
    provider : str
        Where to search for datasets. Currently supported: "huggingface", "torchvision".
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Iterable[Any]
        An iterable of dataset names.

    Examples
    --------
    >>> from jatic_toolbox import list_datasets
    >>> list_datasets(provider="huggingface")
    """
    if provider == "huggingface":
        api = HuggingFaceAPI()
    elif provider == "torchvision":
        api = TorchVisionAPI()
    else:
        raise NotImplementedError(f"Provider, {provider}, not supported.")

    return api.list_datasets(**kwargs)


def load_dataset(
    *,
    dataset_name: str,
    provider: Optional[DATASET_PROVIDERS] = None,
    task: Optional[Literal["image-classification", "object-detection"]] = None,
    split: Optional[str] = None,
    **kwargs: Any,
) -> Union[Dataset[Any], Dict[str, Dataset[Any]]]:
    """
    Load dataset for a given provider.

    Parameters
    ----------
    dataset_name : str
        Name of dataset.
    provider : str
        Where to search for datasets. Currently supported: "huggingface", "torchvision".
    task : str | None (default: None)
        A string of tasks datasets were designed for, such as: "image-classification", "object-detection".
    split : str | None (default: None)
        A string of split to load, such as: "train", "test", "validation".
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Dataset[Any]
        A dataset object.

    Examples
    --------
    >>> from jatic_toolbox import load_dataset
    >>> load_dataset(provider="huggingface", dataset_name="cifar10", task="image-classification")
    """

    if dataset_name in DATASET_REGISTRY:
        kwargs = {**DATASET_REGISTRY[dataset_name], **kwargs}
        dataset_name = kwargs.pop("dataset_name")
        provider = kwargs.pop("provider")
        task = kwargs.pop("task")
        _split = kwargs.pop("split")
        split = split or _split

    if provider == "huggingface":
        api = HuggingFaceAPI()
    elif provider == "torchvision":
        api = TorchVisionAPI()
    else:
        raise NotImplementedError(f"Provider, {provider}, not supported.")

    return api.load_dataset(dataset_name, task=task, split=split, **kwargs)


def list_models(
    *,
    provider: MODEL_PROVIDERS,
    filter_str: Optional[Union[str, List[str]]] = None,
    model_name: Optional[str] = None,
    task: Optional[Union[str, List[str]]] = None,
    **kwargs: Any,
) -> Iterable[Any]:
    """
    List models for a given provider.

    Parameters
    ----------
    provider : str
        Where to search for models. Currently supported: "huggingface", "torchvision".
    filter_str : str | List[str] | None (default: None)
        A string or list of strings that contain complete or partial names for models.
    model_name : str | None (default: None)
        A string that contain complete or partial names for models.
    task : str | List[str] | None (default: None)
        A string or list of strings of tasks models were designed for, such as: "image-classification", "object-detection".
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Iterable[Any]
        An iterable of model names.

    Examples
    --------
    >>> from jatic_toolbox import list_models
    >>> list_models(provider="huggingface", task="image-classification")
    """

    if provider == "huggingface":
        return HuggingFaceAPI().list_models(
            filter_str=filter_str, task=task, model_name=model_name, **kwargs
        )
    elif provider == "torchvision":
        return TorchVisionAPI().list_models(
            filter_str=filter_str, model_name=model_name, task=task, **kwargs
        )

    raise ValueError(f"Provider, {provider}, not supported.")


@overload
def load_model(
    *,
    provider: MODEL_PROVIDERS,
    task: Literal["image-classification"],
    model_name: str,
    **kwargs: Any,
) -> ImageClassifier:
    ...


@overload
def load_model(
    *,
    provider: MODEL_PROVIDERS,
    task: Literal["object-detection"],
    model_name: str,
    **kwargs: Any,
) -> ObjectDetector:
    ...


def load_model(
    *,
    model_name: str,
    provider: Optional[MODEL_PROVIDERS] = None,
    task: Optional[Literal["image-classification", "object-detection"]] = None,
    **kwargs: Any,
) -> Union[ImageClassifier, ObjectDetector]:
    """
    Return a supported model.

    Parameters
    ----------
    model_name : str
        The `model_name` for the model (e.g., "microsoft/resnet-18").
    provider : str
        The provider of the model (e.g., "huggingface"). Currently supported: "huggingface", "torchvision".
    task : str
        The task for the model (e.g., "image-classification").
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]
        A Model object.

    Examples
    --------
    >>> from jatic_toolbox import load_model
    >>> load_model(provider="huggingface", task="image-classification", model_name="microsoft/resnet-18")
    """
    if model_name in MODEL_REGISTRY:
        kwargs = {**MODEL_REGISTRY[model_name], **kwargs}
        provider = kwargs.pop("provider", provider)
        task = kwargs.pop("task", task)
        model_name = kwargs.pop("model_name")

    if provider == "huggingface":
        return HuggingFaceAPI().load_model(task=task, model_name=model_name, **kwargs)
    elif provider == "torchvision":
        assert task is not None, "task must be specified for torchvision models."
        return TorchVisionAPI().load_model(task=task, model_name=model_name, **kwargs)

    raise ValueError(f"Provider, {provider}, not supported.")


def list_metrics(
    *,
    provider: METRIC_PROVIDERS,
    **kwargs: Any,
) -> Iterable[Any]:
    """
    List metrics for a given provider.

    Parameters
    ----------
    provider : str
        Where to search for metrics.
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Iterable[Any]
        An iterable of metric names.

    Examples
    --------
    >>> from jatic_toolbox import list_metrics
    >>> list_metrics(provider="torchmetrics")
    """

    if provider == "torcheval":
        return TorchEvalAPI().list_metrics()
    elif provider == "torchmetrics":
        return TorchMetricsAPI().list_metrics()

    raise ValueError(f"Provider, {provider}, not supported.")


def load_metric(
    *,
    metric_name: str,
    provider: Optional[METRIC_PROVIDERS] = None,
    **kwargs: Any,
) -> Metric:
    """
    Return a Metric object.

    Parameters
    ----------
    metric_name : str
        The `metric_name` for the metric (e.g., "accuracy").
    provider : str
        The provider of the metric (e.g., "torchmetrics"). Currently supported: "torcheval", "torchmetrics".
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Metric
        A Metric object.

    Examples
    --------
    >>> from jatic_toolbox import load_metric
    >>> load_metric(provider="torchmetrics", metric_name="accuracy")
    """
    if metric_name in METRIC_REGISTRY:
        kwargs = {**METRIC_REGISTRY[metric_name], **kwargs}
        provider = kwargs.pop("provider", provider)
        metric_name = kwargs.pop("metric_name")

    if provider == "torcheval":
        return TorchEvalAPI().load_metric(metric_name=metric_name, **kwargs)
    elif provider == "torchmetrics":
        return TorchMetricsAPI().load_metric(metric_name=metric_name, **kwargs)

    raise ValueError(f"Provider, {provider}, not supported.")
