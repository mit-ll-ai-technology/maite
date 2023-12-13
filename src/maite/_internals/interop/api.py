# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Iterable, List, Literal, Optional, TypeVar, overload

from typing_extensions import TypeAlias

from maite._internals.interop.huggingface.api import HuggingFaceAPI
from maite._internals.interop.torcheval.api import TorchEvalAPI
from maite._internals.interop.torchmetrics.api import TorchMetricsAPI
from maite._internals.interop.torchvision.api import TorchVisionAPI
from maite.errors import InvalidArgument
from maite.protocols import (
    ArrayLike,
    Dataset,
    ImageClassifier,
    Metric,
    ObjectDetector,
    SupportsImageClassification,
    SupportsObjectDetection,
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
    provider: DATASET_PROVIDERS | None = None,
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
    >>> from maite import list_datasets
    >>> list_datasets(provider="huggingface")
    """
    if provider is None:
        return list(DATASET_REGISTRY.keys())

    if provider == "huggingface":
        api = HuggingFaceAPI()
    elif provider == "torchvision":
        api = TorchVisionAPI()
    else:
        raise NotImplementedError(f"Provider, {provider}, not supported.")

    return api.list_datasets(**kwargs)


@overload
def load_dataset(
    *,
    dataset_name: str,
    provider: DATASET_PROVIDERS | None = None,
    task: Literal["image-classification"],
    split: str | None = None,
    **kwargs: Any,
) -> Dataset[SupportsImageClassification]:
    ...


@overload
def load_dataset(
    *,
    dataset_name: str,
    provider: DATASET_PROVIDERS | None = None,
    task: Literal["object-detection"],
    split: str | None = None,
    **kwargs: Any,
) -> Dataset[SupportsObjectDetection]:
    ...


@overload
def load_dataset(
    *,
    dataset_name: str,
    provider: DATASET_PROVIDERS | None = None,
    task: None = None,
    split: str | None = None,
    **kwargs: Any,
) -> Dataset[SupportsImageClassification] | Dataset[SupportsObjectDetection]:
    ...


def load_dataset(
    *,
    dataset_name: str,
    provider: DATASET_PROVIDERS | None = None,
    task: Literal["image-classification", "object-detection"] | None = None,
    split: str | None = None,
    **kwargs: Any,
) -> Dataset[SupportsImageClassification] | Dataset[SupportsObjectDetection]:
    """
    Load dataset for a given provider.

    Parameters
    ----------
    dataset_name : str
        Name of dataset.
        If the dataset is not in the registry, it will be passed to the provider interface.
    provider : str | None (default: None)
        Where to search for datasets. Currently supported: "huggingface", "torchvision".
        If None, the provider will be inferred from the registered dataset.
    task : str | None (default: None)
        A string of tasks datasets were designed for, such as: "image-classification", "object-detection".
        If None, the task will be inferred from the registered dataset.
    split : str | None (default: None)
        A string of split to load, such as: "train", "test", "validation".
        If None, the split will be inferred from the registered dataset.
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Dataset[SupportsImageClassification | SupportsObjectDetection]
        A dataset object that supports the given task.

    Examples
    --------
    >>> from maite import load_dataset
    >>> load_dataset(provider="huggingface", dataset_name="cifar10", task="image-classification")
    """

    if dataset_name in DATASET_REGISTRY:
        kwargs = {**DATASET_REGISTRY[dataset_name], **kwargs}
        dataset_name = kwargs.pop("dataset_name")
        provider = kwargs.pop("provider")
        task = kwargs.pop("task")
        _split = kwargs.pop("split")
        split = split or _split

    if task is None:
        raise InvalidArgument(
            f"Task must be specified for loading datasets. Got task={task}."
        )

    if provider == "huggingface":
        api = HuggingFaceAPI()
    elif provider == "torchvision":
        api = TorchVisionAPI()
    else:
        raise InvalidArgument(f"Provider, {provider}, not supported.")

    return api.load_dataset(dataset_name, task=task, split=split, **kwargs)


def list_models(
    *,
    provider: MODEL_PROVIDERS | None = None,
    filter_str: str | List[str] | None = None,
    model_name: str | None = None,
    task: str | List[str] | None = None,
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
    >>> from maite import list_models
    >>> list_models(provider="huggingface", task="image-classification")
    """
    if provider is None:
        return list(MODEL_REGISTRY.keys())

    if provider == "huggingface":
        return HuggingFaceAPI().list_models(
            filter_str=filter_str, task=task, model_name=model_name, **kwargs
        )
    elif provider == "torchvision":
        return TorchVisionAPI().list_models(
            filter_str=filter_str, model_name=model_name, task=task, **kwargs
        )
    else:
        raise InvalidArgument(f"Provider, {provider}, not supported.")


@overload
def load_model(
    *,
    model_name: str,
    provider: MODEL_PROVIDERS | None = None,
    task: Literal["image-classification"],
    **kwargs: Any,
) -> ImageClassifier:
    ...


@overload
def load_model(
    *,
    model_name: str,
    provider: MODEL_PROVIDERS | None = None,
    task: Literal["object-detection"],
    **kwargs: Any,
) -> ObjectDetector:
    ...


@overload
def load_model(
    *,
    model_name: str,
    provider: MODEL_PROVIDERS | None = None,
    task: None = None,
    **kwargs: Any,
) -> ImageClassifier | ObjectDetector:
    ...


def load_model(
    *,
    model_name: str,
    provider: MODEL_PROVIDERS | None = None,
    task: Literal["image-classification", "object-detection"] | None = None,
    **kwargs: Any,
) -> ImageClassifier | ObjectDetector:
    """
    Return a supported model.

    Parameters
    ----------
    model_name : str
        The `model_name` for the model (e.g., "microsoft/resnet-18").
        If the model is not in the registry, it will be passed to the provider interface.
    provider : str | None
        The provider of the model (e.g., "huggingface"). Currently supported: "huggingface", "torchvision".
        If None, the provider will be inferred from the registered model.
    task : str | None
        The task for the model (e.g., "image-classification").
        If None, the task will be inferred from the registered model.
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    ImageClassifier | ObjectDetector
        A Model object that supports the given task.

    Examples
    --------
    >>> from maite import load_model
    >>> load_model(provider="huggingface", task="image-classification", model_name="microsoft/resnet-18")
    """
    if model_name in MODEL_REGISTRY:
        kwargs = {**MODEL_REGISTRY[model_name], **kwargs}
        provider = kwargs.pop("provider", provider)
        task = kwargs.pop("task", task)
        model_name = kwargs.pop("model_name")

    if task is None:
        raise InvalidArgument(
            f"Task must be specified for loading models. Got task={task}."
        )

    if provider == "huggingface":
        api = HuggingFaceAPI()
    elif provider == "torchvision":
        assert task is not None, "task must be specified for torchvision models."
        api = TorchVisionAPI()
    else:
        raise InvalidArgument(f"Provider, {provider}, not supported.")

    return api.load_model(task=task, model_name=model_name, **kwargs)


def list_metrics(
    *,
    provider: METRIC_PROVIDERS | None,
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
    >>> from maite import list_metrics
    >>> list_metrics(provider="torchmetrics")
    """
    if provider is None:
        return list(METRIC_REGISTRY.keys())

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
) -> Metric[[Any, Any], Any]:
    """
    Return a Metric object.

    Parameters
    ----------
    metric_name : str
        The `metric_name` for the metric (e.g., "accuracy").
        If the metric is not in the registry, it will be passed to the provider interface.
    provider : str | None
        The provider of the metric (e.g., "torchmetrics"). Currently supported: "torcheval", "torchmetrics".
        If None, the provider will be inferred from the registered metric.
    **kwargs : Any
        Any keyword supported by provider interface.

    Returns
    -------
    Metric
        A Metric object.

    Examples
    --------
    >>> from maite import load_metric
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
