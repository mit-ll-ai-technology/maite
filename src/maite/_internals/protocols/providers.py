# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# import component generics from generic.py and specialize them for object detection
# domain

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Callable, Literal, Protocol, Union, overload, runtime_checkable

from typing_extensions import Self, TypeAlias

from maite._internals.protocols import (
    generic as gen,
    image_classification as ic,
    object_detection as od,
    task_aliases as al,
)


@runtime_checkable
class ModelProvider(Protocol):
    def help(self, name: str) -> str:
        """
        Get information about the model such as:

            - Instructions for its use
            - Intended purpose
            - Any academic references

        Parameters
        ----------
        name: str
            The key that can be used to retrieve the model, returned by
            ``~ModelProvider.list_models``, same value used to retrieve the
            Model from ``~ModelProvider.load_model``

        Returns
        -------
        output: str
            The informational text.
        """
        ...

    def list_models(
        self,
        *,
        filter_str: str | list[str] | None = None,
        model_name: str | None = None,
        task: al.TaskName | None = None,
    ) -> Iterable[Any]:
        """
        List models for this provider.

        Parameters
        ----------
        filter_str : str | list[str] | None (default: None)
            A string or list of strings that contain complete or partial names for models.
        model_name : str | None (default: None)
            A string that contain complete or partial names for models.
        task : TaskName | None (default: None)
            A string or list of strings of tasks models were designed for, such as: "image-classification", "object-detection".
        **kwargs : Any
            Any keyword supported by this provider interface.

        Returns
        -------
        Iterable[Any]
            An iterable of model names.

        """
        ...

    @overload
    def load_model(
        self, model_name: str, task: Literal["image-classification"]
    ) -> ic.Model:
        ...

    @overload
    def load_model(
        self, model_name: str, task: Literal["object-detection"]
    ) -> od.Model:
        ...

    def load_model(self, model_name: str, task: al.TaskName) -> al.SomeModel:
        """
        Return a supported model.

        Parameters
        ----------
        model_name : str
            The `model_name` for the model (e.g., "microsoft/resnet-18").
        task : str | None
            The task for the model (e.g., "image-classification"). If None the task will be inferred from the model's interface
        **kwargs : Any
            Any keyword supported by provider interface.

        Returns
        -------
        Model
            A Model object that supports the given task.
        """
        ...


@runtime_checkable
class DatasetProvider(Protocol):
    def help(self, name: str) -> str:
        """
        Get information about the dataset such as instructions for its use,
        intended purpose, and relevant references

        Parameters
        ----------
        name: str
            The key that can be used to retrieve the dataset, returned by
            ``~DatasetProvider.list_datasets``, same value used to retrieve the
            Model from ``~DatasetProvider.load_dataset``

        Returns
        -------
        output: str
            The informational text.
        """
        ...

    def list_datasets(self) -> Iterable[str]:
        """
        List datasets for this provider.

        Parameters
        ----------
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Iterable[Any]
            An iterable of dataset names.

        """
        ...

    @overload
    def load_dataset(
        self,
        *,
        dataset_name: str,
        task: Literal["image-classification"],
        split: str | None = None,
    ) -> ic.Dataset:
        ...

    @overload
    def load_dataset(
        self,
        *,
        dataset_name: str,
        task: Literal["object-detection"],
        split: str | None = None,
    ) -> od.Dataset:
        ...

    def load_dataset(
        self,
        *,
        dataset_name: str,
        task: al.TaskName,
        split: str | None = None,
    ) -> al.SomeDataset:
        """
        Load dataset for a given provider.

        Parameters
        ----------
        dataset_name : str
            Name of dataset.
        task : TaskName | None (default: None)
            A string or list of strings of tasks dataset were designed for, such as: "image-classification", "object-detection".
        split : str | None (default: None)
            A string of split to load, such as: "train", "test", "validation".
            If None, the default split will be returned
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Dataset
            A dataset object that supports the given task.

        """
        ...


@runtime_checkable
class MetricProvider(Protocol):
    def help(self, name: str) -> str:
        """
        Get information about the Metric such as:

            - Instructions for its use
            - Intended purpose
            - Any academic references

        Parameters
        ----------
        name: str
            The key that can be used to retrieve the model, returned by
            ``~MetricProvider.list_metric``, same value used to retrieve the
            Model from ``~MetricProvider.load_metric``

        Returns
        -------
        output: str
            The informational text.
        """
        ...

    def list_metrics(self) -> Iterable[Any]:
        """
        List metrics for this provider.

        Parameters
        ----------
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Iterable[Any]
            An iterable of metric names.

        """
        ...

    @overload
    def load_metric(self, metric_name: str) -> ic.Metric:
        ...

    @overload
    def load_metric(self, metric_name: str) -> od.Metric:
        ...

    def load_metric(self, metric_name: str) -> al.SomeMetric:
        # load_metric must return a supertype of all task-specific metrics
        # since gen.Metric is contravariant wrt type argument, gen.Metric[Union[od.TargetBatchType, ic.TargetBatchType,...]]
        # is gen.Metric[<SuperType of (od.TargetBatchType, ic.TargetBatchType,...)>]
        # which is a *subtype* of all task specific metrics (note the reversal of supertype/subtype relationship)
        # when types are passed through contravariant argument of a generic
        """
        Return a Metric object.

        Parameters
        ----------
        metric_name : str
            The `metric_name` for the metric (e.g., "accuracy").
        **kwargs : Any
            Any keyword supported by this provider.

        Returns
        -------
        Metric
            A Metric object.
        """
        ...


# Why an alias union, rather than a base protocol?  There are some applications where
# the concept of AnyProvider is a meaningful type hint However there is no useful
# declaration that could be assigned to a base type say with only the common `help`
# method as a requirement
AnyProvider: TypeAlias = Union[ModelProvider, MetricProvider, DatasetProvider]


#
# ArtifactHub
#
@runtime_checkable
class ArtifactHubEndpoint(Protocol):
    def __init__(self, path: Any):
        """Endpoints are initialized with a path argument specifying any information necessary to find the source via the endpoint's target api"""
        ...

    def get_cache_or_reload(self) -> str | os.PathLike[str]:
        """Create a local copy of the resource in the cache (if needed) and return a path suitable to locate the `hubconf.py` file"""
        ...

    def update_options(self) -> Self:
        """Update update any cached state used by the endpoint

        API tokens, validation options, etc.
        If none of these apply, the method may simply be a no-op
        """
        ...


ModelEntrypoint: TypeAlias = Callable[..., gen.Model]
DatasetEntrypoint: TypeAlias = Callable[..., gen.Dataset]
MetricEntrypoint: TypeAlias = Callable[..., gen.Metric]
AnyEntrypoint: TypeAlias = Union[ModelEntrypoint, DatasetEntrypoint, MetricEntrypoint]
