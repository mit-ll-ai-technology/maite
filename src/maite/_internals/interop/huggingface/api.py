# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Iterable, List, Literal, Tuple, overload

from maite.errors import InvalidArgument
from maite.protocols import (
    Dataset,
    ImageClassifier,
    ObjectDetector,
    SupportsImageClassification,
    SupportsObjectDetection,
)

from ...import_utils import (
    is_hf_datasets_available,
    is_hf_hub_available,
    is_hf_transformers_available,
)

if is_hf_datasets_available():
    import datasets
    from datasets import DatasetDict, IterableDataset

    from .datasets import HuggingFaceObjectDetectionDataset, HuggingFaceVisionDataset

if is_hf_hub_available():
    from huggingface_hub.hf_api import HfApi
    from huggingface_hub.utils.endpoint_helpers import DatasetFilter, ModelFilter

if is_hf_transformers_available():
    from maite.interop.huggingface import (
        HuggingFaceImageClassifier,
        HuggingFaceObjectDetector,
    )


__all__ = ["HuggingFaceAPI"]


class HuggingFaceAPI:
    _SUPPORTED_TASKS: Tuple[str, ...] = ("image-classification", "object-detection")

    def list_datasets(
        self,
        author: str | None = None,
        benchmark: str | List[str] | None = None,
        dataset_name: str | None = None,
        size_categories: str | List[str] | None = None,
        task_categories: str | List[str] | None = None,
        task_ids: str | List[str] | None = None,
        with_community_datasets: bool = False,
    ) -> Iterable[Any]:
        """
        List HuggingFace datasets.

        Parameters
        ----------
        author : str | None (default: None)
            The author of the HuggingFace dataset.
        benchmark : str | List[str] | None (default: None)
            A string or list of strings that can be used to identify datasets on the Hub by their official benchmark.
        dataset_name : str | None (default: None)
            A string or list of strings that can be used to identify datasets on the Hub by its name, such as `SQAC` or `wikineural`.
        size_categories : str | List[str] | None (default: None)
            A string or list of strings that can be used to identify datasets on
            the Hub by the size of the dataset such as `100K<n<1M` or
            `1M<n<10M`.
        task_categories : str | List[str] | None (default: None)
            A string or list of strings that can be used to identify datasets on
            the Hub by the designed task, such as `audio_classification` or
            `named_entity_recognition`.
        task_ids : str | List[str] | None (default: None)
            A string or list of strings that can be used to identify datasets on
            the Hub by the specific task such as `speech_emotion_recognition` or
            `paraphrase`.
        with_community_datasets : bool (default: False)
            Whether to include community HuggingFace datasets in the list.

        Returns
        -------
        Iterable[Any]
            A list of HuggingFace datasets.

        Warning
        -------
        This method requires the `huggingface_hub` package to be installed.

        Examples
        --------
        >>> from maite._internals.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> api.list_datasets(dataset_name="resnet")
        [...]
        """
        if not is_hf_hub_available():  # pragma: no cover
            raise ImportError("HuggingFace Hub is not installed.")

        hf_api = HfApi()

        filt = DatasetFilter(
            author=author,
            benchmark=benchmark,
            dataset_name=dataset_name,
            size_categories=size_categories,
            task_categories=task_categories,
            task_ids=task_ids,
        )

        datasets_list = hf_api.list_datasets(filter=filt, full=False)

        if not with_community_datasets:
            datasets_list = [
                dataset
                for dataset in datasets_list
                if dataset.id is not None and "/" not in dataset.id
            ]

        return datasets_list

    @overload
    def load_dataset(
        self,
        dataset_name: str,
        task: Literal["image-classification"],
        split: str | None = None,
        **kwargs,
    ) -> Dataset[SupportsImageClassification]:
        ...

    @overload
    def load_dataset(
        self,
        dataset_name: str,
        task: Literal["object-detection"],
        split: str | None = None,
        **kwargs,
    ) -> Dataset[SupportsObjectDetection]:
        ...

    def load_dataset(
        self,
        dataset_name: str,
        task: Literal["image-classification", "object-detection"],
        split: str | None = None,
        **kwargs,
    ) -> Dataset[SupportsImageClassification | SupportsObjectDetection]:
        """
        Load a HuggingFace dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the HuggingFace dataset.
        task : Literal["image-classification", "object-detection"] | None (default: None)
            If `None` returns the raw HuggingFace dataset.
            The task to prepare the dataset for during training and evaluation.
        split : str | None (default: None)
            Which split of the data to load.
            If `None`, will return a `dict` with all splits.
            If given, will return a single Dataset.
        **kwargs : Any
            Additional keyword arguments to pass to the HuggingFace dataset.

        Returns
        -------
        Dataset[SupportsImageClassification | SupportsObjectDetection]
            The HuggingFace dataset.

        Raises
        ------
        ImportError
            If HuggingFace Datasets is not installed.

        Examples
        --------
        >>> from maite._internals.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> dataset = api.load_dataset("resnet")
        """
        if not is_hf_datasets_available():  # pragma: no cover
            raise ImportError("HuggingFace Datasets is not installed.")

        if task is None:
            raise InvalidArgument("Task is not specified")
        elif task not in self._SUPPORTED_TASKS:
            raise InvalidArgument(
                f"Task {task} is not supported. Supported tasks are {self._SUPPORTED_TASKS}."
            )

        wrapper_kwargs = {}
        keys = list(kwargs.keys())
        for key in keys:
            if key.endswith("_key"):
                wrapper_kwargs[key] = kwargs.pop(key)

        # TODO: HuggingFace doesn't have a standard on datasets to provide "object-detection"
        # task. We need to check if the dataset is compatible with the task and if not,
        # we load the dataset without the task.
        try:
            dataset = datasets.load_dataset(
                dataset_name, split=split, task=task, **kwargs
            )
        except ValueError as e:  # pragma: no cover
            if "Task object-detection is not compatible" in str(e):
                dataset = datasets.load_dataset(dataset_name, split=split, **kwargs)
            else:
                raise e

        if isinstance(dataset, (dict, DatasetDict)):  # pragma: no cover
            raise InvalidArgument("Split is not specified")

        if isinstance(dataset, IterableDataset):  # pragma: no cover
            raise ValueError(f"IterableDataset is not supported. Got {type(dataset)}.")

        if task == "image-classification":
            return HuggingFaceVisionDataset(dataset, **wrapper_kwargs)
        elif task == "object-detection":
            return HuggingFaceObjectDetectionDataset(dataset, **wrapper_kwargs)

    def list_models(
        self, filter_str: str | List[str] | None = None, **kwargs: Any
    ) -> Iterable[Any]:
        """
        List HuggingFace models.

        Parameters
        ----------
        filter_str : str | List[str] | None (default: None)
            The filter string to use to filter the models.
        **kwargs : Any
            Additional keyword arguments to pass to the HuggingFace model filter.

        Returns
        -------
        Iterable[Any]

        Warnings
        --------
        This method requires the `huggingface_hub` package to be installed.

        Examples
        --------
        >>> from maite._internals.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> api.list_models(model_name="bert")
        """
        if not is_hf_hub_available():  # pragma: no cover
            raise ImportError("HuggingFace Hub is not installed.")

        hf_api = HfApi()
        filt = filter_str
        if filter_str is None:
            filt = ModelFilter(**kwargs)
        models = hf_api.list_models(filter=filt)
        return list(iter(models))

    @overload
    def load_model(
        self,
        task: Literal["image-classification"],
        model_name: str,
        **kwargs: Any,
    ) -> ImageClassifier:
        ...

    @overload
    def load_model(
        self,
        task: Literal["object-detection"],
        model_name: str,
        **kwargs: Any,
    ) -> ObjectDetector:
        ...

    def load_model(
        self,
        task: Literal["image-classification", "object-detection"],
        model_name: str,
        **kwargs: Any,
    ) -> ImageClassifier | ObjectDetector:
        """
        Load a HuggingFace model.

        Parameters
        ----------
        task : str
            The task of the model.
        model_name : str
            The name of the HuggingFace model.
        **kwargs : Any
            Additional keyword arguments to pass to the HuggingFace model.

        Returns
        -------
        ImageClassifier | ObjectDetector
            The loaded model.

        Raises
        ------
        ImportError
            If HuggingFace Transformers is not installed.

        InvalidArgument
            If the task is not supported.

        Examples
        --------
        >>> from maite.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> api.load_model("image-classification", "google/vit-base-patch16-224-in21k")
        """
        if not is_hf_transformers_available():  # pragma: no cover
            raise ImportError("HuggingFace Transformers is not installed.")

        if task is None:
            raise InvalidArgument("Task is not specified")
        elif task not in self._SUPPORTED_TASKS:
            raise InvalidArgument(
                f"Task {task} is not supported. Supported tasks are {self._SUPPORTED_TASKS}."
            )

        if task == "image-classification":
            return HuggingFaceImageClassifier.from_pretrained(model_name, **kwargs)

        if task == "object-detection":
            return HuggingFaceObjectDetector.from_pretrained(model_name, **kwargs)
