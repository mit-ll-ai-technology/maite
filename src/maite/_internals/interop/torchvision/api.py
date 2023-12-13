# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Sequence,
    Tuple,
    cast,
    overload,
)

from maite.errors import InvalidArgument, ToolBoxException
from maite.protocols import (
    Dataset,
    ImageClassifier,
    ObjectDetector,
    SupportsImageClassification,
    SupportsObjectDetection,
)

from ...import_utils import is_torchvision_available

if is_torchvision_available():
    from torchvision import datasets, models
    from torchvision.models import list_models

    from maite.interop.torchvision import (
        TorchVisionClassifier,
        TorchVisionObjectDetector,
    )

    from .datasets import PyTorchVisionDataset, TorchVisionDataset

__all__ = ["TorchVisionAPI"]


def _get_torchvision_dataset(
    dataset_name: str,
) -> Callable[..., Sequence[Any]]:
    from torchvision import datasets

    try:
        return getattr(datasets, dataset_name)
    except AttributeError as e:
        raise InvalidArgument(
            f"Dataset {dataset_name} is not supported by torchvision"
        ) from e


class TorchVisionAPI:
    _SUPPORTED_TASKS: Tuple[str, ...] = ("image-classification", "object-detection")

    def _filter_string(
        self, filter_str: str | List[str] | None, all_models: List[Any]
    ) -> Iterable[Any]:
        """
        Filter strings.

        #TODO: Should I look at fnmatch?

        Parameters
        ----------
        filter_str : Optional[Union[str, List[str]]]
        all_models : List[Any]

        Returns
        -------
        Iterable[Any]
            Filtered list.
        """
        include_filters = []
        if filter_str is not None:
            include_filters = (
                filter_str if isinstance(filter_str, (tuple, list)) else [filter_str]
            )

        out_models = all_models
        if len(include_filters) > 0:
            out_models = []
            for f in include_filters:
                include_models = list(filter(lambda x: f in x, all_models))
                if len(include_models):
                    out_models = set(out_models).union(include_models)

        return list(out_models)

    def list_datasets(self, dataset_name: str | None = None) -> Iterable[str]:
        """
        List torchvision datasets.

        Returns
        -------
        Iterable[str]
            List of torchvision datasets.

        Examples
        --------
        >>> from maite._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.list_datasets()
        [...]
        """
        if not is_torchvision_available():  # pragma: no cover
            raise ImportError("TorchVision is not installed.")

        datasets_list = list(datasets.__all__)
        datasets_list.sort()

        if dataset_name is not None:
            datasets_list = self._filter_string(dataset_name, datasets_list)

        return datasets_list

    @overload
    def load_dataset(
        self,
        dataset_name: str,
        task: Literal["image-classification"],
        split: str | None = None,
        **kwargs: Any,
    ) -> Dataset[SupportsImageClassification]:
        ...

    @overload
    def load_dataset(
        self,
        dataset_name: str,
        task: Literal["object-detection"],
        split: str | None = None,
        **kwargs: Any,
    ) -> Dataset[SupportsObjectDetection]:
        ...

    @overload
    def load_dataset(
        self,
        dataset_name: str,
        task: None = None,
        split: str | None = None,
        **kwargs,
    ) -> Dataset[SupportsImageClassification | SupportsObjectDetection]:
        ...

    def load_dataset(
        self,
        dataset_name: str,
        task: Literal["image-classification", "object-detection"] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> Dataset[SupportsImageClassification | SupportsObjectDetection]:
        """
        Load a dataset from torchvision.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load.
        task : str | None (default: None)
            The task of the dataset.
        split : str | None (default: None)
            The split of the dataset.
        **kwargs : Any
            Any keyword supported by torchvision.

        Returns
        -------
        TorchVisionDataset
            The dataset.

        Examples
        --------
        >>> from maite._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.load_dataset("MNIST", root="data", download=True)
        <maite._internals.interop.torchvision.datasets.TorchVisionDataset object at 0x000001F2B1B5B4C0>
        """
        if not is_torchvision_available():  # pragma: no cover
            raise ImportError("TorchVision is not installed.")

        if task is not None and task not in ("image-classification",):
            raise InvalidArgument(
                f"Task {task} is not supported. Supported tasks are ('image-classification', )."
            )

        fn = _get_torchvision_dataset(dataset_name)

        try:
            dataset = fn(split=split, **kwargs)
        except TypeError:
            train = False
            if split == "train":
                train = True

            try:
                dataset = fn(train=train, **kwargs)
            except TypeError:
                pass

            try:
                dataset = fn(**kwargs)
            except Exception as e:  # pragma: no cover
                raise ToolBoxException(
                    f"Unable to load dataset with `dataset_name={dataset_name}`."
                ) from e

        if TYPE_CHECKING:
            dataset = cast(PyTorchVisionDataset, dataset)

        if task == "image-classification":
            return TorchVisionDataset(dataset)

        raise NotImplementedError("Only `image-classification` task is supported.")

    def list_models(
        self,
        filter_str: str | List[str] | None = None,
        task: str | List[str] | None = None,
        model_name: str | None = None,
    ) -> Iterable[Any]:
        """
        List torchvision models.

        Parameters
        ----------
        filter_str : str | List[str] | None (default: None)
            Filter string, by default None
        task : str | List[str] | None (default: None)
            The task of the model, by default None
        model_name : str | None (default: None)
            The name of the model, by default None.  Replaces filter_str.

        Returns
        -------
        Iterable[Any]
            List of torchvision models.

        Examples
        --------
        >>> from maite._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.list_models()
        """
        if not is_torchvision_available():  # pragma: no cover
            raise ImportError("TorchVision is not installed.")

        all_models = []
        if task is None:
            task = list(self._SUPPORTED_TASKS)

        if not isinstance(task, (list, tuple)):
            task = [task]

        task = list(task)

        if model_name is not None:  # pragma: no cover
            if filter_str is not None:
                warnings.warn("filter_str is ignored when model_name is provided.")
            filter_str = model_name

        for t in task:
            module = None

            if "image-classification" in t:
                module = models

            if "object-detection" in t:
                module = models.detection

            # the following are not supported but provided
            if "optical-flow" in t:  # pragma: no cover
                module = models.optical_flow

            if "quantization" in t:  # pragma: no cover
                module = models.quantization

            if "segmentation" in t:  # pragma: no cover
                module = models.segmentation

            if "video" in t:  # pragma: no cover
                module = models.video

            if module is not None:
                ms = list_models(module=module)

                if filter_str is not None:
                    ms = self._filter_string(filter_str, ms)

                all_models.extend(ms)

        return all_models

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
        Load a TorchVision model.

        Parameters
        ----------
        task : str
            The task of the model.
        model_name : str
            The name of the TorchVision model.
        **kwargs : Any
            Additional keyword arguments to pass to the TorchVision model.

        Returns
        -------
        Classifier | ObjectDetector
            The TorchVision model.

        Raises
        ------
        ImportError
            If TorchVision is not installed.

        InvalidArgument
            If the task is not supported.

        Examples
        --------
        >>> from maite._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.load_model("image-classification", "resnet18")
        <maite._internals.interop.torchvision.models.TorchVisionClassifier object at 0x000001F2B1B5B4C0>
        """
        if not is_torchvision_available():  # pragma: no cover
            raise ImportError("TorchVision is not installed.")

        if "image-classification" in task:
            return TorchVisionClassifier.from_pretrained(model_name, **kwargs)

        if "object-detection" in task:
            return TorchVisionObjectDetector.from_pretrained(model_name, **kwargs)

        raise InvalidArgument(f"Task {task} is not supported.")
