import warnings
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, overload

from typing_extensions import Literal

from jatic_toolbox.protocols import ArrayLike, Classifier, ObjectDetector

from ...import_utils import is_torchvision_available
from .datasets import TorchVisionDataset

__all__ = ["TorchVisionAPI"]


class TorchVisionAPI:
    _SUPPORTED_TASKS: Tuple[str, ...] = ("image-classification", "object-detection")

    def _filter_string(
        self, filter_str: Optional[Union[str, List[str]]], all_models: List[Any]
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

    def list_datasets(self) -> Iterable[str]:
        """
        List torchvision datasets.

        Returns
        -------
        Iterable[str]
            List of torchvision datasets.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.list_datasets()
        [...]
        """
        if not is_torchvision_available():
            warnings.warn("TorchVision is not installed.")
            return []
        from torchvision import datasets

        datasets_list = list(datasets.__all__)
        datasets_list.sort()

        return datasets_list

    def load_dataset(
        self,
        dataset_name: str,
        task: Optional[Literal["image-classification", "object-detection"]] = None,
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> TorchVisionDataset:
        """
        Load a dataset from torchvision.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load.
        task : str
            The task of the dataset.
        **kwargs : Any
            Any keyword supported by torchvision.

        Returns
        -------
        TorchVisionDataset
            The dataset.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.load_dataset("MNIST", root="data", download=True)
        <jatic_toolbox._internals.interop.torchvision.datasets.TorchVisionDataset object at 0x000001F2B1B5B4C0>
        """
        from torchvision import datasets

        fn = getattr(datasets, dataset_name)

        try:
            if split is not None:
                dataset = fn(split=split, **kwargs)
            else:
                dataset = fn(**kwargs)
        except TypeError:
            train = False
            if split == "train":
                train = True

            try:
                dataset = fn(train=train, **kwargs)
            except TypeError as e:
                raise e

        if task == "image-classification":
            return TorchVisionDataset(dataset)

        raise NotImplementedError("Only `image-classification` task is supported.")

    def list_models(
        self,
        filter_str: Optional[Union[str, List[str]]] = None,
        task: Optional[Union[str, List[str]]] = None,
        model_name: Optional[str] = None,
    ) -> Iterable[Any]:
        """
        List torchvision models.

        Parameters
        ----------
        filter_str : Union[str, List[str]] | None (default: None)
            Filter string, by default None
        task : Union[str, List[str]] | None (default: None)
            The task of the model, by default None
        model_name : str | None (default: None)
            The name of the model, by default None

        Returns
        -------
        Iterable[Any]
            List of torchvision models.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.list_models()
        """
        if not is_torchvision_available():
            warnings.warn("TorchVision is not installed.")
            return []

        from torchvision import models
        from torchvision.models import list_models

        all_models = []
        if task is None:
            task = list(self._SUPPORTED_TASKS)

        if not isinstance(task, (list, tuple)):
            task = [task]

        task = list(task)

        if model_name is not None:
            if isinstance(filter_str, str):
                filter_str = [filter_str]

            if filter_str is None:
                filter_str = model_name
            else:
                filter_str.append(model_name)

        for t in task:
            module = None

            if "image-classification" in t:
                module = models

            if "object-detection" in t:
                module = models.detection

            if "optical-flow" in t:
                module = models.optical_flow

            if "quantization" in t:
                module = models.quantization

            if "segmentation" in t:
                module = models.segmentation

            if "video" in t:
                module = models.video

            if module is not None:
                ms = list_models(module=module)

                if filter_str is not None:
                    ms = self._filter_string(filter_str, ms)

                all_models.extend(ms)

        return all_models

    @overload
    def get_model_builder(
        self, task: Literal["image-classification"]
    ) -> Callable[..., Classifier[ArrayLike]]:
        ...

    @overload
    def get_model_builder(
        self, task: Literal["object-detection"]
    ) -> Callable[..., ObjectDetector[ArrayLike]]:
        ...

    def get_model_builder(
        self, task: Literal["image-classification", "object-detection"]
    ) -> Callable[..., Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]]:
        """
        Get the model builder for a given task.

        Parameters
        ----------
        task : Union[str, List[str]]
            The task of the model.

        Returns
        -------
        Callable[..., Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]]
            The model builder.

        Raises
        ------
        ValueError
            If the task is not supported.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.get_model_builder("image-classification")
        <function jatic_toolbox._internals.interop.torchvision.api.TorchVisionAPI.load_model.<locals>.<lambda>(task='image-classification', model_name='resnet18', **kwargs)>
        """
        from jatic_toolbox.interop.torchvision import (
            TorchVisionClassifier,
            TorchVisionObjectDetector,
        )

        if task == "image-classification":
            return TorchVisionClassifier.from_pretrained

        if task == "object-detection":
            return TorchVisionObjectDetector.from_pretrained

        raise ValueError(f"Task {task} is not supported.")

    def load_model(
        self, task: str, model_name: str, **kwargs: Any
    ) -> Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]:
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
        Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]
            The TorchVision model.

        Raises
        ------
        ImportError
            If TorchVision is not installed.

        ValueError
            If the task is not supported.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.torchvision.api import TorchVisionAPI
        >>> api = TorchVisionAPI()
        >>> api.load_model("image-classification", "resnet18")
        <jatic_toolbox._internals.interop.torchvision.models.TorchVisionClassifier object at 0x000001F2B1B5B4C0>
        """
        if not is_torchvision_available():
            raise ImportError("TorchVision is not installed.")

        from jatic_toolbox.interop.torchvision import (
            TorchVisionClassifier,
            TorchVisionObjectDetector,
        )

        if "image-classification" in task:
            return TorchVisionClassifier.from_pretrained(model_name, **kwargs)

        if "object-detection" in task:
            return TorchVisionObjectDetector.from_pretrained(model_name, **kwargs)

        raise ValueError(f"Task {task} is not supported.")
