import warnings
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from jatic_toolbox.protocols import ArrayLike, Classifier, ObjectDetector

from ...import_utils import is_hf_available, is_hf_hub_available
from .datasets import HuggingFaceVisionDataset

__all__ = ["HuggingFaceAPI"]


class HuggingFaceAPI:
    _SUPPORTED_TASKS: Tuple[str, ...] = ("image-classification", "object-detection")

    def list_datasets(
        self,
        author: Optional[str] = None,
        benchmark: Optional[Union[str, List[str]]] = None,
        dataset_name: Optional[str] = None,
        size_categories: Optional[Union[str, List[str]]] = None,
        task_categories: Optional[Union[str, List[str]]] = None,
        task_ids: Optional[Union[str, List[str]]] = None,
        with_community_datasets: bool = False,
    ) -> Iterable[Any]:
        """
        List HuggingFace datasets.

        Parameters
        ----------
        author : str | None (default: None)
            The author of the HuggingFace dataset.
        benchmark : Union[str, List[str]] | None (default: None)
            A string or list of strings that can be used to identify datasets on the Hub by their official benchmark.
        dataset_name : str | None (default: None)
            A string or list of strings that can be used to identify datasets on the Hub by its name, such as `SQAC` or `wikineural`.
        size_categories : Union[str, List[str]] | None (default: None)
            A string or list of strings that can be used to identify datasets on
            the Hub by the size of the dataset such as `100K<n<1M` or
            `1M<n<10M`.
        task_categories : Union[str, List[str]] | None (default: None)
            A string or list of strings that can be used to identify datasets on
            the Hub by the designed task, such as `audio_classification` or
            `named_entity_recognition`.
        task_ids : Union[str, List[str]] | None (default: None)
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
        >>> from jatic_toolbox._internals.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> api.list_datasets(dataset_name="resnet")
        [...]
        """
        if not is_hf_hub_available():
            warnings.warn("HuggingFace Hub is not installed.")
            return []

        from huggingface_hub.hf_api import HfApi
        from huggingface_hub.utils.endpoint_helpers import DatasetFilter

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

    def load_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> HuggingFaceVisionDataset:
        """
        Load a HuggingFace dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the HuggingFace dataset.
        split : str | None (default: None)
            Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
        task : str | None (default: None)
            The task to prepare the dataset for during training and evaluation. Casts the dataset's [`Features`]
            to standardized column names and types as detailed in `datasets.tasks`.
        **kwargs : Any
            Additional keyword arguments to pass to the HuggingFace dataset.

        Returns
        -------
        HuggingFaceVisionDataset
            A Jatic supported dataset.

        Raises
        ------
        ImportError
            If HuggingFace Datasets is not installed.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> dataset = api.load_dataset("resnet")
        """
        if not is_hf_available():
            raise ImportError("HuggingFace Datasets is not installed.")

        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split, task=task, **kwargs)

        return HuggingFaceVisionDataset(dataset)  # type: ignore

    def list_models(
        self,
        filter_str: Optional[Union[str, List[str]]] = None,
        author: Optional[str] = None,
        library: Optional[Union[str, List[str]]] = None,
        language: Optional[Union[str, List[str]]] = None,
        model_name: Optional[str] = None,
        task: Optional[Union[str, List[str]]] = None,
        trained_dataset: Optional[Union[str, List[str]]] = None,
        tags: Optional[Union[str, List[str]]] = None,
    ) -> Iterable[Any]:
        """
        List HuggingFace models.

        Parameters
        ----------
        filter_str : Union[str, List[str]] | None (default: None)
            The filter string to use to filter the models.
        author : str (default: None)
            A string or list of strings that can be used to identify datasets on
            the Hub by the original uploader (author or organization), such as
            `facebook` or `huggingface`.
        library : Union[str, List[str]] | None (default: None)
            A string or list of strings of foundational libraries models were
            originally trained from, such as pytorch, tensorflow, or allennlp.
        language :Union[str, List[str]] | None (default: None)
            A string or list of strings of languages, both by name and country
            code, such as "en" or "English".
        model_name : str | None (default: None)
            A string that contain complete or partial names for models on the
            Hub, such as "bert" or "bert-base-cased".
        task : Union[str, List[str]] | None (default: None)
            A string or list of strings of tasks models were designed for, such
            as: "fill-mask" or "automatic-speech-recognition".
        trained_dataset : Union[str, List[str]] | None (default: None)
            A string tag or a list of string tags of the trained dataset for a
            model on the Hub.
        tags : Union[str, List[str]] | None (default: None)
            A string tag or a list of tags to filter models on the Hub by, such
            as `text-generation` or `spacy`.

        Returns
        -------
        Iterable[Any]

        Warnings
        --------
        This method requires the `huggingface_hub` package to be installed.

        Examples
        --------
        >>> from jatic_toolbox._internals.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> api.list_models(model_name="bert")
        """
        if not is_hf_hub_available():
            warnings.warn("HuggingFace Hub is not installed.")
            return []

        from huggingface_hub.hf_api import HfApi
        from huggingface_hub.utils.endpoint_helpers import ModelFilter

        if task is None:
            task = list(self._SUPPORTED_TASKS)

        hf_api = HfApi()
        filt = filter_str
        if filter_str is None:
            filt = ModelFilter(
                author=author,
                library=library,
                language=language,
                model_name=model_name,
                task=task,
                trained_dataset=trained_dataset,
                tags=tags,
            )

        return hf_api.list_models(filter=filt)

    def get_model_builder(
        self, task: Union[str, List[str]]
    ) -> Callable[..., Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]]:
        """
        Get the model builder for a given task.

        Parameters
        ----------
        task : Union[str, List[str]]
            The task of the model.

        Returns
        -------
        Callable[..., Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]]]
            The model builder.

        Raises
        ------
        ImportError
            If HuggingFace Transformers is not installed.

        ValueError
            If the task is not supported.

        Examples
        --------
        >>> from jatic_toolbox.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> api.get_model_builder("image-classification")
        """
        if not is_hf_available():
            raise ImportError("HuggingFace Transformers is not installed.")

        from jatic_toolbox.interop.huggingface import (
            HuggingFaceImageClassifier,
            HuggingFaceObjectDetector,
        )

        if isinstance(task, str):
            task = [task]

        if "image-classification" in task:
            return HuggingFaceImageClassifier.from_pretrained

        if "object-detection" in task:
            return HuggingFaceObjectDetector.from_pretrained

        raise ValueError(f"Task {task} is not supported.")

    def load_model(
        self, task: Union[str, List[str]], model_name: str, **kwargs: Any
    ) -> Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]:
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
        Union[Classifier[ArrayLike], ObjectDetector[ArrayLike]]
            The loaded model.

        Raises
        ------
        ImportError
            If HuggingFace Transformers is not installed.

        ValueError
            If the task is not supported.

        Examples
        --------
        >>> from jatic_toolbox.interop.huggingface.api import HuggingFaceAPI
        >>> api = HuggingFaceAPI()
        >>> api.load_model("image-classification", "google/vit-base-patch16-224-in21k")
        """
        if not is_hf_available():
            raise ImportError("HuggingFace Transformers is not installed.")

        from jatic_toolbox.interop.huggingface import (
            HuggingFaceImageClassifier,
            HuggingFaceObjectDetector,
        )

        if "image-classification" in task:
            return HuggingFaceImageClassifier.from_pretrained(model_name, **kwargs)

        if "object-detection" in task:
            return HuggingFaceObjectDetector.from_pretrained(model_name, **kwargs)

        raise ValueError(f"Task {task} is not supported.")
