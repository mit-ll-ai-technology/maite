import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from typing_extensions import Literal

from jatic_toolbox.protocols import Dataset, ImageClassifier, ObjectDetector

from ...import_utils import (
    is_hf_datasets_available,
    is_hf_hub_available,
    is_hf_transformers_available,
)
from .datasets import HuggingFaceObjectDetectionDataset, HuggingFaceVisionDataset
from .typing import HuggingFaceDataset

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
        task: Optional[Literal["image-classification", "object-detection"]] = None,
        split: Optional[str] = None,
        **kwargs,
    ) -> Union[Dataset[Any], Dict[str, Dataset[Any]]]:
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
        Union[Dataset, Dict[str, Dataset]]
            The HuggingFace dataset.

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
        if not is_hf_datasets_available():  # pragma: no cover
            raise ImportError("HuggingFace Datasets is not installed.")

        if task is not None and task not in self._SUPPORTED_TASKS:
            raise ValueError(
                f"Task {task} is not supported. Supported tasks are {self._SUPPORTED_TASKS}."
            )

        from datasets import load_dataset

        wrapper_kwargs = {}
        keys = list(kwargs.keys())
        for key in keys:
            if key.endswith("_key"):
                wrapper_kwargs[key] = kwargs.pop(key)

        # TODO: HuggingFace doesn't have a standard on datasets to provide "object-detection"
        # task. We need to check if the dataset is compatible with the task and if not,
        # we load the dataset without the task.
        try:
            dataset = load_dataset(dataset_name, split=split, task=task, **kwargs)
        except ValueError as e:  # pragma: no cover
            if "Task object-detection is not compatible" in str(e):
                dataset = load_dataset(dataset_name, split=split, **kwargs)
            else:
                raise e

        if task is None:
            warnings.warn("Task is not specified. Returning raw HuggingFace dataset.")
            if TYPE_CHECKING:
                assert isinstance(dataset, Dataset)

            return dataset

        if split is None and isinstance(dataset, dict):
            warnings.warn("Split is not specified. Returning raw HuggingFace dataset.")
            return dataset

        if TYPE_CHECKING:
            assert isinstance(dataset, Dataset)
            dataset = cast(HuggingFaceDataset, dataset)

        if task == "image-classification":
            return HuggingFaceVisionDataset(dataset, **wrapper_kwargs)
        elif task == "object-detection":
            return HuggingFaceObjectDetectionDataset(dataset, **wrapper_kwargs)

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
        models = hf_api.list_models(filter=filt)
        return list(iter(models))

    def load_model(
        self,
        task: Optional[Literal["image-classification", "object-detection"]],
        model_name: str,
        **kwargs: Any,
    ) -> Union[ImageClassifier, ObjectDetector]:
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
        if not is_hf_transformers_available():  # pragma: no cover
            raise ImportError("HuggingFace Transformers is not installed.")

        from jatic_toolbox.interop.huggingface import (
            HuggingFaceImageClassifier,
            HuggingFaceObjectDetector,
        )

        if task == "image-classification":
            return HuggingFaceImageClassifier.from_pretrained(model_name, **kwargs)

        if task == "object-detection":
            return HuggingFaceObjectDetector.from_pretrained(model_name, **kwargs)

        raise ValueError(f"Task {task} is not supported.")
