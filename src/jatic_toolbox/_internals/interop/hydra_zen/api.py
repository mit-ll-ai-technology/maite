import importlib
import warnings
from typing import Any, Optional, Type, Union

import hydra_zen
from hydra_zen.typing import Builds
from typing_extensions import Literal, TypeAlias

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.interop.huggingface import (
    HuggingFaceImageClassifier,
    HuggingFaceObjectDetectionDataset,
    HuggingFaceObjectDetector,
    HuggingFaceVisionDataset,
)
from jatic_toolbox.interop.torchvision import (
    TorchVisionClassifier,
    TorchVisionDataset,
    TorchVisionObjectDetector,
)
from jatic_toolbox.protocols import (
    Dataset,
    ImageClassifier,
    Metric,
    ObjectDetectionDataset,
    ObjectDetector,
    VisionDataset,
)

from ...import_utils import (
    is_hf_available,
    is_torcheval_available,
    is_torchmetrics_available,
    is_torchvision_available,
)
from .utils import get_dataclass_docstring, get_torchvision_dataset

__all__ = [
    "get_dataset_builder",
    "get_model_builder",
    "get_metric_builder",
]

SUPPORTED_TASKS: TypeAlias = Literal["image-classification", "object-detection"]


def _get_hf_dataset_builder(
    *,
    dataset_name: str,
    task: SUPPORTED_TASKS,
    split: Optional[str] = None,
    **kwargs: Any,
) -> Type[Builds[Union[Type[VisionDataset], Type[ObjectDetectionDataset]]]]:
    """
    Create a builder for a HuggingFace dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    task : str
        The task of the dataset.
    split : str, optional
        The split of the dataset to use.
    **kwargs : Any
        Additional keyword arguments to pass to the dataset builder.

    Returns
    -------
    Type[Builds[Union[Type[VisionDataset], Type[ObjectDetectionDataset]]]]
        The dataset builder.

    Raises
    ------
    ImportError
        If HuggingFace datasets is not installed.

    Notes
    -----
    This function is a wrapper around `datasets.load_dataset` from HuggingFace datasets.

    See Also
    --------
    datasets.load_dataset : The function that this function wraps.
    """
    if not is_hf_available():  # pragma: no cover
        raise ImportError("HuggingFace Datasets is not installed.")

    if split is None:
        warnings.warn(
            "No split was given. HuggingFace datasets can return a dict with all splits."
        )

    from datasets import load_dataset

    wrapper_kwargs = {}
    keys = list(kwargs.keys())
    for key in keys:
        if key.endswith("_key"):
            wrapper_kwargs[key] = kwargs.pop(key)

    dataset = hydra_zen.builds(
        load_dataset,
        path=dataset_name,
        split=split,
        populate_full_signature=True,
        **kwargs,
    )
    dataset.__doc__ = get_dataclass_docstring(dataset, load_dataset)

    fields = dataset.__dataclass_fields__.keys()
    if "task" in fields:
        dataset.task = task  # type: ignore

    if task == "image-classification":
        cfg = hydra_zen.builds(
            HuggingFaceVisionDataset, dataset=dataset, populate_full_signature=True
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, HuggingFaceVisionDataset)
        return cfg

    elif task == "object-detection":
        cfg = hydra_zen.builds(
            HuggingFaceObjectDetectionDataset,
            dataset=dataset,
            populate_full_signature=True,
            **wrapper_kwargs,
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, HuggingFaceObjectDetectionDataset)
        return cfg
    else:
        raise NotImplementedError(
            f"Unknown task {task}. Supported tasks are {SUPPORTED_TASKS}."
        )


def _get_tv_dataset_builder(
    *,
    dataset_name: str,
    task: SUPPORTED_TASKS,
    split: Optional[str] = None,
    **kwargs: Any,
) -> Type[Builds[Type[TorchVisionDataset]]]:
    """
    Create a builder for a TorchVision dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    task : str
        The task of the dataset.
    split : str, optional
        The split of the dataset to use.
    **kwargs : Any
        Additional keyword arguments to pass to the dataset builder.

    Returns
    -------
    Type[Builds[Type[TorchVisionDataset]]]
        The dataset builder.

    Raises
    ------
    ImportError
        If TorchVision is not installed.

    Notes
    -----
    This function is a wrapper around `torchvision.datasets` from TorchVision.

    See Also
    --------
    torchvision.datasets : The function that this function wraps.
    """
    if not is_torchvision_available():
        raise ImportError("TorchVision is not installed.")

    from torchvision import __version__ as torchvision_version

    if task not in ("image-classification",):
        raise NotImplementedError(
            f"Task {task} is not supported. Supported tasks are ('image-classification', )."
        )

    fn = get_torchvision_dataset(dataset_name)
    dataset = hydra_zen.builds(fn, populate_full_signature=True, **kwargs)
    dataset.__doc__ = get_dataclass_docstring(dataset, fn)

    keys = list(dataset.__dataclass_fields__.keys())

    if split is not None:
        if "split" in keys:
            dataset.split = split  # type: ignore
        elif "train" in keys:
            dataset.train = split == "train"  # type: ignore

    if task == "image-classification":
        cfg = hydra_zen.builds(
            TorchVisionDataset,
            dataset=dataset,
            zen_meta=dict(torchvision_version=torchvision_version),
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, TorchVisionDataset)
        return cfg


def _get_hf_model_builder(
    *,
    task: SUPPORTED_TASKS,
    model_name: str,
    **kwargs: Any,
) -> Type[
    Builds[Union[Type[HuggingFaceImageClassifier], Type[HuggingFaceObjectDetector]]]
]:
    """
    Create a builder for a HuggingFace model.

    Parameters
    ----------
    task : str
        The task of the model.
    model_name : str
        The name of the model.
    **kwargs : Any
        Additional keyword arguments to pass to the model builder.

    Returns
    -------
    Type[Builds[Union[Type[HuggingFaceImageClassifier], Type[HuggingFaceObjectDetector]]]]
        The model builder.

    Raises
    ------
    ImportError
        If HuggingFace Transformers is not installed.

    Notes
    -----
    This function is a wrapper around `transformers.AutoModelForImageClassification` and
    `transformers.AutoFeatureExtractor` from HuggingFace Transformers.

    See Also
    --------
    transformers.AutoModelForImageClassification : The function that this function wraps.
    transformers.AutoFeatureExtractor : The function that this function wraps.
    """
    if not is_hf_available():  # pragma: no cover
        raise ImportError("HuggingFace Transformers is not installed.")

    if task == "image-classification":
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        top_k = kwargs.pop("top_k", None)

        try:
            clf_model = hydra_zen.builds(
                AutoModelForImageClassification.from_pretrained,
                pretrained_model_name_or_path=model_name,
                populate_full_signature=True,
                **kwargs,
            )
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        try:
            processor = hydra_zen.builds(
                AutoFeatureExtractor.from_pretrained,
                pretrained_model_name_or_path=model_name,
                populate_full_signature=True,
                **kwargs,
            )
        except OSError:  # pragma: no cover
            processor = None

        cfg = hydra_zen.builds(
            HuggingFaceImageClassifier,
            model=clf_model,
            processor=processor,
            top_k=top_k,
            populate_full_signature=True,
            **kwargs,
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, HuggingFaceImageClassifier)
        return cfg

    if task == "object-detection":
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        threshold = kwargs.pop("threshold", 0.5)

        try:
            det_model = hydra_zen.builds(
                AutoModelForObjectDetection.from_pretrained,
                pretrained_model_name_or_path=model_name,
                populate_full_signature=True,
                **kwargs,
            )
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        try:
            processor = hydra_zen.builds(
                AutoImageProcessor.from_pretrained,
                pretrained_model_name_or_path=model_name,
                populate_full_signature=True,
                **kwargs,
            )
        except OSError as e:  # noqa: F841
            raise InvalidArgument(e)

        cfg = hydra_zen.builds(
            HuggingFaceObjectDetector,
            model=det_model,
            preprocessor=processor,
            threshold=threshold,
            populate_full_signature=True,
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, HuggingFaceObjectDetector)
        return cfg

    raise NotImplementedError(f"Unsupported task {task}.")


def _get_tv_model_builder(
    *,
    task: SUPPORTED_TASKS,
    model_name: str,
    **kwargs: Any,
) -> Type[Builds[Union[Type[TorchVisionClassifier], Type[TorchVisionObjectDetector]]]]:
    """
    Create a builder for a TorchVision model.

    Parameters
    ----------
    task : str
        The task of the model.
    model_name : str
        The name of the model.
    **kwargs : Any
        Additional keyword arguments to pass to the model builder.

    Returns
    -------
    Type[Builds[Union[Type[TorchVisionClassifier], Type[TorchVisionObjectDetector]]]]
        The model builder.

    Raises
    ------
    ImportError
        If TorchVision is not installed.

    Notes
    -----
    This function is a wrapper around `torchvision.models` from TorchVision.

    See Also
    --------
    torchvision.models : The function that this function wraps.
    """
    if not is_torchvision_available():  # pragma: no cover
        raise ImportError("TorchVision is not installed.")

    if "image-classification" in task:
        cfg = hydra_zen.builds(
            TorchVisionClassifier.from_pretrained,
            name=model_name,
            **kwargs,
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, TorchVisionClassifier)
        return cfg  # type: ignore

    if "object-detection" in task:
        cfg = hydra_zen.builds(
            TorchVisionObjectDetector.from_pretrained,
            name=model_name,
            **kwargs,
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, TorchVisionObjectDetector)
        return cfg  # type: ignore

    raise NotImplementedError(
        f"Unknown task {task}. Supported tasks are {SUPPORTED_TASKS}."
    )


def _get_torchmetrics_metric_builder(
    metric_name: str,
    **kwargs: Any,
) -> Type[Builds[Type[Metric]]]:
    """
    Load a metric builder from TorchMetrics.

    Parameters
    ----------
    metric_name : str
        The name of the metric to load.
    **kwargs : Any
        Additional keyword arguments to pass to the metric builder.

    Returns
    -------
    Type[Builds[Type[Metric]]]
        The metric builder.
    """
    if not is_torchmetrics_available():  # pragma: no cover
        raise ImportError("TorchMetrics is not installed.")

    import torchmetrics

    if metric_name == "MeanAveragePrecision":
        from ..torchmetrics.detection import MeanAveragePrecision

        cfg = hydra_zen.builds(
            MeanAveragePrecision, populate_full_signature=True, **kwargs
        )
        cfg.__doc__ = get_dataclass_docstring(cfg, MeanAveragePrecision)
        return cfg  # type: ignore

    else:
        assert (
            metric_name in torchmetrics.__all__
        ), f"{metric_name} not found in torchmetrics"

        tm_clazz = importlib.import_module("torchmetrics")
        clazz = getattr(tm_clazz, metric_name)
        cfg = hydra_zen.builds(clazz, populate_full_signature=True, **kwargs)
        cfg.__doc__ = get_dataclass_docstring(cfg, clazz)
        return cfg


def _get_torcheval_metric_builder(
    metric_name: str,
    **kwargs: Any,
) -> Type[Builds[Type[Metric]]]:
    """
    Load a metric builder from TorchEval.

    Parameters
    ----------
    metric_name : str
        The name of the metric to load.

    Returns
    -------
    Type[Builds[Type[Metric]]]
        The metric builder.
    """
    if not is_torcheval_available():  # pragma: no cover
        raise ImportError("TorchEval is not installed.")

    import importlib

    import torcheval.metrics as metrics

    assert metric_name in metrics.__all__, f"{metric_name} not found in torcheval"
    tm_clazz = importlib.import_module("torcheval.metrics")
    clazz = getattr(tm_clazz, metric_name)
    cfg = hydra_zen.builds(clazz, populate_full_signature=True, **kwargs)
    cfg.__doc__ = get_dataclass_docstring(cfg, clazz)
    return cfg


def get_dataset_builder(
    *,
    provider: Literal["torchvision", "huggingface"],
    dataset_name: str,
    task: SUPPORTED_TASKS,
    split: Optional[str] = None,
    **kwargs: Any,
) -> Type[Builds[Type[Dataset]]]:
    """
    Construct a builder class for a dataset.

    Parameters
    ----------
    provider : str
        The provider of the dataset.
    dataset_name : str
        The name of the dataset.
    task : str
        The task of the dataset.
    split : str, optional
        The split of the dataset, by default None.
    **kwargs : Any
        Additional keyword arguments to pass to the builder.

    Returns
    -------
    Type[Builds[Type[Dataset]]]
        The builder class.

    Raises
    ------
    NotImplementedError
        If the provider is not supported.
    """
    if provider == "huggingface":
        return _get_hf_dataset_builder(
            dataset_name=dataset_name, task=task, split=split, **kwargs
        )
    elif provider == "torchvision":
        return _get_tv_dataset_builder(
            dataset_name=dataset_name, task=task, split=split, **kwargs
        )

    raise NotImplementedError(
        f"Unknown provider {provider}. Supported providers are ['torchvision', 'huggingface']."
    )


def get_model_builder(
    *,
    provider: Literal["torchvision", "huggingface"],
    model_name: str,
    task: SUPPORTED_TASKS,
    **kwargs: Any,
) -> Type[Builds[Union[Type[ImageClassifier], Type[ObjectDetector]]]]:
    """
    Construct a builder class for a model.

    Parameters
    ----------
    provider : str
        The provider of the model.
    model_name : str
        The name of the model.
    task : str
        The task of the model.
    **kwargs : Any
        Keyword arguments for the model builders.

    Returns
    -------
    Type[Builds]
        The builder class.

    Raises
    ------
    NotImplementedError
        If the provider is not supported.
    """
    if provider == "huggingface":
        return _get_hf_model_builder(model_name=model_name, task=task, **kwargs)
    elif provider == "torchvision":
        return _get_tv_model_builder(model_name=model_name, task=task, **kwargs)

    raise NotImplementedError(
        f"Unknown provider {provider}. Supported providers are ['torchvision', 'huggingface']."
    )


def get_metric_builder(
    *,
    provider: Literal["torchmetrics", "torcheval"],
    metric_name: str,
    **kwargs: Any,
) -> Type[Builds[Type[Metric]]]:
    """
    Construct a builder class for a metric.

    Parameters
    ----------
    provider : str
        The provider of the metric.
    metric_name : str
        The name of the metric.
    **kwargs : Any
        Keyword arguments for the metric builders.

    Returns
    -------
    Type[Builds]
        The builder class.

    Raises
    ------
    NotImplementedError
        If the provider is not supported.
    """
    if provider == "torchmetrics":
        return _get_torchmetrics_metric_builder(metric_name=metric_name, **kwargs)
    elif provider == "torcheval":
        return _get_torcheval_metric_builder(metric_name=metric_name, **kwargs)

    raise NotImplementedError(
        f"Unknown provider {provider}. Supported providers are ['torchmetrics', 'torcheval']."
    )
