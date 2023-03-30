from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from torch import Tensor, nn
from typing_extensions import Self

from jatic_toolbox.protocols import ArrayLike, Classifier, ObjectDetector

__all__ = ["TorchVisionClassifier", "TorchVisionObjectDetector"]


@dataclass
class TorchVisionClassifierOutput:
    logits: Tensor


class TorchVisionClassifier(nn.Module, Classifier[ArrayLike]):
    """
    Wrapper for torchvision image classification models.

    Parameters
    ----------
    model : nn.Module
        TorchVision model.

    Methods
    -------
    list_models(module: Optional[Any] = None) -> Iterable[Any]
        List all available models.

    from_pretrained(name: str, weights: Optional[Union[Callable, str]] = None, **config: Any) -> Self
        Load a pretrained model.

    Examples
    --------
    >>> from jatic_toolbox.interop.torchvision import TorchVisionClassifier
    >>> model = TorchVisionClassifier.from_pretrained("resnet18")
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize a TorchVisionClassifier."""
        super().__init__()
        self._model = model

    def forward(self, data: Tensor) -> TorchVisionClassifierOutput:
        logits = self._model(data)
        return TorchVisionClassifierOutput(logits=logits)

    @classmethod
    def list_models(cls, module: Optional[Any] = None) -> Iterable[Any]:
        from torchvision import models
        from torchvision.models import list_models

        if module is None:
            module = models

        return list_models(module=module)

    @classmethod
    def from_pretrained(
        cls, name: str, weights: Optional[Union[Callable, str]] = None, **config: Any
    ) -> Self:
        from torchvision.models import get_model, get_model_weights

        if weights is None:
            weights = name

        model_weights = get_model_weights(name=weights)
        config["weights"] = model_weights
        model = get_model(name=name, **config)
        return cls(model)


@dataclass
class TorchVisionObjectDetectorOutput:
    boxes: Sequence[Tensor]
    scores: Sequence[Tensor]
    labels: Sequence[Tensor]
    masks: Sequence[Tensor]


class TorchVisionObjectDetector(nn.Module, ObjectDetector[ArrayLike]):
    """
    Wrapper for torchvision object detection models.

    Parameters
    ----------
    model : nn.Module
        TorchVision model.

    Methods
    -------
    list_models(module: Optional[Any] = None) -> Iterable[Any]
        List all available models.

    from_pretrained(name: str, weights: Optional[Union[Callable, str]] = None, **config: Any) -> Self
        Load a pretrained model.

    Examples
    --------
    >>> from jatic_toolbox.interop.torchvision import TorchVisionObjectDetector
    >>> model = TorchVisionObjectDetector.from_pretrained("maskrcnn_resnet50_fpn")
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(self, data: Tensor) -> TorchVisionObjectDetectorOutput:
        outputs = self._model(data)
        all_boxes: Sequence[Tensor] = []
        all_scores: Sequence[Tensor] = []
        all_labels: Sequence[Tensor] = []
        all_masks: Sequence[Tensor] = []
        # dict_keys(['boxes', 'labels', 'scores', 'masks'])
        for output in outputs:
            all_boxes.append(output["boxes"])
            all_labels.append(output["labels"])
            all_scores.append(output["scores"])
            all_masks.append(output["masks"])

        return TorchVisionObjectDetectorOutput(
            boxes=all_boxes, scores=all_scores, labels=all_labels, masks=all_masks
        )

    @classmethod
    def list_models(cls, module: Optional[Any] = None) -> Iterable[Any]:
        from torchvision import models
        from torchvision.models import list_models

        if module is None:
            module = models.detection

        return list_models(module=module)

    @classmethod
    def from_pretrained(
        cls, name: str, weights: Optional[Union[Callable, str]] = None, **config: Any
    ) -> Self:
        from torchvision.models import get_model, get_model_weights

        if weights is None:
            weights = name

        model_weights = get_model_weights(name=weights)
        config["weights"] = model_weights
        model = get_model(name=name, **config)
        return cls(model)
