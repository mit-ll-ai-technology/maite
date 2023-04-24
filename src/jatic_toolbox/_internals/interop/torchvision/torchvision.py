from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import torch as tr
from torch import Tensor, nn
from typing_extensions import Self, TypeAlias

from jatic_toolbox._internals.interop.utils import to_tensor_list
from jatic_toolbox.protocols import ArrayLike, Classifier, ObjectDetector

__all__ = ["TorchVisionClassifier", "TorchVisionObjectDetector"]


TorchVisionProcessor: TypeAlias = Callable[[ArrayLike], Tensor]


@dataclass
class TorchVisionClassifierOutput:
    logits: Tensor


@dataclass
class TorchVisionObjectDetectorOutput:
    boxes: Sequence[Tensor]
    scores: Sequence[Tensor]
    labels: Sequence[Tensor]


class TorchVisionBase(nn.Module):
    def __init__(
        self, model: nn.Module, processor: Optional[TorchVisionProcessor] = None
    ) -> None:
        """Initialize a TorchVisionClassifier."""
        super().__init__()
        self._model = model
        self._processor = processor

    @property
    def device(self) -> tr.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        weights: Optional[str] = None,
        with_processor: bool = True,
        weights_value_name: str = "DEFAULT",
        **config: Any,
    ) -> Self:
        from torchvision.models._api import WeightsEnum, get_model, get_model_weights

        if weights is None:
            weights = name

        model_weights = get_model_weights(name=weights)

        assert issubclass(
            model_weights, WeightsEnum
        ), f"{type(model_weights)} is not a valid model"
        keys = model_weights.__members__.keys()
        assert weights_value_name in keys, f"{weights_value_name} not in {keys}"
        the_model_weights = model_weights[weights_value_name]

        processor = None
        if with_processor:
            processor = the_model_weights.transforms()

        config["weights"] = the_model_weights
        model = get_model(name=name, **config)
        return cls(model, processor)


class TorchVisionClassifier(TorchVisionBase, Classifier[ArrayLike]):
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

    def __init__(
        self, model: nn.Module, processor: Optional[TorchVisionProcessor] = None
    ) -> None:
        """Initialize a TorchVisionClassifier."""
        super().__init__(model, processor)

    def forward(
        self, data: Union[Sequence[ArrayLike], Tensor]
    ) -> TorchVisionClassifierOutput:
        if self._processor is not None:
            data = to_tensor_list(data)
            data = [self._processor(d).to(self.device) for d in data]

        logits = self._model(data)
        return TorchVisionClassifierOutput(logits=logits)

    @classmethod
    def list_models(
        cls, module: Optional[Any] = None
    ) -> Iterable[Any]:  # pragma: no cover
        from torchvision import models
        from torchvision.models import list_models

        if module is None:
            module = models

        return list_models(module=module)


class TorchVisionObjectDetector(TorchVisionBase, ObjectDetector[ArrayLike]):
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

    def __init__(
        self, model: nn.Module, processor: Optional[TorchVisionProcessor] = None
    ) -> None:
        super().__init__(model, processor)

    def forward(
        self, data: Union[Sequence[ArrayLike], Tensor]
    ) -> TorchVisionObjectDetectorOutput:
        if self._processor is not None:
            data = to_tensor_list(data)
            data = [self._processor(d).to(self.device) for d in data]

        outputs = self._model(data)

        all_boxes: Sequence[Tensor] = []
        all_scores: Sequence[Tensor] = []
        all_labels: Sequence[Tensor] = []

        for output in outputs:
            all_boxes.append(output["boxes"])
            all_labels.append(output["labels"])
            all_scores.append(output["scores"])

        return TorchVisionObjectDetectorOutput(
            boxes=all_boxes, scores=all_scores, labels=all_labels
        )

    @classmethod
    def list_models(
        cls, module: Optional[Any] = None
    ) -> Iterable[Any]:  # pragma: no cover
        from torchvision import models
        from torchvision.models import list_models

        if module is None:
            module = models.detection

        return list_models(module=module)
