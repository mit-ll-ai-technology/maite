# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

from torch import Tensor, nn
from typing_extensions import Self, TypeAlias

from maite.errors import InvalidArgument
from maite.protocols import HasDataImage, SupportsArray

from ..base_model import BaseModel

__all__ = ["TorchVisionClassifier", "TorchVisionObjectDetector"]


TorchVisionProcessor: TypeAlias = Callable[[SupportsArray], Tensor]


@dataclass
class TorchVisionClassifierOutput:
    logits: Tensor


@dataclass
class TorchVisionObjectDetectorOutput:
    boxes: Sequence[Tensor]
    scores: Sequence[Tensor]
    labels: Sequence[Union[Tensor, Sequence[str]]]


class TorchVisionBase(nn.Module, BaseModel):
    def __init__(
        self,
        model: nn.Module,
        processor: Optional[TorchVisionProcessor] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a TorchVisionClassifier."""
        super().__init__()
        self._model = model
        self._processor = processor
        self._labels = labels

    def get_labels(self) -> Sequence[str]:
        """Get labels."""
        if self._labels is None:  # pragma: no cover
            raise InvalidArgument("No labels were provided.")
        return self._labels

    def preprocessor(
        self,
        data: SupportsArray,
    ) -> HasDataImage:
        if self._processor is None:  # pragma: no cover
            raise InvalidArgument("No processor was provided.")

        if isinstance(data, Sequence):
            return {"image": [self._processor(i) for i in data]}

        return {"image": self._processor(data)}  # pragma: no cover

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

        try:
            model_weights = get_model_weights(name=weights)
        except ValueError as e:
            raise InvalidArgument(f"Invalid model name: {name}") from e

        assert issubclass(
            model_weights, WeightsEnum
        ), f"{type(model_weights)} is not a valid model"
        keys = model_weights.__members__.keys()
        assert weights_value_name in keys, f"{weights_value_name} not in {keys}"
        the_model_weights = model_weights[weights_value_name]

        processor = None
        if with_processor:
            processor = the_model_weights.transforms()

        labels = the_model_weights.meta["categories"]
        config["weights"] = the_model_weights
        model = get_model(name=name, **config)
        return cls(model, processor, labels)


class TorchVisionClassifier(TorchVisionBase):
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
    >>> from maite.interop.torchvision import TorchVisionClassifier
    >>> model = TorchVisionClassifier.from_pretrained("resnet18")
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Optional[TorchVisionProcessor] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a TorchVisionClassifier."""
        super().__init__(model, processor, labels)

    def forward(
        self, data: Union[HasDataImage, SupportsArray]
    ) -> TorchVisionClassifierOutput:
        images, _ = self._process_inputs(data)
        logits = self._model(images)
        return TorchVisionClassifierOutput(logits=logits)


class TorchVisionObjectDetector(TorchVisionBase):
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
    >>> from maite.interop.torchvision import TorchVisionObjectDetector
    >>> model = TorchVisionObjectDetector.from_pretrained("maskrcnn_resnet50_fpn")
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Optional[TorchVisionProcessor] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(model, processor, labels)

    def forward(
        self, data: Union[HasDataImage, SupportsArray]
    ) -> TorchVisionObjectDetectorOutput:
        images, _ = self._process_inputs(data)

        outputs = self._model(images)

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
