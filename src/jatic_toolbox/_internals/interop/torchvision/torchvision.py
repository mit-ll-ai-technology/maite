from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

from torch import Tensor, nn
from typing_extensions import Self, TypeAlias

from jatic_toolbox._internals.interop.utils import to_tensor_list
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import (
    ArrayLike,
    HasDataImage,
    ImageClassifier,
    ObjectDetector,
    is_list_dict,
)

__all__ = ["TorchVisionClassifier", "TorchVisionObjectDetector"]


TorchVisionProcessor: TypeAlias = Callable[[Sequence[ArrayLike]], Tensor]


@dataclass
class TorchVisionClassifierOutput:
    logits: Tensor


@dataclass
class TorchVisionObjectDetectorOutput:
    boxes: Sequence[Tensor]
    scores: Sequence[Tensor]
    labels: Sequence[Union[Tensor, Sequence[str]]]


class TorchVisionBase(nn.Module):
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
        if self._labels is None:
            raise InvalidArgument("No labels were provided.")
        return self._labels

    @overload
    def preprocessor(
        self,
        data: Sequence[ArrayLike],
        image_key: str = "image",
    ) -> HasDataImage:
        ...

    @overload
    def preprocessor(
        self,
        data: Sequence[HasDataImage],
        image_key: str = "image",
    ) -> Sequence[HasDataImage]:
        ...

    def preprocessor(
        self,
        data: Union[Sequence[ArrayLike], Sequence[HasDataImage]],
        image_key: str = "image",
    ) -> Union[HasDataImage, Sequence[HasDataImage]]:
        if self._processor is None:
            raise InvalidArgument("No processor was provided.")

        if is_list_dict(data):
            out = []
            for d in data:
                data_out = {"image": self._processor(d[image_key])}
                data_out.update({k: v for k, v in d.items() if k != image_key})
                out.append(data_out)

            return out
        else:
            if TYPE_CHECKING:
                data = cast(Sequence[ArrayLike], data)

            images = to_tensor_list(data)
            return {"image": self._processor(images)}

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

        labels = the_model_weights.meta["categories"]
        config["weights"] = the_model_weights
        model = get_model(name=name, **config)
        return cls(model, processor, labels)


class TorchVisionClassifier(TorchVisionBase, ImageClassifier):
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
        self,
        model: nn.Module,
        processor: Optional[TorchVisionProcessor] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a TorchVisionClassifier."""
        super().__init__(model, processor, labels)

    def forward(
        self, data: Union[Mapping[str, ArrayLike], Sequence[ArrayLike], ArrayLike]
    ) -> TorchVisionClassifierOutput:
        if isinstance(data, dict):
            if "image" in data:
                pixel_values = data["image"]
            elif "pixel_values" in data:
                pixel_values = data["pixel_values"]
            else:
                raise InvalidArgument(
                    f"Expected 'image' or 'pixel_values' in data, got {data.keys()}"
                )
        else:
            pixel_values = data

        logits = self._model(pixel_values)
        return TorchVisionClassifierOutput(logits=logits)


class TorchVisionObjectDetector(TorchVisionBase, ObjectDetector):
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
        self,
        model: nn.Module,
        processor: Optional[TorchVisionProcessor] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(model, processor, labels)

    def forward(
        self, data: Union[Mapping[str, ArrayLike], Sequence[ArrayLike], ArrayLike]
    ) -> TorchVisionObjectDetectorOutput:
        if isinstance(data, dict):
            if "image" in data:
                pixel_values = data["image"]
            elif "pixel_values" in data:
                pixel_values = data["pixel_values"]
            else:
                raise InvalidArgument(
                    f"Expected 'image' or 'pixel_values' in data, got {data.keys()}"
                )
        else:
            pixel_values = data

        outputs = self._model(pixel_values)

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
