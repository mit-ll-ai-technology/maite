from dataclasses import dataclass
from typing import Any, Generic, Iterable, List, Optional, Sequence, TypeVar, Union

import torch as tr
from torch import Tensor
from typing_extensions import Self

from jatic_toolbox._internals.interop.utils import to_tensor_list
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ArrayLike, ObjectDetector

from .typing import HuggingFaceProcessor, HuggingFaceWithDetection

__all__ = ["HuggingFaceObjectDetector"]

T = TypeVar("T", bound=ArrayLike)


@dataclass
class HuggingFaceObjectDetectionOutput(Generic[T]):
    boxes: Sequence[T]
    labels: Sequence[T]
    scores: Sequence[T]


class HuggingFaceObjectDetector(ObjectDetector[Tensor]):
    """
    Wrapper for HuggingFace object detection models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForObjectDetection`
    to load the HuggingFace models.
    """

    def __init__(
        self,
        processor: HuggingFaceProcessor,
        model: HuggingFaceWithDetection,
        threshold: float = 0.5,
    ) -> None:
        """
        Initialize HuggingFaceObjectDetector.

        Parameters
        ----------
        processor : Callable[[Sequence[ArrayLike]], BatchFeature]
            A HuggingFace feature extractor for a given model.

        model : Callable[[Tensor, ...], HasObjectDetections]
            A HuggingFace object detection model.

        Examples
        --------
        >>> from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
        >>> processor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        >>> hf_model = HuggingFaceObjectDetector(processor, model)
        """
        super().__init__()
        self.processor = processor
        self.model = model
        self.threshold = threshold

    @classmethod
    def list_models(
        cls, task: Optional[str] = "object-detection", **kwargs: Any
    ) -> Iterable[Any]:  # pragma: no cover
        from huggingface_hub.hf_api import HfApi
        from huggingface_hub.utils.endpoint_helpers import ModelFilter

        hf_api = HfApi()
        filt = ModelFilter(task=task, **kwargs)
        models = hf_api.list_models(filter=filt)
        return [m.modelId for m in models]

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> Self:  # pragma: no cover
        """
        Load a HuggingFace model from pretrained weights.

        Uses `AutoFeatureExtractor` and `AutoModelForObjectDetection`.

        Parameters
        ----------
        model : str
            The `model id` of a pretrained object detector from HuggingFace.

        **kwargs : Any
            Keyword arguments for HuggingFace AutoFeatureExtractor and AutoModelForObjectDetection.

        Returns
        -------
        HuggingFaceObjectDetector
            The JATIC Toolbox wrapper for a HuggingFace object detector.

        Examples
        --------
        >>> hf_image_classifier = HuggingFaceObjectDetector.from_pretrained(model="facebook/detr-resnet-50")
        """
        from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

        processor: HuggingFaceProcessor
        det_model: HuggingFaceWithDetection

        try:
            processor = AutoFeatureExtractor.from_pretrained(model, **kwargs)
            det_model = AutoModelForObjectDetection.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        return cls(processor, det_model)

    def __call__(
        self, data: Union[ArrayLike, Sequence[ArrayLike]]
    ) -> HuggingFaceObjectDetectionOutput[Tensor]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : ArrayLike
            An array of images.  Inputs can be `NDArray` or `torch.Tensor`.

        Returns
        -------
        HuggingFaceObjectDetectionOutput
            An object detection object containing bounding boxes with corresponding scores.

        Examples
        --------
        First create a random NumPy image array:

        >>> import numpy as np
        >>> image = np.random.uniform(0, 255, size=(200, 200, 3))

        Load a HuggingFace object detection model and execute on
        the above image:

        >>> hf_object_detector = HuggingFaceObjectDetector.from_pretrained(model="facebook/detr-resnet-50")
        >>> detections = hf_object_detector([image])

        We can check to verify the output contains `boxes` and `scores` attributes:

        >>> from jatic_toolbox.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        data = to_tensor_list(data)
        features = self.processor(images=data, return_tensors="pt")
        outputs = self.model(**features)

        target_sizes = tr.IntTensor([tuple(img.shape[:2]) for img in data])
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )

        output_labels: List[Tensor] = []
        output_scores: List[Tensor] = []
        output_boxes: List[Tensor] = []
        for i in range(len(results)):
            boxes = results[i]["boxes"]
            scores = results[i]["scores"]
            labels = results[i]["labels"]

            output_boxes.append(boxes)
            output_scores.append(scores)
            output_labels.append(labels)

        return HuggingFaceObjectDetectionOutput(
            boxes=output_boxes, labels=output_labels, scores=output_scores
        )
