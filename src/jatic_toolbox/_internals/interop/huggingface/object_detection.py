from typing import Any, Iterable, List, Optional, Sequence, TypeVar, Union

import numpy as np
import torch as tr
from torch import nn
from typing_extensions import Self

from jatic_toolbox._internals.interop.utils import to_tensor_list
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import (
    ArrayLike,
    HasDetectionLogits,
    HasObjectDetections,
    ObjectDetector,
)

from .typing import (
    HuggingFaceObjectDetectionOutput,
    HuggingFaceObjectDetectionPostProcessor,
    HuggingFaceProcessor,
    HuggingFaceWithDetection,
)

__all__ = ["HuggingFaceObjectDetector"]

T = TypeVar("T", bound=ArrayLike)


class HuggingFaceObjectDetector(nn.Module, ObjectDetector[ArrayLike]):
    """
    Wrapper for HuggingFace object detection models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForObjectDetection`
    to load the HuggingFace models.
    """

    def __init__(
        self,
        model: HuggingFaceWithDetection,
        processor: Optional[HuggingFaceProcessor] = None,
        post_processor: Optional[HuggingFaceObjectDetectionPostProcessor] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Initialize HuggingFaceObjectDetector.

        Parameters
        ----------
        model : Callable[[Tensor, ...], HasObjectDetections]
            A HuggingFace object detection model.

        processor : Callable[[Sequence[ArrayLike]], BatchFeature]
            A HuggingFace feature extractor for a given model.

        post_processor : Callable[[HasObjectDetections, float, Any], HFProcessedDetection]
            A HuggingFace post processor for a given model.

        Examples
        --------
        >>> from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
        >>> processor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        >>> hf_model = HuggingFaceObjectDetector(processor, model)
        """
        super().__init__()
        self.model = model
        self.processor = processor
        self.post_processor = post_processor
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
    def from_pretrained(
        cls,
        model: str,
        *,
        with_processor: bool = True,
        with_post_processor: bool = True,
        **kwargs: Any,
    ) -> Self:  # pragma: no cover
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
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor: Optional[HuggingFaceProcessor]
        det_model: HuggingFaceWithDetection

        try:
            det_model = AutoModelForObjectDetection.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        if with_processor or with_post_processor:
            try:
                processor = AutoImageProcessor.from_pretrained(model, **kwargs)
            except OSError as e:  # noqa: F841
                raise InvalidArgument(e)

            if with_processor and with_post_processor:
                return cls(
                    det_model,
                    processor,
                    post_processor=processor.post_process_object_detection,
                )
            elif not with_post_processor:
                return cls(det_model, processor)
            else:
                return cls(
                    det_model, post_processor=processor.post_process_object_detection
                )

        return cls(det_model)

    def forward(
        self, data: Union[ArrayLike, Sequence[ArrayLike]]
    ) -> Union[HasObjectDetections[ArrayLike], HasDetectionLogits[ArrayLike]]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : Union[ArrayLike, Sequence[ArrayLike]]
            A single image or a sequence of images to extract object detection from.

        Returns
        -------
        HuggingFaceObjectDetectionOutput[Tensor]
            A dataclass containing the object detection results.

        Raises
        ------
        NotImplementedError
            If the model does not have a processor.

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
        if self.processor is None:
            data = tr.as_tensor(data)
            target_sizes = tr.IntTensor([tuple(img.shape[:2]) for img in data])
            outputs = self.model(data)
        else:
            data = to_tensor_list(data)
            target_sizes = tr.IntTensor(
                [tuple(np.asarray(img).shape[:2]) for img in data]
            )
            features = self.processor(images=data, return_tensors="pt")
            features.to(self.model.device)

            outputs = self.model(**features)

        if self.post_processor is None:
            return outputs

        assert isinstance(outputs, HasDetectionLogits)
        results = self.post_processor(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )

        if isinstance(results, list):
            output_labels: List[tr.Tensor] = []
            output_scores: List[tr.Tensor] = []
            output_boxes: List[tr.Tensor] = []
            for result in results:
                boxes = tr.as_tensor(result["boxes"])
                scores = tr.as_tensor(result["scores"])
                labels = tr.as_tensor(result["labels"])

                output_boxes.append(boxes)
                output_scores.append(scores)
                output_labels.append(labels)

            return HuggingFaceObjectDetectionOutput(
                boxes=output_boxes, labels=output_labels, scores=output_scores
            )
        else:
            assert isinstance(results, HasObjectDetections)
            return results
