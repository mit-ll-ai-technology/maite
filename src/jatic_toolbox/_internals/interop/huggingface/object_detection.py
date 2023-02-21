import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence

import numpy as np
import torch as tr
from numpy.typing import NDArray
from PIL.Image import Image
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from typing_extensions import TypeAlias

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ImageType, ObjectDetector, ShapedArray
from jatic_toolbox.utils.validation import check_type

__all__ = ["HuggingFaceObjectDetector"]

HFProcessedDetection: TypeAlias = List[
    Dict[Literal["scores", "labels", "boxes"], tr.Tensor]
]


@dataclass
class HuggingFaceObjectDetectionOutput:
    boxes: List[NDArray]
    scores: List[List[Dict[str, float]]]


class HuggingFaceObjectDetector(ObjectDetector):
    """
    Wrapper for HuggingFace object detection models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForObjectDetection`
    to load the HuggingFace models.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """
        Initialize HuggingFaceObjectDetector.

        Parameters
        ----------
        model : str
            The `model id` of a pretrained object detector stored on HuggingFace.

        **kwargs : Any
            Keyword arguments for HuggingFace AutoFeatureExtractor and AutoModelForObjectDetection.

        Examples
        --------
        >>> hf_object_detector = HuggingFaceObjectDetector(model="facebook/detr-resnet-50")
        """
        super().__init__()
        check_type("model", model, str)

        self._model = model

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model, **kwargs
            )
            self.model = AutoModelForObjectDetection.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

    def __call__(self, data: Sequence[ImageType]) -> HuggingFaceObjectDetectionOutput:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : Sequence[ImageType]
            An array of images.  Inputs can be `PIL.Image`, `NDArray`, or `torch.Tensor`
            but HuggingFace converts all types to NumPy for feature extraction.

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

        >>> hf_object_detector = HuggingFaceObjectDetector(model="facebook/detr-resnet-50")
        >>> detections = hf_object_detector([image])

        We can check to verify the output contains `boxes` and `scores` attributes:

        >>> from jatic_toolbox.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        arr_iter: List[ShapedArray] = []
        for img in data:
            check_type("img", img, (Image, np.ndarray, tr.Tensor))
            if isinstance(img, tr.Tensor):
                warnings.warn(
                    "HuggingFace feature extractors convert input data to NumPy arrays (input data type: `torch.Tensor`)"
                )

            arr_iter.append(np.asarray(img))

        with tr.no_grad():
            inputs = self.feature_extractor(images=arr_iter, return_tensors="pt")
            outputs = self.model(**inputs)

            target_sizes = tr.IntTensor(
                [[img.shape[0], img.shape[1]] for img in arr_iter]
            )
            results: HFProcessedDetection = (
                self.feature_extractor.post_process_object_detection(
                    outputs, target_sizes=target_sizes
                )
            )
            scores: tr.Tensor = tr.softmax(outputs.logits, dim=-1).numpy()

        output_scores: List[List[Dict[str, float]]] = []
        output_boxes: List[NDArray] = []
        for i in range(len(data)):  # pragma: no cover
            output_boxes.append(results[i]["boxes"].numpy())
            output_scores.append(self._process_scores(scores[i]))

        return HuggingFaceObjectDetectionOutput(
            boxes=output_boxes, scores=output_scores
        )

    def _process_scores(self, scores) -> List[Dict[str, float]]:
        """
        Process scores.

        Added to implement test.
        """
        scores_i = []
        for j in range(len(scores)):
            scores_i.append({k: scores[j][k].item() for k in range(len(scores[j]))})
        return scores_i
