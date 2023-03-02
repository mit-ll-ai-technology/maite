from collections import UserDict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, TypeVar, Union

import numpy as np
import torch as tr
from torch import Tensor
from typing_extensions import Protocol, Self, TypeAlias

from jatic_toolbox._internals.protocols import HasObjectDetections
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ArrayLike, ObjectDetector
from jatic_toolbox.utils.validation import check_type

__all__ = ["HuggingFaceObjectDetector"]


T = TypeVar("T", bound=ArrayLike)

HFProcessedDetection: TypeAlias = List[
    Dict[Literal["scores", "labels", "boxes"], tr.Tensor]
]


class BatchFeature(UserDict[str, T]):
    ...


class HuggingFaceProcessor(Protocol[T]):
    def __call__(
        self,
        images: Sequence[T],
        return_tensors: Union[bool, str] = "pt",
        **kwargs: Any,
    ) -> BatchFeature[T]:
        ...

    def post_process_object_detection(
        self, outputs: HasObjectDetections[T], threshold: float, target_sizes: Any
    ) -> HFProcessedDetection:
        ...


class HuggingFaceModel(Protocol[T]):
    def __call__(self, pixel_values: T, **kwargs: Any) -> HasObjectDetections[T]:
        ...


@dataclass
class HuggingFaceObjectDetectionOutput:
    boxes: List[Tensor]
    labels: List[Tensor]
    scores: List[Tensor]


class HuggingFaceObjectDetector(ObjectDetector[Tensor]):
    """
    Wrapper for HuggingFace object detection models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForObjectDetection`
    to load the HuggingFace models.
    """

    def __init__(
        self,
        processor: HuggingFaceProcessor[Tensor],
        model: HuggingFaceModel[Tensor],
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

        processor: HuggingFaceProcessor[Tensor]
        det_model: HuggingFaceModel[Tensor]

        try:
            processor = AutoFeatureExtractor.from_pretrained(model, **kwargs)
            det_model = AutoModelForObjectDetection.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        return cls(processor, det_model)

    def __call__(self, data: Sequence[Tensor]) -> HuggingFaceObjectDetectionOutput:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : Sequence[ArrayLike]
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

        >>> hf_object_detector = HuggingFaceObjectDetector.from_pretrained(model="facebook/detr-resnet-50")
        >>> detections = hf_object_detector([image])

        We can check to verify the output contains `boxes` and `scores` attributes:

        >>> from jatic_toolbox.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        arr_iter: List[tr.Tensor] = []
        for img in data:
            check_type("img", img, (np.ndarray, tr.Tensor))
            if isinstance(img, tr.Tensor):
                arr_iter.append(img)
            else:
                arr_iter.append(tr.as_tensor(img))

        with tr.no_grad():
            inputs = self.processor(images=arr_iter, return_tensors="pt")
            outputs = self.model(**inputs)

            target_sizes = tr.IntTensor(
                [[img.shape[0], img.shape[1]] for img in arr_iter]
            )
            results: HFProcessedDetection = (
                self.processor.post_process_object_detection(
                    outputs, threshold=self.threshold, target_sizes=target_sizes
                )
            )

        output_labels: List[Tensor] = []
        output_scores: List[Tensor] = []
        output_boxes: List[Tensor] = []
        for i in range(len(data)):  # pragma: no cover
            output_boxes.append(results[i]["boxes"])
            output_scores.append(results[i]["scores"])
            output_labels.append(results[i]["labels"])

        return HuggingFaceObjectDetectionOutput(
            boxes=output_boxes, labels=output_labels, scores=output_scores
        )
