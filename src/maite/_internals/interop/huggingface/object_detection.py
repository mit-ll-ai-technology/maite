# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tr
from typing_extensions import Self

from maite.errors import InvalidArgument
from maite.protocols import HasDataImage, SupportsArray

from .base import BaseHFModel
from .typing import (
    HuggingFaceDetectorOutput,
    HuggingFaceDetectorPredictions,
    HuggingFaceObjectDetectionPostProcessor,
    HuggingFacePostProcessorInput,
    HuggingFaceProcessor,
    HuggingFaceWithDetection,
)

__all__ = ["HuggingFaceObjectDetector"]


class HuggingFaceObjectDetector(BaseHFModel):
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
        threshold: Optional[float] = 0.5,
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

        threshold : float, optional
            The threshold for the model, by default 0.5.

        Examples
        --------
        >>> from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
        >>> processor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        >>> hf_model = HuggingFaceObjectDetector(processor, model)
        """
        super().__init__(
            model=model, processor=processor, post_processor=post_processor
        )
        self._threshold = threshold

    def preprocessor(
        self,
        data: SupportsArray,
    ) -> Union[HasDataImage, Sequence[HasDataImage]]:
        """
        Preprocess images for a HuggingFace object detector.

        Parameters
        ----------
        images : Sequence[ArrayLike]
            The images to preprocess.

        Returns
        -------
        tr.Tensor
            The preprocessed images.

        Examples
        --------
        """
        if self._processor is None:  # pragma: no cover
            raise InvalidArgument("No processor was provided.")

        assert isinstance(data, (list, tuple))
        target_sizes = [tuple(np.asarray(img).shape[:2]) for img in data]
        image_features = self._processor(images=data, return_tensors="pt")[
            "pixel_values"
        ]

        out = HasDataImage(image=image_features)
        out["target_size"] = target_sizes  # type: ignore
        return out

    def post_processor(
        self,
        model_outputs: HuggingFaceDetectorOutput,
        threshold: float = 0.5,
        target_sizes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> HuggingFaceDetectorPredictions:
        """
        Post process the outputs of a HuggingFace object detector.

        Parameters
        ----------
        model_outputs : HuggingFaceDetectorOutput
            The outputs of the model.
        threshold : float
            The threshold to use for the post processing.
        target_sizes : Sequence[Tuple[int, ...]]
            The shape of each image.

        Returns
        -------
        HuggingFaceDetectorPredictions
            The post processed outputs.

        Examples
        --------
        """
        assert self._post_processor is not None, "No post processor was provided."

        pp_input = HuggingFacePostProcessorInput(
            logits=model_outputs.logits, pred_boxes=model_outputs.boxes
        )
        results = self._post_processor(
            pp_input, threshold=threshold, target_sizes=target_sizes
        )

        if isinstance(results, list):
            output_labels: List[tr.Tensor] = []
            output_scores: List[tr.Tensor] = []
            output_boxes: List[tr.Tensor] = []
            for result in results:
                boxes = result["boxes"]
                scores = result["scores"]
                labels = result["labels"]

                output_boxes.append(boxes)
                output_scores.append(scores)
                output_labels.append(labels)

            return HuggingFaceDetectorPredictions(
                boxes=output_boxes, labels=output_labels, scores=output_scores
            )
        else:
            assert not isinstance(results, Sequence)
            return results

    @classmethod
    def from_pretrained(
        cls,
        model: str,
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
            The MAITE wrapper for a HuggingFace object detector.

        Examples
        --------
        >>> hf_image_classifier = HuggingFaceObjectDetector.from_pretrained(model="facebook/detr-resnet-50")
        """
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor: Optional[HuggingFaceProcessor]
        det_model: HuggingFaceWithDetection

        threshold = kwargs.pop("threshold", 0.5)

        try:
            det_model = AutoModelForObjectDetection.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        try:
            processor = AutoImageProcessor.from_pretrained(model, **kwargs)
        except OSError as e:  # noqa: F841
            raise InvalidArgument(e)

        post_processor = getattr(processor, "post_process_object_detection", None)

        return cls(
            det_model,
            processor,
            post_processor=post_processor,
            threshold=threshold,
        )

    def forward(
        self, data: Union[SupportsArray, HasDataImage]
    ) -> Union[HuggingFaceDetectorOutput, HuggingFaceDetectorPredictions]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : ArrayLike | Sequence[ArrayLike] | HasDataImage | Sequence[HasDataImage]
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

        >>> from maite.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        images, target_size = self._process_inputs(data)
        if target_size is None:
            target_size = []

            assert isinstance(images, tr.Tensor) or isinstance(images, Sequence)
            for img in images:
                assert hasattr(img, "shape"), "Image must have a shape attribute."
                shape = getattr(img, "shape")
                target_size.append([shape[-2], shape[-1]])

        if TYPE_CHECKING:
            assert isinstance(self.model, HuggingFaceWithDetection)

        outputs = self.model(images)
        results = HuggingFaceDetectorOutput(
            logits=outputs.logits, boxes=outputs.pred_boxes
        )

        if self._threshold is not None:
            return self.post_processor(
                results, threshold=self._threshold, target_sizes=target_size
            )

        return results
