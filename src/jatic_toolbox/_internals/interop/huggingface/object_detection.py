from collections import UserDict
from typing import Any, List, Optional, Sequence, TypeVar, Union, overload

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
    HasImagesDict,
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
        self._processor = processor

        if processor is not None and post_processor is None:
            if hasattr(processor, "post_process_object_detection"):
                post_processor = processor.post_process_object_detection

        self._post_processor = post_processor
        self._threshold = threshold
        self._labels = list(model.config.id2label.values())

    @overload
    def preprocessor(
        self,
        data: Sequence[ArrayLike],
        image_key: str = "image",
    ) -> HasImagesDict:
        ...

    @overload
    def preprocessor(
        self,
        data: Sequence[HasImagesDict],
        image_key: str = "image",
    ) -> Sequence[HasImagesDict]:
        ...

    def preprocessor(
        self,
        data: Union[Sequence[ArrayLike], Sequence[HasImagesDict]],
        image_key: str = "image",
    ) -> Union[HasImagesDict, Sequence[HasImagesDict]]:
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

        if isinstance(data[0], dict):
            images = to_tensor_list([d[image_key] for d in data])
            target_sizes = [tuple(np.asarray(img).shape[:2]) for img in images]
            image_features = self._processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]

            out = []
            for d, image, ts in zip(data, image_features, target_sizes):
                data_out = {"image": image, "target_size": ts}
                data_out.update({k: v for k, v in d.items() if k != image_key})
                out.append(data_out)

            return out

        else:
            images = to_tensor_list(data)
            target_sizes = [tuple(np.asarray(img).shape[:2]) for img in images]
            image_features = self._processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]
            assert isinstance(image_features, tr.Tensor)
            return {"image": image_features, "target_size": target_sizes}

    def post_processor(
        self, model_outputs: HasDetectionLogits, **kwargs: Any
    ) -> HasObjectDetections:
        """
        Post process the outputs of a HuggingFace object detector.

        Parameters
        ----------
        outputs : Any
            The outputs of the model.

        Returns
        -------
        Any
            The post processed outputs.

        Examples
        --------
        """
        if self._post_processor is None:  # pragma: no cover
            raise InvalidArgument("No post processor was provided.")

        target_sizes = model_outputs.get("target_size", None)

        results = self._post_processor(
            model_outputs, threshold=self._threshold, target_sizes=target_sizes
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

        threshold = kwargs.pop("threshold", 0.5)

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
                    threshold=threshold,
                )
            elif not with_post_processor:
                return cls(det_model, processor)
            else:
                return cls(
                    det_model,
                    post_processor=processor.post_process_object_detection,
                    threshold=threshold,
                )

        return cls(det_model, threshold=threshold)

    def forward(
        self,
        data: Union[HasImagesDict, ArrayLike],
    ) -> HasDetectionLogits[ArrayLike]:
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
        target_size = None
        if isinstance(data, (dict, UserDict)):
            target_size = data.get("target_size", None)

            if "image" in data:
                pixel_values = data["image"]
            elif "pixel_values" in data:
                pixel_values = data["pixel_values"]
            else:
                raise InvalidArgument(
                    f"Expected 'image' or 'pixel_values' in data, got {data.keys()}"
                )
        else:
            pixel_values = tr.as_tensor(data)

        outputs = self.model(pixel_values)
        return outputs.__class__({"target_size": target_size, **outputs})
