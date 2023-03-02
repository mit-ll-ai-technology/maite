from collections import UserDict
from typing import Any, Sequence, TypeVar

from torch import Tensor
from typing_extensions import Protocol, Self

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ArrayLike, Classifier, HasLogits

__all__ = ["HuggingFaceImageClassifier"]


T = TypeVar("T", bound=ArrayLike)


class BatchFeature(UserDict[str, T]):
    ...


class HuggingFaceProcessor(Protocol[T]):
    def __call__(self, images: Sequence[T], **kwargs: Any) -> BatchFeature[T]:
        ...


class HuggingFaceModel(Protocol[T]):
    def __call__(self, pixel_values: T, **kwargs: Any) -> HasLogits[T]:
        ...


class HuggingFaceImageClassifier(Classifier[Tensor]):
    """
    Wrapper for HuggingFace image classifiation models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForImageClassification`
    to load the HuggingFace models.
    """

    def __init__(
        self, processor: HuggingFaceProcessor[Tensor], model: HuggingFaceModel[Tensor]
    ) -> None:
        """
        Initialize HuggingFaceImageClassifier.

        Parameters
        ----------
        processor : HuggingFaceProcessor[Tensor]
            A HuggingFace feature extractor for a given model.

        model : HuggingFaceModel[Tensor]
            A HuggingFace image classification model.

        Examples
        --------
        >>> from transformers import AutoFeatureExtractor, AutoModelForImageClassification
        >>> processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        >>> model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        >>> hf_model = HuggingFaceImageClassifier(processor, model)
        """
        super().__init__()
        self.processor = processor
        self.model = model

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> Self:  # pragma: no cover
        """
        Load a HuggingFace model from pretrained weights.

        Uses `AutoFeatureExtractor` and `AutoModelForImageClassification`.

        Parameters
        ----------
        model : str
            The `model id` of a pretrained image classifier stored on HuggingFace.

        **kwargs : Any
            Keyword arguments for HuggingFace AutoFeatureExtractor and AutoModelForImageClassification.

        Returns
        -------
        HuggingFaceImageClassifier
            The JATIC Toolbox wrapper for a HuggingFace image classifier.

        Examples
        --------
        >>> hf_image_classifier = HuggingFaceImageClassifier.from_pretrained(model="microsoft/resnet-50")
        """
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        try:
            processor: HuggingFaceProcessor[
                Tensor
            ] = AutoFeatureExtractor.from_pretrained(model, **kwargs)
            clf_model: HuggingFaceModel[
                Tensor
            ] = AutoModelForImageClassification.from_pretrained(model, **kwargs)
        except OSError as e:
            raise InvalidArgument(e)

        return cls(processor, clf_model)

    def __call__(self, data: Sequence[Tensor]) -> HasLogits[Tensor]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        img_iter : Iterable[torch.Tensor]
            An array of images.

        Returns
        -------
        List[ObjectDetectionOutput]
            A list of object detection bounding boxes with corresponding scores.

        Examples
        --------
        First create a random NumPy image array:

        >>> import numpy as np
        >>> image = np.random.uniform(0, 255, size=(200, 200, 3))

        Load a HuggingFace object detection model and execute on
        the above image:

        >>> hf_object_detector = HuggingFaceImageClassifier(model="facebook/detr-resnet-50")
        >>> detections = hf_object_detector([image])

        We can check to verify the output contains `boxes` and `scores` attributes:

        >>> from jatic_toolbox.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        inputs: BatchFeature[Tensor] = self.processor(images=data, return_tensors="pt")
        outputs: HasLogits[Tensor] = self.model(**inputs)
        return outputs
