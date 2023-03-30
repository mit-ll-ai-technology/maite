from typing import Any, Iterable, Optional, Sequence, Union

from torch import nn
from typing_extensions import Self

from jatic_toolbox._internals.interop.utils import to_tensor_list
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ArrayLike, Classifier, HasLogits

from .typing import HuggingFaceProcessor, HuggingFaceWithLogits

__all__ = ["HuggingFaceImageClassifier"]


class HuggingFaceImageClassifier(nn.Module, Classifier[ArrayLike]):
    """
    Wrapper for HuggingFace image classifiation models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForImageClassification`
    to load the HuggingFace models.
    """

    def __init__(
        self,
        model: HuggingFaceWithLogits,
        processor: Optional[HuggingFaceProcessor] = None,
    ) -> None:
        """
        Initialize HuggingFaceImageClassifier.

        Parameters
        ----------
        processor : HuggingFaceProcessor
            A HuggingFace feature extractor for a given model.

        model : HuggingFaceModel
            A HuggingFace image classification model.

        Examples
        --------
        >>> from transformers import AutoFeatureExtractor, AutoModelForImageClassification
        >>> processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        >>> model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        >>> hf_model = HuggingFaceImageClassifier(processor, model)
        """
        super().__init__()
        self.model = model
        self.processor = processor

    @classmethod
    def list_models(
        cls, task: Optional[str] = "image-classification", **kwargs: Any
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

        processor: Optional[HuggingFaceProcessor]
        clf_model: HuggingFaceWithLogits

        try:
            processor = AutoFeatureExtractor.from_pretrained(model, **kwargs)
        except OSError:  # pragma: no cover
            processor = None

        try:
            clf_model = AutoModelForImageClassification.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        return cls(clf_model, processor)

    def forward(
        self, data: Union[ArrayLike, Sequence[ArrayLike]]
    ) -> HasLogits[ArrayLike]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : ArrayLike
            An array of images.

        Returns
        -------
        List[ObjectDetectionOutput]
            A list of object detection bounding boxes with corresponding scores.

        Examples
        --------
        First create a random NumPy image array:

        >>> import numpy as np
        >>> image = np.random.uniform(0, 255, size=(1, 200, 200, 3))

        Load a HuggingFace object detection model and execute on
        the above image:

        >>> hf_object_detector = HuggingFaceImageClassifier(model="facebook/detr-resnet-50")
        >>> detections = hf_object_detector(image)

        We can check to verify the output contains `boxes` and `scores` attributes:

        >>> from jatic_toolbox.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        if self.processor is not None:
            data = to_tensor_list(data)
            features = self.processor(images=data, return_tensors="pt")
            features.to(self.model.device)
            outputs: HasLogits = self.model(**features)
        else:
            import torch as tr

            outputs = self.model(tr.as_tensor(data))
        return outputs
