import warnings
from typing import Any, List, Sequence, Union

import numpy as np
import torch as tr
from PIL.Image import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import (
    ArrayLike,
    Classifier,
    HasLogits,
    HasProbs,
    ImageType,
)
from jatic_toolbox.utils.validation import check_type

from ..utils import ClassificationOutput

__all__ = ["HuggingFaceImageClassifier"]


class HuggingFaceImageClassifier(Classifier):
    """
    Wrapper for HuggingFace image classifiation models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForImageClassification`
    to load the HuggingFace models.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """
        Initialize HuggingFaceImageClassifier.

        Parameters
        ----------
        model : str
            The `model id` of a pretrained image classifier stored on HuggingFace.

        **kwargs : Any
            Keyword arguments for HuggingFace AutoFeatureExtractor and AutoModelForImageClassification.

        Examples
        --------
        >> import numpy as np
        >> data = np.random.uniform(0, 255, size=(200, 200, 3))
        >> hf_image_classifier = HuggingFaceImageClassifier(model="")
        >> classifier_output = hf_image_classifier([data])
        """
        super().__init__()
        check_type("model", model, str)

        self._model = model

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model, **kwargs
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                model, **kwargs
            )
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

    def __call__(self, data: Sequence[ImageType]) -> Union[HasLogits, HasProbs]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        img_iter : Iterable[PIL.Image.Image | numpy.ndarray | torch.Tensor]
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
        arr_iter: List[ArrayLike] = []
        for img in data:
            check_type("img", img, (Image, np.ndarray, tr.Tensor))
            if isinstance(img, tr.Tensor):
                warnings.warn(
                    "HuggingFace feature extractors convert input data to NumPy arrays (input data type: `torch.Tensor`)"
                )

            np_img = np.asarray(img)
            arr_iter.append(np_img)

        with tr.no_grad():
            inputs = self.feature_extractor(images=arr_iter, return_tensors="pt")
            outputs = self.model(**inputs)
            assert hasattr(outputs, "logits")

        return ClassificationOutput(logits=outputs.logits)
