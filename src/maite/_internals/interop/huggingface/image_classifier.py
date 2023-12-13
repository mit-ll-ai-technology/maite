# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import torch as tr
from typing_extensions import Self

from maite.errors import InvalidArgument
from maite.protocols import HasDataImage, HasLogits, SupportsArray

from .base import BaseHFModel
from .typing import (
    HuggingFacePredictions,
    HuggingFaceProbs,
    HuggingFaceProcessor,
    HuggingFaceWithLogits,
)

__all__ = ["HuggingFaceImageClassifier"]


class HuggingFaceImageClassifier(BaseHFModel):
    """
    Wrapper for HuggingFace image classifiation models.

    This interface uses `AutoFeatureExtractor` and `AutoModelForImageClassification`
    to load the HuggingFace models.
    """

    def __init__(
        self,
        model: HuggingFaceWithLogits,
        processor: Optional[HuggingFaceProcessor] = None,
        top_k: Optional[int] = None,
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
        super().__init__(model=model, processor=processor)
        self._top_k = top_k

    def preprocessor(
        self,
        data: SupportsArray,
    ) -> HasDataImage:
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
        assert self._processor is not None, "No processor was provided."
        assert isinstance(data, (list, tuple))

        image_features = self._processor(images=data, return_tensors="pt")[
            "pixel_values"
        ]
        assert isinstance(image_features, tr.Tensor)
        return {"image": image_features}

    def post_processor(
        self, outputs: HasLogits
    ) -> Union[HuggingFacePredictions, HuggingFaceProbs]:
        """
        Postprocess the outputs of a HuggingFace image classifier.

        Parameters
        ----------
        outputs : HasLogits
            The outputs of a HuggingFace image classifier.

        Returns
        -------
        HuggingFaceProbs | HuggingFacePredictions
            The postprocessed outputs of a HuggingFace image classifier.

        Examples
        --------
        """
        probs = tr.as_tensor(outputs.logits).softmax(dim=-1)

        if self._top_k is None:
            return HuggingFaceProbs(probs=probs)

        top_k = self._top_k
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        scores, ids = probs.topk(top_k)
        ids = ids.tolist()

        if TYPE_CHECKING:
            ids = cast(List[List[int]], ids)

        model_labels = list(self.get_labels())
        labels = [[model_labels[_id] for _id in _ids] for _ids in ids]
        return HuggingFacePredictions(scores=scores, labels=labels)

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> Self:
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
            The MAITE wrapper for a HuggingFace image classifier.

        Examples
        --------
        >>> hf_image_classifier = HuggingFaceImageClassifier.from_pretrained(model="microsoft/resnet-50")
        """
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        processor: Optional[HuggingFaceProcessor] = None
        clf_model: HuggingFaceWithLogits

        top_k = kwargs.pop("top_k", None)

        try:
            clf_model = AutoModelForImageClassification.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        try:
            processor = AutoFeatureExtractor.from_pretrained(model, **kwargs)
        except OSError:  # pragma: no cover
            processor = None

        return cls(clf_model, processor, top_k=top_k)

    def forward(
        self, data: Union[HasDataImage, SupportsArray]
    ) -> Union[HuggingFacePredictions, HuggingFaceProbs]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : Union[HasDataImage, ArrayLike, Sequence[ArrayLike]]
            The data to extract object detection from.

        Returns
        -------
        HasLogits
            The object detection results.

        Examples
        --------
        First create a random NumPy image array:

        >>> import numpy as np
        >>> image = np.random.uniform(0, 255, size=(1, 200, 200, 3))

        Load a HuggingFace object detection model and execute on
        the above image:

        >>> hf_object_detector = HuggingFaceImageClassifier(model="facebook/detr-resnet-50")
        >>> detections = hf_object_detector(image)

        We can check to verify the output contains `pred_boxes` and `logits` attributes:

        >>> from maite.protocols import HasLogits
        >>> assert isinstance(detections, HasLogits)
        """
        images, _ = self._process_inputs(data)

        if TYPE_CHECKING:
            assert isinstance(self.model, HuggingFaceWithLogits)

        results = self.model(images)
        return self.post_processor(results)
