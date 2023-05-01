from collections import UserDict
from typing import Any, Optional, Sequence, Union, overload

import torch as tr
from torch import nn
from typing_extensions import Self

from jatic_toolbox._internals.interop.utils import to_tensor_list
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ArrayLike, Classifier, HasLogits

from .typing import (
    HasImagesDict,
    HuggingFacePostProcessedImages,
    HuggingFaceProcessor,
    HuggingFaceWithLogits,
)

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
        super().__init__()
        self.model = model
        self._processor = processor
        self._labels = list(model.config.id2label.values())
        self._top_k = top_k

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
            image_features = self._processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]

            out = []
            for d, image in zip(data, image_features):
                data_out = {"image": image}
                data_out.update({k: v for k, v in d.items() if k != image_key})
                out.append(data_out)
            return out

        else:
            images = to_tensor_list(data)
            image_features = self._processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]
            assert isinstance(image_features, tr.Tensor)
            return {image_key: image_features}

    def post_processor(self, outputs: HasLogits) -> HuggingFacePostProcessedImages:
        """
        Postprocess the outputs of a HuggingFace image classifier.

        Parameters
        ----------
        outputs : HasLogits
            The outputs of a HuggingFace image classifier.

        Returns
        -------
        HuggingFacePostProcessedImages
            The postprocessed outputs of a HuggingFace image classifier.

        Examples
        --------
        """
        probs = outputs.logits.softmax(dim=-1)

        if self._top_k is None:
            labels = [list(self.model.config.id2label.values())] * probs.shape[0]
            return HuggingFacePostProcessedImages(probs=probs, labels=labels)

        top_k = self._top_k
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        scores, ids = probs.topk(top_k)
        ids = ids.tolist()
        labels = [[self.model.config.id2label[_id] for _id in _ids] for _ids in ids]
        return HuggingFacePostProcessedImages(probs=scores, labels=labels)

    @classmethod
    def from_pretrained(
        cls, model: str, with_processor: bool = True, **kwargs: Any
    ) -> Self:
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

        processor: Optional[HuggingFaceProcessor] = None
        clf_model: HuggingFaceWithLogits

        top_k = kwargs.pop("top_k", None)

        try:
            clf_model = AutoModelForImageClassification.from_pretrained(model, **kwargs)
        except OSError as e:  # pragma: no cover
            raise InvalidArgument(e)

        if with_processor:
            try:
                processor = AutoFeatureExtractor.from_pretrained(model, **kwargs)
            except OSError:  # pragma: no cover
                processor = None

        return cls(clf_model, processor, top_k=top_k)

    def forward(self, data: Union[HasImagesDict, ArrayLike]) -> HasLogits[ArrayLike]:
        """
        Extract object detection for HuggingFace Object Detection models.

        Parameters
        ----------
        data : Union[HasImagesDict, ArrayLike, Sequence[ArrayLike]]
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

        >>> from jatic_toolbox.protocols import HasLogits
        >>> assert isinstance(detections, HasLogits)
        """
        if isinstance(data, (dict, UserDict)):
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

        return self.model(pixel_values)
