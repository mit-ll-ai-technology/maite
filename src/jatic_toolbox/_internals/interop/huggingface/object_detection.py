import warnings
from typing import Any, Iterable, Sequence

import numpy as np
import torch as tr
from PIL.Image import Image
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols.array import ArrayLike
from jatic_toolbox.protocols.object_detection import (
    BoundingBox,
    ObjectDetection,
    ObjectDetectionOutput,
)
from jatic_toolbox.utils.validation import check_type

__all__ = ["HuggingFaceObjectDetector", "HuggingFaceBoundingBox"]


class HuggingFaceBoundingBox(BoundingBox):
    """
    Implementation of `BoundingBox` for HuggingFace output.
    """

    def __init__(self, bbox: Sequence[float]):
        """
        Initialize HuggingFace bounding box.

        Parameters
        ----------
        bbox : Sequence[float]
            Bounding box as [x1, y1, x2, y2].

        Examples
        --------
        >> hf_bbox = HuggingFaceBoundingBox([0, 1, 2, 3])
        >> hf_bbox.min_vertex
        [0, 1]
        >> hf_bbox.max_vertex
        [2, 3]
        """
        x1, y1, x2, y2 = bbox
        self.min_vertex = [x1, y1]
        self.max_vertex = [x2, y2]


class HuggingFaceObjectDetector(ObjectDetection):
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
        >> import numpy as np
        >> data = np.random.uniform(0, 255, size=(200, 200, 3))
        >> hf_object_detector = HuggingFaceObjectDetector(model="facebook/detr-resnet-50")
        >> detection_output = hf_object_detector([data])
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

    def __call__(
        self, img_iter: Iterable[ArrayLike]
    ) -> Iterable[ObjectDetectionOutput]:
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
        >> import numpy as np
        >> image = np.random.uniform(0, 255, size=(200, 200, 3))
        >> hf_object_detector = HuggingFaceObjectDetector(model="facebook/detr-resnet-50")
        >> detections = hf_object_detector([image])
        """
        arr_iter = []
        for img in img_iter:
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
            results = self.feature_extractor.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )
            scores = tr.softmax(outputs.logits, dim=-1).numpy()

        dets = []
        for i, _ in enumerate(img_iter):
            boxes = results[i]["boxes"].numpy()
            dets.append(self._hfboxes_to_jatic(boxes, scores[i]))
        return dets

    def _hfboxes_to_jatic(
        self, boxes: Iterable[ArrayLike], scores: Iterable[ArrayLike]
    ) -> ObjectDetectionOutput:
        """Convert HuggingFace Bounding Boxes to JATIC Bounding Boxes"""
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)

        output_scores = []
        output_boxes = []
        for j in range(len(boxes)):
            output_boxes.append(HuggingFaceBoundingBox(boxes[j]))
            output_scores.append(
                {k: scores[j][k].item() for k in range(len(scores[j]))}
            )
        return ObjectDetectionOutput(boxes=output_boxes, scores=output_scores)
