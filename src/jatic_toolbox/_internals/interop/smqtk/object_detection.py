import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pooch
import torch as tr
from numpy.typing import NDArray
from PIL.Image import Image
from smqtk_detection.impls.detect_image_objects import centernet

from jatic_toolbox.errors import InternalError, InvalidArgument
from jatic_toolbox.protocols import ImageType, ObjectDetector
from jatic_toolbox.utils.validation import check_type

if not centernet.usable:  # pragma: no cover
    raise InternalError(
        "The following packages must be installed for SMQTK-Detection CenterNet: `torch`, `cv2`, `numba`"
    )

__all__ = ["CenterNetVisdrone"]

_MODELS = {
    "resnet50": dict(
        url="https://data.kitware.com/api/v1/item/623259f64acac99f426f21db/download",
        known_hash="2f412e3d1e783a551c9810ce5c7f6e05e9bda2705b1e1509a54bf3474b7b2084",
        fname="centernet-resnet50.pth",
    ),
    "resnet18": dict(
        url="https://data.kitware.com/api/v1/item/623de4744acac99f42f05fb1/download",
        known_hash="3b75fb60f1e85464e4cd0ee33cf0b018717968368a1022b10240ff27757efb4b",
        fname="centernet-resnet18.pth",
    ),
    "res2net50": dict(
        url="https://data.kitware.com/api/v1/item/623e18464acac99f42f40a4e/download",
        known_hash="0e3c436860eb9f5794f25540605d904a9a5e5cb4009effb9b5e05b3ecdacbcc9",
        fname="centernet-res2net50.pth",
    ),
}


@dataclass
class SMQTKObjectDetectionOutput:
    boxes: List[NDArray]
    scores: List[List[Dict[str, float]]]


class CenterNetVisdrone(ObjectDetector):
    """
    Wrapper for CenterNet model pretrained on the visdrone2019 dataset.
    """

    def __init__(self, model: str = "resnet50", **kwargs: Any):
        """
        Initialize CenterNetVisdrone.

        Parameters
        ----------
        model : str (default: "resnet50")
            The named model architecture (see `jatic_toolbox.interop.smqtk.centernet._MODELS`).

        **kwargs : Any
            Keyword arguments for SMQTK-Detection CenterNetVisdrone class.

        Examples
        --------
        >>> centernet = CenterNetVisdrone(model="resnet50")
        """
        check_type("model name", model, str)
        centernet_model_file = self._get_model_file(model)
        self._detector = centernet.CenterNetVisdrone(
            arch=model, model_file=centernet_model_file, **kwargs
        )

    def _get_model_file(self, model: str) -> str:
        """
        Get model weights file.

        Parameters
        ----------
        model : str
            The named model architecture (see `jatic_toolbox.interop.smqtk.centernet._MODELS`).

        Returns
        -------
        str
            The location of the model weights file.
        """

        if model not in _MODELS:
            model_archs = ",".join(list(_MODELS.keys()))
            raise InvalidArgument(
                f"SMQTK-Detecton CenterNet architecture `{model}` not one of {model_archs}"
            )  # pragma: no cover

        return pooch.retrieve(**_MODELS[model])

    def __call__(self, data: Sequence[ImageType]) -> SMQTKObjectDetectionOutput:
        """
        Object Detector for CenterNet.

        Parameters
        ----------
        data: Sequence[ImageType]
            An array of images.  Inputs can be `PIL.Image`, `NDArray`, or `torch.Tensor`
            .

        Returns
        -------
        SMQTKObjectDetectionOutput
            A list of object detection bounding boxes with corresponding scores.

        Examples
        --------
        First create a random NumPy image array:

        >>> import numpy as np
        >>> image = np.random.uniform(0, 255, size=(200, 200, 3))

        >> centernet = CenterNetVisdrone(model="resnet50")
        >>> detections = centernet([image])

        We can check to verify the output contains `boxes` and `scores` attributes:

        >>> from jatic_toolbox.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        arr_iter = []
        for img in data:
            check_type("img", img, (Image, np.ndarray, tr.Tensor))
            if isinstance(img, tr.Tensor):
                warnings.warn(
                    "SMQTK expects NumPy arrays (input data type: `torch.Tensor`)"
                )
                img = img.detach().cpu().numpy()
            arr_iter.append(np.asarray(img))

        smqt_output = self._detector(arr_iter)

        all_boxes: List[NDArray] = []
        all_scores: List[List[Dict[str, float]]] = []
        for dets in smqt_output:
            boxes = []
            scores = []
            for bbox, label_score in dets:
                flatten_box = np.hstack([bbox.min_vertex, bbox.max_vertex])
                boxes.append(flatten_box)
                scores.append(label_score)

            all_boxes.append(np.asarray(boxes))
            all_scores.append(scores)

        return SMQTKObjectDetectionOutput(boxes=all_boxes, scores=all_scores)
