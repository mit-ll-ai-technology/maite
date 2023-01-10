import warnings
from typing import Any, Iterable, List

import numpy as np
import pooch
import torch as tr
from PIL.Image import Image
from smqtk_detection.impls.detect_image_objects import centernet

from jatic_toolbox.errors import InternalError, InvalidArgument
from jatic_toolbox.protocols.array import ArrayLike
from jatic_toolbox.protocols.object_detection import (
    ObjectDetection,
    ObjectDetectionOutput,
)

# from jatic_toolbox.protocols import ArrayLike, ObjectDetection, ObjectDetectionOutput
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


class CenterNetVisdrone(ObjectDetection):
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
        >> centernet = CenterNetVisdrone(model="resnet50")
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

    def __call__(self, img_iter: Iterable[ArrayLike]) -> List[ObjectDetectionOutput]:
        """
        Object Detector for CenterNet.

        Parameters
        ----------
        img_iter : Iterable[ArrayLike]
            An array of images.

        Returns
        -------
        List[ObjectDetectionOutput]
            A list of object detection bounding boxes with corresponding scores.

        Examples
        --------
        >> import numpy as np
        >> image = np.random.uniform(0, 255, size=(200, 200, 3))
        >> centernet = CenterNetVisdrone(model="resnet50")
        >> detections = centernet([image])
        """
        arr_iter = []
        for img in img_iter:
            check_type("img", img, (Image, np.ndarray, tr.Tensor))
            if isinstance(img, tr.Tensor):
                warnings.warn(
                    "SMQTK expects NumPy arrays (input data type: `torch.Tensor`)"
                )
            arr_iter.append(np.asarray(img))

        arr_iter = [np.asarray(img) for img in img_iter]
        smqt_output = self._detector(arr_iter)

        detector_output = []
        for dets in smqt_output:
            boxes = []
            scores = []
            for bbox, label_score in dets:
                boxes.append(bbox)
                scores.append(label_score)

            detector_output.append(ObjectDetectionOutput(boxes=boxes, scores=scores))

        return detector_output
