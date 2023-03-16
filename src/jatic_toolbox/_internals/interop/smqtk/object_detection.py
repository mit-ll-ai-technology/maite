from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Protocol, Self, TypeAlias

from jatic_toolbox._internals.interop.utils import is_numpy_array, is_torch_tensor
from jatic_toolbox.errors import InternalError, InvalidArgument
from jatic_toolbox.protocols import ArrayLike, ObjectDetector

__all__ = ["CenterNet"]

# TODO: Need to get rid of this from the toolbox.
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
    """Implements output for JATIC Object Detections."""

    boxes: List[NDArray[Any]]
    scores: List[NDArray[Any]]
    labels: List[List[Hashable]]


class _AxisAlignedBoundingBox(Protocol):
    """Protocol for SMQTK Axis Aligned Bounding Box."""

    __slots__ = "min_vertex", "max_vertex"


class AxisAlignedBoundingBox(NamedTuple):
    min_vertex: Sequence[Union[int, float]]
    max_vertex: Sequence[Union[int, float]]


SMQTKAxisAlignedBoxes: TypeAlias = Iterable[
    Iterable[Tuple[_AxisAlignedBoundingBox, Dict[Hashable, float]]]
]


class SMQTDetector(Protocol):
    def __call__(self, img_iter: Iterable[NDArray[Any]]) -> SMQTKAxisAlignedBoxes:
        ...


def _get_model_file(model: str) -> str:  # pragma: no cover
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
    import pooch

    if model not in _MODELS:  # pragma: no cover
        model_archs = ",".join(list(_MODELS.keys()))
        raise InvalidArgument(
            f"SMQTK-Detecton CenterNet architecture `{model}` not one of {model_archs}"
        )

    return pooch.retrieve(**_MODELS[model])


class CenterNet(ObjectDetector[NDArray[Any]]):
    """
    Wrapper for CenterNet model pretrained on the visdrone2019 dataset.
    """

    def __init__(self, detector: SMQTDetector):
        """
        Initialize CenterNet.

        Parameters
        ----------
        model : str (default: "resnet50")
            The named model architecture (see `jatic_toolbox.interop.smqtk.centernet._MODELS`).

        **kwargs : Any
            Keyword arguments for SMQTK-Detection CenterNet class.

        Examples
        --------
        >>> from smqtk_detection.impls.detect_image_objects import centernet
        >>> detector = centernet.CenterNet(arch="resnet50", model_file=centernet_model_file)
        >>> centernet = CenterNet(detector)
        """
        self._detector = detector

    @classmethod
    def list_models(
        cls, *args: Any, **kwargs: Any
    ) -> Iterable[str]:  # pragma: no cover
        # TODO: There are more models supported.
        return [
            "resnet18",
            "resnet50",
            "res2net50",
        ]

    @classmethod
    def from_pretrained(
        cls, model: str = "resnet50", **kwargs: Any
    ) -> Self:  # pragma: no cover
        """
        Load pretrained model for `smqtk_detection.impls.detect_image_objects.centernet.CenterNet`.

        Parameters
        ----------
        model : str (default: "resnet50")
            The `model id` of a pretrained detector for `CenterNet`.

        **kwargs : Any
            Keyword arguments for `CenterNet`.

        Returns
        -------
        CenterNet
            The JATIC Toolbox wrapper for `CenterNet`.

        Examples
        --------
        >>> hf_image_classifier = CenterNet.from_pretrained(model="resnet50")
        """
        from smqtk_detection.impls.detect_image_objects import centernet

        if not centernet.usable:  # pragma: no cover
            raise InternalError(
                "The following packages must be installed for SMQTK-Detection CenterNet: `torch`, `cv2`, `numba`"
            )

        model_file = _get_model_file(model)
        detector: SMQTDetector = centernet.CenterNetVisdrone(
            arch=model, model_file=model_file, **kwargs
        )
        return cls(detector)

    def __call__(
        self, data: Union[ArrayLike, Sequence[ArrayLike]]
    ) -> SMQTKObjectDetectionOutput:
        """
        Object Detector for CenterNet.

        Parameters
        ----------
        data : Sequence[ArrayLike]
            An array of images.  Inputs can be `PIL.Image`, `np.ndarray`, or `torch.Tensor`
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

        >> centernet = CenterNet(model="resnet50")
        >>> detections = centernet([image])

        We can check to verify the output contains `boxes` and `scores` attributes:

        >>> from jatic_toolbox.protocols import HasObjectDetections
        >>> assert isinstance(detections, HasObjectDetections)
        """
        img_data: Iterable[ArrayLike]
        if isinstance(data, Sequence):
            assert isinstance(data[0], ArrayLike)
            img_data = [np.asarray(x) for x in data]
        elif is_torch_tensor(data):
            if TYPE_CHECKING:
                from torch import Tensor

                assert isinstance(data, Tensor)
            img_data = [np.asarray(x.detach().cpu().numpy()) for x in data]

        elif is_numpy_array(data):
            if TYPE_CHECKING:
                assert isinstance(data, np.ndarray)
            img_data = [np.asarray(x) for x in data]

        else:
            raise InvalidArgument(f"Unsupported JATIC data type {type(data)}.")

        smqt_output = self._detector(img_data)

        # extract output into JATIC format
        output_labels: List[List[Hashable]] = []
        output_scores: List[np.ndarray] = []
        output_boxes: List[np.ndarray] = []
        for dets in smqt_output:
            boxes: List[np.ndarray] = []
            scores: List[float] = []
            labels: List[Hashable] = []
            for bbox, label_score in dets:
                flatten_box = np.hstack([bbox.min_vertex, bbox.max_vertex])
                for k, v in label_score.items():
                    boxes.append(flatten_box)
                    scores.append(v)
                    labels.append(k)

            output_boxes.append(np.stack(boxes))
            output_scores.append(np.asarray(scores))
            output_labels.append(labels)

        return SMQTKObjectDetectionOutput(
            boxes=output_boxes, scores=output_scores, labels=output_labels
        )
