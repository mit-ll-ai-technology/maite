from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..import_utils import is_torch_available
from ..protocols import ArrayLike


def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not.

    Safe to call even if torch is not installed.
    """

    def _is_torch(x):
        import torch

        return isinstance(x, torch.Tensor)

    return False if not is_torch_available() else _is_torch(x)


def is_numpy_array(x):
    """Tests if `x` is a numpy array or not."""
    return isinstance(x, np.ndarray)


@dataclass
class ClassificationOutput:
    # doc-ignore: EX01
    """
    Object detection output.

    Parameters
    ----------
    logits : ArrayLike
        An array of logits for each image.
    """
    logits: ArrayLike


@dataclass
class ObjectDetectionOutput:
    # doc-ignore: EX01
    """
    Object detection output.

    Parameters
    ----------
    boxes : Sequence[ArrayLike]
        Detection boxes for each data point.  The number of detections
        can vary for each image.

    scores : Sequence[Sequence[Dict[str, float]]]
        The label (key) and score (value) for each detection bounding box.
    """
    boxes: Sequence[ArrayLike]
    scores: Sequence[ArrayLike]
    labels: Sequence[Any]
