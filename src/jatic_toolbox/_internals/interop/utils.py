from dataclasses import dataclass
from typing import Any, Dict, Sequence

from ..protocols import ArrayLike


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
    scores: Sequence[Sequence[Dict[Any, float]]]
