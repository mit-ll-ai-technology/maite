# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from typing_extensions import TypeAlias
from ultralytics import YOLO
from ultralytics.engine.results import Results
from yolov5.models.common import AutoShape, Detections

from maite.protocols import ModelMetadata, object_detection as od

YOLOObjectDetectionResults: TypeAlias = Union[Detections, list[Results]]
YOLOModel: TypeAlias = Union[YOLO, AutoShape]


@dataclass
class ObjectDetectionTargets:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class YoloObjectDetector:
    """MAITE-wrapped object detection YOLO model.

    Wrapped YOLO model which adheres to MAITE protocols. This wrapped model is intended
    to be used as-is for basic use cases of object detection.

    Examples
    --------
    Import relevant Python libraries.

    >>> import numpy as np
    >>> from ultralytics import YOLO
    >>> from maite.interop.models import YoloObjectDetector
    >>> from maite.protocols import object_detection as od

    Load an Ultralytics-hosted YOLOv5 model, 'yolov5nu',

    >>> yolov5_model = YOLO("yolov5nu")
    >>> wrapped_yolov5_model = YoloObjectDetector(
    ...     model=yolov5_model,
    ...     id='YOLOv5nu',
    ...     index2label=yolov5_model.names
    ... )

    or, load an Ultralytics YOLOv5 model from a local filepath.

    >>> yolov5_model = YOLO("./yolov5nu.pt")
    >>> wrapped_yolov5_model = YoloObjectDetector(
    ...     model=yolov5_model,
    ...     id='YOLOv5nu',
    ...     index2label=yolov5_model.names
    ... )

    Load an Ultralytics-hosted YOLOv8 model, 'yolov8n',

    >>> yolov8_model = YOLO("yolov8n")
    >>> wrapped_yolov8_model = YoloObjectDetector(
    ...     model=yolov8_model,
    ...     id='YOLOv8n',
    ...     index2label=yolov8_model.names
    ... )

    or, load an Ultralytics YOLOv8 model from a local filepath.

    >>> yolov8_model = YOLO("./yolov8n.pt")
    >>> wrapped_yolov8_model = YoloObjectDetector(
    ...     model=yolov8_model,
    ...     id='YOLOv8n',
    ...     index2label=yolov8_model.names
    ... )

    Perform object detection inference with the model.

    >>> N_DATAPOINTS = 5  # datapoints in dataset
    >>> C = 3  # number of color channels
    >>> H = 5  # img height
    >>> W = 6  # img width
    >>> batch_data: od.InputBatchType = list(np.random.rand(N_DATAPOINTS, C, H, W))
    >>> model_results: od.TargetBatchType = wrapped_yolov8_model(batch_data)
    >>> print(model_results)
    [ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32))]

    Notes
    -----
    Only Ultralytics YOLOv5 and YOLOv8 models are currently supported.
    """

    def __init__(
        self,
        model: YOLOModel,
        id: Optional[str] = None,
        index2label: Optional[dict[int, str]] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : YOLO | AutoShape
            A loaded Ultralytics YOLO model. The model can be loaded via the
            `ultralytics` or `yolov5` Python Library. Models must be either YOLOv5 or
            YOLOv8 and must be designed for the object detection task.

        id : Optional[str], (default=None)
            An id or name identifying the model.

        index2label : Optional[dict[int, str]], (default=None)
            A mapping from integer class index to string name.

        **kwargs
            Additional keyword arguments for configuring the model's prediction process. These arguments
            (such as `verbose`, `conf`, and `device` for YOLOv8), will be passed at inference time to the
            underlying native model.

            For `ultralytics` loaded models, refer to the
            `Ultralytics Docs <https://docs.ultralytics.com/modes/predict/#inference-arguments>`_
            for allowed keyword arguments.

            For `yolov5` loaded legacy models, refer to the
            `YOLOv5 model <https://github.com/ultralytics/yolov5/blob/30e4c4f09297b67afedf8b2bcd851833ddc9dead/models/common.py#L243-L252>_
            for allowed keyword arguments, as stated in the `Ultralytics YOLOv5 Docs <https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#simple-example>`_.

        """
        self.model = model
        self.kwargs = kwargs

        # Add model metadata
        if id is None:
            id = "MAITE-wrapped YOLO Model"

        self.metadata: ModelMetadata = (
            ModelMetadata(id=id)
            if index2label is None
            else ModelMetadata(id=id, index2label=index2label)
        )

    @staticmethod
    def _format_results(
        results: YOLOObjectDetectionResults,
    ) -> list[ObjectDetectionTargets]:
        """Format the object detections results.

        Format the results from the underlying YOLO model to the expected MAITE output.

        Parameters
        ----------
        results : Union[Detections, list]
            If the model was loaded with the `ultralytics` library, then `results` is a
            List of ultralytics.Results. Otherwise, the model is assumed to have been
            loaded with the legacy `yolov5` library and `results` are yolov5.Detections.

        Returns
        -------
        list[ObjectDetectionTargets]
            List of object detections, defined by boxes, labels, and scores.
        """
        all_detections: list[ObjectDetectionTargets] = []

        if isinstance(results, list):
            for result in results:
                detections = result.boxes
                if detections is None:
                    continue
                detections = detections.cpu().numpy()
                boxes = np.array(detections.xyxy)
                labels = np.array(detections.cls, dtype=np.uint8)
                scores = np.array(detections.conf)
                all_detections.append(
                    ObjectDetectionTargets(boxes=boxes, labels=labels, scores=scores)
                )
        else:
            # Results from yolov5 engine are `Detections` objects
            for result in results.pred:
                if result.shape[0] == 0:
                    boxes = np.zeros((0, 4))
                    labels = np.zeros(0, dtype=np.uint8)
                    scores = np.zeros(0)
                else:
                    boxes = np.zeros((result.shape[0], 4))
                    labels = np.zeros(result.shape[0], dtype=np.uint8)
                    scores = np.zeros(result.shape[0])
                    for idx, detection in enumerate(result):
                        # if detection[0].numel() != 0: # 0 on the xyxy ?
                        boxes[idx, :4] = detection[:4]
                        labels[idx] = detection[5].int()
                        scores[idx] = detection[4]

                all_detections.append(
                    ObjectDetectionTargets(boxes=boxes, labels=labels, scores=scores)
                )

        return all_detections

    def __call__(self, batch: od.InputBatchType) -> list[ObjectDetectionTargets]:
        """
        Parameters
        ----------
        batch : od.InputBatchType
            Sequence of batch images to perform inference on.

        Returns
        -------
        list[ObjectDetectionTargets]
            List of object detections, defined by boxes, labels, and scores.

        Raises
        ------
        Exception
            General exception for any error that occurs when processing inference
            results. To minimize the possibility of error, please ensure the model being
            wrapped is a YOLOv5 or YOLOv8 model designed for the object detection task.
        """
        # Convert to NumPy arrays (w/shape-(H,W,C)) for consistent preprocessing by native model
        # - i.e., the same preprocessing that would be applied when source is filename or PIL image
        if isinstance(batch[0], torch.Tensor):
            # Type ignore because ArrayLike is not guaranteed to have .cpu(), however,
            # we know for a fact it is a torch.Tensor here
            batch = [b.cpu().numpy().transpose((1, 2, 0)) for b in batch]  # type: ignore
        elif isinstance(batch[0], np.ndarray):
            # Type ignore because ArrayLike is not guaranteed to have .transpose,
            # however, we know for a fact it is an np.ndarray here
            batch = [b.transpose((1, 2, 0)) for b in batch]  # type: ignore
        else:
            batch = [np.array(b).transpose((1, 2, 0)) for b in batch]

        # Perform inference on batch
        results = self.model(batch, **self.kwargs)

        try:
            return self._format_results(results)
        except Exception:
            raise Exception(
                "MAITE could not process the model's inference result. Please ensure the model is either YOLOv5 or YOLOv8 designed for the object detection task."
            )


__all__ = ["YoloObjectDetector"]
