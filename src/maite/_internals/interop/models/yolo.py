# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from typing_extensions import Sequence, TypeAlias, cast
from ultralytics.engine.results import Results
from ultralytics.models import YOLO
from yolov5.models.common import AutoShape, Detections

from maite.protocols import ModelMetadata
from maite.protocols import object_detection as od

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

    Notes
    -----
    Only Ultralytics YOLOv5 and YOLOv8 models are currently supported.

    Examples
    --------

    Import relevant Python libraries.

    >>> import io
    >>> import contextlib
    >>> import numpy as np
    >>> from typing_extensions import Sequence
    >>> from ultralytics.models import YOLO
    >>> from maite.interop.models.yolo import YoloObjectDetector
    >>> from maite.protocols import object_detection as od

    In the following code, we capture stdout and stderr to make automated docstring testing easier. This is not necessary for typical use.

    Load an Ultralytics-hosted YOLOv5 model, 'yolov5nu',

    >>> with (
    ...     contextlib.redirect_stdout(io.StringIO()),
    ...     contextlib.redirect_stderr(io.StringIO()),
    ... ):
    ...     yolov5_model = YOLO("yolov5nu")
    >>> metadata = ModelMetadata(id="YOLOv5nu", index2label=yolov5_model.names)
    >>> wrapped_yolov5_model = YoloObjectDetector(yolov5_model, metadata)

    or, load an Ultralytics YOLOv5 model from a local filepath.

    >>> with (
    ...     contextlib.redirect_stdout(io.StringIO()),
    ...     contextlib.redirect_stderr(io.StringIO()),
    ... ):
    ...     yolov5_model = YOLO("./yolov5nu.pt")
    >>> metadata = ModelMetadata(id="YOLOv5nu", index2label=yolov5_model.names)
    >>> wrapped_yolov5_model = YoloObjectDetector(yolov5_model, metadata)

    Load an Ultralytics-hosted YOLOv8 model, 'yolov8n',

    >>> with (
    ...     contextlib.redirect_stdout(io.StringIO()),
    ...     contextlib.redirect_stderr(io.StringIO()),
    ... ):
    ...     yolov8_model = YOLO("yolov8n")
    >>> metadata = ModelMetadata(id="YOLOv8n", index2label=yolov8_model.names)
    >>> wrapped_yolov8_model = YoloObjectDetector(yolov8_model, metadata)

    or, load an Ultralytics YOLOv8 model from a local filepath.

    >>> with (
    ...     contextlib.redirect_stdout(io.StringIO()),
    ...     contextlib.redirect_stderr(io.StringIO()),
    ... ):
    ...     yolov8_model = YOLO("./yolov8n.pt")
    >>> metadata = ModelMetadata(id="YOLOv8n", index2label=yolov8_model.names)
    >>> wrapped_yolov8_model = YoloObjectDetector(yolov8_model, metadata)

    Perform object detection inference with the model.

    >>> N_DATAPOINTS = 5  # datapoints in dataset
    >>> C = 3  # number of color channels
    >>> H = 5  # img height
    >>> W = 6  # img width
    >>> batch_data: Sequence[od.InputType] = list(np.random.rand(N_DATAPOINTS, C, H, W))
    >>> model_results: Sequence[od.TargetType] = wrapped_yolov8_model(batch_data)
    >>> print(model_results)
    [ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32)), ObjectDetectionTargets(boxes=array([], shape=(0, 4), dtype=float32), labels=array([], dtype=uint8), scores=array([], dtype=float32))]
    """

    def __init__(
        self,
        model: Union[YOLO, AutoShape],
        metadata: ModelMetadata,
        yolo_inference_args: Optional[dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        model : YOLO | AutoShape
            A loaded Ultralytics YOLO model. The model can be loaded via the
            `ultralytics` or `yolov5` Python Library. Models must be either YOLOv5 or
            YOLOv8 and must be designed for the object detection task.

        metadata : ModelMetadata
            A typed dictionary containing at least an 'id' field of type str.

        yolo_inference_args : Optional[dict[str, Any]], (default=None)
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
        self.metadata = metadata
        self.yolo_inference_args: dict[str, Any] = (
            yolo_inference_args if yolo_inference_args is not None else {}
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

    def __call__(self, batch: Sequence[od.InputType]) -> list[ObjectDetectionTargets]:
        # doc-ignore: EX01
        """
        Make a model prediction for inputs in input batch.

        Parameters
        ----------
        batch : Sequence[od.InputType]
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
            batch = cast(Sequence[torch.Tensor], batch)
            batch = [b.cpu().numpy().transpose((1, 2, 0)) for b in batch]
        elif isinstance(batch[0], np.ndarray):
            # Type ignore because ArrayLike is not guaranteed to have .transpose,
            # however, we know for a fact it is an np.ndarray here
            batch = cast(Sequence[np.ndarray], batch)
            batch = [b.transpose((1, 2, 0)) for b in batch]
        else:
            batch = [np.array(b).transpose((1, 2, 0)) for b in batch]

        # Perform inference on batch
        results = self.model(batch, **self.yolo_inference_args)

        try:
            return self._format_results(results)
        except Exception:
            raise Exception(
                "MAITE could not process the model's inference result. Please ensure the model is either YOLOv5 or YOLOv8 designed for the object detection task."
            )


__all__ = ["YoloObjectDetector"]
