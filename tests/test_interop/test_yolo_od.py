import contextlib
import io

from ultralytics.models import YOLO

from maite.interop.models.yolo import YoloObjectDetector
from maite.protocols import ModelMetadata


def test_load_yolo_wrapper():
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yolov5_model = YOLO("yolov5nu")

    metadata = ModelMetadata(id="test", index2label=yolov5_model.names)
    YoloObjectDetector(yolov5_model, metadata)
