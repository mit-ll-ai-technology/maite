import contextlib
import io

import pytest

from maite._internals.import_utils import (
    is_ultralytics_available,
    is_yolov5_available,
)

# Skip this whole module if dependencies are not installed (avoiding bad imports)
# (This is redundant behind conftest.py collect_ignore_glob)
if not is_yolov5_available() or not is_ultralytics_available():
    pytest.skip("test module requires both yolov5 and ultralytics packages", allow_module_level=True)

from ultralytics.models import YOLO

from maite.interop.models.yolo import YoloObjectDetector
from maite.protocols import ModelMetadata

pytestmark = pytest.mark.yolo_models


def test_load_yolo_wrapper():
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yolov5_model = YOLO("yolov5nu")

    metadata = ModelMetadata(id="test", index2label=yolov5_model.names)
    YoloObjectDetector(yolov5_model, metadata)
