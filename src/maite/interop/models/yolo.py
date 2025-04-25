# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import warnings

from maite._internals.import_utils import (
    is_torch_available,
    is_ultralytics_available,
    is_yolov5_available,
)

if is_torch_available() and is_yolov5_available() and is_ultralytics_available():
    from maite._internals.interop.models.yolo import YoloObjectDetector

    __all__ = ["YoloObjectDetector"]
else:
    warnings.warn(
        "The `YoloObjectDetector` wrapper requires the torch, yolov5, and ultralytics packages, "
        "which can be installed with the command: `pip install maite[yolo-models]`."
    )
