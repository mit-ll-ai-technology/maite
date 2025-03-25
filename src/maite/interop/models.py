# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

try:
    from maite._internals.interop.models.yolo import YoloObjectDetector
except ModuleNotFoundError:
    from typing import List

    from maite._internals.import_utils import (
        is_torch_available,
        is_ultralytics_available,
        is_yolov5_available,
    )

    modules: List[str] = []
    if not is_torch_available():
        modules.append("torch")
    if not is_yolov5_available():
        modules.append("yolov5")
    if not is_ultralytics_available():
        modules.append("ultralytics")
    raise ModuleNotFoundError(
        f"`YoloObjectDetector` requires the following missing package dependencies: {modules}. "
        "Please install them via `pip install maite[yolo-models]`."
    )
except Exception:
    raise Exception


__all__ = ["YoloObjectDetector"]
