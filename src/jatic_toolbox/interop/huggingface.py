# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from jatic_toolbox._internals.import_utils import is_torch_available

if is_torch_available():
    from jatic_toolbox._internals.interop.huggingface.datasets import (
        HuggingFaceObjectDetectionDataset,
        HuggingFaceVisionDataset,
    )
    from jatic_toolbox._internals.interop.huggingface.image_classifier import (
        HuggingFaceImageClassifier,
    )
    from jatic_toolbox._internals.interop.huggingface.object_detection import (
        HuggingFaceObjectDetector,
    )

    __all__ = [
        "HuggingFaceObjectDetector",
        "HuggingFaceImageClassifier",
        "HuggingFaceObjectDetectionDataset",
        "HuggingFaceVisionDataset",
    ]
