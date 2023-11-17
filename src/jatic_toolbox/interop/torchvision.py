# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from jatic_toolbox._internals.import_utils import is_torchvision_available

if is_torchvision_available():
    from jatic_toolbox._internals.interop.torchvision.datasets import TorchVisionDataset
    from jatic_toolbox._internals.interop.torchvision.torchvision import (
        TorchVisionClassifier,
        TorchVisionObjectDetector,
    )

    __all__ = [
        "TorchVisionObjectDetector",
        "TorchVisionClassifier",
        "TorchVisionDataset",
    ]
