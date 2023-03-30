from jatic_toolbox._internals.import_utils import is_torchvision_available

if is_torchvision_available():
    from jatic_toolbox._internals.interop.torchvision.torchvision import (
        TorchVisionClassifier,
        TorchVisionObjectDetector,
    )

    __all__ = ["TorchVisionObjectDetector", "TorchVisionClassifier"]
