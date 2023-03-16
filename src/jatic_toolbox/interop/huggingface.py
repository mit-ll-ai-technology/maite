from jatic_toolbox._internals.import_utils import is_torch_available

if is_torch_available():
    from jatic_toolbox._internals.interop.huggingface.image_classifier import (
        HuggingFaceImageClassifier,
    )
    from jatic_toolbox._internals.interop.huggingface.object_detection import (
        HuggingFaceObjectDetector,
    )

    __all__ = ["HuggingFaceObjectDetector", "HuggingFaceImageClassifier"]
