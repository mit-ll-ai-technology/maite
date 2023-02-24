from jatic_toolbox._internals.interop.import_utils import is_hf_available

if is_hf_available():
    from jatic_toolbox._internals.interop.huggingface.image_classifier import (
        HuggingFaceImageClassifier,
    )
    from jatic_toolbox._internals.interop.huggingface.object_detection import (
        HuggingFaceObjectDetector,
    )

    __all__ = ["HuggingFaceObjectDetector", "HuggingFaceImageClassifier"]
