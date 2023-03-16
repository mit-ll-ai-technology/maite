from jatic_toolbox._internals.import_utils import is_torch_available

if is_torch_available():
    from jatic_toolbox._internals.interop.augmentation.wrappers import (
        AugmentationWrapper,
    )

    __all__ = ["AugmentationWrapper"]
