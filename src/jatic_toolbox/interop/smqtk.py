from jatic_toolbox._internals.interop.import_utils import is_smqtk_available

if is_smqtk_available():
    from jatic_toolbox._internals.interop.smqtk.object_detection import (
        _MODELS,
        CenterNetVisdrone,
    )

    __all__ = ["CenterNetVisdrone", "_MODELS"]
