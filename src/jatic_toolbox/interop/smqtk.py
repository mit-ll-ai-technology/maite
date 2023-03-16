from jatic_toolbox._internals.import_utils import is_numpy_available

if is_numpy_available():
    from jatic_toolbox._internals.interop.smqtk.object_detection import CenterNet

    __all__ = ["CenterNet"]
