import importlib
import importlib.util


def is_numpy_available():
    return importlib.util.find_spec("numpy") is not None


def is_torch_available():
    return importlib.util.find_spec("torch") is not None


def is_hf_datasets_available():
    return importlib.util.find_spec("datasets") is not None


def is_hf_transformers_available():
    return importlib.util.find_spec("transformers") is not None


def is_hf_available():
    return is_hf_transformers_available() and is_hf_datasets_available()


def is_hydra_zen_available():
    return importlib.util.find_spec("hydra_zen") is not None


def is_augly_available():
    return importlib.util.find_spec("augly") is not None


def is_smqtk_available():
    return importlib.util.find_spec("smqtk_detection") is not None


def is_pytest_available():
    return importlib.util.find_spec("pytest") is not None
