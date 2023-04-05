import importlib
import importlib.util


def is_numpy_available():
    return importlib.util.find_spec("numpy") is not None


def is_torch_available():
    return importlib.util.find_spec("torch") is not None


def is_torchvision_available():
    return importlib.util.find_spec("torchvision") is not None


def is_timm_available():
    return importlib.util.find_spec("timm") is not None


def is_hf_hub_available():
    return importlib.util.find_spec("huggingface_hub") is not None


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


def is_torchmetrics_available():
    return importlib.util.find_spec("torchmetrics") is not None


def is_tqdm_available():
    return importlib.util.find_spec("tqdm") is not None


def is_pil_available():
    return importlib.util.find_spec("PIL") is not None
