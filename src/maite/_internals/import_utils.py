# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

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


def is_pytest_available():
    return importlib.util.find_spec("pytest") is not None


def is_torchmetrics_available():
    return importlib.util.find_spec("torchmetrics") is not None


def is_tqdm_available():
    return importlib.util.find_spec("tqdm") is not None


def is_pil_available():
    return importlib.util.find_spec("PIL") is not None


def is_torcheval_available():
    return importlib.util.find_spec("torcheval") is not None


def is_hypothesis_available():
    return importlib.util.find_spec("hypothesis") is not None


if is_pytest_available():
    import pytest

    requires_torch = pytest.mark.skipif(
        not is_torch_available(), reason="test requires torch"
    )
    requires_torchvision = pytest.mark.skipif(
        not is_torchvision_available(), reason="test requires torchvision"
    )
    requires_timm = pytest.mark.skipif(
        not is_timm_available(), reason="test requires timm"
    )
    requires_hf_hub = pytest.mark.skipif(
        not is_hf_hub_available(), reason="test requires huggingface_hub"
    )
    requires_hf_datasets = pytest.mark.skipif(
        not is_hf_datasets_available(), reason="test requires datasets"
    )
    requires_hf_transformers = pytest.mark.skipif(
        not is_hf_transformers_available(), reason="test requires transformers"
    )
    requires_torchmetrics = pytest.mark.skipif(
        not is_torchmetrics_available(), reason="test requires torchmetrics"
    )
    requires_pil = pytest.mark.skipif(
        not is_pil_available(), reason="test requires PIL"
    )
    requires_torcheval = pytest.mark.skipif(
        not is_torcheval_available(), reason="test requires torcheval"
    )
