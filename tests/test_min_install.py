# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest

import maite
from maite._internals.import_utils import (
    is_hf_datasets_available,
    is_hf_hub_available,
    is_hf_transformers_available,
    is_torcheval_available,
    is_torchmetrics_available,
    is_torchvision_available,
)


@pytest.mark.skipif(
    (
        is_torchvision_available()
        or is_hf_datasets_available()
        or is_hf_transformers_available()
        or is_hf_hub_available()
    ),
    reason="test requires the following not be installed: torchvision, datasets, transformers and huggingface_hub",
)

@pytest.mark.skipif(
    (is_torcheval_available() or is_torchmetrics_available()),
    reason="test requires the following not be installed: torchmetrics and torcheval",
)
