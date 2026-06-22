# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import random

import numpy as np
import torch
from hypothesis import given

# PT019, hypothesis.errors.InvalidArgument is raised if `given` is not
# called with at least one argument.


@given(...)
def test_python_random_controlled_by_hypothesis(_dummy: None):  # noqa: PT019
    # hypothesis sets initial seed to 0 within scope of test
    pre_seed = random.random()  # noqa: S311
    random.seed(0)
    assert pre_seed == random.random()  # noqa: S311


@given(...)
def test_torch_random_controlled_by_hypothesis(_dummy: None):  # noqa: PT019
    # hypothesis sets initial seed to 0 within scope of test
    pre_seed = torch.rand((1,)).item()
    torch.manual_seed(0)
    assert pre_seed == torch.rand((1,)).item()


@given(...)
def test_numpy_random_controlled_by_hypothesis(_dummy: None):  # noqa: PT019
    # hypothesis sets initial seed to 0 within scope of test
    pre_seed = np.random.random((1,)).item()  # noqa: NPY002
    np.random.seed(0)  # noqa: NPY002
    assert pre_seed == np.random.random((1,)).item()  # noqa: NPY002
