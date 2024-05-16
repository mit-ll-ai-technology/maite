# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# flake8: noqa

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import settings

from maite._internals import import_utils
from maite.testing.pytest import cleandir  # noqa: F401
from tests import all_dummy_subpkgs

st.register_type_strategy(st.DataObject, st.data())

settings.register_profile("cicd", max_examples=10, deadline=None)

if bool(os.environ.get("CI_JOB_ID")):
    print("*** Running in CI, using CI settings ***")
    settings.load_profile("cicd")

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []


def _safe_find_spec(pkg: str):
    # returning `None` means that module/subpackage is not installed
    try:
        return find_spec(pkg)
    except ModuleNotFoundError:
        return None


for subpkg in all_dummy_subpkgs:
    if _safe_find_spec(f"maite_dummy.{subpkg}") is None:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                str((Path(__file__).parent / f"dummy_projects/{subpkg}").absolute()),
            ]
        )

if import_utils.is_torch_available():
    from hypothesis import register_random

    class TorchRandomWrapper:
        def __init__(self):
            # This class provides a shim that matches torch to stdlib random,
            # and lets us avoid importing torch until it's already in use.
            from torch import default_generator

            self.seed = default_generator.manual_seed
            self.getstate = default_generator.get_state
            self.setstate = default_generator.set_state

    r = TorchRandomWrapper()
    # Note: do not specifying TorchRandomWrapper() inline. It will be garbage collected
    register_random(r)
else:
    collect_ignore_glob.append("requires_torch/**")

if not import_utils.is_torch_available():
    collect_ignore_glob.append("*torch*.py")

if not (
    import_utils.is_hf_transformers_available()
    and import_utils.is_hf_datasets_available()
    and import_utils.is_hf_hub_available()
):
    collect_ignore_glob.append("*huggingface*.py")

if not import_utils.is_torchvision_available():
    collect_ignore_glob.append("*torchvision*.py")

if not import_utils.is_torchmetrics_available():
    collect_ignore_glob.append("*torchmetrics*.py")

if not import_utils.is_torcheval_available():
    collect_ignore_glob.append("*torcheval*.py")
