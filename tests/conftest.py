# flake8: noqa

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import hypothesis.strategies as st
import pkg_resources

from jatic_toolbox.testing.pytest import cleandir  # noqa: F401
from tests import all_dummy_subpkgs

st.register_type_strategy(st.DataObject, st.data())

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []


def _safe_find_spec(pkg: str):
    # returning `None` means that module/subpackage is not installed
    try:
        return find_spec(pkg)
    except ModuleNotFoundError:
        return None


for subpkg in all_dummy_subpkgs:
    if _safe_find_spec(f"jatic_dummy.{subpkg}") is None:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                str((Path(__file__).parent / f"dummy_projects/{subpkg}").absolute()),
            ]
        )

if "torch" in sys.modules:
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


_installed = {pkg.key for pkg in pkg_resources.working_set}


hydra_zen_installed = "hydra-zen" in _installed
if not hydra_zen_installed:
    collect_ignore_glob.append("*hydra_zen*.py")


huggingface_installed = True
for _module_name in ("datasets", "transformers", "huggingface-hub"):
    if _module_name not in _installed:
        huggingface_installed = False
        break
if not huggingface_installed:
    collect_ignore_glob.append("*huggingface*.py")


smqtk_installed = True
for _module_name in ("smqtk-detection", "numba"):
    if _module_name not in _installed:
        smqtk_installed = False
        break
if not smqtk_installed or sys.version_info >= (3, 10):
    collect_ignore_glob.append("*smqtk*.py")

augly_installed = "augly" in _installed
if not augly_installed:
    collect_ignore_glob.append("*augly*.py")
