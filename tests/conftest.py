# flake8: noqa

import sys

import hypothesis.strategies as st
import pkg_resources

from jatic_toolbox.testing.pytest import cleandir  # noqa: F401

st.register_type_strategy(st.DataObject, st.data())

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []

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
