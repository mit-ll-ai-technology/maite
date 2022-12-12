# flake8: noqa

import sys

import hypothesis.strategies as st

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
