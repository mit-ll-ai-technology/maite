# flake8: noqa

import hypothesis.strategies as st
from hypothesis import register_random

from jatic_toolbox.testing.pytest_fixtures import cleandir  # noqa: F401

st.register_type_strategy(st.DataObject, st.data())


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
