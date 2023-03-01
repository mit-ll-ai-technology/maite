from jatic_toolbox._internals.import_utils import is_numpy_available, is_torch_available
from jatic_toolbox._internals.interop.utils import is_numpy_array, is_torch_tensor


def test_is_torch_tensor():
    if is_torch_available():
        import torch as tr

        assert is_torch_tensor(tr.zeros(1))
    else:
        assert not is_torch_tensor([1, 2])


def test_is_numpy_array():
    if is_numpy_available():
        import numpy as np

        assert is_numpy_array(np.zeros(1))
    else:
        assert not is_numpy_array([1, 2])
