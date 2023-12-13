# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from maite._internals.import_utils import (
    is_numpy_available,
    is_pil_available,
    is_torch_available,
)
from maite._internals.interop.utils import is_numpy_array, is_pil_image, is_torch_tensor


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


def test_is_pil_image():
    if is_pil_available():
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.zeros((10, 10, 3)).astype(np.uint8))

        assert is_pil_image(img)
    else:
        assert not is_pil_image([1, 2])
