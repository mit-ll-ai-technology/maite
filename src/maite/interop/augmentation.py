# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from maite._internals.import_utils import is_torch_available

if is_torch_available():
    from maite._internals.interop.augmentation.wrappers import AugmentationWrapper

    __all__ = ["AugmentationWrapper"]
