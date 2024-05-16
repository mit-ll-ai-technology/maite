# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# This module is designed to intentionally include dependencies (i.e. mygrad)
# that have not been installed by default. maite should handle this
# gracefully.

from mygrad import tensor

__all__ = ["func_needs_mygrad"]


def func_needs_mygrad() -> None:
    tensor(1)
    return None
