# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
Utilities for validating documentation strings for maite projects.
"""

from maite._internals.testing.docs import (
    NumpyDocErrorCode,
    NumPyDocResults,
    validate_docstring,
)

__all__ = ["validate_docstring", "NumpyDocErrorCode", "NumPyDocResults"]
