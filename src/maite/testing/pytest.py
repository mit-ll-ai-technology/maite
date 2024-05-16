# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from maite._internals.import_utils import is_pytest_available

if is_pytest_available():
    from maite._internals.testing.pytest import cleandir
else:
    raise ImportError(
        "maite.testing.pytest requires that pytest be installed as a dependency."
    )  # pragma: no cover

__all__ = ["cleandir"]
