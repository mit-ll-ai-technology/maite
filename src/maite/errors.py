# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

__all__ = ["MaiteException", "InternalError", "InvalidArgument"]


class MaiteException(Exception):
    # doc-ignore: NOQA
    """Base exception thrown by the MAITE."""


class InternalError(MaiteException, AssertionError):
    # doc-ignore: NOQA
    """An internal function was misconfigured or misused."""


class InvalidArgument(MaiteException, TypeError):
    # doc-ignore: NOQA
    """A MAITE interface was passed a bad value or type."""
