# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

__all__ = ["ToolBoxException", "InternalError", "ToolBoxException"]


class ToolBoxException(Exception):
    # doc-ignore: NOQA
    """Base exception thrown by the toolbox."""


class InternalError(ToolBoxException, AssertionError):
    # doc-ignore: NOQA
    """An internal function was misconfigured or misused."""


class InvalidArgument(ToolBoxException, TypeError):
    # doc-ignore: NOQA
    """A toolbox interface was passed a bad value or type."""
