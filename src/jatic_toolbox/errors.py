__all__ = ["ToolBoxException", "InternalError", "ToolBoxException"]


class ToolBoxException(Exception):
    """Base exception thrown by the toolbox"""

    # doc-ignore: NOQA


class InternalError(ToolBoxException, AssertionError):
    """An internal function was misconfigured or misused."""

    # doc-ignore: NOQA


class InvalidArgument(ToolBoxException, TypeError):
    """A toolbox interface was passed a bad value or type."""

    # doc-ignore: NOQA
