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
