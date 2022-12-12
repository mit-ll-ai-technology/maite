class ToolBoxException(Exception):
    """Base exception thrown by the toolbox"""


class InternalError(ToolBoxException, AssertionError):
    """An internal function was misconfigured or misused."""


class InvalidArgument(ToolBoxException, TypeError):
    """A toolbox interface was passed a bad value or type."""
