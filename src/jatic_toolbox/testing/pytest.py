from warnings import warn

try:
    from jatic_toolbox._internals.testing.pytest import cleandir
except ImportError as e:  # pragma: no cover
    warn(
        "jatic_toolbox.testing.pytest requires that pytest be installed as a dependency."
    )
    raise e

__all__ = ["cleandir"]
