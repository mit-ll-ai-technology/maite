from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    try:
        from ._version import version as __version__
    except ImportError:  # pragma: no cover
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str
