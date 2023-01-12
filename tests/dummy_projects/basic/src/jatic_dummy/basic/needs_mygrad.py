# This module is designed to intentionally include dependencies (i.e. mygrad)
# that have not been installed by default. jatic-toolbox should handle this
# gracefully.

from mygrad import tensor

__all__ = ["func_needs_mygrad"]


def func_needs_mygrad() -> None:
    tensor(1)
    return None
