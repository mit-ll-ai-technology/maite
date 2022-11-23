"""
Utilities for checking user inputs to interfaces and raising legible, explicit error
messages.
"""

from jatic_toolbox._internals.validation import check_domain, check_type

__all__ = ["check_type", "check_domain"]
