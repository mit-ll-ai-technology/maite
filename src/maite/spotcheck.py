# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT
"""MAITE runtime verification engine: 'spotcheck'"""

import warnings

from maite._internals.import_utils import is_beartype_available

if is_beartype_available():
    from maite._internals.spotcheck.proxy_factory import spotcheck_component
    from maite._internals.spotcheck.spotcheck_node import SpotcheckError
    from maite._internals.spotcheck.spotcheck_tasks import spotcheck

    SpotcheckError.__module__ = "maite.spotcheck"

    __all__ = ["spotcheck", "spotcheck_component", "SpotcheckError"]
else:
    warnings.warn(
        "The `spotcheck` runtime verification functionality requires beartype, "
        "which can be installed via `experimental` extra",
        stacklevel=2,
    )
