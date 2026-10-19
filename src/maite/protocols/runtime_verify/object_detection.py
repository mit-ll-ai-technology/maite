# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verifiable object-detection primitive variants (carry beartype ``Is[...]``).

Each symbol here is the ``Annotated[Base, Is[...]]`` form of the corresponding plain
primitive in :py:mod:`maite.protocols.object_detection`. They carry beartype value-based
validators for use with :py:func:`maite.spotcheck.spotcheck_component`, and are NOT usable
with ``isinstance``. See the canonical plain symbols (linked per-member below) for the
semantics of each type.
"""

import warnings
from typing import TypeAlias

from maite._internals.import_utils import is_beartype_available

if is_beartype_available():
    from maite._internals.protocols.runtime_verify.object_detection import (  # isort:skip
        Image as _Image,
        InputType as _InputType,
        TargetType as _TargetType,
        ObjectDetectionTarget as _ObjectDetectionTarget,
    )

    ObjectDetectionTarget: TypeAlias = _ObjectDetectionTarget
    """Runtime-verifiable variant of :py:obj:`maite.protocols.object_detection.ObjectDetectionTarget`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.object_detection.ObjectDetectionTarget` for its semantics
    (`boxes`/`labels`/`scores` and their shapes).
    """

    Image: TypeAlias = _Image
    """Runtime-verifiable variant of :py:obj:`maite.protocols.object_detection.Image`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.object_detection.Image` for its semantics
    (`(C, H, W)` shape expectations).
    """

    InputType: TypeAlias = _InputType
    """Runtime-verifiable variant of :py:obj:`maite.protocols.object_detection.InputType`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.object_detection.InputType` for its semantics.
    """

    TargetType: TypeAlias = _TargetType
    """Runtime-verifiable variant of :py:obj:`maite.protocols.object_detection.TargetType`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.object_detection.TargetType` for its semantics.
    """

    __all__ = ["ObjectDetectionTarget", "Image", "InputType", "TargetType"]
else:
    warnings.warn(
        "The `runtime_verify` protocol variants require beartype, which can be installed via `experimental` extra",
        stacklevel=2,
    )
