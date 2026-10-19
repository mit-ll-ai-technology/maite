# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verifiable multi-object-tracking primitive variants.

Each symbol here is the ``Annotated[...]`` form of the corresponding plain primitive in
:py:mod:`maite.protocols.multiobject_tracking`, intended for use with
:py:func:`maite.spotcheck.spotcheck_component`. None are usable with ``isinstance``.
See the canonical plain symbols (linked per-member below) for the semantics of each type.
(Note `VideoStream` and `InputType` currently carry no value-based ``Is[...]`` predicate;
they are included for parity with the other task namespaces.)
"""

import warnings
from typing import TypeAlias

from maite._internals.import_utils import is_beartype_available

if is_beartype_available():
    from maite._internals.protocols.runtime_verify.multiobject_tracking import (  # isort:skip
        VideoStream as _VideoStream,
        MultiobjectTrackingTarget as _MultiobjectTrackingTarget,
        InputType as _InputType,
        TargetType as _TargetType,
    )

    MultiobjectTrackingTarget: TypeAlias = _MultiobjectTrackingTarget
    """Runtime-verifiable variant of :py:obj:`maite.protocols.multiobject_tracking.MultiobjectTrackingTarget`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.multiobject_tracking.MultiobjectTrackingTarget` for its
    semantics (`frame_tracks`).
    """

    VideoStream: TypeAlias = _VideoStream
    """Runtime-verifiable variant of :py:obj:`maite.protocols.multiobject_tracking.VideoStream`.

    Currently carries no value-based ``Is[...]`` predicate, and (as a subscripted generic)
    is NOT usable with ``isinstance``. See :py:obj:`maite.protocols.multiobject_tracking.VideoStream`
    for its semantics.
    """

    InputType: TypeAlias = _InputType
    """Runtime-verifiable variant of :py:obj:`maite.protocols.multiobject_tracking.InputType`.

    Currently carries no value-based ``Is[...]`` predicate, and is NOT usable with
    ``isinstance``. See :py:obj:`maite.protocols.multiobject_tracking.InputType` for its
    semantics.
    """

    TargetType: TypeAlias = _TargetType
    """Runtime-verifiable variant of :py:obj:`maite.protocols.multiobject_tracking.TargetType`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.multiobject_tracking.TargetType` for its semantics.
    """

    __all__ = ["VideoStream", "MultiobjectTrackingTarget", "InputType", "TargetType"]
else:
    warnings.warn(
        "The `runtime_verify` protocol variants require beartype, which can be installed via `experimental` extra",
        stacklevel=2,
    )
