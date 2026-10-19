# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verifiable image-classification primitive variants (carry beartype ``Is[...]``).

Each symbol here is the ``Annotated[Base, Is[...]]`` form of the corresponding plain
primitive in :py:mod:`maite.protocols.image_classification`. They carry beartype value-based
validators for use with :py:func:`maite.spotcheck.spotcheck_component`, and are NOT usable
with ``isinstance``.
"""

import warnings
from typing import TypeAlias

from maite._internals.import_utils import is_beartype_available

if is_beartype_available():
    from maite._internals.protocols.runtime_verify.image_classification import (  # isort:skip
        Image as _Image,
        ImgClassification as _ImgClassification,
        InputType as _InputType,
        TargetType as _TargetType,
    )

    Image: TypeAlias = _Image
    """Runtime-verifiable variant of :py:obj:`maite.protocols.image_classification.Image`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.image_classification.Image` for its
    semantics (`(C, H, W)` shape expectations).
    """

    ImgClassification: TypeAlias = _ImgClassification
    """Runtime-verifiable variant of :py:obj:`maite.protocols.image_classification.ImgClassification`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.image_classification.ImgClassification` for its
    semantics (`(Cl,)` shape expectations).
    """

    InputType: TypeAlias = _InputType
    """Runtime-verifiable variant of :py:obj:`maite.protocols.image_classification.InputType`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.image_classification.InputType` for its semantics.
    """

    TargetType: TypeAlias = _TargetType
    """Runtime-verifiable variant of :py:obj:`maite.protocols.image_classification.TargetType`.

    Carries beartype ``Is[...]`` validators; NOT usable with ``isinstance``.
    See :py:obj:`maite.protocols.image_classification.TargetType` for its semantics.
    """

    __all__ = ["Image", "ImgClassification", "InputType", "TargetType"]
else:
    warnings.warn(
        "The `runtime_verify` protocol variants require beartype, which can be installed via `experimental` extra",
        stacklevel=2,
    )
    __all__ = []
