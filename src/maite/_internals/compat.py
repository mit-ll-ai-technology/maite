# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import sys

from maite._internals.import_utils import is_beartype_available

__all__ = ["TypedDict", "Is"]

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


# `Is` is beartype's validator-factory used to attach lightweight predicates to
# `Annotated` hints (e.g. `Annotated[Image, Is[is_3dim]]`). beartype is an optional
# ("experimental") dependency, so we mediate its import here the same way we do for
# `TypedDict`: when beartype is installed we re-export the real factory; otherwise we
# provide an inert mock so `Annotated[X, Is[pred]]` still constructs. The mock's metadata
# is never consumed because `maite.spotcheck` refuses to expose `spotcheck` without beartype.
if is_beartype_available():
    from beartype.vale import Is  # type: ignore (intentionally shadowing hard-coded `Is` of different type)
else:

    class _VapidValidator:
        # tolerate future `Is[a] & Is[b]` / `Is[a] | Is[b]` combinator use
        def __and__(self, other: object) -> "_VapidValidator":
            return self

        def __or__(self, other: object) -> "_VapidValidator":
            return self

        def __repr__(self) -> str:
            return "Is[<beartype-unavailable>]"

    class Is:  # deliberately mirrors the beartype.vale.Is name
        def __class_getitem__(cls, item: object) -> "_VapidValidator":
            return _VapidValidator()
