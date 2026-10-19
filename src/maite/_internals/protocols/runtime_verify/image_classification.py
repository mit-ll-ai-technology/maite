# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verification typing for the image-classification AI problem.

This module owns everything ``spotcheck`` needs that must *not* leak into the
isinstance-able public primitives:

* the value-based predicate functions (shape checks),
* the ``Annotated[Base, Is[...]]`` semantic aliases that carry those predicates,
* the per-task ``TV_SUB_MAP`` (``TypeVar -> Annotated`` alias), and
* ``RUNTIME_VERIFY_LOOKUP`` (``id(plain primitive) -> Annotated`` alias), the
  identity-keyed table ``spotcheck`` uses to re-attach predicates at runtime.

The dependency direction is strictly ``runtime_verify -> protocols``: the plain
(isinstance-able, non-``Annotated``) primitives live in
:mod:`maite._internals.protocols.image_classification`; here we import each one and
wrap it, never the reverse. Wrapping the *imported* plain primitive (rather than
re-referencing its underlying base) keeps the base single-sourced, so the two
forms cannot drift apart.
"""

from collections.abc import Mapping
from typing import Annotated, TypeVar

import numpy as np
from typing_extensions import TypeForm

from maite._internals.compat import Is
from maite._internals.protocols import generic as gen
from maite._internals.protocols import image_classification as ic
from maite.protocols import ArrayLike, DatumMetadata


# Define predicates with which to enrich semantic aliases for verifiability
def is_3dim(x: ArrayLike) -> bool:
    return np.asarray(x).ndim == 3


def is_1dim(x: ArrayLike) -> bool:
    return np.asarray(x).ndim == 1


# Semantic aliases carrying the value-based predicates. The base is taken from the plain
# primitives imported from ``protocols.image_classification``.

# Idempotency of hint enrichment is guaranteed structurally by ``_hint_substitute``,
# which treats an ``Annotated`` whose base is itself a lookup key as already-enriched -- see
# ``maite._internals.spotcheck.generic_hint_resolution``.)

#  We wrap ``ic.Image.__value__`` rather than ``ic.Image`` itself for beartype digestion:
# ``ic.Image`` is a ``typing_extensions.TypeAliasType`` (a backport), which beartype cannot
# digest on Python < 3.12 -- it only recognizes the native ``typing.TypeAliasType``.
# This issue is being tracked here: https://github.com/beartype/beartype/issues/693
Image = Annotated[ic.Image.__value__, Is[is_3dim]]
ImgClassification = Annotated[ic.ImgClassification.__value__, Is[is_1dim]]

InputType = Image
TargetType = ImgClassification
DatumMetadataType = DatumMetadata

# --- spotcheck TypeVar substitution map ---
# (consumed by maite._internals.spotcheck.spotcheck_tasks). Maps every variance flavor
# of the generic protocol TypeVars to a single "semantic alias" per role, so a polymorphic
# task (e.g. `evaluate`/`predict`) that parametrizes protocols by its own TypeVars can be
# closed at spotcheck time. When beartype is available the aliases carry lightweight
# validators; otherwise they collapse to the plain types and add no extra validation.
# (The *_in TypeVars are currently unused by any protocol; kept for forward-compat.)
# The `Is[...]` predicates are inert when beartype is unavailable (see maite._internals.compat).

TV_SUB_MAP: Mapping[TypeVar, TypeForm] = {
    gen.InputType_co: Image,
    gen.InputType_cn: Image,
    gen.InputType_in: Image,
    gen.TargetType_co: ImgClassification,
    gen.TargetType_cn: ImgClassification,
    gen.TargetType_in: ImgClassification,
    gen.DatumMetadataType_co: DatumMetadataType,
    gen.DatumMetadataType_cn: DatumMetadataType,
    gen.DatumMetadataType_in: DatumMetadataType,
}

# --- identity-keyed plain -> Annotated lookup ---
# Keyed on object identity (``id``) of the plain primitives, because IC's ``Image``
# and ``ImgClassification`` are both backed by the same ``ArrayLike`` base and differ
# only by predicate: a structure-keyed lookup could not tell them apart. spotcheck
# uses this table to re-attach predicates to the plain primitives after generic
# TypeVar substitution.
RUNTIME_VERIFY_LOOKUP: dict[int, TypeForm] = {
    id(ic.Image): Image,
    id(ic.ImgClassification): ImgClassification,
}
