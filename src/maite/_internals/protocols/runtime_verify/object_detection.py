# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verification typing for the object-detection AI problem.

This module owns everything ``spotcheck`` needs that must *not* leak into the
isinstance-able public primitives:

* the value-based predicate functions (shape / cardinality checks),
* the ``Annotated[Base, Is[...]]`` semantic aliases that carry those predicates,
* the per-task ``TV_SUB_MAP`` (``TypeVar -> Annotated`` alias), and
* ``RUNTIME_VERIFY_LOOKUP`` (``id(plain primitive) -> Annotated`` alias), the
  identity-keyed table ``spotcheck`` uses to re-attach predicates at runtime.

The dependency direction is strictly ``runtime_verify -> protocols``: the plain
(isinstance-able, non-``Annotated``) primitives live in
:mod:`maite._internals.protocols.object_detection`; here we import each one and wrap
it, never the reverse. Wrapping the *imported* plain primitive (rather than
re-referencing its underlying base) keeps the base single-sourced, so the two forms
cannot drift apart.
"""

from collections.abc import Mapping
from typing import Annotated, TypeVar

import numpy as np
from typing_extensions import TypeForm

from maite._internals.compat import Is
from maite._internals.protocols import generic as gen
from maite._internals.protocols import object_detection as od
from maite.protocols import ArrayLike, DatumMetadata


# Define predicates with which to enrich semantic aliases for verifiability
def is_3dim(x: ArrayLike) -> bool:
    return np.asarray(x).ndim == 3


def has_matching_cardinality(x: od.ObjectDetectionTarget) -> bool:
    boxes_arr = np.asarray(x.boxes)
    scores_arr = np.asarray(x.scores)
    labels_arr = np.asarray(x.labels)

    return (boxes_arr.size == scores_arr.size == labels_arr.size == 0) or (
        boxes_arr.shape[0] == scores_arr.shape[0] == labels_arr.shape[0]
    )


def boxes_has_4_cols(x: od.ObjectDetectionTarget) -> bool:
    boxes_arr = np.asarray(x.boxes)
    return boxes_arr.size == 0 or (len(boxes_arr.shape) > 1 and boxes_arr.shape[1] == 4)


def is_scores_col_sum_lt_1(x: od.ObjectDetectionTarget) -> bool:
    scores = np.asarray(x.scores)
    # scores may be (N_DETECTIONS,) with one score per detection, or
    # (N_DETECTIONS, N_CLASSES) with per-class scores summed across classes.
    if scores.ndim == 1:
        return bool(np.all(scores < 1 + 1e-6))
    return bool(np.all(scores.sum(1) < 1 + 1e-6))


# Semantic aliases carrying the value-based predicates. The base is taken from the plain
# primitives imported from ``protocols.object_detection``.

# Idempotency of hint enrichment is guaranteed structurally by ``_hint_substitute``,
# which treats an ``Annotated`` whose base is itself a lookup key as already-enriched -- see
# ``maite._internals.spotcheck.generic_hint_resolution``.)

#  We wrap ``od.Image.__value__`` rather than ``od.Image`` itself for beartype digestion:
# ``od.Image`` is a ``typing_extensions.TypeAliasType`` (a backport), which beartype cannot
# digest on Python < 3.12 -- it only recognizes the native ``typing.TypeAliasType``.
# This issue is being tracked here: https://github.com/beartype/beartype/issues/693

Image = Annotated[od.Image.__value__, Is[is_3dim]]
ObjectDetectionTarget = Annotated[
    od.ObjectDetectionTarget,
    Is[has_matching_cardinality],
    Is[boxes_has_4_cols],
    Is[is_scores_col_sum_lt_1],
]

InputType = Image
TargetType = ObjectDetectionTarget
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
    gen.TargetType_co: ObjectDetectionTarget,
    gen.TargetType_cn: ObjectDetectionTarget,
    gen.TargetType_in: ObjectDetectionTarget,
    gen.DatumMetadataType_co: DatumMetadataType,
    gen.DatumMetadataType_cn: DatumMetadataType,
    gen.DatumMetadataType_in: DatumMetadataType,
}

# --- identity-keyed plain -> Annotated lookup ---
# Keyed on object identity (``id``) of the plain primitives. spotcheck uses this table
# to re-attach predicates to the plain primitives after generic TypeVar substitution.
RUNTIME_VERIFY_LOOKUP: dict[int, TypeForm] = {
    id(od.Image): Image,
    id(od.ObjectDetectionTarget): ObjectDetectionTarget,
}
