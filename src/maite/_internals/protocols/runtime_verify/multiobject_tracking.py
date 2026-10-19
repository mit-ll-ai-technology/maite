# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verification typing for the multi-object-tracking AI problem.

This module owns everything ``spotcheck`` needs that must *not* leak into the
isinstance-able public primitives:

* the value-based predicate functions (cardinality / shape checks),
* the ``Annotated[Base, Is[...]]`` semantic aliases that carry those predicates,
* the per-task ``TV_SUB_MAP`` (``TypeVar -> Annotated`` alias), and
* ``RUNTIME_VERIFY_LOOKUP`` (``id(plain primitive) -> Annotated`` alias), the
  identity-keyed table ``spotcheck`` uses to re-attach predicates at runtime.

The dependency direction is strictly ``runtime_verify -> protocols``: the plain
(isinstance-able, non-``Annotated``) primitives live in
:mod:`maite._internals.protocols.multiobject_tracking`; here we import each one and
wrap it, never the reverse. Wrapping the *imported* plain primitive (rather than
re-referencing its underlying base) keeps the base single-sourced, so the two forms
cannot drift apart.

``VideoStream`` carries no value predicate (there is nothing to verify about a frame
iterable beyond its structure), so it has no ``Annotated`` counterpart and no
``RUNTIME_VERIFY_LOOKUP`` entry; the plain primitive is used directly in the
``TV_SUB_MAP`` to close the input TypeVars.
"""

from collections.abc import Mapping
from typing import Annotated, TypeVar

import numpy as np
from typing_extensions import TypeForm

from maite._internals.compat import Is
from maite._internals.protocols import generic as gen
from maite._internals.protocols import multiobject_tracking as mot


# Define predicates with which to enrich semantic aliases for verifiability
def has_compatible_track_cardinality(x: mot.MultiobjectTrackingTarget) -> bool:
    for frame_track in x.frame_tracks:
        boxes_arr = np.asarray(frame_track.boxes)
        labels_arr = np.asarray(frame_track.labels)
        scores_arr = np.asarray(frame_track.scores)
        track_ids_arr = np.asarray(frame_track.track_ids)

        # all must have size 0 OR equal first dimension
        if not (
            (boxes_arr.size == labels_arr.size == scores_arr.size == track_ids_arr.size == 0)
            or (boxes_arr.shape[0] == labels_arr.shape[0] == scores_arr.shape[0] == track_ids_arr.shape[0])
        ):
            return False
    return True


def boxes_has_4_cols(x: mot.MultiobjectTrackingTarget) -> bool:
    for frame_track in x.frame_tracks:
        boxes_arr = np.asarray(frame_track.boxes)
        if not (boxes_arr.size == 0 or boxes_arr.shape[1] == 4):
            return False
    return True


# Semantic aliases carrying the value-based predicates. The base is taken from the plain
# primitives imported from ``protocols.multiobject_tracking``.

# Idempotency of hint enrichment is guaranteed structurally by ``_hint_substitute``,
# which treats an ``Annotated`` whose base is itself a lookup key as already-enriched -- see
# ``maite._internals.spotcheck.generic_hint_resolution``.)

#  We wrap ``mot.VideoStream.__value__`` rather than ``mot.VideoStream`` itself for beartype digestion:
# ``mot.VideoStream`` is a ``typing_extensions.TypeAliasType`` (a backport), which beartype cannot
# digest on Python < 3.12 -- it only recognizes the native ``typing.TypeAliasType``.
# This issue is being tracked here: https://github.com/beartype/beartype/issues/693

VideoStream = mot.VideoStream.__value__
MultiobjectTrackingTarget = Annotated[
    mot.MultiobjectTrackingTarget,
    Is[has_compatible_track_cardinality],
    Is[boxes_has_4_cols],
]

InputType = VideoStream
TargetType = MultiobjectTrackingTarget
DatumMetadataType = mot.DatumMetadata

# --- spotcheck TypeVar substitution map ---
# (consumed by maite._internals.spotcheck.spotcheck_tasks). Maps every variance flavor
# of the generic protocol TypeVars to a single "semantic alias" per role, so a polymorphic
# task (e.g. `evaluate`/`predict`) that parametrizes protocols by its own TypeVars can be
# closed at spotcheck time. When beartype is available the aliases carry lightweight
# validators; otherwise they collapse to the plain types and add no extra validation.
# (The *_in TypeVars are currently unused by any protocol; kept for forward-compat.)
# The `Is[...]` predicates are inert when beartype is unavailable (see maite._internals.compat).

TV_SUB_MAP: Mapping[TypeVar, TypeForm] = {
    gen.InputType_co: VideoStream,
    gen.InputType_cn: VideoStream,
    gen.InputType_in: VideoStream,
    gen.TargetType_co: MultiobjectTrackingTarget,
    gen.TargetType_cn: MultiobjectTrackingTarget,
    gen.TargetType_in: MultiobjectTrackingTarget,
    gen.DatumMetadataType_co: DatumMetadataType,
    gen.DatumMetadataType_cn: DatumMetadataType,
    gen.DatumMetadataType_in: DatumMetadataType,
}

# --- identity-keyed plain -> beartype-digestible replacement lookup ---
# Keyed on object identity (``id``) of the plain primitives. spotcheck uses this table to
# rewrite the plain-primitive leaves after generic TypeVar substitution. ``MultiobjectTrackingTarget``
# is replaced by its predicate-bearing ``Annotated`` form; ``VideoStream`` is replaced by its
# beartype-digestible base (no predicate) so the bare ``TypeAliasType`` leaf never reaches beartype.
RUNTIME_VERIFY_LOOKUP: dict[int, TypeForm] = {
    id(mot.VideoStream): VideoStream,
    id(mot.MultiobjectTrackingTarget): MultiobjectTrackingTarget,
}
