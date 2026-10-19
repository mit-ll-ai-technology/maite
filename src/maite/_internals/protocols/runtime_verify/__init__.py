# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verification typing for MAITE's AI-problem protocols.

This package is the single home for everything ``spotcheck`` needs but that must
*not* leak into the isinstance-able public primitives:

* the value-based predicate functions (shape / cardinality checks),
* the ``Annotated[Base, Is[...]]`` semantic aliases that carry those predicates,
* the per-task ``TV_SUB_MAP`` (``TypeVar -> Annotated`` alias) consumed by
  ``maite._internals.spotcheck.spotcheck_tasks``.

The dependency direction is strictly ``runtime_verify.<ai_problem> -> protocols.<ai_problem>``:
these modules import the *plain* protocol classes and base types from
``maite._internals.protocols.<ai_problem>`` and wrap them, never the reverse. Keeping the
predicates out of ``protocols.<ai_problem>`` is what lets the base protocol classes there
be parametrized by plain (isinstance-able, non-``Annotated``) primitives while
``spotcheck`` re-attaches the predicates at runtime.

``RUNTIME_VERIFY_LOOKUP`` below merges the per-task identity-keyed lookups into a single
``{id(plain primitive) -> Annotated form}`` table. ``spotcheck`` (via
``proxy_factory``) uses it to re-attach value predicates to member-hint leaves after
generic TypeVar substitution. Merging is safe: the keys are ``id(...)`` of distinct
primitive objects across tasks, so they cannot collide.
"""

from typing_extensions import TypeForm

from maite._internals.protocols.runtime_verify import image_classification as _ic
from maite._internals.protocols.runtime_verify import multiobject_tracking as _mot
from maite._internals.protocols.runtime_verify import object_detection as _od

RUNTIME_VERIFY_LOOKUP: dict[int, TypeForm] = {
    **_ic.RUNTIME_VERIFY_LOOKUP,
    **_od.RUNTIME_VERIFY_LOOKUP,
    **_mot.RUNTIME_VERIFY_LOOKUP,
}
