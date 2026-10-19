"""Phase 5 tests for the isinstance-able primitives + runtime-verify re-injection.

These lock in the three requirements of the ``958-isinstance-and-spotchecking`` work:

* **Req 1** -- the semantic-alias primitives exposed from ``maite.protocols.<task>`` are
  *not* ``Annotated`` (so the protocol-class-backed ones are usable with ``isinstance``);
  the ``Annotated`` forms live under ``maite.protocols.runtime_verify.<task>``.
* **Req 2** -- an exposed union-backed primitive statically/repr-wise shows as its alias
  name (``Image``), not the expanded base.
* **Req 3 / Phase 4** -- ``spotcheck`` re-attaches the value predicates at runtime via the
  identity-keyed ``leaf_rewrite`` seam, so the now-plain primitives are still verified. The
  critical guarantee is that ``Image`` (3-dim) and ``ImgClassification`` (1-dim) are neither
  swapped nor merged during that re-attachment.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, get_args, get_origin

import numpy as np
import pytest
from typing_extensions import TypeForm

import maite.protocols.image_classification as ic
import maite.protocols.multiobject_tracking as mot
import maite.protocols.object_detection as od
import maite.protocols.runtime_verify.image_classification as rv_ic
import maite.protocols.runtime_verify.multiobject_tracking as rv_mot
import maite.protocols.runtime_verify.object_detection as rv_od
from maite._internals.spotcheck.generic_hint_resolution import _hint_substitute, resolve_protocol_members
from maite._internals.spotcheck.proxy_factory import (
    _ENRICH,
    RedundantSpotcheckWarning,
    _members_carry_value_predicate,
    spotcheck_component,
)
from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata
from maite.spotcheck import SpotcheckError

# ---------------------------------------------------------------------------
# Req 1: public primitives non-Annotated; runtime_verify counterparts Annotated
# ---------------------------------------------------------------------------

# The semantic-alias primitives that must stay plain (non-Annotated) on the public surface.
PLAIN_PRIMITIVES = [
    ic.Image,
    ic.ImgClassification,
    od.Image,
    od.ObjectDetectionTarget,
    mot.VideoStream,
    mot.MultiobjectTrackingTarget,
]

# runtime_verify counterparts that carry `Is[...]` predicates -> must be Annotated.
# (mot.VideoStream carries no predicate and is deliberately excluded: its runtime_verify
# form is the bare `.__value__` base, so it is *not* Annotated -- see below.)
PREDICATE_BEARING_RV_PRIMITIVES = [
    rv_ic.Image,
    rv_ic.ImgClassification,
    rv_od.Image,
    rv_od.ObjectDetectionTarget,
    rv_mot.MultiobjectTrackingTarget,
]


@pytest.mark.parametrize("primitive", PLAIN_PRIMITIVES)
def test_public_primitive_is_not_annotated(primitive: object) -> None:
    assert get_origin(primitive) is not Annotated


@pytest.mark.parametrize("rv_primitive", PREDICATE_BEARING_RV_PRIMITIVES)
def test_runtime_verify_primitive_is_annotated(rv_primitive: object) -> None:
    assert get_origin(rv_primitive) is Annotated


def test_runtime_verify_videostream_has_no_annotated_wrapper() -> None:
    # VideoStream carries no value predicate, so its runtime_verify form is the bare
    # beartype-digestible base (not Annotated). Documented parity exception.
    assert get_origin(rv_mot.VideoStream) is not Annotated


# ---------------------------------------------------------------------------
# Req 1: protocol-class-backed primitives are usable with isinstance
# ---------------------------------------------------------------------------


def test_object_detection_target_is_isinstance_able() -> None:
    class _Target:
        boxes = np.zeros((0, 4))
        scores = np.zeros((0,))
        labels = np.zeros((0,))

    # The whole point of keeping the primitive non-Annotated: this must not raise
    # and must recognize a structural implementer.
    assert isinstance(_Target(), od.ObjectDetectionTarget)


def test_multiobject_tracking_target_is_isinstance_able() -> None:
    class _Target:
        frame_tracks = []

    assert isinstance(_Target(), mot.MultiobjectTrackingTarget)


# ---------------------------------------------------------------------------
# Req 2: exposed primitive reveals as its alias name, not the expanded base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primitive", "expected_name"),
    [
        (ic.Image, "Image"),
        (ic.ImgClassification, "ImgClassification"),
        (od.Image, "Image"),
        (mot.VideoStream, "VideoStream"),
    ],
)
def test_public_primitive_reveals_as_alias_name(primitive: object, expected_name: str) -> None:
    assert getattr(primitive, "__name__", None) == expected_name


# ---------------------------------------------------------------------------
# Phase 4: no RedundantSpotcheckWarning for the standard task protocols
# ---------------------------------------------------------------------------

STANDARD_PROTOCOLS: list[TypeForm[Any]] = [
    ic.Dataset,
    ic.Model,
    od.Dataset,
    od.Model,
    mot.Dataset,
    mot.Model,
]


@pytest.mark.parametrize("protocol", STANDARD_PROTOCOLS)
def test_standard_protocol_carries_value_predicate_after_enrichment(protocol: TypeForm[Any]) -> None:
    # The plain primitives ride in through `__orig_bases__` predicate-free; the
    # `leaf_rewrite=_ENRICH` seam re-attaches the `Is[...]` validators at resolution time.
    members = resolve_protocol_members(protocol, leaf_rewrite=_ENRICH)
    assert _members_carry_value_predicate(members) is True


@pytest.mark.parametrize("protocol", STANDARD_PROTOCOLS)
def test_standard_protocol_wrap_does_not_warn(protocol: TypeForm[Any]) -> None:
    from maite._internals.spotcheck import proxy_factory

    # cache_clear so the once-per-(protocol, config) warning would actually fire if present.
    proxy_factory._build_proxy_class.cache_clear()  # noqa: SLF001
    with warnings.catch_warnings():
        warnings.simplefilter("error", RedundantSpotcheckWarning)
        spotcheck_component(protocol)  # must not raise: predicates re-attached by _ENRICH


# ---------------------------------------------------------------------------
# Critical ambiguity regression: Image (3-dim) vs ImgClassification (1-dim)
# must be neither swapped nor merged during predicate re-attachment.
# ---------------------------------------------------------------------------

_IMG_3DIM = np.zeros((3, 4, 4), dtype=np.float32)  # valid Image (is_3dim)
_LABEL_1DIM = np.zeros((4,), dtype=np.float32)  # valid ImgClassification (is_1dim)


def _make_ic_dataset(input_val: ArrayLike, target_val: ArrayLike) -> ic.Dataset:
    """A minimal single-item IC Dataset returning the given input/target per item.

    ``len == 1`` makes the wrap-time state-verification sampling deterministic (index 0).
    """

    class _DS:
        metadata: DatasetMetadata = {"id": "ambiguity-probe"}

        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int, /) -> tuple[ArrayLike, ArrayLike, DatumMetadata]:
            if i >= 1:
                raise IndexError("out of range")
            return input_val, target_val, {"id": i}

    return _DS()


def _wrap_ic_dataset(input_val: ArrayLike, target_val: ArrayLike) -> None:
    """Wrap an IC Dataset impl; raises SpotcheckError if a sampled item violates a predicate."""
    spotcheck_component(ic.Dataset)(_make_ic_dataset(input_val, target_val))


def test_ic_good_impl_passes_proving_not_swapped_or_merged() -> None:
    # A genuinely valid item (3-dim input, 1-dim target) must pass. This is the
    # discriminating case: a *swap* (Image<->ImgClassification) or a *merge* (both
    # positions sharing one predicate) would reject this good data.
    _wrap_ic_dataset(_IMG_3DIM, _LABEL_1DIM)  # must not raise


def test_ic_ambiguous_impl_fails_on_both() -> None:
    # 1-dim input (wrong for Image) AND 3-dim target (wrong for ImgClassification):
    # both are wrong under the correct assignment, so this must be rejected.
    with pytest.raises(SpotcheckError):
        _wrap_ic_dataset(_LABEL_1DIM, _IMG_3DIM)


def test_ic_input_position_independently_enforces_is_3dim() -> None:
    # Only the input is wrong (1-dim); target is valid (1-dim). The input position
    # must enforce is_3dim independently of the target position.
    with pytest.raises(SpotcheckError):
        _wrap_ic_dataset(_LABEL_1DIM, _LABEL_1DIM)


def test_ic_target_position_independently_enforces_is_1dim() -> None:
    # Only the target is wrong (3-dim); input is valid (3-dim). The target position
    # must enforce is_1dim independently of the input position.
    with pytest.raises(SpotcheckError):
        _wrap_ic_dataset(_IMG_3DIM, _IMG_3DIM)


# ---------------------------------------------------------------------------
# _ENRICH idempotency (guards against double-wrapping on a second pass)
# ---------------------------------------------------------------------------


def _annotated_meta_count(hint: object) -> int:
    """Number of ``Is[...]`` metadata entries on an ``Annotated`` hint (0 if not Annotated)."""
    return len(get_args(hint)) - 1 if get_origin(hint) is Annotated else 0


@pytest.mark.parametrize("primitive", PLAIN_PRIMITIVES)
def test_enrich_is_idempotent_on_plain_primitive(primitive: TypeForm[Any]) -> None:
    from maite._internals.protocols.runtime_verify import RUNTIME_VERIFY_LOOKUP

    once = _hint_substitute(primitive, _ENRICH)
    twice = _hint_substitute(once, _ENRICH)
    # First pass swaps the plain primitive for its identity-keyed runtime-verify form.
    assert once is RUNTIME_VERIFY_LOOKUP[id(primitive)]
    # Second pass is a fixed point: nothing is re-wrapped and the predicate count is
    # unchanged. This must hold for *every* backing -- including the class-backed forms
    # (od.ObjectDetectionTarget, mot.MultiobjectTrackingTarget) whose Annotated base is
    # itself a lookup key, which is exactly the case a naive walk would double-enrich.
    assert twice is once
    assert _annotated_meta_count(twice) == _annotated_meta_count(once)


@pytest.mark.parametrize("rv_primitive", PREDICATE_BEARING_RV_PRIMITIVES)
def test_enrich_is_noop_on_already_annotated_runtime_verify_form(rv_primitive: TypeForm[Any]) -> None:
    # The polymorphic-task (P2) leaf shape: a pinned tv_sub_map substitutes the Annotated
    # runtime-verify form directly, so the final `_ENRICH`/`_hint_substitute` pass sees an
    # already-enriched leaf and must leave it untouched (no doubled predicates).
    before = _annotated_meta_count(rv_primitive)
    after = _hint_substitute(rv_primitive, _ENRICH)
    assert after is rv_primitive
    assert _annotated_meta_count(after) == before


def test_enrich_leaves_unknown_leaf_untouched() -> None:
    assert _hint_substitute(int, _ENRICH) is int
