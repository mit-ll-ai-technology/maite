"""Test whether parametrized generics in a typehint are correctly handled via `resolve_protocol_members`

What should happen is that for typehints referring to parametrized generics (potentially even strings)
we can resolve the typehint into the non-generic equivalent. This permits downstream handlers of
the resulting ProtocolMembers-typed value to properly install state / behavior validation taps

Design: classic example-based testing
- Instantiate a variety of classes at different levels of relationship to parametrized generics
- Run `resolve_protocol_members` on each
"""

from typing import Any, Generic, Protocol, TypeAlias, TypeVar

import pytest
from typing_extensions import TypeForm

from maite._internals.spotcheck.generic_hint_resolution import (
    SubstitutionConflictError,
    UninferrableSubstitutionError,
    UnresolvedTypeVarsError,
    resolve_protocol_members,
)

T_co = TypeVar("T_co", covariant=True)
T_cn = TypeVar("T_cn", contravariant=True)
T_in = TypeVar("T_in", contravariant=False, covariant=False)
T_in2 = TypeVar("T_in2", contravariant=False, covariant=False)
T = TypeVar("T")


# Note: we're constructing these classes so that they all *should* become the same
# after resolve_protocol_members
class SomeOrdinaryProtoBaseline(Protocol):
    a_plain_attr: int

    @property
    def a_prop(self) -> list[str]: ...

    def a_method(self, x: float, *args: tuple[int], **kwargs: tuple[float]) -> "str": ...


class SomeGenericProto(Protocol, Generic[T_in, T_in2, T_cn, T_co]):
    a_plain_attr: T_in

    @property
    def a_prop(self) -> list[T_in2]: ...

    def a_method(self, x: T_cn, *args: tuple[int], **kwargs: tuple[float]) -> T_co: ...


class SomePartiallyParmdGenericProto(SomeGenericProto[int, str, float, T_co], Protocol): ...


class SomeParmdGenericProto(SomeGenericProto[int, str, float, str], Protocol): ...


SomeTypeAlias: TypeAlias = SomeGenericProto[int, str, float, str]

PROTOCOL_TYPEHINTS_FAILING_RESOLUTION: tuple[TypeForm[Any], ...] = (SomeGenericProto, SomePartiallyParmdGenericProto)

PROTOCOL_TYPEHINTS_RESOLVING_TO_BASELINE: tuple[TypeForm[Any], ...] = (
    SomeOrdinaryProtoBaseline,
    SomeGenericProto[int, str, float, str],
    SomeParmdGenericProto,
    SomeTypeAlias,
)


class SomeGeneric(Protocol, Generic[T_co]):
    def meth(self) -> T_co: ...


@pytest.mark.parametrize("proto_hint", PROTOCOL_TYPEHINTS_FAILING_RESOLUTION)
def test_failure_in_unclosed(proto_hint: TypeForm[Any]):
    """Trying to resolve protocol members on generic should raise an error"""
    with pytest.raises(UnresolvedTypeVarsError):
        resolve_protocol_members(proto_hint, require_closed=True)


@pytest.mark.parametrize("proto_hint", PROTOCOL_TYPEHINTS_RESOLVING_TO_BASELINE)
def test_equivalent_resolutions(proto_hint: TypeForm[Any]):
    """Check that resolving types on SomeOrdinaryProtocol_baseline
    results in same value as resolving any other TypeForm"""
    assert resolve_protocol_members(SomeOrdinaryProtoBaseline, require_closed=True) == resolve_protocol_members(
        proto_hint
    )


# ---------------------------------------------------------------------------
# Caller-pinned TypeVar substitution (`pinned_tv_sub_map`)
# ---------------------------------------------------------------------------


def test_pinned_closes_partial_generic():
    """Pinning the last free TypeVar of a partially-parametrized protocol closes it,
    yielding the same members as the baseline (which pins T_co -> str)."""
    resolved = resolve_protocol_members(SomePartiallyParmdGenericProto, pinned_tv_sub_map={T_co: str})
    assert resolved == resolve_protocol_members(SomeOrdinaryProtoBaseline)


def test_pinned_closes_bare_generic():
    """A bare (unparametrized) generic protocol can be closed purely via the pin map."""
    resolved = resolve_protocol_members(SomeGeneric, pinned_tv_sub_map={T_co: int})
    assert resolved.methods["meth"].return_annotation is int


def test_pinned_conflict_raises():
    """Pinning a TypeVar to a value the class itself parametrizes differently is a conflict."""
    # SomeParmdGenericProto fixes T_in -> int via SomeGenericProto[int, str, float, str];
    # pinning T_in -> float contradicts that derivation.
    with pytest.raises(SubstitutionConflictError):
        resolve_protocol_members(SomeParmdGenericProto, pinned_tv_sub_map={T_in: float})


# ---------------------------------------------------------------------------
# `require_closed=False` — accept vacuous checking of open members
# ---------------------------------------------------------------------------


def test_require_closed_false_allows_open_members():
    """With require_closed=False an unresolved generic returns rather than raising,
    leaving free TypeVars in place in the resolved members."""
    members = resolve_protocol_members(SomeGenericProto, require_closed=False)
    assert members.attrs["a_plain_attr"] is T_in


# ---------------------------------------------------------------------------
# Inconsistent MRO parametrization (`UninferrableSubstitutionError`)
# ---------------------------------------------------------------------------


class _DiamondBase(Protocol, Generic[T_co]):
    def g(self) -> T_co: ...


class _DiamondMidInt(_DiamondBase[int], Protocol): ...


class _DiamondMidStr(_DiamondBase[str], Protocol): ...


# Statically ill-typed on purpose: the two mid classes bind _DiamondBase's TypeVar to
# incompatible types. This is exactly the inconsistency the resolver must detect at runtime,
# so pyright's (correct) static objection is suppressed for this fixture only.
class _DiamondLeaf(_DiamondMidInt, _DiamondMidStr, Protocol):  # pyright: ignore[reportGeneralTypeIssues]
    ...


def test_uninferrable_substitution_raises():
    """A hierarchy that binds the same base TypeVar to two different types across
    MRO frames cannot be resolved to a single substitution map."""
    with pytest.raises(UninferrableSubstitutionError):
        resolve_protocol_members(_DiamondLeaf)


# ---------------------------------------------------------------------------
# Non-protocol inputs
# ---------------------------------------------------------------------------


class _NotAProtocol(Generic[T]): ...


def test_non_protocol_alias_raises():
    """A parametrized alias whose origin is not a Protocol is rejected."""
    with pytest.raises(TypeError):
        resolve_protocol_members(_NotAProtocol[int])
