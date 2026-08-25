"""
Unit tests for the high-level ``spotcheck`` decorator and its supporting hint
machinery in ``maite._internals.spotcheck.spotcheck_tasks``.

The module separates *mechanism* (``_reduce`` -- whitelist-free structural
reduction) from *policy* (``hint_matches_component_protocol`` -- which owns the
component-protocol whitelist and the ``Union`` handling rules). A third helper,
``_reduce_to_forwardable_hint``, produces the bare Protocol class / parametrized
alias that the decorator forwards to ``spotcheck_component`` (which rejects
``Optional``/``Union``). The suite covers each of these plus the decorator
end-to-end.

Unions are spelled PEP-604 (``X | Y``) by default; a handful of tests use the
``typing.Union`` / ``typing.Optional`` spelling (with a targeted ``noqa``) to
exercise the ``typing.Union`` origin branch, since the source keys on
``origin in (Union, UnionType)``.
"""

from collections.abc import Callable, Mapping
from types import UnionType
from typing import Annotated, Any, Optional, Protocol, TypeAlias, TypeVar, Union, get_origin

import pytest
from beartype.vale import Is
from typing_extensions import TypeAliasType, TypeForm

from maite._internals.spotcheck import spotcheck_tasks
from maite._internals.spotcheck.spotcheck_node import SpotcheckError
from maite._internals.spotcheck.spotcheck_tasks import (
    COMPONENT_PROTOCOL_BASE_CLASSES,
    UnresolvableUnionHintError,
    _reduce,
    _reduce_to_forwardable_hint,
    hint_matches_component_protocol,
    spotcheck,
)
from maite.protocols.generic import (
    Dataset,
    DatasetMetadata,
    DatumMetadata,
    Model,
)

# ==============================================================================
# Fixtures: a checkable parametrized Dataset alias + conforming / violating impls
# ==============================================================================

# Predicates live on Annotated TypeAliases (see spotcheck_component docstring).
PosInt: TypeAlias = Annotated[int, Is[lambda x: x > 0]]
DS: TypeAlias = Dataset[PosInt, str, DatumMetadata]


class GoodDataset:
    """A Dataset[PosInt, str, DatumMetadata] implementer that honors PosInt."""

    metadata: DatasetMetadata = {"id": "ExampleDataset"}

    def __getitem__(self, i: int) -> tuple[int, str, DatumMetadata]:
        if i >= len(self):
            raise IndexError("Index out of range")
        return (1, "asdf", DatumMetadata(id="OnlyDatum"))

    def __len__(self) -> int:
        return 2


class BadDataset(GoodDataset):
    """Violates the PosInt promise by yielding a negative input element."""

    def __getitem__(self, i: int) -> tuple[int, str, DatumMetadata]:
        _ = i  # index ignored: every datum is deliberately bad
        return (-1, "asdf", DatumMetadata(id="OnlyDatum"))


# A user protocol that descends from a component base (should match).
class DerivedDataset(Dataset[int, str, DatumMetadata], Protocol):
    """A user-defined Protocol subclassing a component base."""


# A protocol unrelated to any component base (should not match).
class SomeProtocol(Protocol):
    x: int


# A plain, non-protocol class (should not match).
class PlainClass:
    x: int = 0


# Sorted at module scope (deterministic parametrize IDs). Kept out of the
# parametrize call so pyright infers the lambda param from the set element
# (`type`) rather than back-propagating parametrize's argvalue element type.
_COMPONENT_BASES = sorted(COMPONENT_PROTOCOL_BASE_CLASSES, key=lambda c: c.__name__)

# Explicit TypeAliasTypes for the "unwrap TypeAliasType" reducer tests. Defined
# at module scope with matching names (pyright requires both, and a TypeAliasType
# may only be assigned in module/class scope).
NestedListAlias = TypeAliasType("NestedListAlias", list[int])
NestedDatasetAlias = TypeAliasType("NestedDatasetAlias", DS)


# ==============================================================================
# _reduce  (whitelist-free structural reduction)
# ==============================================================================


def test_reduce_plain_type_returned_as_is() -> None:
    assert _reduce(int) is int


def test_reduce_none_and_nonetype_collapse_to_none() -> None:
    assert _reduce(None) is None
    assert _reduce(type(None)) is None


def test_reduce_generic_alias_to_origin() -> None:
    assert _reduce(list[int]) is list


def test_reduce_parametrized_protocol_to_origin() -> None:
    assert _reduce(DS) is Dataset


def test_reduce_annotated_strips_to_inner() -> None:
    assert _reduce(Annotated[Dataset, "meta"]) is Dataset


def test_reduce_pep604_union_left_intact() -> None:
    u = int | str
    assert get_origin(_reduce(u)) in (Union, UnionType)  # in python 3.14 runtime objects will be the same


def test_reduce_typing_union_left_intact() -> None:
    u = Union[int, str]  # noqa: UP007 -- exercise the typing.Union origin branch
    assert _reduce(u) is u  # policy layer, not _reduce, decides union fate


def test_reduce_typealias_unwrapped_then_reduced() -> None:
    assert _reduce(NestedListAlias) is list


# ==============================================================================
# hint_matches_component_protocol  (policy: whitelist + Union rules)
# ==============================================================================


@pytest.mark.parametrize("base", _COMPONENT_BASES)
def test_each_component_base_matches(base: type) -> None:
    assert hint_matches_component_protocol(base) is True


def test_parametrized_component_matches() -> None:
    assert hint_matches_component_protocol(DS) is True


def test_descendant_protocol_matches() -> None:
    assert hint_matches_component_protocol(DerivedDataset) is True


def test_non_component_protocol_does_not_match() -> None:
    assert hint_matches_component_protocol(SomeProtocol) is False


def test_plain_class_does_not_match() -> None:
    assert hint_matches_component_protocol(PlainClass) is False


def test_builtin_does_not_match() -> None:
    assert hint_matches_component_protocol(int) is False


def test_none_does_not_match() -> None:
    assert hint_matches_component_protocol(None) is False


@pytest.mark.parametrize(
    "hint",
    [
        Dataset | None,
        Annotated[Dataset, "meta"],
        Annotated[Dataset | None, "meta"],
    ],
)
def test_optional_and_annotated_component_match(hint: TypeForm[Any]) -> None:
    """Rule (a): Optional[Component] (and Annotated wrappers) defer to the component."""
    assert hint_matches_component_protocol(hint) is True


def test_typing_optional_component_matches() -> None:
    hint = Optional[Dataset]  # noqa: UP045 -- exercise the typing.Union origin branch
    assert hint_matches_component_protocol(hint) is True


@pytest.mark.parametrize(
    "hint",
    [
        Dataset | int,
        Dataset | Model,
        Dataset | Model | None,  # Optional[...] of a genuine multi-component union
    ],
)
def test_multi_member_union_naming_component_raises(hint: TypeForm[Any]) -> None:
    """Rule (b): a genuine multi-member union that names a component is fatal."""
    with pytest.raises(UnresolvableUnionHintError):
        hint_matches_component_protocol(hint)


def test_typing_union_naming_component_raises() -> None:
    hint = Union[Dataset, Model]  # noqa: UP007 -- exercise the typing.Union origin branch
    with pytest.raises(UnresolvableUnionHintError):
        hint_matches_component_protocol(hint)


@pytest.mark.parametrize("hint", [int | str, int | None])
def test_union_without_component_is_false(hint: TypeForm[Any]) -> None:
    """A union naming no component is silently ignored (no raise, just False)."""
    assert hint_matches_component_protocol(hint) is False


def test_typing_optional_scalar_is_false() -> None:
    hint = Optional[int]  # noqa: UP045 -- exercise the typing.Union origin branch
    assert hint_matches_component_protocol(hint) is False


def test_unresolvable_union_hint_carries_hint() -> None:
    hint = Dataset | Model
    with pytest.raises(UnresolvableUnionHintError) as excinfo:
        hint_matches_component_protocol(hint)
    assert excinfo.value.hint == hint


# ==============================================================================
# _reduce_to_forwardable_hint  (strip Optional/Annotated, PRESERVE parametrization)
# ==============================================================================


def test_forwardable_parametrized_alias_preserved() -> None:
    assert _reduce_to_forwardable_hint(DS) == DS


def test_forwardable_pep604_optional_stripped_preserving_parametrization() -> None:
    assert _reduce_to_forwardable_hint(DS | None) == DS


def test_forwardable_typing_optional_stripped() -> None:
    hint = Optional[DS]  # noqa: UP045 -- exercise the typing.Union origin branch
    assert _reduce_to_forwardable_hint(hint) == DS


def test_forwardable_annotated_stripped() -> None:
    assert _reduce_to_forwardable_hint(Annotated[DS, "meta"]) == DS


def test_forwardable_annotated_optional_stripped() -> None:
    assert _reduce_to_forwardable_hint(Annotated[DS | None, "meta"]) == DS


def test_forwardable_plain_protocol_preserved() -> None:
    assert _reduce_to_forwardable_hint(Dataset) is Dataset


def test_forwardable_genuine_union_is_none() -> None:
    assert _reduce_to_forwardable_hint(Dataset | Model) is None


def test_forwardable_none_is_none() -> None:
    assert _reduce_to_forwardable_hint(None) is None


def test_forwardable_optional_scalar_kept() -> None:
    assert _reduce_to_forwardable_hint(int | None) is int


def test_forwardable_typealias_preserves_parametrization() -> None:
    assert _reduce_to_forwardable_hint(NestedDatasetAlias) == DS


# ==============================================================================
# spotcheck decorator -- argument wrapping (bare and factory forms)
# ==============================================================================


@spotcheck
def _consume_bare(ds: DS) -> str:
    _ = ds  # only structural conformance / wrapping matters here
    return "ok"


@spotcheck(rate=1.0)
def _consume_factory(ds: DS) -> str:
    _ = ds
    return "ok"


@pytest.mark.parametrize("task", [_consume_bare, _consume_factory])
def test_component_arg_good_impl_passes(task: Callable[[DS], str]) -> None:
    assert task(GoodDataset()) == "ok"


@pytest.mark.parametrize("task", [_consume_bare, _consume_factory])
def test_component_arg_bad_impl_raises(task: Callable[[DS], str]) -> None:
    with pytest.raises(SpotcheckError):
        task(BadDataset())


# ==============================================================================
# spotcheck decorator -- Optional argument handling
# ==============================================================================


@spotcheck
def _consume_optional(ds: DS | None) -> None:
    _ = ds


def test_optional_arg_none_passes_through() -> None:
    assert _consume_optional(None) is None


def test_optional_arg_good_impl_passes() -> None:
    assert _consume_optional(GoodDataset()) is None


def test_optional_arg_bad_impl_raises() -> None:
    # Reaching SpotcheckError proves the Optional was stripped and the
    # parametrized Dataset hint was forwarded to spotcheck_component.
    with pytest.raises(SpotcheckError):
        _consume_optional(BadDataset())


# ==============================================================================
# spotcheck decorator -- return-value wrapping (plain and Optional)
# ==============================================================================


@spotcheck
def _produce(bad: bool) -> DS:
    return BadDataset() if bad else GoodDataset()


def test_return_good_impl_is_wrapped_and_usable() -> None:
    result = _produce(False)
    assert len(result) == 2  # proxy forwards transparently


def test_return_bad_impl_raises() -> None:
    with pytest.raises(SpotcheckError):
        _produce(True)


@spotcheck
def _produce_optional(which: str) -> DS | None:
    if which == "none":
        return None
    return BadDataset() if which == "bad" else GoodDataset()


def test_optional_return_none_passes_through() -> None:
    assert _produce_optional("none") is None


def test_optional_return_good_impl_is_wrapped() -> None:
    result = _produce_optional("good")
    assert result is not None
    assert len(result) == 2


def test_optional_return_bad_impl_raises() -> None:
    with pytest.raises(SpotcheckError):
        _produce_optional("bad")


# ==============================================================================
# spotcheck decorator -- no-op and decoration-time union rejection
# ==============================================================================


@spotcheck
def _plain(x: int) -> int:
    return x


def test_no_component_hints_returns_function_unwrapped() -> None:
    """When nothing is component-typed, the decorator returns *func* untouched."""
    assert _plain(3) == 3
    assert not hasattr(_plain, "__wrapped__")


def test_component_union_param_raises_at_decoration() -> None:
    with pytest.raises(UnresolvableUnionHintError):

        @spotcheck
        def _bad(ds: Dataset | int) -> None:
            _ = ds


# ==============================================================================
# spotcheck decorator -- forwarding plumbing (hint reduction + kwargs pass-through)
# ==============================================================================


def test_forwarding_passes_reduced_hint_and_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper forwards the *reduced* hint plus rate/override to spotcheck_component,
    and skips None-valued (Optional) arguments entirely."""
    calls: list[tuple[TypeForm[Any], float, bool, Mapping[TypeVar, TypeForm[Any]]]] = []

    def spy(
        protocol: TypeForm[Any],
        *,
        rate: float,
        override_existing: bool,
        tv_sub_map: Mapping[TypeVar, TypeForm[Any]],
    ) -> object:
        calls.append((protocol, rate, override_existing, tv_sub_map))

        def passthrough(impl: object):
            return impl

        return passthrough

    monkeypatch.setattr(spotcheck_tasks, "spotcheck_component", spy)

    @spotcheck(rate=0.25, override_existing=True)
    def consume(ds: DS | None) -> None:
        _ = ds

    inst = GoodDataset()
    consume(inst)

    assert len(calls) == 1
    proto_arg, rate_arg, override_arg, tv_sub_map = calls[0]
    assert proto_arg == DS  # Optional stripped, parametrization preserved
    assert rate_arg == 0.25
    assert override_arg is True
    assert tv_sub_map is None

    calls.clear()
    consume(None)  # None-valued Optional argument is never forwarded
    assert calls == []
