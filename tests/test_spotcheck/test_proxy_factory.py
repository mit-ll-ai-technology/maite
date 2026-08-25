"""
Unit tests for the dynamic spotchecking proxy factory using pytest.

This suite uses a highly parameterized approach to test behavioral parity
and type violation detection across various interaction modes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Protocol, TypeVar, cast

import pytest
from typing_extensions import TypeForm

# Correct, localized imports per structural guidelines
from maite._internals.spotcheck import proxy_factory
from maite._internals.spotcheck.protocol_introspection import (
    ProtocolMembers,
    protocol_members,
)
from maite._internals.spotcheck.proxy_factory import (
    _ProxyBase,
    _ProxyConfig,
    _unwrap_proxied,
    spotcheck_component,
)
from maite._internals.spotcheck.spotcheck_node import (
    SpotcheckError,
    VapidMethodError,
    VapidPropertyError,
)

T = TypeVar("T")


class RefProto(Protocol):
    """A complex protocol exercising all proxy capabilities."""

    x: int
    y: str

    @property
    def dependent_prop(self) -> str: ...

    def some_method(self, a: int, b: str) -> bool: ...

    def __call__(self, val: float) -> float: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> str: ...


class RefImpl:
    """A valid, correct implementation of RefProto."""

    def __init__(self) -> None:
        self.x = 10
        self.y = "hello"
        self._items = ["zero", "one", "two"]

    @property
    def dependent_prop(self) -> str:
        return f"Value is {self.x}"

    def some_method(self, a: int, b: str) -> bool:
        return len(b) > a

    def __call__(self, val: float) -> float:
        return val * 2.0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> str:
        return self._items[index]

    def helper(self) -> str:
        """A non-mandated method. Must pass through the proxy."""
        return "I am a helper"


class RefImplWIter(RefImpl):
    """A valid, correct implementation of RefProto with explicit __iter__ method"""

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)


# a lookup allowing us to fetch conformant values for a given hint
EXAMPLES: dict[TypeForm[Any], Any] = {int: 1, str: "a", float: 1.0, bool: True}

# a lookup allowing us to fetch non-conformant values for a given hint
COUNTEREXAMPLES: dict[TypeForm[Any], Any] = {int: "a", str: 1, float: "a", bool: 0}

# ==============================================================================
# Structural Segmentation and Dynamic Test Data Generation
# ==============================================================================

_MEMBERS: ProtocolMembers = protocol_members(RefProto)
_VARS: Mapping[str, Any] = vars(RefProto)

PLAIN_ATTRS: list[str] = [n for n in _MEMBERS.attrs if not isinstance(_VARS.get(n), property)]
PROP_ATTRS: list[str] = [n for n in _MEMBERS.attrs if isinstance(_VARS.get(n), property)]
METHODS: list[str] = list(_MEMBERS.methods.keys())

# Methods that enable non-destructive sequence verification at wrap time
WRAP_TIME_CHECKED_METHODS: list[str] = ["__getitem__"]

METHOD_PARAMS: list[tuple[str, str, TypeForm[Any]]] = [
    (meth_name, param.name, param.annotation)
    for meth_name, sig in _MEMBERS.methods.items()
    for param in sig.parameters.values()
    if param.name != "self" and param.annotation in COUNTEREXAMPLES
]


def _get_bad_impl_instance(name_to_break: str) -> RefImpl:
    """Factory to dynamically create a 'mutant' instance that is invalid in one specific way."""
    if name_to_break in PLAIN_ATTRS:
        mutant = RefImpl()
        hint = _MEMBERS.attrs[name_to_break]
        if hint is None:
            raise TypeError("Cannot mutate protocol impl for unhinted attr")
        setattr(mutant, name_to_break, COUNTEREXAMPLES[hint])
        return mutant

    if name_to_break in PROP_ATTRS:
        hint = _MEMBERS.attrs[name_to_break]
        if hint is None:
            raise TypeError("Cannot mutate protocol impl for unhinted prop")
        bad_val = COUNTEREXAMPLES[hint]
        # Dynamically create a new class with the single broken property
        MutantImpl = type(  # noqa: N806, MutantImpl is a variable, but it is also a class, so it should be CapWords/PascalCase
            "MutantImpl",
            (RefImpl,),
            {name_to_break: property(lambda self, val=bad_val: val)},  # noqa: ARG005 (deliberately ignoring 'self')
        )
        return cast(RefImpl, MutantImpl())

    if name_to_break in METHODS:
        hint = _MEMBERS.methods[name_to_break].return_annotation
        bad_val = COUNTEREXAMPLES.get(hint)
        # Dynamically create a new class with the single broken method
        MutantImpl = type(  # noqa: N806, MutantImpl is a variable, but it is also a class, so it should be CapWords/PascalCase
            "MutantImpl",
            (RefImpl,),
            {name_to_break: lambda self, *args, **kwargs: bad_val},  # noqa: ARG005 (deliberately ignoring all args)
        )
        return cast(RefImpl, MutantImpl())

    raise ValueError(f"Unknown member to break: {name_to_break}")


def _get_bad_impl_instance_witer(name_to_break: str) -> RefImplWIter:
    """Factory to dynamically create a 'mutant' instance that is invalid in one specific way
    (returning impl with an __iter__ method)"""
    if name_to_break in PLAIN_ATTRS:
        mutant = RefImplWIter()
        hint = _MEMBERS.attrs[name_to_break]
        if hint is None:
            raise TypeError("Cannot mutate protocol impl for unhinted attr")
        setattr(mutant, name_to_break, COUNTEREXAMPLES[hint])
        return mutant

    if name_to_break in PROP_ATTRS:
        hint = _MEMBERS.attrs[name_to_break]
        if hint is None:
            raise TypeError("Cannot mutate protocol impl for unhinted prop")
        bad_val = COUNTEREXAMPLES[hint]
        # Dynamically create a new class with the single broken property
        MutantImpl = type(  # noqa: N806, MutantImpl is a variable, but it is also a class, so it should be CapWords/PascalCase
            "MutantImpl",
            (RefImplWIter,),
            {name_to_break: property(lambda self, val=bad_val: val)},  # noqa: ARG005 (deliberately ignoring 'self')
        )
        return cast(RefImplWIter, MutantImpl())

    if name_to_break in METHODS:
        hint = _MEMBERS.methods[name_to_break].return_annotation
        bad_val = COUNTEREXAMPLES.get(hint)
        # Dynamically create a new class with the single broken method
        MutantImpl = type(  # noqa: N806, MutantImpl is a variable, but it is also a class, so it should be CapWords/PascalCase
            "MutantImpl",
            (RefImplWIter,),
            {name_to_break: lambda self, *args, **kwargs: bad_val},  # noqa: ARG005 (deliberately ignoring all args)
        )
        return cast(RefImplWIter, MutantImpl())

    # hard-wiring a breaking case for dunder not mandated by protocol
    if name_to_break in ["__iter__"]:
        bad_val = iter([1.0, 2.0])
        # Dynamically create a new class with the single broken method
        MutantImpl = type(  # noqa: N806, MutantImpl is a variable, but it is also a class, so it should be CapWords/PascalCase
            "MutantImpl",
            (RefImplWIter,),
            {"__iter__": lambda self, *args, **kwargs: bad_val},  # noqa: ARG005 (deliberately ignoring all args)
        )
        return cast(RefImplWIter, MutantImpl())

    raise ValueError(f"Unknown member to break: {name_to_break}")


# Using select builtins (right now just 'iter') on "bad" mutations of impls
# should be caught by spotcheck at iteration time
BADIMPLS_FOR_ITERATION: list[RefProto] = [
    _get_bad_impl_instance(name_to_break="__getitem__"),
    _get_bad_impl_instance_witer(name_to_break="__iter__"),
]
# Note that a bad __getitem__ alone will not trigger spotcheck if 'iter(Proxy)' is called since
# built-in calls are forwarded "by API" to preserve fallbacks
# The two cases are:
# 1: __iter__ is NOT on impl) Proxy.__iter__ -> iter(Proxy.__wrapped__) -> Proxy.__wrapped__.__getitem__ (unchecked!)
# 2: __iter__ is on impl) Proxy.__iter__ -> iter(Proxy.__wrapped__) -> Proxy.__iter__ (unchecked! return)

SCRIPTS: dict[str, list[tuple[str, Callable[[Any], Any]]]] = {
    "reads": [
        ("x", lambda o: o.x),
        ("y", lambda o: o.y),
        ("helper", lambda o: o.helper()),
        ("dependent_prop", lambda o: o.dependent_prop),
        ("iteration_over", lambda o: [print(i) for i in o]),
    ],
    "calls": [
        ("some_method", lambda o: o.some_method(1, "foo")),
        ("__call__", lambda o: o(5.0)),
        ("__len__", lambda o: len(o)),
        ("__getitem__", lambda o: o[1]),
        ("negative-index", lambda o: o[-1]),
    ],
    "state_update": [
        ("write_x", lambda o: setattr(o, "x", 20)),
        ("read_x_back", lambda o: o.x),
    ],
    "impl_errors": [("index_error", lambda o: o[999])],
}

# ==============================================================================
# Behavioral Parity Testing (Transparent Proxying)
# ==============================================================================


@pytest.mark.parametrize("script_name", sorted(SCRIPTS))
def test_behavioral_parity(script_name: str) -> None:
    """Ensure valid operations and native errors are identical on raw and proxied objects."""

    def execute_trace(obj: RefImpl | RefProto) -> list[tuple[str, str, Any]]:
        results = []
        errors: list[Exception] = []  # build up list of errors
        for desc, func in SCRIPTS[script_name]:
            try:
                results.append((desc, "Success", func(obj)))
            except Exception as e:  # noqa: PERF203, BLE001 (performance overhead not significant/blind exception forwarded purely for parity)
                errors.append(e)
                results.append((desc, "Error", type(e)))

        return results

    raw_trace = execute_trace(RefImpl())
    proxy_trace = execute_trace(spotcheck_component(RefProto)(RefImpl()))
    assert raw_trace == proxy_trace


# ==============================================================================
# Typehint Violation Detection Testing
# ==============================================================================


@pytest.mark.parametrize("name", PLAIN_ATTRS + WRAP_TIME_CHECKED_METHODS)
def test_wrap_time_detection(name: str) -> None:
    """Eager state verification should catch bad initial states."""
    bad_impl = _get_bad_impl_instance(name)
    with pytest.raises(SpotcheckError):
        spotcheck_component(RefProto)(bad_impl)


@pytest.mark.parametrize("name", PROP_ATTRS)
def test_bad_property_read(name: str) -> None:
    """Bad property returns are caught on access."""
    bad_impl = _get_bad_impl_instance(name)
    proxy = spotcheck_component(RefProto)(bad_impl)
    with pytest.raises(SpotcheckError):
        getattr(proxy, name)


@pytest.mark.parametrize("name", PLAIN_ATTRS)
def test_bad_attribute_write(name: str) -> None:
    """Setting a plain attribute to a bad type raises an error on write."""
    proxy = spotcheck_component(RefProto)(RefImpl())
    hint = _MEMBERS.attrs[name]
    if hint is None:
        raise TypeError("Cannot test mutation of unhinted attribute")
    bad_value = COUNTEREXAMPLES[hint]
    with pytest.raises(SpotcheckError):
        setattr(proxy, name, bad_value)


@pytest.mark.parametrize(("method_name", "param_name", "param_hint"), METHOD_PARAMS)
def test_bad_method_argument(method_name: str, param_name: str, param_hint: TypeForm[Any]) -> None:
    """Bad arguments passed to a method are caught before execution."""
    proxy = spotcheck_component(RefProto)(RefImpl())
    sig = _MEMBERS.methods[method_name]
    valid_kwargs = {
        p.name: EXAMPLES[p.annotation] for p in sig.parameters.values() if p.name != "self" and p.annotation in EXAMPLES
    }
    valid_kwargs[param_name] = COUNTEREXAMPLES[param_hint]
    with pytest.raises(SpotcheckError):
        getattr(proxy, method_name)(**valid_kwargs)


@pytest.mark.parametrize("badimpl", BADIMPLS_FOR_ITERATION)
def test_break_iteration(badimpl: RefProto) -> None:
    """Ensure built-in iteration is broken if corresponding method has bad type hint"""

    print("---enter test---")
    # use __getitem__
    print(f"{badimpl[0]=}")

    if hasattr(badimpl, "__iter__"):
        print('hasattr(badimpl,"__iter__")')

    with pytest.raises(SpotcheckError):
        for _ in spotcheck_component(RefProto)(badimpl):
            ...


_WRAP_TIME_CHECKPOINT: dict[str, type[Exception]] = {
    "__len__": TypeError,  # CPython complains sooner about __len__ returning a non-int
    "__getitem__": SpotcheckError,
}


@pytest.mark.parametrize("name", METHODS)
def test_bad_method_return_val(name: str) -> None:
    """Bad returns from methods are caught after execution."""
    bad_impl = _get_bad_impl_instance(name)

    # Special sequence-related dunders are pre-sampled at wrap time
    # so we expect failure before method call even happens
    if name in _WRAP_TIME_CHECKPOINT:
        with pytest.raises(_WRAP_TIME_CHECKPOINT[name]):
            spotcheck_component(RefProto)(bad_impl)
        return

    proxy = spotcheck_component(RefProto)(bad_impl)
    sig = _MEMBERS.methods[name]
    valid_kwargs = {
        p.name: EXAMPLES[p.annotation] for p in sig.parameters.values() if p.name != "self" and p.annotation in EXAMPLES
    }

    with pytest.raises(SpotcheckError):
        getattr(proxy, name)(**valid_kwargs)


# ==============================================================================
# Rate Parameter
# ==============================================================================


class _RecordingCheck:
    """A check_procedure stand-in that records every invocation.

    Optionally delegates to a real check so validity semantics are preserved.
    """

    def __init__(self, delegate: Callable[[object, TypeForm[Any]], None] | None = None) -> None:
        self.calls: list[tuple[object, TypeForm[Any]]] = []
        self._delegate = delegate

    def __call__(self, value: object, hint: TypeForm[Any]) -> None:
        self.calls.append((value, hint))
        if self._delegate is not None:
            self._delegate(value, hint)


@pytest.mark.parametrize("bad_rate", [-0.1, 1.1, 2.0, -1.0])
def test_rate_out_of_range_rejected(bad_rate: float) -> None:
    """Rates outside [0.0, 1.0] must be rejected at config construction and via the public API."""
    with pytest.raises(ValueError, match="rate must be in interval"):
        _ProxyConfig(rate=bad_rate)
    with pytest.raises(ValueError, match="rate must be in interval"):
        spotcheck_component(RefProto, rate=bad_rate)(RefImpl())


@pytest.mark.parametrize("good_rate", [0.0, 0.5, 1.0])
def test_rate_in_range_accepted(good_rate: float) -> None:
    """Boundary and interior rates are accepted."""
    assert _ProxyConfig(rate=good_rate).rate == good_rate


def test_rate_zero_skips_runtime_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """At rate=0.0 the runtime taps (get/set/call) never invoke the check procedure.

    Note: eager wrap-time state verification is *not* rate-gated, so we record the
    wrap-time call count and assert nothing further fires during interaction.
    """
    # Force the RNG draw above 0.0 so the gate is deterministic (random() in [0, 1)).
    monkeypatch.setattr(proxy_factory._rng, "random", lambda: 0.5)  # noqa: SLF001, intentional private-member access
    rec = _RecordingCheck()
    proxy = spotcheck_component(RefProto, rate=0.0, check_procedure=rec)(RefImpl())

    n_wrap_time = len(rec.calls)
    _ = proxy.x
    _ = proxy.dependent_prop
    proxy.some_method(1, "foo")
    proxy(2.0)
    proxy.x = 20

    assert len(rec.calls) == n_wrap_time  # assert no checks happened


def test_rate_one_runs_runtime_checks() -> None:
    """At rate=1.0 the runtime taps invoke the check procedure."""
    rec = _RecordingCheck()
    proxy = spotcheck_component(RefProto, rate=1.0, check_procedure=rec)(RefImpl())

    before = len(rec.calls)
    _ = proxy.x
    assert len(rec.calls) > before


@pytest.mark.parametrize(("draw", "expect_check"), [(0.4, True), (0.6, False)])
def test_rate_gate_consults_rng(monkeypatch: pytest.MonkeyPatch, draw: float, expect_check: bool) -> None:
    """With rate=0.5, a draw <= rate performs the check; a draw > rate skips it."""
    monkeypatch.setattr(proxy_factory._rng, "random", lambda: draw)  # noqa: SLF001, intentional private-member access
    rec = _RecordingCheck()
    proxy = spotcheck_component(RefProto, rate=0.5, check_procedure=rec)(RefImpl())

    before = len(rec.calls)
    _ = proxy.x
    assert (len(rec.calls) > before) is expect_check


def test_custom_check_procedure_is_used() -> None:
    """A user-supplied check_procedure replaces the default backend."""

    class BoomError(Exception):
        pass

    def always_fail(value: object, hint: TypeForm[Any]) -> None:
        raise BoomError(f"{value!r} checked against {hint!r}")

    # Wrap-time state verification calls the procedure on `x`, so it fires immediately.
    with pytest.raises(BoomError):
        spotcheck_component(RefProto, check_procedure=always_fail)(RefImpl())


# ==============================================================================
# Double-Wrapping
# ==============================================================================


def test_double_wrap_warns_and_returns_same_object() -> None:
    """Re-wrapping a proxy (override_existing=False) warns and returns it untouched."""
    proxy = spotcheck_component(RefProto)(RefImpl())
    with pytest.warns(RuntimeWarning, match="already a proxy"):
        again = spotcheck_component(RefProto)(proxy)
    assert again is proxy


def test_double_wrap_override_rewraps_underlying_impl() -> None:
    """override_existing=True unwraps one layer and rewraps the *original* impl (no nesting)."""
    impl = RefImpl()
    proxy = spotcheck_component(RefProto)(impl)

    rewrapped = spotcheck_component(RefProto, override_existing=True)(proxy)

    assert rewrapped is not proxy
    inner = _unwrap_proxied(rewrapped)
    assert inner is impl
    assert not isinstance(inner, _ProxyBase)


# ==============================================================================
# Vapid (unhinted) Members
# ==============================================================================


class UnderhintedAttributeProto(Protocol):
    a = 5


class UnderhintedPropertyProto(Protocol):
    @property
    def unhinted_prop(self): ...


class UnderhintedMethodArgProto(Protocol):
    def unhinted_arg(self, a) -> int: ...


class UnderhintedMethodReturnProto(Protocol):
    def unhinted_arg(self, a): ...


class UnderhintedImpl:
    """A structural implementer of UnderhintedProto with typehints missing from UnderhintedProto"""

    unhinted_prop = 7

    def unhinted_return(self) -> int:
        return 1

    def unhinted_arg(self, a: int) -> int:
        return a


def test_unhinted_attribute_raises_at_wrap_time() -> None:
    """Eager state verification raises for an attribute with no readable hint."""
    with pytest.raises(TypeError, match="is not a property, callable, staticmethod, classmethod, or hinted attribute"):
        spotcheck_component(UnderhintedAttributeProto)(UnderhintedImpl())


def test_unhinted_property_return_raises_at_wrap_time() -> None:
    """Eager state verification raises for a property with no readable hint."""
    with pytest.raises(VapidPropertyError, match="Unhinted property cannot be runtime-checked"):
        spotcheck_component(UnderhintedPropertyProto)(UnderhintedImpl())  # type: ignore (static check redundant)


def test_unhinted_method_arg_raises_at_wrap_time() -> None:
    """Eager state verification raises for an attribute with no readable hint."""
    with pytest.raises(VapidMethodError, match="Unhinted method cannot be runtime-checked"):
        spotcheck_component(UnderhintedMethodArgProto)(UnderhintedImpl())


def test_unhinted_method_return_raises_at_wrap_time() -> None:
    """Eager state verification raises for a method with no readable return hint."""
    with pytest.raises(VapidMethodError, match="Unhinted method cannot be runtime-checked"):
        spotcheck_component(UnderhintedMethodReturnProto)(UnderhintedImpl())  # type: ignore (static check redundant)


# ==============================================================================
# Wrap-time State Verification Warnings
# ==============================================================================


def test_missing_protocol_attr_warns_at_wrap_time() -> None:
    """An implementer missing a promised plain attribute warns at wrap time."""

    class MissingY(RefImpl):
        def __init__(self) -> None:
            super().__init__()
            del self.y

    with pytest.raises(AttributeError, match="Implementer is missing protocol attribute"):
        spotcheck_component(RefProto)(MissingY())


# ==============================================================================
# Variadic Parameter Checking
# ==============================================================================


class VarProto(Protocol):
    """Protocol exercising *args / **kwargs annotation checking."""

    def collect(self, *args: int, **kwargs: str) -> None: ...


class VarImpl:
    def collect(self, *args: int, **kwargs: str) -> None:
        _ = (args, kwargs)  # consumed only for structural conformance
        return


def test_variadic_positional_elements_checked() -> None:
    """Each *args element is checked against the variadic annotation."""
    proxy: Any = spotcheck_component(VarProto)(VarImpl())
    proxy.collect(1, 2, 3)  # all ints -> ok
    bad_positional: Any = "bad"
    with pytest.raises(SpotcheckError):
        proxy.collect(1, bad_positional)


def test_variadic_keyword_values_checked() -> None:
    """Each **kwargs value is checked against the variadic annotation."""
    proxy: Any = spotcheck_component(VarProto)(VarImpl())
    proxy.collect(a="ok", b="fine")  # all strs -> ok
    bad_keyword: Any = 5
    with pytest.raises(SpotcheckError):
        proxy.collect(a=bad_keyword)


# ==============================================================================
# Fallback Dunder Routing
# ==============================================================================


class MinimalProto(Protocol):
    """A protocol declaring no dunders, so fallbacks are installed."""

    x: int


class MinimalImpl:
    def __init__(self) -> None:
        self.x = 1

    def __len__(self) -> int:
        return 3


def test_fallback_dunder_routes_to_wrapped() -> None:
    """A fallback dunder (not a protocol method) forwards to the wrapped object."""
    proxy: Any = spotcheck_component(MinimalProto)(MinimalImpl())
    assert len(proxy) == 3


def test_fallback_dunder_unsupported_raises_typeerror() -> None:
    """A fallback dunder the wrapped object lacks raises an informative TypeError."""
    proxy: Any = spotcheck_component(MinimalProto)(MinimalImpl())
    with pytest.raises(TypeError):
        _ = 1 in proxy
