"""A module for dynamically generating spotchecking proxies for protocol implementers."""

from __future__ import annotations

import functools
import inspect
import operator
import random
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeVar, cast, get_origin

from typing_extensions import TypeForm

from maite._internals.spotcheck.generic_hint_resolution import resolve_protocol_members
from maite._internals.spotcheck.protocol_introspection import ProtocolMembers
from maite._internals.spotcheck.spotcheck_node import (
    CheckProcedure,
    VapidAttributeError,
    VapidMethodError,
    VapidPropertyError,
    spotcheck_node,
)

_rng = random.Random()  # noqa: S311 (not used for cryptographic purposes, but for spotcheck sampling)

# ---------------------------------------------------------------------------
# Types and Configuration
# ---------------------------------------------------------------------------

P = TypeVar("P")
T = TypeVar("T")


@dataclass(frozen=True)
class _ProxyConfig:
    """Internal configuration for proxy generation.

    Attributes
    ----------
    check_procedure : CheckProcedure
        The callable used to validate values against type hints.
    rate : float
        The fraction of opportunities the check should run.
    tv_sub_map : frozenset[tuple[TypeVar, TypeForm[Any]]]
        Caller-pinned TypeVar substitutions, stored as a hashable, immutable,
        order-insensitive set of items. Typically empty. Kept hashable so the
        whole config remains a valid ``lru_cache`` key for ``_build_proxy_class``
        (a ``dict``/``MappingProxyType`` field would be unhashable and break it).

    """

    check_procedure: CheckProcedure = spotcheck_node
    rate: float = 1.0
    tv_sub_map: frozenset[tuple[TypeVar, TypeForm[Any]]] = frozenset()

    def __post_init__(self) -> None:
        if not (0 <= self.rate <= 1.0):
            raise ValueError(f"rate must be in interval [0.0, 1.0], got {self.rate}")

    @property
    def pinned_subs(self) -> Mapping[TypeVar, TypeForm[Any]]:
        """Read-only mapping view of the caller-pinned TypeVar substitutions."""
        return MappingProxyType(dict(self.tv_sub_map))


# ---------------------------------------------------------------------------
# Constants & Base Class
# ---------------------------------------------------------------------------

# For these dunder accesses, forward associated built-in on '__wrapped__'
# (We can't naively forward dunders and retain behavioral parity because
# built-ins have a thin orchestration layer with fallbacks. So if a user
# ran `iter(p)` on some proxy, looking up '__iter__' on wrapped impl
# is NOT the same (and worse than) as forwarding to iter(p.__wrapper__))

# This list of _FALLBACK_DUNDER_FORWARDS could be increased if the methods
# required by existing protocol classes were increased. If we broaden
# consideration beyond core MAITE protocols, we might adopt a dedicated
# proxying library (e.g., wrapt)
_FALLBACK_DUNDER_FORWARDS: dict[str, Callable] = {
    "__next__": next,
    "__len__": len,
    "__repr__": repr,
    "__getitem__": operator.getitem,
    "__contains__": operator.contains,
    "__call__": lambda w, *a, **kw: w(*a, **kw),
}


class _ProxyBase:
    """Static base class for all dynamically generated proxies.

    Provides shared forwarding logic and statically-known properties.

    """

    __proxy_config__: ClassVar[_ProxyConfig]
    __proxy_members__: ClassVar[ProtocolMembers]
    __wrapped__: object  # wrapped object

    def __init__(self: object, wrapped: object) -> None:
        object.__setattr__(self, "__wrapped__", wrapped)

    # Fallback attribute retrieval hook: forward to underlying wrapped object
    def __getattr__(self: object, name: str) -> object:
        target = object.__getattribute__(self, "__wrapped__")
        return getattr(target, name)

    def __setattr__(self, name: str, value: object) -> None:
        # setting attribute with name '__wrapped__' is special
        # because it is not a passthrough
        if name == "__wrapped__":
            object.__setattr__(self, name, value)
            return

        cls = type(self)

        if name in cls.__proxy_members__.attrs:
            # if name was promised by protocol class attrs, route
            # mutation through property getters/setters (built-in validation)
            object.__setattr__(self, name, value)
            return

        target = object.__getattribute__(self, "__wrapped__")
        setattr(target, name, value)


# ---------------------------------------------------------------------------
# Internal Factory Helpers
# ---------------------------------------------------------------------------


def _checked_property(
    name: str,
    hint: TypeForm[Any] | None,
    config: _ProxyConfig,
) -> property:
    """Construct a checked property descriptor."""

    # Check that the property we are creating will be validated
    # against a non-None hint
    if hint is None:
        raise VapidPropertyError(prop_obj=name)

    def fget(self: object) -> object:
        target = object.__getattribute__(self, "__wrapped__")
        value = getattr(target, name)
        if config.rate >= 1 or (_rng.random() <= config.rate):
            config.check_procedure(value, hint)
        return value

    def fset(self: object, value: object) -> None:
        if config.rate >= 1 or (_rng.random() <= config.rate):
            config.check_procedure(value, hint)
        target = object.__getattribute__(self, "__wrapped__")
        setattr(target, name, value)

    return property(fget, fset)


def _checked_method(
    name: str,
    sig: inspect.Signature,
    config: _ProxyConfig,
) -> Callable[..., object]:
    """Construct a checked method wrapper."""

    # Cached method kind - detected once on first call
    method_kind: Literal["static", "class", "instance"] | None = None

    # Check that method signature is not missing any hints (at wrap time)
    for param in sig.parameters.values():
        hint = param.annotation
        valid_unhinted_member_names = ("self", "cls")
        if hint is inspect.Parameter.empty and param.name not in valid_unhinted_member_names:
            warnings.warn(
                f"Unhinted method arguments for 'self', or 'cls' should be "
                f"have names in the set {valid_unhinted_member_names} to ensure they are "
                f"not mistaken for ordinary unhinted arguments during runtime verification.",
                stacklevel=2,
            )
            raise VapidMethodError(method_obj=name)

    # Check that method signature return type is not missing hints (at wrap time)
    if sig.return_annotation is inspect.Signature.empty:
        raise VapidMethodError(method_obj=name)

    def method(self: object, *args: Any, **kwargs: Any) -> object:  # noqa ANN401, proxy wrapping requires Any
        nonlocal method_kind  # let us determine this once on first call and store for future calls

        target = object.__getattribute__(self, "__wrapped__")
        func = getattr(target, name)

        # Detect method kind once on first call
        if method_kind is None:
            if inspect.isfunction(func):
                method_kind = "static"
            elif inspect.ismethod(func) and isinstance(func.__self__, type):
                method_kind = "class"
            else:
                method_kind = "instance"

        # Bind arguments for validation based on method kind
        if method_kind == "static":
            bound_args = sig.bind(*args, **kwargs)
        elif method_kind == "class":
            bound_args = sig.bind(type(target), *args, **kwargs)
        else:  # instance
            bound_args = sig.bind(target, *args, **kwargs)

        bound_args.apply_defaults()

        # decide whether to do check at all in this call
        do_check = config.rate >= 1 or (_rng.random() <= config.rate)

        # Get parameter names for position-based skipping
        param_names = list(sig.parameters.keys())

        for arg_name, arg_val in bound_args.arguments.items():
            # Skip the first parameter for instance and class methods BY POSITION
            # This handles protocols where the first param is not named 'self' or 'cls'
            if method_kind in ("instance", "class") and param_names and arg_name == param_names[0]:
                continue

            param = sig.parameters[arg_name]
            hint = param.annotation

            if hint is inspect.Parameter.empty:
                raise VapidMethodError(method_obj=self)

            if do_check:
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    for item in arg_val:
                        config.check_procedure(item, hint)
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    for item in arg_val.values():
                        config.check_procedure(item, hint)
                else:
                    config.check_procedure(arg_val, hint)

        if sig.return_annotation is inspect.Signature.empty:
            raise VapidMethodError(method_obj=self)

        # Call the implementation - func is already bound correctly for instance/class methods
        result = func(*args, **kwargs)

        if do_check:
            config.check_procedure(result, sig.return_annotation)

        return result

    method.__name__ = name
    return method


def _fallback_dunder(name: str, func_form: Callable[..., object]) -> Callable[..., object]:
    """Construct a fallback dunder method that routes to the wrapped object."""

    def fallback(self: object, *args: Any, **kwargs: Any) -> object:  # noqa: ANN401, fallback wrapping is necessary blind to signatures
        target = object.__getattribute__(self, "__wrapped__")
        return func_form(target, *args, **kwargs)

    fallback.__name__ = name
    return fallback


@functools.lru_cache(maxsize=128)
def _build_proxy_class(
    protocol: TypeForm[Any],
    config: _ProxyConfig,
) -> type[_ProxyBase]:
    """Dynamically construct a proxy class using the type metaclass.

    Returns ``type[_ProxyBase]`` (not bare ``type``) so callers keep static
    access to the ``ClassVar`` members declared on :class:`_ProxyBase`
    (``__proxy_members__`` / ``__proxy_config__``). The three-arg ``type(...)``
    constructor is statically typed as bare ``type``, so a single ``cast`` at the
    ``return`` records what we know: the class we just built derives from
    ``_ProxyBase``. It does *not* claim the instances satisfy any protocol -- that
    promise is asserted separately, at the wrapper boundary in ``wrap_with_config``.
    """

    # Resolve entries of protocol that have TypeVars. Caller-pinned substitutions
    # are carried on the config but not yet consumed here; wire them in when ready:
    #   members = resolve_protocol_members(protocol, pinned_tv_sub_map=dict(config.tv_sub_map) or None)
    members = resolve_protocol_members(protocol, pinned_tv_sub_map=config.pinned_subs)

    namespace: dict[str, object] = {
        "__proxy_config__": config,
        "__proxy_members__": members,
    }

    for attr_name, hint in members.attrs.items():
        namespace[attr_name] = _checked_property(attr_name, hint, config)

    for meth_name, sig in members.methods.items():
        namespace[meth_name] = _checked_method(meth_name, sig, config)

    # Mirror 'free iterator' (created when no '__iter__' defined but '__getitem__(i: int, /)' exists and
    # raises IndexError for out-of-bounds indexing *in* proxy to avoid the
    # proxy.__iter__ -> iter(proxy.__wrapped__) routing that would otherwise not be checked.

    if "__getitem__" in members.methods and "__iter__" not in members.methods:

        def iter_w_checked_fallback(self: _ProxyBase) -> Iterator[Any]:
            impl = self.__wrapped__
            getitem_return_hint = members.methods["__getitem__"].return_annotation
            do_check = config.rate >= 1 or (_rng.random() <= config.rate)
            if hasattr(type(impl), "__iter__"):
                # just forward to inner __iter__ and verify return
                for item in impl:  # type: ignore , see 'hasattr' check
                    if do_check:
                        spotcheck_node(item, getitem_return_hint)
                    yield item
            else:
                # imitate 'free iterator' construct (getting validation
                # through Proxy class' __getitem__)
                i = 0
                while True:
                    try:
                        item = self[i]  # type: ignore , we know proxy has '__getitem__'
                        if do_check:
                            spotcheck_node(item, getitem_return_hint)
                    except IndexError:
                        return
                    yield item
                    i += 1

        namespace["__iter__"] = iter_w_checked_fallback

    for dunder in _FALLBACK_DUNDER_FORWARDS:
        if dunder not in members.methods:
            # if it is already mandated by protocol,
            # (we have already forwarded 'attrs' or 'methods'
            #  via property objects or type-checked methods)
            namespace[dunder] = _fallback_dunder(dunder, _FALLBACK_DUNDER_FORWARDS[dunder])

    proto_name = getattr(protocol, "__name__", "Protocol")
    class_name = f"{proto_name}Proxy"
    return cast("type[_ProxyBase]", type(class_name, (_ProxyBase,), namespace))


# ---------------------------------------------------------------------------
# State Verification
# ---------------------------------------------------------------------------


def _spotcheck_impl_state(
    impl: object,
    members: ProtocolMembers,
    config: _ProxyConfig,
) -> None:
    """
    Perform immediate state verification on non-destructively-observable attributes
    and heuristic sequence lookups.
    """
    # 1. Verification of plain (non-dynamic) attributes
    for name, hint in members.attrs.items():
        if hint is None:
            # don't tolerate missing attribute hints, gives impression of safety
            raise VapidAttributeError(name)

        try:
            raw = inspect.getattr_static(impl, name)
        except AttributeError as e:
            raise type(e)(
                f"Implementer is missing protocol attribute {name!r}. "
                f"Must be present to perform runtime validation: {e}"
            ) from e

        # Descriptors (like properties) have a __get__ method on their class.
        # We skip them to prevent potentially destructive/expensive side effects.
        if not hasattr(type(raw), "__get__"):
            value = getattr(impl, name)
            config.check_procedure(value, hint)

    # 2. Non-Destructive Method-based Sampling (relying on convention)
    # Some structural patterns are expected to be checkable non-destructively by convention.
    # We can expand this section to include more attributes amenable to random access
    # checking or side-effect-free sampling as needed.
    if "__len__" in members.methods and "__getitem__" in members.methods:
        getitem_sig = members.methods["__getitem__"]
        item_hint = getitem_sig.return_annotation

        if item_hint is not inspect.Signature.empty:
            # We use an Any cast to easily leverage standard sequence protocols dynamically
            impl_seq = cast(Any, impl)
            length = len(impl_seq)
            if length > 0:
                index = _rng.randint(0, length - 1)
                config.check_procedure(impl_seq[index], item_hint)


def _unwrap_proxied(obj: T) -> T:
    """Remove one proxy layer if *obj* is a proxy from this factory.

    Typed as identity (``T -> T``): a proxy is always statically typed as the
    protocol it stands in for (the ``Callable[[P], P]`` contract of
    :func:`spotcheck_component`), so removing the wrapper preserves the caller's
    static type. The runtime layer-removal is reflected with a ``cast``.
    """
    if isinstance(obj, _ProxyBase):
        return cast(T, object.__getattribute__(obj, "__wrapped__"))
    return obj


# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------


def wrap_with_config(
    protocol: type[P],
    config: _ProxyConfig | None = None,
    override_existing: bool = False,
) -> Callable[[P], P]:
    """Create a wrapper function that wraps implementers of a protocol."""

    active_config = config or _ProxyConfig()
    proxy_cls = _build_proxy_class(protocol, active_config)

    # The proxy class has access to the cached introspected ProtocolMembers
    members = proxy_cls.__proxy_members__

    def wrapper(impl: P) -> P:
        if isinstance(impl, _ProxyBase):
            if not override_existing:
                warnings.warn(
                    f"Object {impl!r} is already a proxy. Skipping wrapper application. "
                    "Use `override_existing=True` to force rewrapping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return impl
            # override_existing is true and impl is a _ProxyBase, unwrap before continuing
            impl = _unwrap_proxied(impl)

        # Perform eager state verification on simple attributes and non-destructively observable attributes
        _spotcheck_impl_state(impl, members, active_config)

        # The proxy forwards every protocol member to `impl` at runtime (via checked
        # descriptors and `__getattr__`), so it stands in for a `P`. A static checker
        # cannot verify this -- the members are injected dynamically -- so the claim
        # is discharged by an explicit `cast`. This is the single, deliberate boundary
        # where the structural promise is asserted; keeping it a `cast(P, ...)` rather
        # than leaking through `Any` localizes the unchecked assumption to one line.
        return cast(P, proxy_cls(impl))

    return wrapper


def spotcheck_component(
    protocol: type[P],
    rate: float = 1.0,
    check_procedure: CheckProcedure = spotcheck_node,
    override_existing: bool = False,
    tv_sub_map: Mapping[TypeVar, TypeForm[Any]] | None = None,
) -> Callable[[P], P]:
    """Create a constructor for component-checking proxy class

    Parameters
    ----------
    protocol : P
        The protocol defining the types to enforce. May be a type, or GenericAlias representing a protocol.
    rate : float
        The fraction of opportunities the behavior-based check should run, defaults to 1.0.
        (Note: state verification always runs at wrap time.)
    check_procedure : CheckProcedure | None, optional
        A callable used to perform spotcheck validations, defaults to maite-packaged `beartype` backend.
    override_existing : bool, optional
        If True, re-wraps existing proxies. If False, returns them untouched.
    tv_sub_map : Mapping[TypeVar, TypeForm[Any]] | None, optional
        Caller-supplied TypeVar substitutions used to close any free TypeVars in
        *protocol*. Typically omitted (defaults to an empty mapping).

    Returns
    -------
    Callable[[P], P]
        Callable that will wrap component implementer in a spotchecking proxy.

    Raises
    ------
    TypeError
        If *protocol* is neither an instance of ``type`` (a plain, possibly
        unparametrized Protocol class) nor a generic alias (a parametrized
        Protocol such as ``Dataset[int]``). This is an affirmative precondition:
        callers are expected to hand in a concrete Protocol reference together
        with an implementer, so ``Optional[...]`` / ``Union[...]`` hints are not
        meaningful here and are rejected (unions specifically fall through this
        coarse gate as aliases and are then rejected by ``resolve_protocol_members``
        because their origin is not a Protocol).

    Examples
    --------

    >>> from maite.protocols.generic import Dataset, DatasetMetadata, DatumMetadata
    >>> from maite._internals.spotcheck.proxy_factory import spotcheck_component
    >>> from typing_extensions import Annotated, TypeAlias
    >>> from beartype.vale import Is
    >>> PosInt = Annotated[int, Is[lambda x: x > 0]]  # store predicates on TypeAliases!
    >>> DatasetProto: TypeAlias = Dataset[PosInt, str, DatumMetadata]  # Define component expectations in type

    Define a Dataset implementer that obeys 'PosInt' promise of Dataset[PosInt, str, DatumMetadata] type

    >>> class GoodDummyDataset:
    ...     metadata: DatasetMetadata = {"id": "ExampleDataset"}
    ...
    ...     def __getitem__(self, i: int) -> tuple[int, str, DatumMetadata]:
    ...         if i >= len(self):
    ...             raise IndexError("Index out of range")
    ...         return (1, "asdf", DatumMetadata(id="OnlyDatum"))
    ...
    ...     def __len__(self) -> int:
    ...         return 2

    Define Dataset implementer that *violates* 'PosInt' promise of Dataset[PosInt, str, DatumMetadata] type

    >>> class BadDummyDataset(GoodDummyDataset):
    ...     def __getitem__(self, i: int) -> tuple[int, str, DatumMetadata]:
    ...         return (-1, "asdf", DatumMetadata(id="OnlyDatum"))

    Valid implementers is wrapped and can be interacted with as objects of the protocol type

    >>> checked_good_dataset: DatasetProto = spotcheck_component(DatasetProto)(GoodDummyDataset())
    >>> for datum in iter(checked_good_dataset):
    ...     print(f"{datum=}")
    datum=(1, 'asdf', {'id': 'OnlyDatum'})
    datum=(1, 'asdf', {'id': 'OnlyDatum'})

    Invalid implementers are caught by `spotcheck_component` (this occurs at wrap-time for non-destructive observations)

    >>> checked_bad_dataset = spotcheck_component(DatasetProto)(BadDummyDataset())  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    maite.spotcheck.SpotcheckError: ...
    """
    if not (isinstance(protocol, type) or get_origin(protocol) is not None):
        raise TypeError(
            "spotcheck_component requires `protocol` to be a Protocol class or a "
            f"parametrized Protocol alias (e.g. Dataset or Dataset[int]); got {protocol!r}."
        )

    # -- pass through config-related args to _ProxyConfig constructor --
    # (We are effectively writing defaults to _ProxyConfig here and not in config
    # but spotcheck_component API clarity seems worth it)
    config = _ProxyConfig(
        check_procedure=check_procedure,
        rate=rate,
        tv_sub_map=frozenset((tv_sub_map or {}).items()),
    )

    return wrap_with_config(
        protocol,
        config,
        override_existing=override_existing,
    )
