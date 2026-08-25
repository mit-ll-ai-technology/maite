"""Resolve generic hints within signatures resulting from protocol introspection

This module's key contribution is `resolve_protocol_members`, which takes a typehint
for a protocol class (that may be a parametrized generic or a subclass of one) and
returns a fully-populated ProtocolMembers object with all TypeVars substituted appropriately.
Users can provide their own substitution maps of TypeVars to typehints (through `pinned_tv_sub_map`)
to enable wrapping generic components.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, TypeVar, cast, get_args, get_origin

from typing_extensions import TypeForm, get_original_bases, is_protocol

from maite._internals.spotcheck.protocol_introspection import ProtocolMembers, protocol_members

__all__ = ["UnresolvedTypeVarsError", "resolve_protocol_members"]

TypeVarSubMap = Mapping[TypeVar, TypeForm[Any]]


class UnresolvedTypeVarsError(TypeError):
    """TypeVars remain in a member hint (open type expression)."""

    def __init__(self, protocol: object, open_members: dict[str, tuple[TypeVar, ...]]) -> None:
        self.open_members = open_members
        detail = "; ".join(f"{name}: {', '.join(str(tv) for tv in tvs)}" for name, tvs in sorted(open_members.items()))

        super().__init__(
            f"{protocol!r} resolves to open member hints ({detail}); supply the "
            f"missing parametrizations or pass require_closed=False to accept "
            f"vacuous checking of these members"
        )


class SubstitutionConflictError(TypeError):
    """Derived parametrization disagrees with caller-pinned binding."""

    def __init__(self, typevar: TypeVar, pinned: TypeForm[Any], derived: TypeForm[Any]) -> None:
        self.typevar = typevar
        self.pinned = pinned
        self.derived = derived

        super().__init__(
            f"caller pinned {typevar!r} to {pinned!r}, but the protocol's own "
            f"parametrization derives {typevar!r} = {derived!r}; drop the entry "
            f"from typevar_subs or fix the conflicting parametrization"
        )


class UninferrableSubstitutionError(TypeError):
    """MRO-derived parametrization frames disagree; no single substitution
    map can be inferred from class hierarchy"""

    def __init__(self, typevar: TypeVar, leafward: TypeForm[Any], baseward: TypeForm[Any]) -> None:
        self.typevar = typevar
        self.leafward = leafward
        self.baseward = baseward

        super().__init__(
            f"MRO-derived substitution for {typevar!r} is uninferrable: a "
            f"leaf-ward frame binds {leafward!r} but a base-ward frame binds "
            f"{baseward!r}; the class hierarchy parametrizes the same base "
            f"inconsistently and must be fixed at definition site"
        )


def _substitute(hint: TypeForm[Any], sub_map: TypeVarSubMap) -> TypeForm[Any]:
    """
    Close *hint* over *env*. Subscription on a generic alias performs
    recursive, order-aware TypeVar replacement, so no manual tree walk
    needed; unmapped TypeVars are left in place (the guard decides policy)
    """
    if isinstance(hint, TypeVar):
        return sub_map.get(hint, hint)  # type: ignore (TypeVars are TypeForm[Any], so there should be no issue)
    params = getattr(hint, "__parameters__", ())
    if params:
        return hint[tuple(sub_map.get(p, p) for p in params)]  # type: ignore[index],  if *params* exists we can count on *hint* being subscriptable
    return hint


def _free_typevars(hint: TypeForm[Any]) -> tuple[TypeVar, ...]:
    if isinstance(hint, TypeVar):
        return (hint,)
    found: list[TypeVar] = []
    for arg in get_args(hint):
        found.extend(_free_typevars(arg))
    return tuple(dict.fromkeys(found))  # keep order and dedupe


def _typevar_subs_from_class(
    cls: type, derived_tv_submap: TypeVarSubMap | None = None, pinned_tv_submap: TypeVarSubMap | None = None
) -> TypeVarSubMap:
    """
    Start with pinned_tv_submap and then compose parametrization frames leaf-to-base
    along the MRO to arrive at a single TypeVarSubMap. Raise on discovered conflicts.

    Each class records the *expressions* it used to parametrize its bases in
    ``__orig_bases__``. Each time a parametrized base is provided, we can use
    that reference to build up our 'submap'. During the walk of the *cls* mro, a
    frame may mention TypeVars bound by classes closer to the leaf (multi-hop
    chains), so every frame's values are substituted through the substitution map
    accumulated so far before merging. Key collisions during the walk raise
    a Spotcheck error and are expected to be rare. (This would happen if the
    same TypeVar were used with different bindings at 2 or more levels of the
    mro). The *seed* argument supplies the leaf-most override available to the
    caller and can be used to manually parametrize an otherwise unparametrized
    set of TypeVars in the cls mro.

    *pinned* is a caller-supplied TypeVar substitution map that can permit
    proper TypeVar resolution at the resolve_protocol_members level with a
    non-closed generic protocol class. If *pinned* is provided and it conflicts
    with class-derived TypeVar substitution, we raise a SubstitutionConflictError.
    """

    tv_sub_map: TypeVarSubMap = dict(pinned_tv_submap or {})
    pinned_tv_keys = frozenset(tv_sub_map)

    def safe_merge(tv: TypeVar, hint: TypeForm[Any]) -> None:
        """Try adding item (*tv*: *hint*) into tv_submap, looking for conflicts either with
        user-provided pinned_tv_submap or from earlier in MRO."""
        resolved = _substitute(hint, tv_sub_map)
        if tv in pinned_tv_keys:
            # A caller pin supplies the type for an otherwise-open parametrization.
            # If the class derives a *concrete* (TypeVar-free) binding that disagrees,
            # that is a genuine conflict. If the derived binding is itself still an
            # open TypeVar -- as when a polymorphic task (e.g. ``evaluate``) parametrizes
            # a protocol by its own TypeVars -- the pin legitimately closes it and wins.
            if not _free_typevars(resolved) and resolved != tv_sub_map[tv]:
                raise SubstitutionConflictError(tv, tv_sub_map[tv], resolved)
            return
        if tv in tv_sub_map:
            if resolved != tv_sub_map[tv]:
                raise UninferrableSubstitutionError(tv, tv_sub_map[tv], resolved)
            return
        tv_sub_map[tv] = resolved  # key isn't present already

    # attempt to merge pinned substitution map with derived substitution map
    for tv, hint in (derived_tv_submap or {}).items():
        safe_merge(tv, hint)

    # Walk through classes in MRO starting with *cls* and going toward 'object'
    # (Note that mro elements have no subscripts! We need to use '__parameters__'
    # attribute to get those)
    for klass in cls.__mro__:
        # Fetch a list of classes provided as bases to *klass* in its declaration
        # (could be of type 'type' for non-generic base classes or could be of
        # type '_GenericAlias' for subscripted suffix, whether parametrized with
        # real types or TypeVars.)
        for base in get_original_bases(klass):
            origin = get_origin(base)
            if origin is None or not getattr(origin, "__parameters__", ()):
                continue  # either klass is unsubscripted or has no open type parameters

            for tv, value in zip(origin.__parameters__, get_args(base), strict=True):
                safe_merge(tv, value)  # update tv_sub_map, looking for collisions

    return tv_sub_map


def _resolved_signature(sig: inspect.Signature, env: TypeVarSubMap) -> inspect.Signature:
    params = [
        p.replace(annotation=_substitute(p.annotation, env)) if p.annotation is not inspect.Parameter.empty else p
        for p in sig.parameters.values()
    ]

    ret = sig.return_annotation
    if ret is not inspect.Signature.empty:
        ret = _substitute(ret, env)
    return sig.replace(parameters=params, return_annotation=ret)


def resolve_protocol_members(
    protocol: TypeForm[Any],
    require_closed: bool = True,
    pinned_tv_sub_map: TypeVarSubMap | None = None,
) -> ProtocolMembers:
    """
    Introspect *protocol with all derivable TypeVar substitution applied.

    The single resolution point: parametrized aliases, plain protocols, and
    subclasses of parametrized generics (any numberof hops) all route through
    here; future variants extend this function alone
    """

    # resolve *protocol* hint to nearest actual type
    # get relevant class for this nearest type name and
    # keep track of any parameters used in subscription
    # (these are not directly on that type since its generic)

    if isinstance(protocol, type):
        origin_cls, tv_sub_map = protocol, _typevar_subs_from_class(protocol, pinned_tv_submap=pinned_tv_sub_map)
    else:
        # protocol is a GenericAlias, so we need to look at its unsubscripted base.
        # This is also the single point that rejects hints which are not protocols:
        # unions (`typing.Union`/`types.UnionType`), `Optional[...]`, `Annotated[...]`,
        # and non-protocol generics (`list[int]`) all have an origin that is either
        # not a type or not a Protocol, so they raise here rather than silently
        # producing a proxy for an ambiguous/non-protocol hint.
        origin = get_origin(protocol)
        if not (isinstance(origin, type) and is_protocol(origin)):
            raise TypeError(f"{protocol!r} is neither a class nor a parametrized Protocol alias")

        derived_tv_sub_map: TypeVarSubMap = dict(
            zip(getattr(origin, "__parameters__", ()), get_args(protocol), strict=True)
        )
        origin_cls, tv_sub_map = (
            origin,
            _typevar_subs_from_class(origin, derived_tv_submap=derived_tv_sub_map, pinned_tv_submap=pinned_tv_sub_map),
        )

    members = protocol_members(origin_cls)

    # _substitute is dynamic hint-tree rewrite; the checker cannot follow
    # subscription-based substitution, so the TypeForm claim is restated here

    attrs: dict[str, TypeForm[Any] | None] = {
        name: None if hint is None else cast(TypeForm[Any], _substitute(hint, tv_sub_map))
        for name, hint in members.attrs.items()
    }

    methods = {name: _resolved_signature(sig, tv_sub_map) for name, sig in members.methods.items()}

    if require_closed:
        open_members: dict[str, tuple[TypeVar, ...]] = {}
        for name, hint in attrs.items():
            if hint is not None and (tvs := _free_typevars(hint)):
                open_members[name] = tvs
        for name, sig in methods.items():
            tvs = tuple(
                dict.fromkeys(
                    tv
                    for ann in (
                        *(p.annotation for p in sig.parameters.values()),
                        sig.return_annotation,
                    )
                    if ann is not inspect.Parameter.empty
                    for tv in _free_typevars(ann)
                )
            )
            if tvs:
                open_members[name] = tvs
        if open_members:
            raise UnresolvedTypeVarsError(protocol, open_members)

    return ProtocolMembers(attrs=attrs, methods=methods)
