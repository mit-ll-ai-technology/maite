"""Resolve generic hints within signatures resulting from protocol introspection

This module's key contribution is `resolve_protocol_members`, which takes a typehint
for a protocol class (that may be a parametrized generic or a subclass of one) and
returns a fully-populated ProtocolMembers object with all TypeVars substituted appropriately.
Users can provide their own substitution maps of TypeVars to typehints (through `pinned_tv_sub_map`)
to enable wrapping generic components.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from types import UnionType
from typing import Annotated, Any, TypeVar, Union, cast, get_args, get_origin

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


def _typevar_substitute(hint: TypeForm[Any], sub_map: TypeVarSubMap) -> TypeForm[Any]:
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


def _hint_substitute(hint: TypeForm[Any], fn: Callable[[TypeForm[Any]], TypeForm[Any]]) -> TypeForm[Any]:
    """Walk *hint*'s type-expression tree, offering *fn* every node.

    Perform substitution for type hint (a sibling of :func:`_typevar_substitute`, but keyed
    on node *identity* rather than TypeVar binding). *fn* is offered each node --
    the whole hint first, then every nested argument recursively -- and returns
    either a replacement or the node unchanged. A node that *fn* replaces is taken
    as-is and not descended into; composite hints (``Annotated``, unions, and
    subscripted generic aliases such as ``tuple`` / ``Sequence``) that *fn* leaves
    alone are rebuilt from their possibly-rewritten arguments. Leaves *fn* does not
    touch are returned unchanged, so an unchanged subtree returns the identical
    object (and a fully-unchanged tree returns *hint* itself).

    This helper embeds no MAITE-domain knowledge (of AI problems, tasks, or
    primitives): which leaves get rewritten -- and into what -- lives entirely in
    *fn*. (This is the seam ``spotcheck`` uses to re-attach value predicates to plain
    primitive leaves after generic TypeVar substitution.)

    **Idempotency.** ``_hint_substitute(_hint_substitute(h, fn), fn) == _hint_substitute(h, fn)``
    for any *fn* whose replacement outputs are ``Annotated`` forms wrapping the very
    leaf they replace (as ``spotcheck``'s enrichment is). Such an output has an
    ``Annotated`` base that is *itself* a replacement target, so a naive second walk
    would descend into it, re-match, and double the metadata. The ``Annotated`` branch
    guards exactly that case: when the immediate base is one *fn* would replace, the
    node is an already-enriched form and is returned unchanged (a fixed point).
    Unrelated ``Annotated`` nodes (metadata over a non-target base) and composite
    bases are still descended into normally.
    """
    replaced = fn(hint)
    if replaced is not hint:
        return replaced  # fn claimed this node; do not descend into the replacement

    origin = get_origin(hint)
    if origin is None:
        return hint  # bare leaf fn declined to rewrite

    if origin is Annotated:
        # Rewrite only the wrapped base; the metadata (Is[...] validators, etc.) is
        # opaque to a type rewrite and is carried through untouched.
        base, *metadata = get_args(hint)
        if fn(base) is not base:
            # `base` is itself a replacement target -> this Annotated node is an
            # already-enriched form; descending would re-match `base` and double the
            # metadata. Treat it as a fixed point.
            return hint
        new_base = _hint_substitute(base, fn)
        if new_base is base:
            return hint
        return cast(TypeForm[Any], Annotated[(new_base, *metadata)])

    args = get_args(hint)
    if not args:
        return hint

    new_args = tuple(_hint_substitute(a, fn) for a in args)
    if new_args == args:
        return hint

    if origin is Union or origin is UnionType:
        # `Union[...]` subscripted with the runtime arg-tuple collapses a 1-tuple to
        # the sole member and accepts any typeforms (Annotated leaves included), which
        # a `|` fold would not reliably do across special forms.
        return cast(TypeForm[Any], Union[new_args])  # noqa: UP007

    single = new_args[0] if len(new_args) == 1 else new_args
    return cast(TypeForm[Any], origin[single])  # origin is a subscriptable generic alias


def _rewrite_signature_leaves(
    sig: inspect.Signature, fn: Callable[[TypeForm[Any]], TypeForm[Any]]
) -> inspect.Signature:
    params = [
        p.replace(annotation=_hint_substitute(p.annotation, fn)) if p.annotation is not inspect.Parameter.empty else p
        for p in sig.parameters.values()
    ]
    ret = sig.return_annotation
    if ret is not inspect.Signature.empty:
        ret = _hint_substitute(ret, fn)
    return sig.replace(parameters=params, return_annotation=ret)


def _free_typevars(hint: TypeForm[Any]) -> tuple[TypeVar, ...]:
    if isinstance(hint, TypeVar):
        return (hint,)
    found: list[TypeVar] = []
    for arg in get_args(hint):
        found.extend(_free_typevars(arg))
    return tuple(dict.fromkeys(found))  # keep order and dedupe


def _safe_merge(
    tv: TypeVar,
    hint: TypeForm[Any],
    tv_sub_map: dict[TypeVar, TypeForm[Any]],
    *,
    pinned_tv_keys: frozenset[TypeVar],
) -> None:
    """Merge item (*tv*: *hint*) into *tv_sub_map*, looking for conflicts either with
    the user-provided pinned substitutions or with a binding discovered earlier in the MRO."""
    resolved = _typevar_substitute(hint, tv_sub_map)
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

    # attempt to merge pinned substitution map with derived substitution map
    for tv, hint in (derived_tv_submap or {}).items():
        _safe_merge(tv, hint, tv_sub_map, pinned_tv_keys=pinned_tv_keys)

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
                _safe_merge(
                    tv, value, tv_sub_map, pinned_tv_keys=pinned_tv_keys
                )  # update tv_sub_map, looking for collisions

    return tv_sub_map


def _resolved_signature(sig: inspect.Signature, env: TypeVarSubMap) -> inspect.Signature:
    params = [
        p.replace(annotation=_typevar_substitute(p.annotation, env))
        if p.annotation is not inspect.Parameter.empty
        else p
        for p in sig.parameters.values()
    ]

    ret = sig.return_annotation
    if ret is not inspect.Signature.empty:
        ret = _typevar_substitute(ret, env)
    return sig.replace(parameters=params, return_annotation=ret)


def resolve_protocol_members(
    protocol: TypeForm[Any],
    *,
    require_closed: bool = True,
    pinned_tv_sub_map: TypeVarSubMap | None = None,
    leaf_rewrite: Callable[[TypeForm[Any]], TypeForm[Any]] | None = None,
) -> ProtocolMembers:
    """
    Introspect *protocol with all derivable TypeVar substitution applied.

    The single resolution point: parametrized generic protocols, plain protocols,
    and subclasses of parametrized generics (any number of hops) all route through
    here; future variants extend this function alone

    *leaf_rewrite*, when supplied, is applied as a final pass to every resolved
    attribute hint and method-signature annotation (after TypeVar substitution and
    the ``require_closed`` check) via :func:`_hint_substitute`. It is the seam through
    which ``spotcheck`` re-attaches value predicates to plain primitive leaves; it
    receives each hint node and returns a replacement or the node unchanged. It runs
    after closedness is decided, so it cannot affect TypeVar resolution.
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
            raise TypeError(f"{protocol!r} is neither a protocol class nor a parametrized generic protocol")

        derived_tv_sub_map: TypeVarSubMap = dict(
            zip(getattr(origin, "__parameters__", ()), get_args(protocol), strict=True)
        )
        origin_cls, tv_sub_map = (
            origin,
            _typevar_subs_from_class(origin, derived_tv_submap=derived_tv_sub_map, pinned_tv_submap=pinned_tv_sub_map),
        )

    members = protocol_members(origin_cls)

    # _typevar_substitute is dynamic hint-tree rewrite; the checker cannot follow
    # subscription-based substitution, so the TypeForm claim is restated here

    attrs: dict[str, TypeForm[Any] | None] = {
        name: None if hint is None else cast(TypeForm[Any], _typevar_substitute(hint, tv_sub_map))
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

    if leaf_rewrite is not None:
        attrs = {
            name: None if hint is None else cast(TypeForm[Any], _hint_substitute(hint, leaf_rewrite))
            for name, hint in attrs.items()
        }
        methods = {name: _rewrite_signature_leaves(sig, leaf_rewrite) for name, sig in methods.items()}

    return ProtocolMembers(attrs=attrs, methods=methods)
