"""Define a higher-level API that permits wrapping all arguments of a callable
that are MAITE protocol types. (This permits MAITE tasks to be decorated for
light-weight opt-in to spotchecking)
"""

import functools
import inspect
from collections.abc import Callable, Mapping
from types import GenericAlias, MappingProxyType, UnionType
from typing import Annotated, Any, ParamSpec, TypeVar, Union, get_args, get_origin, get_type_hints, overload

from typing_extensions import TypeAliasType, TypeForm, is_protocol

from maite._internals.protocols.image_classification import TV_SUB_MAP as _IMAGE_CLASSIFICATION_TV_SUB_MAP
from maite._internals.protocols.multiobject_tracking import TV_SUB_MAP as _MULTIOBJECT_TRACKING_TV_SUB_MAP
from maite._internals.protocols.object_detection import TV_SUB_MAP as _OBJECT_DETECTION_TV_SUB_MAP
from maite._internals.spotcheck.proxy_factory import _unwrap_proxied, spotcheck_component
from maite.protocols.generic import (
    Augmentation,
    DataLoader,
    Dataset,
    Metric,
    Model,
)

R = TypeVar("R")
P = ParamSpec("P")

# store these to screen typehints for those that are related

COMPONENT_PROTOCOL_BASE_CLASSES: set[type] = {Dataset, DataLoader, Augmentation, Model, Metric}

# Map AI problem labels to TypeVar->TypeForm substitutions
# (so `spotcheck` can be used with polymorphic functions with the use of 'ai_problem' argument
# that picks out the TypeVar->TypeForm substitutions to hand to spotcheck_component.)

AI_PROBLEM_TV_SUB_MAPS: Mapping[str, Mapping[TypeVar, TypeForm[Any]]] = MappingProxyType(
    {
        "image_classification": _IMAGE_CLASSIFICATION_TV_SUB_MAP,
        "object_detection": _OBJECT_DETECTION_TV_SUB_MAP,
        "multiobject_tracking": _MULTIOBJECT_TRACKING_TV_SUB_MAP,
    }
)


class UndefinedAIProblemError(TypeError):
    """Provided AI problem is undefined in MAITE and thus doesn't correspond to any known set of MAITE primitives"""

    def __init__(self, ai_problem_label: str) -> None:
        self.ai_problem_label = ai_problem_label

        super().__init__(
            f"AI problem with label '{ai_problem_label}' is unknown to MAITE, "
            "and cannot be used to specify expected primitive types "
            "(required for runtime validation)."
        )


class UnresolvableUnionHintError(TypeError):
    """A hint is a multi-member ``Union`` that names a component protocol.

    Such a hint names more than one candidate class, so there is no single
    protocol to wrap independent of the runtime implementation. Note that
    ``Optional[X]`` -- a two-member union of one type and ``None`` -- is *not*
    considered unresolvable: it defers to ``X``.
    """

    def __init__(self, hint: TypeForm[Any]) -> None:
        self.hint = hint

        super().__init__(
            f"{hint!r} is a Union naming multiple candidate classes (at least one of which "
            "is a component protocol), so a unique protocol to wrap cannot be deduced "
            "independent of the implementation."
        )


def _reduce(hint: TypeForm[Any] | None) -> TypeForm[Any] | None:
    """Structurally reduce *hint* toward the single class it denotes.

    Strips ``Annotated`` / ``TypeAliasType`` wrappers and resolves generic
    aliases to their origin class. Returns:

    * a ``type`` when the hint denotes a single concrete class,
    * the ``Union`` object itself (``typing.Union`` or PEP-604 ``X | Y``) left
      intact, so the whitelist-aware policy layer can inspect its members,
    * ``None`` for ``None`` / ``NoneType`` and anything otherwise irreducible.

    This is deliberately whitelist-free: *which* classes count as component
    protocols -- and *whether* a union is tolerated -- is decided downstream in
    :func:`hint_matches_component_protocol`, where the whitelist lives.
    """

    seen: set[int] = set()  # guard against TypeAliasType self-reference loops

    while id(hint) not in seen:
        seen.add(id(hint))

        # base cases
        if hint is None or hint is type(None):
            return None

        if isinstance(hint, type) and not isinstance(hint, GenericAlias):
            # isinstance(list[int], type) == True in 3.10
            return hint

        if isinstance(hint, TypeAliasType):  # unwrap TypeAliasType (3.12+)
            hint = hint.__value__
            continue

        origin = get_origin(hint)
        if isinstance(hint, GenericAlias):  # catch list[...] and dict[...]
            return origin

        if origin is Annotated:  # unwrap 'Annotated[<hint>, ...]'
            hint = get_args(hint)[0]
        elif origin in (Union, UnionType):
            return hint  # leave union intact; policy layer applies the whitelist
        elif origin is not None:
            hint = origin  # generic alias -> origin class (user generics, list/dict, ...)
        else:
            return None

    return None


def hint_matches_component_protocol(
    hint: TypeForm[Any] | None,
    targets: set[type] = COMPONENT_PROTOCOL_BASE_CLASSES,
) -> bool:
    """Check whether *hint* refers to a descendant of a member of *targets*.

    Union policy (whitelist-dependent, hence handled here rather than in
    :func:`_reduce`):

    * ``Optional[X]`` (a two-member union of one type and ``None``) defers to ``X``.
    * A genuine multi-member union raises :class:`UnresolvableUnionHint` if any
      member is itself a component protocol (we cannot pick a unique one to wrap);
      otherwise it is ignored and ``False`` is returned.
    """

    core = _reduce(hint)
    if core is None:
        return False

    if get_origin(core) in (Union, UnionType):
        args = get_args(core)
        non_none = [a for a in args if a is not type(None)]
        # rule (a): exactly Optional[X] -> defer to the single non-None member
        if len(args) == 2 and len(non_none) == 1:
            return hint_matches_component_protocol(non_none[0], targets)
        # rule (b): a genuine multi-member union is fatal only if it names a component
        if any(hint_matches_component_protocol(a, targets) for a in non_none):
            raise UnresolvableUnionHintError(core)
        return False

    # core is a single concrete class (see _reduce contract)
    if not isinstance(core, type):
        raise TypeError("Expected 'core' to be of type 'type'")

    return is_protocol(core) and bool(targets & set(core.__mro__))


def _reduce_to_forwardable_hint(hint: TypeForm[Any] | None) -> TypeForm[Any] | None:
    """Reduce *hint* to a form directly forwardable to ``spotcheck_component``.

    Strips ``Annotated`` / ``TypeAliasType`` wrappers and collapses ``Optional[X]``
    to ``X`` while *preserving* any generic parametrization (so ``Dataset[int]``
    survives as ``Dataset[int]`` rather than being reduced to ``Dataset``). Returns
    ``None`` for genuine multi-member unions and for anything otherwise
    irreducible. This is what lets a decorated task forward a clean Protocol class
    or parametrized alias -- never an ``Optional``/``Union`` that
    ``spotcheck_component`` would reject.
    """

    seen: set[int] = set()

    while id(hint) not in seen:
        seen.add(id(hint))

        if hint is None or hint is type(None):
            return None

        if isinstance(hint, TypeAliasType):  # unwrap TypeAliasType (3.12+)
            hint = hint.__value__
            continue

        origin = get_origin(hint)
        if origin is Annotated:  # unwrap 'Annotated[<hint>, ...]'
            hint = get_args(hint)[0]
        elif origin in (Union, UnionType):
            args = get_args(hint)
            non_none = [a for a in args if a is not type(None)]
            if len(args) == 2 and len(non_none) == 1:
                hint = non_none[0]  # unwrap Optional, keep parametrization
            else:
                return None  # genuine multi-member union -> nothing single to forward
        else:
            return hint  # plain type or parametrized alias -> forward as-is

    return None


@overload
def spotcheck(
    maite_task: Callable[P, R], /, *, rate: float = ..., override_existing: bool = ..., ai_problem: str | None = ...
) -> Callable[P, R]: ...


@overload
def spotcheck(
    maite_task: None = ..., /, *, rate: float = ..., override_existing: bool = ..., ai_problem: str | None = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def spotcheck(
    maite_task: Callable[P, R] | None = None,
    /,
    *,
    rate: float = 1.0,
    override_existing: bool = False,
    ai_problem: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """
    Apply spotcheck_component to arguments of *maite_task* that are hinted as
    MAITE component protocols. This opts in to runtime validation at use time.

    Parameters
    ----------
    maite_task: Callable[P, R]
        The MAITE task callable (accepting or returning some values typed as MAITE component protocols)
    rate : float
        The fraction of opportunities the behavior-based check should run, defaults to 1.0.
        (Note: state verification always runs at wrap time.)
    override_existing : bool, optional
        If True, re-wraps existing proxies. Defaults to False.
    ai_problem : str | None, optional
        The string label of an AI problem (needed only when MAITE task is generic over multiple
        AI problems.)

    Raises
    ------
    UndefinedAIProblemError
        At decoration time, if an ``ai_problem`` is provided that does
        not correspond to AI problem known to MAITE.

    UnresolvableUnionHintError
        At decoration time, if a parameter or return hint is a genuine
        multi-member ``Union`` that names a component protocol (there is no
        unique protocol to wrap). ``Optional[Component]`` is fine and defers to
        the wrapped component.

    SubstitutionConflictError
        When applying ``spotcheck`` decorator to callables specific to an AI problem,
        if the provided ``ai_problem`` argument corresponds to MAITE primitive types
        that differ from primitive types implied by component argument hints, this
        error is raised.
    """

    # populate typeVar substitution map if 'ai_problem' was provided
    tv_sub_map: Mapping[TypeVar, TypeForm[Any]] | None = None
    if ai_problem is not None:
        if ai_problem in AI_PROBLEM_TV_SUB_MAPS:
            tv_sub_map = AI_PROBLEM_TV_SUB_MAPS[ai_problem]
        else:
            raise UndefinedAIProblemError(ai_problem_label=ai_problem)

    def deco(func: Callable[P, R]) -> Callable[P, R]:

        # Find all arguments to *maite_task* that are typed as MAITE component protocols
        # (Note we need to set include_extras=True to avoid stripping 'Annotated' for nested spotcheck_component calls)
        task_hints: dict[str, TypeForm[Any]] = get_type_hints(func, include_extras=True)
        # e.g. {'ds': od.Dataset, 'aug': od.Augmentation, ... 'return': od.Dataset}

        args_to_wrap = frozenset(
            n for n, h in task_hints.items() if n != "return" and hint_matches_component_protocol(h)
        )

        check_return = hint_matches_component_protocol(task_hints.get("return"))

        if not (args_to_wrap or check_return):
            return func

        # Precompute the hint forwarded to spotcheck_component for each wrapped slot.
        # Optional/Annotated/TypeAlias wrappers are stripped here so spotcheck_component
        # receives a bare Protocol class or parametrized alias (its precondition),
        # never an Optional/Union it would reject.
        forward_hints: dict[str, TypeForm[Any]] = {}
        for name in (*args_to_wrap, *(("return",) if check_return else ())):
            reduced = _reduce_to_forwardable_hint(task_hints[name])
            if reduced is not None:
                forward_hints[name] = reduced

        sig = inspect.signature(func)

        # define wrapper that applies spotcheck
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # wrap arguments
            if args_to_wrap:
                bound = sig.bind(*args, **kwargs)  # canonicalize arguments
                for name in args_to_wrap & bound.arguments.keys():
                    # guard against None-valued implementers for 'Optional'-typed slots
                    if bound.arguments[name] is not None and name in forward_hints:
                        bound.arguments[name] = spotcheck_component(
                            forward_hints[name],  # type: ignore , assume _reduce_to_forwardable doesn't change assignability
                            rate=rate,
                            override_existing=override_existing,
                            tv_sub_map=tv_sub_map,
                        )(bound.arguments[name])
                result = func(*bound.args, **bound.kwargs)
            else:
                result = func(*args, **kwargs)

            if result is not None and check_return:
                # check return value, but don't return wrapped object
                spotcheck_component(
                    forward_hints["return"],  # type: ignore , assume _reduce_to_forwardable doesn't change assignability
                    rate=rate,
                    override_existing=True,  # override existing proxy in case task passes it through
                    tv_sub_map=tv_sub_map,
                )(result)

            return _unwrap_proxied(result)  # unwrap in case result was wrapped on entering task

        return wrapper

    # enables calling in decorator form (where maite_task is 'None'
    # and deco call is deferred) or single-pass wrapping form.
    if maite_task is None:
        return deco
    return deco(maite_task)
