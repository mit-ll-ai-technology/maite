"""A module for inspecting and extracting members from protocol classes."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, get_type_hints

from typing_extensions import TypeForm, get_protocol_members, is_protocol


@dataclass(frozen=True)
class ProtocolMembers:
    """
    Hold resolved versions of the required attributes and method signatures of a protocol.

    Attributes
    ----------
    attrs : Mapping[str, TypeForm[Any] | None]
        A mapping of attribute names to their resolved type hints. Write-only
        properties are mapped to `None`.
    methods : Mapping[str, inspect.Signature]
        A mapping of method names to their resolved inspection signatures.
    """

    attrs: Mapping[str, TypeForm[Any] | None]
    methods: Mapping[str, inspect.Signature]

    def __post_init__(self: "ProtocolMembers") -> None:
        """Ensure the internal mappings are truly read-only using MappingProxyType."""
        object.__setattr__(self, "attrs", MappingProxyType(self.attrs))
        object.__setattr__(self, "methods", MappingProxyType(self.methods))

    @property
    def names(self: "ProtocolMembers") -> frozenset[str]:
        """
        Return a frozenset of all member names (attributes and methods).

        Returns
        -------
        frozenset[str]
            A union of attribute and method names defined in the protocol.
        """
        return frozenset(self.attrs) | frozenset(self.methods)


def _resolved_signature(func: Callable[..., Any]) -> inspect.Signature:
    """
    Generate a signature for a callable, resolving forward references.

    Parameters
    ----------
    func : Callable[..., Any]
        The function or method to inspect and resolve type hints for.

    Returns
    -------
    inspect.Signature
        A new signature object with resolved type hints.

    Raises
    ------
    TypeError
        If type hints cannot be resolved due to a type resolution error.
    NameError
        If type hints contain unresolvable forward references (e.g., a string
        annotation referring to a name that does not exist in any accessible scope).
    """
    signature: inspect.Signature = inspect.signature(func)
    try:
        resolved_hints: dict[str, TypeForm[Any]] = get_type_hints(
            func,
            include_extras=True,
        )
    except (TypeError, NameError) as e:
        # Fallback if hints cannot be resolved
        raise type(e)(f"Protocol members expected to be resolvable: {e}") from e

    resolved_params: list[inspect.Parameter] = [
        p.replace(annotation=resolved_hints.get(p.name, p.annotation)) for p in signature.parameters.values()
    ]
    resolved_return_annotation: Any = resolved_hints.get(
        "return",
        signature.return_annotation,
    )

    return signature.replace(
        parameters=resolved_params,
        return_annotation=resolved_return_annotation,
    )


def protocol_members(proto: TypeForm[Any]) -> ProtocolMembers:
    """
    Extract all required members and signatures from a protocol.

    This function requires that all type hints in the protocol are resolvable.
    If any class-level attribute annotation, property getter return annotation,
    or method signature contains an unresolvable forward reference, a conspicuous
    error will be raised.

    Parameters
    ----------
    proto : TypeForm[Any]
        The protocol class to inspect.

    Returns
    -------
    ProtocolMembers
        A deeply immutable object containing the protocol's members.

    Raises
    ------
    TypeError
        If `proto` is not a valid protocol, if a member is not supported, or
        if type hints cannot be resolved due to a type resolution error.
    NameError
        If any type hint (class-level attribute, property getter return type,
        or method signature annotation) contains an unresolvable forward reference
        (e.g., a string annotation referring to a name that does not exist in any
        accessible scope).
    """
    if not (isinstance(proto, type) and is_protocol(proto)):
        msg: str = f"Provided type {proto} is not a valid Protocol."
        raise TypeError(msg)

    attrs: dict[str, TypeForm[Any] | None] = {}
    methods: dict[str, inspect.Signature] = {}

    try:
        # Note: returns only class-level variable annotations
        # (i.e., NOT instance methods, property objects, static methods, or class methods)
        var_hints: dict[str, TypeForm[Any]] = get_type_hints(
            proto,
            include_extras=True,
        )
    except (TypeError, NameError) as e:
        # Fail loudly if hints cannot be resolved
        raise type(e)(f"Protocol members expected to be resolvable: {e}") from e

    member_names: frozenset[str] = get_protocol_members(proto)

    for name in member_names:
        # Fast path for class-level annotated attributes
        if name in var_hints:
            attrs[name] = var_hints[name]
            continue

        # Introspect declaration without triggering any descriptors
        raw: Any = inspect.getattr_static(proto, name)

        if isinstance(raw, property):
            # Read-only attribute spelled as a property on protocol: its type
            # is the getter's resolved return annotation
            if raw.fget is None:  # write-only property, returns 'None' legitimately
                attrs[name] = None
            else:
                try:
                    attrs[name] = get_type_hints(
                        raw.fget,
                        include_extras=True,
                    ).get("return")
                except (TypeError, NameError) as e:
                    # Fail loudly if hints cannot be resolved
                    raise type(e)(f"Protocol members expected to be resolvable: {e}") from e
        elif isinstance(raw, (staticmethod, classmethod)):
            methods[name] = _resolved_signature(raw.__func__)
        elif callable(raw):
            methods[name] = _resolved_signature(raw)
        else:
            # Unhinted ordinary class attributes on protocol are not represented in varhints,
            # so they will raise this TypeError rather than a VapidAttributeError
            msg: str = (
                f"Protocol member {name!r} is not a property, callable, "
                f"staticmethod, classmethod, or hinted attribute: {raw!r}"
            )
            raise TypeError(msg)

    return ProtocolMembers(attrs=attrs, methods=methods)
