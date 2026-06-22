# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, TypeGuard, TypeVar

from maite._internals.compat import TypedDict


# Can't bound typevar with TypedDict directly because it is actually a metaclass,
# So we can create an empty TypedDict for this
class TypedDictClass(TypedDict): ...


K = TypeVar("K")
T = TypeVar("T")
Td = TypeVar("Td", bound=TypedDictClass)


def is_list_of_type(d: Any, guard: type[T] = dict) -> TypeGuard[list[T]]:  # noqa: ANN401, deliberate use of 'Any' type
    """
    Check if object is a list of dictionaries.

    Parameters
    ----------
    d : Any
        The object to check.
    guard : type[T]
        The type guard of the dictionaries. Defaults to dict.

    Returns
    -------
    TypeGuard[list[T]]
        True if object is a list of dictionaries.

    Examples
    --------
    >>> is_list_dict([{"a": 1}, {"b": 2}])
    True
    """
    return isinstance(d, (list, tuple)) and isinstance(d[0], guard)


def is_list_dict(d: Any, _guard: type[T] = dict[Any, Any]) -> TypeGuard[list[T]]:  # noqa: ANN401, deliberate use of 'Any' type
    """
    Check if object is a list of dictionaries.

    Parameters
    ----------
    d : Any
        The object to check.
    guard : type[T]
        The type guard of the dictionaries. Defaults to dict.

    Returns
    -------
    TypeGuard[list[T]]
        True if object is a list of dictionaries.

    Examples
    --------
    >>> is_list_dict([{"a": 1}, {"b": 2}])
    True
    """
    return isinstance(d, (list, tuple)) and isinstance(d[0], dict)


def is_typed_dict(object_: Any, target: type[Td]) -> TypeGuard[Td]:  # noqa: ANN401, deliberate use of 'Any' type
    """
    Check if object is a typed dictionary.

    Parameters
    ----------
    object_ : Any
        The object to check.

    target : type[T]
        The type of the dictionary.

    Returns
    -------
    TypeGuard[T]
        True if object is a typed dictionary.

    Examples
    --------
    >>> from typing_extensions import TypedDict
    >>> class Foo(TypedDict):
    ...     a: int
    >>> is_typed_dict({"a": 1}, Foo)
    True
    """
    if not isinstance(object_, dict):
        return False

    k_obj = set(object_.keys())
    ks = set(target.__annotations__.keys())

    if hasattr(target, "__total__") and target.__total__:
        return all(k in k_obj for k in ks)

    return any(k in k_obj for k in ks)
