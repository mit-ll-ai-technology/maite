# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# pyright: strict

from __future__ import annotations

from collections.abc import Collection
from enum import Enum, EnumMeta
from itertools import chain
from typing import Any, Callable, Protocol, TypeVar, Union, overload, runtime_checkable

from maite.errors import InvalidArgument

T = TypeVar("T")
N = TypeVar("N", int, float)
C = TypeVar("C", bound="Comparable")


def _tick(x: str):
    return "`" + x + "`"


def _safe_name(x: Any, ticked: bool = True) -> str:
    out = getattr(x, "__name__", str(x))
    return _tick(out) if ticked else out


@runtime_checkable
class Comparable(Protocol):
    def __eq__(self, __x: Any) -> bool:
        ...

    def __lt__(self: C, __x: C) -> bool:
        ...

    def __gt__(self: C, __x: C) -> bool:
        ...

    def __le__(self: C, __x: C) -> bool:
        ...

    def __ge__(self: C, __x: C) -> bool:
        ...


class Unsatisfiable(AssertionError):
    ...


def check_type(
    name: str, arg: T, type_: Union[type, tuple[type, ...]], *, optional: bool = False
) -> T:
    """
    Check that an argument is an instance of one or more types.

    Parameters
    ----------
    name : str
        The argument's name.

    arg : T (Any)
        The argument.

    type_ : type | tuple[type, ...]
        The type that `arg` should belong to. If multiple types are provided
        then checks that `arg` is an instance of at least one of them.

    optional : bool, keyword, optional (default=False)
        If ``True``, then ``arg`` can be None.

    Returns
    -------
    T
        The input value, `arg`, unchanged.

    Raises
    ------
    InvalidArgument
        `arg` is not of the expected type.

    Examples
    --------
    >>> from maite.utils.validation import check_type
    >>> check_type('apple', 1, int)
    1

    >>> try:
    ...     check_type('apple', 1, bool)
    ... except:
    ...     print("maite.errors.InvalidArgument: Expected `apple` to be of type `bool`. Got `1` (type: `int`).")
    maite.errors.InvalidArgument: Expected `apple` to be of type `bool`. Got `1` (type: `int`).

    >>> check_type('apple', 1, (int, bool))
    1

    >>> print(check_type('apple', None, (int, bool), optional=True))
    None
    """
    if optional and arg is None:
        return arg

    if not isinstance(arg, type_):
        if isinstance(type_, tuple):
            clause = f"of types: {', '.join(_safe_name(t) for t in type_)}"
        else:
            clause = f"of type {_safe_name(type_)}"

        raise InvalidArgument(
            f"Expected `{name}` to be {'`None` or ' if optional else ''}{clause}. Got "
            f"`{arg}` (type: {_safe_name(type(arg))})."
        )
    return arg


@overload
def check_domain(
    name: str,
    arg: C,
    *,
    lower: Comparable,
    upper: Comparable | None = None,
    incl_low: bool = ...,
    incl_up: bool = ...,
    lower_name: str = ...,
    upper_name: str = ...,
) -> C:
    ...


@overload
def check_domain(
    name: str,
    arg: C,
    *,
    lower: Comparable | None = None,
    upper: Comparable,
    incl_low: bool = ...,
    incl_up: bool = ...,
    lower_name: str = ...,
    upper_name: str = ...,
) -> C:
    ...


def check_domain(
    name: str,
    arg: C,
    *,
    lower: Comparable | None = None,
    upper: Comparable | None = None,
    incl_low: bool = True,
    incl_up: bool = True,
    lower_name: str = "",
    upper_name: str = "",
) -> C:
    """
    Check that an argument falls within `[lower <=] arg [<= upper]`.

    Parameters
    ----------
    name : str
        The argument's name.

    arg : Comparable
        The value to be checked.

    lower : Comparable|None
        The lower bound of the domain. This bound is not checked
        if unspecified.

    upper : Comparable|None
        The upper bound of the domain. This bound is not checked
        if unspecified.

    incl_low : bool, optional (default=True)
        If `True`, the lower bound is inclusive.

    incl_up : bool, optional (default=True)
        If `True`, the upper bound is inclusive.

    lower_name : str = ""
        If specified, includes the name of the lower bound in the
        error message.

    upper_name : str = ""
        If specified, includes the name of the upper bound in the
        error message.

    Returns
    -------
    Comparable
        The input value, `arg`, unchanged.

    Raises
    ------
    InvalidArgument
        `arg` does not satisfy the inequality.

    Unsatisfiable
        An internal assertion error when the provided domain
        bounds cannot be satisfied.

    Examples
    --------
    >>> from maite.utils.validation import check_domain
    >>> try:
    ...     check_domain("x", 1, lower=20)
    ... except:
    ...     print("maite.errors.InvalidArgument: `x` must satisfy `20 <= x`.  Got: `1`.")
    maite.errors.InvalidArgument: `x` must satisfy `20 <= x`.  Got: `1`.

    >>> try:
    ...     check_domain("x", 1, lower=1, incl_low=False)
    ... except:
    ...     print("maite.errors.InvalidArgument: `x` must satisfy `1 < x`.  Got: `1`.")
    maite.errors.InvalidArgument: `x` must satisfy `1 < x`.  Got: `1`.

    >>> check_domain("x", 1, lower=1, incl_low=True) # ok
    1

    >>> check_domain("x", 0.0, lower=-10, upper=10)  # ok
    0.0
    """
    # check internal params
    check_type("name", name, str)
    check_type("incl_low", incl_low, bool)
    check_type("incl_up", incl_up, bool)

    if lower is not None and upper is not None:
        if incl_up and incl_low:
            if not (lower <= upper):
                raise Unsatisfiable(f"{lower} <= {upper}")
        elif not lower < upper:
            raise Unsatisfiable(f"{lower} < {upper}")
    elif lower is None and upper is None:
        raise Unsatisfiable("Neither `lower` nor `upper` were specified.")

    min_satisfied = (
        (lower <= arg if incl_low else lower < arg) if lower is not None else True
    )
    max_satisfied = (
        (arg <= upper if incl_up else arg < upper) if upper is not None else True
    )

    if not min_satisfied or not max_satisfied:
        lsymb = "<=" if incl_low else "<"
        rsymb = "<=" if incl_up else "<"

        err_msg = f"`{name}` must satisfy `"

        if lower is not None:
            if lower_name:
                lower = f"{lower_name}={lower}"
            err_msg += f"{lower} {lsymb} "

        err_msg += f"{name}"

        if upper is not None:
            if upper_name:
                upper = f"{upper_name}={upper}"
            err_msg += f" {rsymb} {upper}"

        err_msg += f"`.  Got: `{arg}`."

        raise InvalidArgument(err_msg)
    return arg


class SupportsEq(Protocol):
    def __eq__(self, __o: object) -> bool:
        ...


def check_one_of(
    name: str,
    arg: T,
    collection: Union[Collection[Any], type[Enum]],
    *vals: SupportsEq,
    requires_identity: bool = False,
) -> T:
    """
    Check that `arg` is a member of `collection` or of `vals`.

    Parameters
    ----------
    name : str
        The argument's name.

    arg : T (Any)
        The argument.

    collection : Collection | type[Enum]
        Any collection (i.e., supports `__contains__` and `__iter__`) or enum type.

    *vals : Any
        Additional values to check `arg` against.

    requires_identity : bool, optional (default=False)
        If `True`, all (non hash-based) membership checks are performed element-wise
        using `x is e` rather  than the default `x is e or x == e`.

        This can be helpful for ensuring strict identity, e.g., preventing `1` from
        matching against `True`.

    Returns
    -------
    T
        The input value, `arg`, unchanged.

    Raises
    ------
    InvalidArgument
        `arg` is not of a member of `collections` nor `vals`.

    Unsatisfiable
        An internal assertion error when the provided collection is empty.

    Examples
    --------
    >>> from maite.utils.validation import check_one_of
    >>> try:
    ...     check_one_of("foo", None, [1, 2])
    ... except:
    ...     print("maite.errors.InvalidArgument: Expected `foo` to be one of: 1, 2. Got `None`.")
    maite.errors.InvalidArgument: Expected `foo` to be one of: 1, 2. Got `None`.

    Including `None` as an acceptable value

    >>> print(check_one_of("foo", None, [1, 2], None))
    None

    Enforcing strict identity using `requires_identity`:

    >>> check_one_of("foo", 1, [True])  # `1` == `True`
    1

    >>> try:
    ...     check_one_of("foo", 1, [True], requires_identity=True)
    ... except:
    ...     print("maite.errors.InvalidArgument: Expected `foo` to be: True. Got `1`.")
    maite.errors.InvalidArgument: Expected `foo` to be: True. Got `1`.

    Support for enums:

    >>> from enum import Enum
    >>> class Pet(Enum):
    ...     cat = 1
    ...     dog = 2

    >>> try:
    ...     check_one_of("bar", 88, Pet)
    ... except:
    ...     print("maite.errors.InvalidArgument: Expected `bar` to be one of: Pet.cat, Pet.dog. Got `88`.")
    maite.errors.InvalidArgument: Expected `bar` to be one of: Pet.cat, Pet.dog. Got `88`.
    >>> check_one_of("bar", Pet.cat, Pet)
    <Pet.cat: 1>
    """

    if isinstance(collection, EnumMeta):
        if isinstance(arg, collection):
            return arg
    elif requires_identity:
        if any(arg is x for x in chain(collection, vals)):
            return arg
    elif arg in collection or arg in vals:
        return arg

    values = sorted({str(x) for x in chain(collection, vals)})
    if not values:
        raise Unsatisfiable("`collections` and `args` are both empty.")

    raise InvalidArgument(
        f"Expected `{name}` to be{' one of' if len(values) > 1 else ''}: "
        f"{', '.join(values)}. Got `{arg}`."
    )


def chain_validators(*validators: Callable[[str, Any], Any]) -> Callable[[str, T], T]:
    """
    Enable validators to be chained together.

    Validators are functions like `(name: str, arg: T, [...]) -> T`. This is meant to be
    used with partial'd validators, where only the `name` and `arg` fields need be
    populated.

    Parameters
    ----------
    *validators : Callable[[str, T], T]
        Accepts `name` and `arg`, and returns `arg` if it is a valid input, otherwise
        should raise `InvalidArgument`.

    Returns
    -------
    Callable[[str, T], T]
        A function of signature `chain(name: str, arg: T) -> T` that calls each
        validator as `val(name, arg)` in order from low-index to high-index.

    Examples
    --------
    >>> from maite.utils.validation import check_type, check_domain, chain_validators
    >>> from functools import partial
    >>> is_int = partial(check_type, type_=int)
    >>> is_pos = partial(check_domain, lower=0)
    >>> check_pos_int = chain_validators(is_int, is_pos)
    >>> check_pos_int("foo", 10)
    10
    >>> try:
    ...     check_pos_int("foo", ["a"])
    ... except:
    ...     print("maite.errors.InvalidArgument: Expected `foo` to be of type `int`. Got `['a']` (type: `list`).")
    maite.errors.InvalidArgument: Expected `foo` to be of type `int`. Got `['a']` (type: `list`).
    >>> try:
    ...     check_pos_int("foo", -1)
    ... except:
    ...     print("maite.errors.InvalidArgument: `foo` must satisfy `0 <= foo`.  Got: `-1`.")
    maite.errors.InvalidArgument: `foo` must satisfy `0 <= foo`.  Got: `-1`.
    """

    def chain(name: str, arg: T) -> T:
        for v in validators:
            v(name, arg)
        return arg

    return chain
