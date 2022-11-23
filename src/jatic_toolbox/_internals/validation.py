from typing import Any, Optional, Tuple, TypeVar, Union, cast, overload

from typing_extensions import Protocol

from .errors import InvalidArgument

T = TypeVar("T", bound=Any)
N = TypeVar("N", int, float)
C = TypeVar("C", bound="Comparable")


def _safe_name(x: Any, ticked: bool = True) -> str:
    out = getattr(x, "__name__", str(x))
    return f"`{out}`" if ticked else out


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
    name: str, arg: T, type_: Union[type, Tuple[type, ...]], *, optional: bool = False
) -> T:
    """Checks that an argument is an instance of one or more types.

    Parameters
    ----------
    name : str
        The argument's name.

    arg : T (Any)
        The argument

    type_ : type | tuple[type, ...]

    optional : bool, keyword, optional (default=False)
        If ``True``, then ``arg`` can be None."""
    if optional and arg is None:
        return arg

    if not isinstance(arg, type_):
        if isinstance(type_, tuple):
            clause = f"of types: {', '.join(_safe_name(t) for t in type_)}"
        else:
            clause = f"of type {_safe_name(type_)}"

        raise InvalidArgument(
            f"Expected `{name}` to be {'None or ' if optional else ''}{clause}. Got {arg} "
            f"(type: {_safe_name(type(arg))})."
        )
    return arg


@overload
def check_domain(
    name: str,
    value: C,
    *,
    min_: Comparable,
    max_: Optional[Comparable] = None,
    incl_min: bool = ...,
    incl_max: bool = ...,
    lower_name: str = ...,
    upper_name: str = ...,
) -> C:
    ...


@overload
def check_domain(
    name: str,
    value: C,
    *,
    min_: Optional[Comparable] = None,
    max_: Comparable,
    incl_min: bool = ...,
    incl_max: bool = ...,
    lower_name: str = ...,
    upper_name: str = ...,
) -> C:
    ...


def check_domain(
    name: str,
    value: C,
    *,
    min_: Optional[Comparable] = None,
    max_: Optional[Comparable] = None,
    incl_min: bool = True,
    incl_max: bool = True,
    lower_name: str = "",
    upper_name: str = "",
) -> C:
    """


    Examples
    --------
    >>> domain_check("x", 1, min_=20)
    InvalidArgument: `x` must satisfy 20 <= x  Got: 1

    >>> domain_check("x", 1, min_=1, incl_min=False)
    InvalidArgument: `x` must satisfy 1 < x  Got: 1

    >>> domain_check("x", 1, min_=1, incl_min=True) # ok
    1
    >>> domain_check("x", 0.0, min_=-10, max_=10)  # ok
    0.0

    Raises
    ------
    InvalidArgument"""
    # check internal params
    check_type("name", name, str)
    check_type("incl_min", incl_min, bool)
    check_type("incl_max", incl_max, bool)

    if min_ is not None and max_ is not None:
        if incl_max and incl_min:
            if not (min_ <= max_):
                raise Unsatisfiable(f"{min_} <= {max_}")
        elif not min_ < max_:
            raise Unsatisfiable(f"{min_} < {max_}")
    elif min_ is None and max_ is None:
        raise Unsatisfiable("Neither `min_` nor `max_` were specified.")

    min_satisfied = (
        (min_ <= value if incl_min else min_ < value) if min_ is not None else True
    )
    max_satisfied = (
        (value <= max_ if incl_max else value < max_) if max_ is not None else True
    )

    if not min_satisfied or not max_satisfied:
        lsymb = "<=" if incl_min else "<"
        rsymb = "<=" if incl_max else "<"

        err_msg = f"`{name}` must satisfy"

        if min_ is not None:
            if lower_name:  # pragma: no cover
                min_ = f"{lower_name}(= {min_})"
            err_msg += f" {min_} {lsymb}"

        err_msg += f" {name}"

        if max_ is not None:
            if upper_name:
                max_ = f"{upper_name}(= {max_})"
            err_msg += f" {rsymb} {max_}"

        err_msg += f"  Got: {value}"

        raise InvalidArgument(err_msg)
    return cast(C, value)
