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
        The argument.

    type_ : type | tuple[type, ...]

    optional : bool, keyword, optional (default=False)
        If ``True``, then ``arg`` can be None.

    Returns
    -------
    T

    Raises
    ------
    InvalidArgument
        `arg` is not of the expected type.

    Examples
    --------
    >>> check_type('apple', 1, int)
    1
    >>> check_type('apple', 1, bool)
    InvalidArgument: Expected `apple` to be of type `bool`. Got 1 (type: `int`).
    >>> check_type('apple', 1, (int, bool))
    1
    >>> check_type('apple', None, (int, bool), optional=True)
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
            f"Expected `{name}` to be {'`None` or ' if optional else ''}{clause}. Got {arg} "
            f"(type: {_safe_name(type(arg))})."
        )
    return arg


@overload
def check_domain(
    name: str,
    value: C,
    *,
    lower: Comparable,
    upper: Optional[Comparable] = None,
    incl_low: bool = ...,
    incl_up: bool = ...,
    lower_name: str = ...,
    upper_name: str = ...,
) -> C:
    ...


@overload
def check_domain(
    name: str,
    value: C,
    *,
    lower: Optional[Comparable] = None,
    upper: Comparable,
    incl_low: bool = ...,
    incl_up: bool = ...,
    lower_name: str = ...,
    upper_name: str = ...,
) -> C:
    ...


def check_domain(
    name: str,
    value: C,
    *,
    lower: Optional[Comparable] = None,
    upper: Optional[Comparable] = None,
    incl_low: bool = True,
    incl_up: bool = True,
    lower_name: str = "",
    upper_name: str = "",
) -> C:
    """
    Checks that an argument falls within `[lower <=] arg [<= upper]`

    Parameters
    ----------
    name : str
        The argument's name.

    value : Comparable

    lower : Optional[Comparable]
        The lower bound of the domain. This bound is not checked
        if unspecified.

    upper : Optional[Comparable]
        The upper bound of the domain. This bound is not checked
        if unspecified.

    incl_low : bool, optional (default=True)
        If `True`, the lower bound is inclusive.

    incl_up : bool, optional (default=True)
        If `True`, the upper bound is inclusive.

    lower_name: str = ""
        If specified, includes the name of the lower bound in the
        error message.

    upper_name: str = ""
        If specified, includes the name of the upper bound in the
        error message.

    Returns
    -------
    value : Comparable

    Raises
    ------
    InvalidArgument
        `value` does not satisfy the inequality.

    Unsatisfiable
        An internal assertion error when the provided domain
        bounds cannot be satisfied.

    Examples
    --------
    >>> domain_check("x", 1, lower=20)
    InvalidArgument: `x` must satisfy 20 <= x  Got: 1

    >>> domain_check("x", 1, lower=1, incl_low=False)
    InvalidArgument: `x` must satisfy 1 < x  Got: 1

    >>> domain_check("x", 1, lower=1, incl_low=True) # ok
    1
    >>> domain_check("x", 0.0, lower=-10, upper=10)  # ok
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
        (lower <= value if incl_low else lower < value) if lower is not None else True
    )
    max_satisfied = (
        (value <= upper if incl_up else value < upper) if upper is not None else True
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

        err_msg += f"`.  Got: `{value}`."

        raise InvalidArgument(err_msg)
    return cast(C, value)
