"""Define spotcheck runtime exceptions and a 'node-level' callable that
applies verification given a realized implementer and a typehint it purports
to satisfy.
"""

from collections.abc import Callable
from typing import Any, TypeAlias, cast

from beartype.door import die_if_unbearable
from beartype.roar import BeartypeDoorHintViolation
from typing_extensions import TypeForm


class SpotcheckError(TypeError):
    """
    A value failed a spotcheck against a typehint

    Subclasses ``TypeError`` so existing handlers keep working while letting
    proxy layers distinguish this failure from a ``TypeError`` raised by the
    wrapped implementer's own code

    Parameters
    ----------
    value : object
        The value that failed validation.
    hint : TypeForm[Any]
        The typehint the value was checked against.
    detail : str | None, optional
        A human-readable explanation of *why* the value failed -- e.g. beartype's
        violation message, which names the specific ``Is[...]`` validator (predicate)
        that was violated. Appended to the base message when present.
    failed_predicates : list[str] | None, optional
        Human labels for the specific ``Is[...]`` predicate(s) that failed (or raised),
        recovered by re-running the hint's validators. When present these are hoisted to
        the *first* line of the message so the offending predicate is obvious at a glance,
        rather than buried inside beartype's longer ``detail``.
    """

    def __init__(
        self,
        value: object,
        hint: TypeForm[Any],
        detail: str | None = None,
        failed_predicates: list[str] | None = None,
    ) -> None:
        self.value = value
        self.hint = hint
        self.detail = detail
        self.failed_predicates = list(failed_predicates) if failed_predicates else []

        lines: list[str] = []
        if self.failed_predicates:
            lines.append(f"failed predicate(s): {', '.join(self.failed_predicates)}")
        lines.append(f"{value!r} does not satisfy {hint!r}")
        if detail:
            lines.append(detail)
        super().__init__("\n".join(lines))


class VapidAttributeError(TypeError):
    """
    Missing type hint for a candidate protocol class attribute results in 'vapid' checking

    Subclasses ``TypeError`` so existing handlers keep working while letting
    proxy layers distinguish this failure from a ``TypeError`` raised by the
    wrapped implementer's own code
    """

    def __init__(self, attr_name: str) -> None:
        super().__init__(
            f"Unhinted attribute cannot be runtime-checked meaningfully. "
            f"Add hints for all attributes of object owning attribute {attr_name!r} "
            f"to enable proper runtime checking."
        )


class VapidPropertyError(TypeError):
    """
    Missing type hint for a candidate protocol class property results in 'vapid' checking

    Subclasses ``TypeError`` so existing handlers keep working while letting
    proxy layers distinguish this failure from a ``TypeError`` raised by the
    wrapped implementer's own code
    """

    def __init__(self, prop_obj: object) -> None:
        super().__init__(
            f"Unhinted property cannot be runtime-checked meaningfully. "
            f"Add hints for all properties of object owning property {prop_obj!r} "
            f"to enable proper runtime checking."
        )


class VapidMethodError(TypeError):
    """
    Missing type hint for a candidate protocol class method results in 'vapid' checking

    Subclasses ``TypeError`` so existing handlers keep working while letting
    proxy layers distinguish this failure from a ``TypeError`` raised by the
    wrapped implementer's own code
    """

    def __init__(self, method_obj: object) -> None:
        super().__init__(
            f"Unhinted method cannot be runtime-checked meaningfully. "
            f"Add hints for all properties of object owning method {method_obj!r} "
            f"to enable proper runtime checking."
        )


# TypeAlias for the verification callable footprint
CheckProcedure: TypeAlias = Callable[[object, TypeForm[Any]], None]


def _describe_failed_predicates(value: object, hint: TypeForm[Any]) -> list[str]:
    """Re-run the ``Is[...]`` validators carried by *hint* to name the offender(s).

    Runs on the failure path only. For an ``Annotated[T, Is[p1], Is[p2], ...]`` hint,
    each validator is re-applied to *value*; those that return falsy -- or raise a
    non-structural error -- are reported so the caller can surface the precise
    predicate at the top of the ``SpotcheckError`` message instead of leaving it
    buried in beartype's diagnosis.

    Best-effort and side-effect-free: returns ``[]`` when *hint* carries no
    ``Is[...]`` metadata (plain types, containers such as ``Sequence[...]``) or when
    a validator cannot be introspected. ``AttributeError`` from a predicate is treated
    as a *structural* mismatch (a missing attribute, already explained by beartype),
    not a value-level predicate failure, and is skipped.

    Parameters
    ----------
    value : object
        The value that failed ``die_if_unbearable``.
    hint : TypeForm[Any]
        The hint it was checked against.

    Returns
    -------
    list[str]
        Human labels for the failing predicate(s), e.g. ``["is_1dim"]`` or
        ``["is_scores_col_sum_lt_1 (raised AxisError: ...)"]``. Empty when nothing
        could be attributed.
    """
    offenders: list[str] = []
    for validator in getattr(hint, "__metadata__", ()):
        is_valid = getattr(validator, "is_valid", None)
        if not callable(is_valid):
            continue  # not a beartype Is[...] validator -- skip
        # ``repr(validator)`` is reliably ``"beartype.vale.Is[<predicate>]"``; the
        # validator's ``get_repr`` attribute is not consistently a method, so avoid it.
        repr_str = repr(validator)
        name = repr_str[repr_str.find("Is[") + 3 : -1] if "Is[" in repr_str else repr_str
        try:
            failed = not is_valid(value)
        except AttributeError:
            continue  # structural mismatch; beartype's message already covers it
        except Exception as exc:  # noqa: BLE001 -- any predicate-internal error names the predicate
            offenders.append(f"{name} (raised {type(exc).__name__}: {exc})")
            continue
        if failed:
            offenders.append(name)
    return offenders


# default verification callable
def spotcheck_node(value: object, hint: TypeForm[Any]) -> None:
    """
    Default "leaf-level" check on instances, backed by beartype's ``die_if_unbearable``.

    On success ``die_if_unbearable`` returns quickly (no message is built). On failure
    it raises a ``BeartypeDoorHintViolation`` whose message names the specific
    ``Is[...]`` validator (predicate) that was violated; we re-raise that as a
    ``SpotcheckError`` with the beartype detail folded in and chained via ``from``.

    Raises
    ------
        SpotcheckError : if *value* does not satisfy *hint*

        _BeartypeUtilCallableException: if hint is a bare string (unresolved)
    """

    try:
        die_if_unbearable(value, cast(Any, hint))
    except BeartypeDoorHintViolation as exc:
        raise SpotcheckError(
            value, hint, detail=str(exc), failed_predicates=_describe_failed_predicates(value, hint)
        ) from exc
    except Exception as exc:
        # Catch-all: a predicate raised something other than a hint violation. Naming the
        # predicate is especially valuable here, since the raw message does not.
        raise SpotcheckError(
            value, hint, detail=str(exc), failed_predicates=_describe_failed_predicates(value, hint)
        ) from exc
