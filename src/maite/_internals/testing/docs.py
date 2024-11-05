# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# flake8: noqa

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection
from inspect import getsource, isclass
from itertools import chain, zip_longest
from typing import Any, Callable, Literal, Protocol, TypedDict, Union, cast, overload

from typing_extensions import NotRequired, ReadOnly, TypeAlias

from maite._internals.validation import check_type
from maite.errors import InvalidArgument

from ..utils import is_typed_dict

NumpyDocErrorCode: TypeAlias = Literal[
    "GL01",
    "GL02",
    "GL03",
    "GL05",
    "GL06",
    "GL07",
    "GL08",
    "GL09",
    "GL10",
    "SS01",
    "SS02",
    "SS03",
    "SS04",
    "SS05",
    "SS06",
    "ES01",
    "PR01",
    "PR02",
    "PR03",
    "PR04",
    "PR05",
    "PR06",
    "PR07",
    "PR08",
    "PR09",
    "PR10",
    "RT01",
    "RT02",
    "RT03",
    "RT04",
    "RT05",
    "YD01",
    "SA01",
    "SA02",
    "SA03",
    "SA04",
    "EX01",
    "NOQA",
]

ERRORCODES: set[NumpyDocErrorCode] = set(NumpyDocErrorCode.__args__)


class _C:
    ...


AUTO_INIT_DOC = _C.__init__.__doc__


class _NumpyDocValidate(TypedDict):
    # doc-ignore: NOQA
    type: Literal["function", "type"]
    docstring: str
    deprecated: bool
    file: str
    file_line: int
    errors: list[tuple[NumpyDocErrorCode, str]]


class NumPyDocResults(TypedDict):
    # doc-ignore: NOQA
    error_count: int
    errors: dict[NumpyDocErrorCode, list[str]]
    file: str
    file_line: int
    ignored_errors: ReadOnly[NotRequired[dict[NumpyDocErrorCode, list[str]]]]


class NumPyDocResultsWithIgnored(NumPyDocResults):
    # doc-ignore: NOQA
    ignored_errors: dict[NumpyDocErrorCode, list[str]]


doc_ignore_re = re.compile(r"#\s?doc-ignore:(.*)")
_comma_or_whitespace = re.compile(r"[,\s+]")


def _get_numpy_tags(obj: Any) -> set[NumpyDocErrorCode]:
    """Searches source code for # doc-ignore: <list of error codes>."""
    try:
        src = getsource(obj)
    except TypeError:  # pragma: no cover
        # handles classes defined in `__main__`
        return set()
    if isclass(obj):
        src = src[: src.find("def ")]
    joined_tags = ",".join([x.strip() for x in re.findall(doc_ignore_re, src)])
    joined_tags = {x.strip() for x in re.split(_comma_or_whitespace, joined_tags)}
    return ERRORCODES & joined_tags


@overload
def validate_docstring(
    obj: Any,
    ignore: Collection[NumpyDocErrorCode] = ...,
    method_ignore: Collection[NumpyDocErrorCode] | None = ...,
    property_ignore: Collection[NumpyDocErrorCode] | None = ...,
    include_ignored_errors: Literal[True] = ...,
    ignore_via_comments_allowed: bool = ...,
) -> NumPyDocResultsWithIgnored:
    ...


@overload
def validate_docstring(
    obj: Any,
    ignore: Collection[NumpyDocErrorCode] = ...,
    method_ignore: Collection[NumpyDocErrorCode] | None = ...,
    property_ignore: Collection[NumpyDocErrorCode] | None = ...,
    include_ignored_errors: bool = ...,
    ignore_via_comments_allowed: bool = ...,
) -> NumPyDocResults:
    ...


def validate_docstring(
    obj: Any,
    ignore: Collection[NumpyDocErrorCode] = ("SA01",),
    method_ignore: Collection[NumpyDocErrorCode] | None = None,
    property_ignore: Collection[NumpyDocErrorCode] | None = None,
    include_ignored_errors: bool = False,
    ignore_via_comments_allowed: bool = True,
) -> Union[NumPyDocResults, NumPyDocResultsWithIgnored]:
    # fmt: off
    """
    Validate an object's docstring against the NumPy docstring standard [1]_.

    The body of a function or class can include a comment of the format:: 

       # doc-ignore: <CODE1> <CODE2> [...]
    
    where each <CODEN> is any error code listed in the Notes section. This will cause
    `validate_docstring` to ignore said error code during its analysis.

    When `obj` is a class object, the `__doc__` attributes of `obj`, `obj.__init__` and
    of all of the public methods and properties of `obj` will be validated. The
    doc strings of `doc` and `doc.__init__` are considered in conjunction with one
    another so that users need not specify redundant documentation.

    This function require that `numpydoc` is installed.

    Parameters
    ----------
    obj : Any
        A function, class, module, or method whose docstring will be validated.
        If `obj` is a class, its public methods and properties will be traversed and
        their doc strings will be included in the validation process.

    ignore : Collection[ErrorCode], optional (default=['SA01'])
        One or more error codes to be excluded from the reported errors. See the
        Notes section for a list of error codes. NOQA can be specified to ignore
        *all* errors.

    method_ignore : Collection[ErrorCode] | None
        For method docstring. One or more error codes to be excluded from the reported
        errors. If not specified, defers to the codes specified in `ignore`.

    property_ignore : Collection[ErrorCode] | None
        For property doc strings. One or more error codes to be excluded from the
        reported errors. If not specified, defers to the codes specified in `ignore`.

    include_ignored_errors : bool, optional (default=False)
        If `True`, include the errors that were ignored during the validation.

    ignore_via_comments_allowed : bool, optional (default=True)
        If `True` then the source code of `obj` will be parsed for comments of the form
        # doc-ignore: <CODE1> <CODE2> [...] to extract additional error codes that 
        will be ignored during the validation process. Class properties are not 
        supported.

    Returns
    -------
    NumPyDocResults
        A dictionary with the following fields.
            - error_count : int
            - errors : dict[ErrorCode, list[str]]
            - ignored_errors : NotEquired[dict[ErrorCode, list[str]]]
            - file : str
            - file_line : int

    Notes
    -----
    The following are the error codes that can be returned by this validation function:
    - NOQA: Can be specified to ignore all error codes.
    - GL01: Docstring text (summary) should start in the line immediately after the opening quotes.
    - GL02: Closing quotes should be placed in the line after the last text in the docstring.
    - GL03: Double line break found.
    - GL05: Tabs found at the start of line.
    - GL06: Found unknown section.
    - GL07: Sections are in the wrong order.
    - GL08: The object does not have a docstring.
    - GL09: Deprecation warning should precede extended summary.
    - GL10: reST directives {directives} must be followed by two colons.
    - SS01: No summary found.
    - SS02: Summary does not start with a capital letter.
    - SS03: Summary does not end with a period.
    - SS04: Summary contains heading whitespaces.
    - SS05: Summary must start with infinitive verb, not third person.
    - SS06: Summary should fit in a single line.
    - ES01: No extended summary found.
    - PR01: Signature parameter not documented.
    - PR02: Unknown parameters included in Parameters.
    - PR03: Wrong parameters order.
    - PR04: Parameter has no type
    - PR05: Parameter type should not finish with ".".
    - PR06: Parameter type should be changed.
    - PR07: Parameter has no description.
    - PR08: Parameter description should start with a capital letter.
    - PR09: Parameter description should finish with ".".
    - PR10: Parameter requires a space before the colon.
    - RT01: No Returns section found
    - RT02: The first line of the Returns section should contain only the type.
    - RT03: Return value has no description.
    - RT04: Return value description should start with a capital letter.
    - RT05: Return value description should finish with ".".
    - YD01: No Yields section found.
    - SA01: See Also section not found.
    - SA02: Missing period at end of description for See Also reference.
    - SA03: Description should be capitalized for See Also reference.
    - SA04: Missing description for See Also reference.
    - EX01: No examples section found.

    References
    ----------
    .. [1] https://numpydoc.readthedocs.io/en/latest/format.html

    Examples
    --------
    >>> from maite.testing.docs import validate_docstring
    >>> from maite.testing.documentation.documentation_dependencies import person
    
    Let's ignore the need for an Extended Summary and a See Also section.

    >>> validate_docstring(person, ignore=('ES01', 'SA01'), include_ignored_errors=True)
    {'error_count': 0, 'errors': {}, ...'ignored_errors': {'ES01': ['No extended summary found'], 'SA01': ['See Also section not found']}...
    
    Using comments to skip validation.

    >>> def f():
    ...     # doc-ignore: GL08
    ...     return
    >>> validate_docstring(f)   # doctest: +ELLIPSIS
    {'error_count': 0, 'errors': {}, 'file': ..., 'file_line': 1}
    >>> validate_docstring(f, ignore_via_comments_allowed=False)   # doctest: +ELLIPSIS
    {'error_count': 1, 'errors': {'GL08': ['The object does not have a docstring']}, 'file': ..., 'file_line': 1}
    """
    # fmt: on
    try:
        from numpydoc.docscrape import ClassDoc
        from numpydoc.validate import get_doc_object, validate
    except ImportError:
        raise ImportError("`numpydoc` must be installed in order to use this function.")

    check_type("ignore_via_comments_allowed", ignore_via_comments_allowed, bool)

    ignore = set(ignore)

    if method_ignore is None:
        method_ignore = ignore

    if property_ignore is None:
        property_ignore = ignore

    method_ignore = set(method_ignore)
    property_ignore = set(property_ignore)

    get_tags = (
        _get_numpy_tags
        if ignore_via_comments_allowed
        else lambda _: cast(set[NumpyDocErrorCode], set())
    )

    for _name, _codes in [
        ("ignore", ignore),
        ("method_ignore", method_ignore),
        ("property_ignore", property_ignore),
    ]:
        if not _codes <= ERRORCODES:
            unknown = ", ".join(sorted(_codes - ERRORCODES))
            raise InvalidArgument(
                f"`{_name}` contains the following elements that are not valid error "
                f"code(s): {unknown}"
            )
    validate = cast(Callable[[Any], _NumpyDocValidate], validate)

    errors: dict[NumpyDocErrorCode, list[str]] = defaultdict(list)
    ignored_errors: dict[NumpyDocErrorCode, list[str]] = defaultdict(list)

    def update_errors(
        new_errors: list[tuple[NumpyDocErrorCode, str]],
        ignore: set[NumpyDocErrorCode],
        prefix: str = "",
    ) -> None:
        if "NOQA" in ignore:
            return

        for err_code, err_msg in new_errors:
            if err_code not in ignore:
                errors[err_code].append(prefix + err_msg)
            else:
                ignored_errors[err_code].append(prefix + err_msg)

    if is_typed_dict(obj):
        ignore.add("NOQA")
    elif isinstance(obj, type) and issubclass(obj, Protocol):
        ignore.add("NOQA")

    doc_obj = get_doc_object(obj)

    results = validate(doc_obj)
    results_codes = {c for c, _ in results["errors"]}

    update_errors(results["errors"], ignore=get_tags(obj) | ignore)

    # We want to avoid forcing redundant sections between the class-level
    # docstring and the docstring for the class' __init__. Here we ignore
    # an error in one section if the error is not present in the other section.
    #
    # E.g., the class-level doc can be missing the Parameters section as long
    # as the __init__ is not also missing that section.
    if isinstance(doc_obj, ClassDoc):
        init_results = validate(get_doc_object(obj.__init__))
        if obj.__init__.__doc__ == AUTO_INIT_DOC:
            init_results["errors"].append(
                ("GL08", "The object does not have a docstring")
            )

        init_codes = {c for c, _ in init_results["errors"]}

        if "GL08" not in init_codes:
            update_errors(
                init_results["errors"],
                prefix=f"{obj.__name__}.__init__: ",
                ignore=get_tags(obj.__init__) | ignore,
            )
            # Ignore 'missing section' errors unless the error occurs in both the
            # class docstring and in the __init__ docstring
            resolved: list[NumpyDocErrorCode] = [
                code
                for code in errors
                if (code.endswith("01") or code == "GL08")
                and ((code not in init_codes) ^ (code not in results_codes))
            ]

            for item in resolved:
                errors.pop(item)

        del init_results

        obj = cast(Any, obj)
        for _ignore, name in chain(
            zip_longest([], doc_obj.properties, fillvalue=property_ignore),
            zip_longest([], doc_obj.methods, fillvalue=method_ignore),
        ):
            if isinstance(obj, type) and issubclass(obj, Protocol):
                # protocol attributes don't need docstrings
                break
            if name not in obj.__dict__:
                # don't scan inherited methods/attrs
                continue

            assert isinstance(name, str)
            _ignore = cast(set[NumpyDocErrorCode], _ignore)
            _member = getattr(obj, name)
            attr_results = validate(get_doc_object(_member))
            prefix = f"{obj.__name__}.{name}: "
            update_errors(
                attr_results["errors"],
                prefix=prefix,
                ignore=get_tags(_member) | _ignore,
            )

    if include_ignored_errors:
        out = NumPyDocResultsWithIgnored(
            error_count=sum((len(x) for x in errors.values())),
            errors=dict(errors),
            file=results["file"],
            file_line=results["file_line"],
            ignored_errors=dict(ignored_errors),
        )
    else:
        out = NumPyDocResults(
            error_count=sum((len(x) for x in errors.values())),
            errors=dict(errors),
            file=results["file"],
            file_line=results["file_line"],
        )

    return out
