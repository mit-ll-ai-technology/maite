from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Collection, Dict, List, Set, Tuple, cast

from typing_extensions import Literal, NotRequired, TypeAlias, TypedDict

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
]

ERRORCODES: Set[NumpyDocErrorCode] = set(NumpyDocErrorCode.__args__)  # type: ignore


class _NumpyDocValidate(TypedDict):
    type: Literal["function", "type"]
    docstring: str
    deprecated: bool
    file: str
    file_line: int
    errors: List[Tuple[NumpyDocErrorCode, str]]


class NumPyDocResults(TypedDict):
    error_count: int
    errors: Dict[NumpyDocErrorCode, List[str]]
    file: str
    file_line: int
    ignored_errors: NotRequired[Dict[NumpyDocErrorCode, List[str]]]


def validate_docstring(
    obj: Any,
    ignore: Collection[NumpyDocErrorCode] = ("SA01",),
    include_ignored_errors: bool = False,
) -> NumPyDocResults:
    """
    Validate an object's docstring against the NumPy docstring standard [1]_.

    This is a light wrapper around the `numpydoc.validate.validate` function; it
    requires that `numpydoc` is installed.

    Parameters
    ----------
    obj : Any
        A function, class, module, or method whose docstring will be validated.
        If `obj` is a class, its public methods and properties will be traversed and
        their doc strings will be included in the validation process.

    ignore : Collection[ErrorCode], optional (default=['SA01'])
        One or more error codes to be excluded from the reported errors. See the
        Notes section for a list of error codes.

    include_ignored_errors : bool, optional (default=False)
        If `True`, include the errors that were ignored during the validation.

    Returns
    -------
    ValidationResults
        A dictionary with the following fields.
            - error_count : int
            - errors : Dict[ErrorCode, List[str]]
            - ignored_errors : Dict[ErrorCode, List[str]]
            - file : str
            - file_line : int

    Notes
    -----
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
    >>> def person(name: str, age: int):
    ...     '''
    ...     Enter person ID info.
    ...
    ...     Parameters
    ...     ----------
    ...     name : str
    ...         The person's first name.
    ...
    ...     age : int
    ...         The person's age.
    ...
    ...     Returns
    ...     -------
    ...     None
    ...         Nothing.
    ...
    ...     Examples
    ...     --------
    ...     >>> person('Brad', 22)
    ...     '''
    ...     ...

    Let's ignore the need for an Extended Summary and a See Also section.

    >>> out = validate_docstring(person, ignore=('ES01', 'SA01'))
    {'error_count': 0,
     'errors': {},
     'ignored_errors': {'ES01': ['No extended summary found'],
      'SA01': ['See Also section not found']},
     'file': 'home/scratch.py',
     'file_line': 1}
    """
    try:
        from numpydoc.docscrape import ClassDoc
        from numpydoc.validate import get_doc_object, validate
    except ImportError:
        raise ImportError("`numpydoc` must be installed in order to use this function.")

    ignore = set(ignore)
    if not ignore <= ERRORCODES:
        unknown = ", ".join(sorted(ignore - ERRORCODES))
        raise ValueError(
            f"`ignore` contains the following elements that are not valid error "
            f"code(s): {unknown}"
        )

    validate = cast(Callable[[Any], _NumpyDocValidate], validate)

    errors: Dict[NumpyDocErrorCode, List[str]] = defaultdict(list)
    ignored_errors: Dict[NumpyDocErrorCode, List[str]] = defaultdict(list)

    def update_errors(
        new_errors: List[Tuple[NumpyDocErrorCode, str]], prefix: str = ""
    ) -> None:
        for err_code, err_msg in new_errors:
            if err_code not in ignore:
                errors[err_code].append(prefix + err_msg)
            else:
                ignored_errors[err_code].append(prefix + err_msg)

    doc_obj = get_doc_object(obj)

    results = validate(doc_obj)

    update_errors(results["errors"])

    if isinstance(doc_obj, ClassDoc):
        if hasattr(obj, "__init__"):
            init_results = validate(get_doc_object(obj.__init__))
            if "GL08" in init_results["errors"]:
                init_results = None
        else:
            init_results = None

        if init_results:
            # Ignore 'missing section' errors unless the error occurs in both the
            # class docstring and in the __init__ docstring
            if "GLO8" in results["errors"]:
                # class does not have docstring; defer to __init__
                results = init_results
            else:
                update_errors(
                    init_results["errors"], prefix=f"{obj.__name__}.__init__: "
                )

                init_codes = set(c for c, _ in init_results["errors"])
                results_codes = set(c for c, _ in results["errors"])
                resolved: List[NumpyDocErrorCode] = [
                    code
                    for code in errors
                    if code.endswith("01")
                    and ((code not in init_codes) ^ (code not in results_codes))
                ]

                for item in resolved:
                    errors.pop(item)

                for code, msgs in errors.items():
                    if code.endswith("01") and len(msgs) == 2:
                        msgs.pop()

        del init_results

        for name in chain(doc_obj.properties, doc_obj.methods):
            attr_results = validate(get_doc_object(getattr(obj, name)))
            prefix = f"{obj.__name__}.{name}: "
            update_errors(attr_results["errors"], prefix=prefix)

    out = NumPyDocResults(
        error_count=sum((len(x) for x in errors.values()), start=0),
        errors=dict(errors),
        file=results["file"],
        file_line=results["file_line"],
    )
    if include_ignored_errors:
        out["ignored_errors"] = dict(ignored_errors)
    return out
