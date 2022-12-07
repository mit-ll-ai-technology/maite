import pytest

from jatic_toolbox.testing.docs import validate_docstring


def form_doc(
    summary: str = "",
    extended: str = "",
    params: str = "",
    returns: str = "",
    see_also: str = "",
    examples: str = "",
):
    doc = "\n"
    if summary:
        doc += summary + "\n\n"
    if extended:
        doc += extended + "\n"
    if params:
        doc += f"""
Parameters
----------
{params}
"""

    if returns:
        doc += f"""
Returns
-------
{returns}
"""

    if see_also:
        doc += f"""
See Also
--------
{see_also}
"""
    if examples:
        doc += f"""
Examples
--------
{examples}
"""
    if doc.startswith("\n\n"):
        doc = doc[1:]
    return doc


def make_func(
    summary: str = "",
    extended: str = "",
    params: str = "",
    returns: str = "",
    see_also: str = "",
    examples: str = "",
):
    def f(x, y):
        ...

    f.__doc__ = form_doc(summary, extended, params, returns, see_also, examples)
    return f


def make_class(class_doc, init_doc, method_doc: str = "", property_doc: str = ""):
    class Class:
        def __init__(self, x: int) -> None:
            ...

        if method_doc:

            def method(self, y: str):
                ...

        if property_doc:

            @property
            def prop(self):
                ...

    Class.__doc__ = class_doc
    Class.__init__.__doc__ = init_doc
    if method_doc:
        Class.method.__doc__ = method_doc
    if property_doc:
        Class.prop.__doc__ = method_doc
    return Class


@pytest.mark.parametrize(
    "obj",
    [
        make_func(
            "Compute result.",
            "Uses math to do thing.",
            params="x : int\n    About x.\ny : str\n    About y.",
            returns="int\n    The result.",
            examples=">>> 1 + 1\n2",
        ),
        make_class(
            form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                "",
                examples=">>> 1+1\n2",
            ),
            form_doc(params="x : str\n    About x."),
        ),
    ],
)
def test_good_doc(obj):
    results = validate_docstring(obj, ignore=("SA01",))
    assert results["error_count"] == 0, results["errors"]


bad_doc = make_func(
    "Compute result.",
    "Uses math to do thing.",
    params="z : int\n    About x.\ny : str\n    About y.",
    returns="int\n    The result.",
)


@pytest.mark.parametrize(
    "obj, ignore_codes, error_codes",
    [
        (bad_doc, ["SA01"], ["EX01", "PR01", "PR02"]),
        (bad_doc, ["EX01", "SA01"], ["PR01", "PR02"]),
        (bad_doc, ["EX01", "SA01", "PR01"], ["PR02"]),
        (bad_doc, ["EX01", "SA01", "PR01", "PR02"], []),
    ],
)
def test_bad_doc(obj, ignore_codes, error_codes):
    results = validate_docstring(obj, ignore=ignore_codes)
    assert set(results["errors"]) == set(error_codes), results["errors"]
