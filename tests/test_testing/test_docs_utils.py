from itertools import chain

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


def make_class(
    class_doc: str = "",
    init_doc: str = "",
    method_doc: str = "",
    property_doc: str = "",
):
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
        Class.prop.__doc__ = property_doc
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
        make_class(
            form_doc(
                examples=">>> 1+1\n2",
            ),
            form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                params="x : str\n    About x.",
            ),
        ),
        make_class(
            "",
            form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                params="x : str\n    About x.",
                examples=">>> 1+1\n2",
            ),
        ),
        make_class(
            form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                params="x : str\n    About x.",
                examples=">>> 1+1\n2",
            ),
            "",
        ),
        make_class(
            form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                params="x : str\n    About x.",
                examples=">>> 1+1\n2",
            ),
            "",
            method_doc=form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                params="y : str\n    About x.",
                examples=">>> 1+1\n2",
            ),
        ),
        pytest.param(
            make_class(),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="BadClass",
        ),
        pytest.param(
            make_func(),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="bad_func",
        ),
    ],
)
def test_good_doc(obj):
    results = validate_docstring(obj, ignore=("SA01",))
    assert results["error_count"] == 0, results["errors"]


bad_doc_func = make_func(
    "Compute result.",
    "Uses math to do thing.",
    params="z : int\n    About x.\ny : str\n    About y.",
    returns="int\n    The result.",
)

bad_doc_class = make_class(
    form_doc(
        "A class thing.",
        "This class does things.\nIt does lots of things.",
        "",
        examples=">>> 1+1\n2",
    ),
    form_doc(params="x : str\n    About x."),
)


@pytest.mark.parametrize(
    "obj, ignore_codes, error_codes",
    [
        (bad_doc_func, ["SA01"], ["EX01", "PR01", "PR02"]),
        (bad_doc_func, ["EX01", "SA01"], ["PR01", "PR02"]),
        (bad_doc_func, ["EX01", "SA01", "PR01"], ["PR02"]),
        (bad_doc_func, ["EX01", "SA01", "PR01", "PR02"], []),
        (
            make_class(
                form_doc(
                    "A class thing.",
                    "This class does things.\nIt does lots of things.",
                    "",
                    examples=">>> 1+1\n2",
                ),
                form_doc(params="x : str\n    About x."),
            ),
            ["SA01"],
            [],
        ),
        (
            make_class(
                form_doc(
                    "A class thing.",
                    "This class does things.\nIt does lots of things.",
                ),
                form_doc(params="x : str\n    About x."),
            ),
            ["SA01", "EX01"],
            [],
        ),
        (
            make_class(
                form_doc(
                    "A class thing.",
                    "This class does things.\nIt does lots of things.",
                ),
                "",
            ),
            ["SA01", "EX01", "PR01"],
            [],
        ),
    ],
)
def test_bad_doc(obj, ignore_codes, error_codes):
    results = validate_docstring(obj, ignore=ignore_codes)
    assert set(results["errors"]) == set(error_codes), results["errors"]
    assert results["error_count"] == sum(len(v) for v in results["errors"].values())


def test_ignore_method():
    class_bad_method = make_class(
        method_doc=form_doc(
            "A class thing.",
            "This class does things.\nIt does lots of things.",
            params="y : str\n    About x.",
        ),
    )
    results1 = validate_docstring(class_bad_method, method_ignore=["EX01", "SA01"])
    assert results1["error_count"] == 1, results1["errors"]

    results2 = validate_docstring(class_bad_method)
    assert results2["error_count"] == 2
    assert "GL08" in results2["errors"]
    del results2["errors"]["GL08"]
    assert all(
        msg.startswith("Class.method") for msg in chain(*results2["errors"].values())
    )


def test_ignore_property():
    class_bad_method = make_class(
        property_doc=form_doc(
            "A class thing.",
            "This class does things.\nIt does lots of things.",
        ),
    )
    results1 = validate_docstring(class_bad_method, property_ignore=["EX01", "SA01"])
    assert results1["error_count"] == 1, results1["errors"]

    results2 = validate_docstring(class_bad_method)
    assert results2["error_count"] == 2
    assert "GL08" in results2["errors"]
    del results2["errors"]["GL08"]
    assert all(
        msg.startswith("Class.prop") for msg in chain(*results2["errors"].values())
    )
