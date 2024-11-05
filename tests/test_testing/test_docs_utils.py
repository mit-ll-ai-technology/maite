# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from itertools import chain
from typing import Protocol, TypedDict

import pytest
from pytest import param

from maite._internals.testing.docs import _get_numpy_tags
from maite.errors import InvalidArgument
from maite.testing.docs import validate_docstring


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
    class_doc: str | None = None,
    init_doc: str | None = None,
    method_doc: str = "",
    property_doc: str = "",
):
    class Class:
        if init_doc is not None:

            def __init__(self, x: int) -> None:
                ...

        if method_doc:

            def method(self, y: str):
                ...

        if property_doc:

            @property
            def prop(self):
                ...

    if class_doc is not None:
        Class.__doc__ = class_doc

    if init_doc is not None:
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
            form_doc(params="x : int\n    About x."),
        ),
        make_class(
            init_doc=form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                params="x : int\n    About x.",
                examples=">>> 1+1\n2",
            ),
        ),
        make_class(
            form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                params="x : int\n    About x.",
                examples=">>> 1+1\n2",
            ),
            init_doc="",
        ),
        make_class(
            form_doc(
                "A class thing.",
                "This class does things.\nIt does lots of things.",
                examples=">>> 1+1\n2",
            ),
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
        param(bad_doc_func, ["SA01"], ["EX01", "PR01", "PR02"], id="check_ignore0"),
        param(bad_doc_func, ["EX01", "SA01"], ["PR01", "PR02"], id="check_ignore1"),
        param(bad_doc_func, ["EX01", "SA01", "PR01"], ["PR02"], id="check_ignore2"),
        param(bad_doc_func, ["EX01", "SA01", "PR01", "PR02"], [], id="check_ignore3"),
        param(
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
            id="ignore extended summary",
        ),
        param(
            make_class(
                form_doc(
                    "A class thing.",
                    "This class does things.\nIt does lots of things.",
                ),
                form_doc(params="x : str\n    About x."),
            ),
            ["SA01", "EX01"],
            [],
            id="ignore examples",
        ),
        param(
            make_class(
                form_doc(
                    "A class thing.",
                    "This class does things.\nIt does lots of things.",
                ),
                "",
            ),
            ["SA01", "EX01", "PR01"],
            [],
            id="ignore params and examples",
        ),
        param(
            make_class(
                form_doc(
                    "A class thing.",
                    "This class does things.\nIt does lots of things.",
                    params="x : str\n    About x.",
                    examples=">>> 1+1\n2",
                )
            ),
            ["SA01"],
            ["PR02"],
            id="error: documented param x is unknown",
        ),
        param(make_class(), [], ["GL08"], id="error: No class docstring"),
    ],
)
def test_bad_doc(obj, ignore_codes, error_codes):
    results = validate_docstring(obj, ignore=ignore_codes, include_ignored_errors=True)
    assert set(results["errors"]) == set(error_codes), results["errors"]
    assert set(results["ignored_errors"]) == set(ignore_codes)
    assert results["error_count"] == sum(len(v) for v in results["errors"].values())


def test_bad_error_code():
    with pytest.raises(InvalidArgument, match=r"NOTACODE"):
        validate_docstring(make_class(), ignore=["NOTACODE"])  # type: ignore


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


def tagged_f():
    # doc-ignore: GL08
    # doc-ignore: RT03 RT05
    return


class TaggedClass:
    # doc-ignore: GL08,GL02    RT01  ,  EX01

    def prop(self):
        """Bye"""
        # doc-ignore: YD01,SS06
        ...

    def f(self):
        """Hello"""
        # doc-ignore: SA04 not-a-code EX01, SA04
        ...


@pytest.mark.parametrize(
    "obj,expected_codes",
    [
        (tagged_f, {"GL08", "RT03", "RT05"}),
        (TaggedClass, {"GL08", "GL02", "RT01", "EX01"}),
        (TaggedClass.f, {"EX01", "SA04"}),
        (TaggedClass.prop, {"YD01", "SS06"}),
    ],
)
def test_tag_parsing(obj, expected_codes):
    assert _get_numpy_tags(obj) == expected_codes


class A:
    # doc-ignore: GL08

    def f(self):
        """Hello"""
        # doc-ignore: SA04 not-a-code ES01 SS03, EX01
        ...


class B:
    # doc-ignore: GL08

    def f(self):
        """Hello"""
        # doc-ignore: SA04 not-a-code NOQA
        ...


@pytest.mark.parametrize("obj, num_err", [(tagged_f, 1), (A, 4), (B, 4)])
def test_validates_respects_comments(obj, num_err):
    assert validate_docstring(obj)["error_count"] == 0, validate_docstring(obj)
    assert (
        validate_docstring(obj, ignore_via_comments_allowed=False)["error_count"]
        == num_err
    )


class Parent:
    def f(self):
        ...


class Child(Parent):
    # doc-ignore: GL08
    #
    # method f should not be scanned
    ...


class AProtocol(Protocol):
    def meth(self):
        ...


class ATypedDict(TypedDict):
    x: int


@pytest.mark.parametrize("obj", [Child, AProtocol, ATypedDict])
def test_no_scan(obj):
    assert validate_docstring(obj)["error_count"] == 0, validate_docstring(obj)
