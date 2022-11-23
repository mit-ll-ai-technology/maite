# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from typing import Union

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings

from jatic_toolbox._internals.errors import InvalidArgument
from jatic_toolbox.utils.validation import check_domain, check_type


def everything_except(excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


any_types = st.from_type(type)


@settings(max_examples=10)
@given(
    name=st.sampled_from(["name_a", "name_b"]),
    target_type=st.shared(any_types, key="target_type"),
    value=st.shared(any_types, key="target_type").flatmap(everything_except),
)
def test_check_type_catches_bad_type(name, target_type, value):
    with pytest.raises(InvalidArgument):
        check_type(name, arg=value, type_=target_type)


@given(
    target_type=st.shared(any_types, key="target_type"),
    value=st.shared(any_types, key="target_type").flatmap(st.from_type),
)
def test_check_type_passes_good_type(target_type, value):
    check_type("dummy", arg=value, type_=target_type)


@given(...)
def test_check_multiple_types(value: Union[str, int]):
    out = check_type("dummy", value, type_=(str, int))
    assert out == value


def test_no_bounds():
    with pytest.raises(AssertionError):
        check_domain("arg", 1)  # type: ignore


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            dict(value=1, min_=1, max_=1, incl_min=False, incl_max=False),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="1 < ... < 1",
        ),
        pytest.param(
            dict(value=1, min_=1, max_=1, incl_min=True, incl_max=False),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="1 <= ... < 1",
        ),
        pytest.param(
            dict(value=1, min_=1, max_=1, incl_min=True, incl_max=True),
            id="1 <= ... <= 1",
        ),
        pytest.param(
            dict(value=1, min_=1, max_=1, incl_min=False, incl_max=True),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="1 < ... <= 1",
        ),
        pytest.param(
            dict(value=1, min_=2, max_=1, incl_min=False, incl_max=False),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="2 < ... < 1",
        ),
        pytest.param(
            dict(value=1, min_=2, max_=1, incl_min=True, incl_max=False),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="2 <= ... < 1",
        ),
        pytest.param(
            dict(value=1, min_=2, max_=1, incl_min=False, incl_max=True),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="2 < ... <= 1",
        ),
        pytest.param(
            dict(value=1, min_=2, max_=1, incl_min=True, incl_max=True),
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="2 <= ... <= 1",
        ),
    ],
)
def test_min_max_ordering(kwargs):
    check_domain("dummy", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            dict(value=1, min_=1, incl_min=False),
            marks=pytest.mark.xfail(raises=InvalidArgument, strict=True),
            id="lower:1 < value:1",
        ),
        pytest.param(dict(value=1, min_=1, incl_min=True), id="lower:1 <= value:1"),
        pytest.param(
            dict(value=1, max_=1, incl_max=False),
            marks=pytest.mark.xfail(raises=InvalidArgument, strict=True),
            id="value:1 < upper:1",
        ),
        pytest.param(dict(value=1, max_=1, incl_max=True), id="value:1 <= upper:1"),
        pytest.param(
            dict(value=1, min_=1, max_=1, incl_min=True, incl_max=True),
            id="lower:1 <= value:1 <= upper:1",
        ),
    ],
)
def test_bad_inequality(kwargs):
    check_domain("dummy", **kwargs)


@given(
    lower=(st.none() | st.floats(allow_nan=False, min_value=-1e6, max_value=1e6)),
    upper=(st.none() | st.floats(allow_nan=False, min_value=-1e6, max_value=1e6)),
    data=st.data(),
    incl_max=st.booleans(),
    incl_min=st.booleans(),
)
def test_valid_inequalities(
    lower, upper, data: st.DataObject, incl_max: bool, incl_min: bool
):
    if lower is None:
        incl_min = True
    if upper is None:
        incl_max = True

    if lower is None and upper is None:
        assume(False)
        assert False

    if lower is not None and upper is not None:
        lower, upper = (upper, lower) if upper < lower else (lower, upper)

    if incl_min is False or incl_max is False and lower == upper:
        assume(False)
        assert False
    if lower is None and upper is None:
        assume(False)
        assert False

    value = (
        data.draw(
            st.floats(
                min_value=lower,
                max_value=upper,
                exclude_max=not incl_max,
                exclude_min=not incl_min,
            ),
            label="value",
        )
        if lower != upper
        else lower
    )

    check_domain(
        "dummy",
        value=value,
        min_=lower,
        max_=upper,
        incl_max=incl_max,
        incl_min=incl_min,
    )
