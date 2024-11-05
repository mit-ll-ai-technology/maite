# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import re
from collections.abc import Collection
from enum import Enum, Flag
from functools import partial
from typing import Any, NamedTuple, Union

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings

from maite.errors import InvalidArgument
from maite.utils.validation import (
    chain_validators,
    check_domain,
    check_one_of,
    check_type,
)


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
    arg=st.shared(any_types, key="target_type").flatmap(everything_except),
)
def test_check_type_catches_bad_type(name, target_type, arg):
    with pytest.raises(InvalidArgument):
        check_type(name, arg=arg, type_=target_type)


@settings(deadline=None)
@given(
    target_type=st.shared(any_types, key="target_type"),
    arg=st.shared(any_types, key="target_type").flatmap(st.from_type),
)
def test_check_type_passes_good_type(target_type, arg):
    check_type("dummy", arg=arg, type_=target_type)


@given(...)
def test_check_multiple_types(arg: Union[str, int, None]):
    out = check_type("dummy", arg, type_=(str, int), optional=True)
    assert out == arg


def test_no_bounds():
    with pytest.raises(AssertionError):
        check_domain("arg", 1)  # pyright: ignore


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"arg": 1, "lower": 1, "upper": 1, "incl_low": False, "incl_up": False},
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="1 < ... < 1",
        ),
        pytest.param(
            {"arg": 1, "lower": 1, "upper": 1, "incl_low": True, "incl_up": False},
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="1 <= ... < 1",
        ),
        pytest.param(
            {"arg": 1, "lower": 1, "upper": 1, "incl_low": True, "incl_up": True},
            id="1 <= ... <= 1",
        ),
        pytest.param(
            {"arg": 1, "lower": 1, "upper": 1, "incl_low": False, "incl_up": True},
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="1 < ... <= 1",
        ),
        pytest.param(
            {"arg": 1, "lower": 2, "upper": 1, "incl_low": False, "incl_up": False},
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="2 < ... < 1",
        ),
        pytest.param(
            {"arg": 1, "lower": 2, "upper": 1, "incl_low": True, "incl_up": False},
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="2 <= ... < 1",
        ),
        pytest.param(
            {"arg": 1, "lower": 2, "upper": 1, "incl_low": False, "incl_up": True},
            marks=pytest.mark.xfail(raises=AssertionError, strict=True),
            id="2 < ... <= 1",
        ),
        pytest.param(
            {"arg": 1, "lower": 2, "upper": 1, "incl_low": True, "incl_up": True},
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
            {"arg": 1, "lower": 1, "incl_low": False},
            marks=pytest.mark.xfail(raises=InvalidArgument, strict=True),
            id="lower:1 < arg:1",
        ),
        pytest.param({"arg": 1, "lower": 1, "incl_low": True}, id="lower:1 <= arg:1"),
        pytest.param(
            {"arg": 1, "upper": 1, "incl_up": False},
            marks=pytest.mark.xfail(raises=InvalidArgument, strict=True),
            id="arg:1 < upper:1",
        ),
        pytest.param({"arg": 1, "upper": 1, "incl_up": True}, id="arg:1 <= upper:1"),
        pytest.param(
            {"arg": 1, "lower": 1, "upper": 1, "incl_low": True, "incl_up": True},
            id="lower:1 <= arg:1 <= upper:1",
        ),
    ],
)
def test_bad_inequality(kwargs):
    check_domain("dummy", **kwargs)


@given(
    lower=(st.none() | st.floats(allow_nan=False, min_value=-1e6, max_value=1e6)),
    upper=(st.none() | st.floats(allow_nan=False, min_value=-1e6, max_value=1e6)),
    data=st.data(),
    incl_up=st.booleans(),
    incl_low=st.booleans(),
)
def test_valid_inequalities(
    lower, upper, data: st.DataObject, incl_up: bool, incl_low: bool
):
    if lower is None:
        incl_low = True
    if upper is None:
        incl_up = True

    if lower is None and upper is None:
        assume(False)
        assert False

    if lower is not None and upper is not None:
        lower, upper = (upper, lower) if upper < lower else (lower, upper)

    if incl_low is False or incl_up is False and lower == upper:
        assume(False)
        assert False
    if lower is None and upper is None:
        assume(False)
        assert False

    arg = (
        data.draw(
            st.floats(
                min_value=lower,
                max_value=upper,
                exclude_max=not incl_up,
                exclude_min=not incl_low,
            ),
            label="arg",
        )
        if lower != upper
        else lower
    )

    check_domain(
        "dummy",
        arg=arg,
        lower=lower,
        upper=upper,
        incl_up=incl_up,
        incl_low=incl_low,
    )


class CheckOneOfInputs(NamedTuple):
    arg: Any
    collection: Collection = []
    vals: tuple[Any, ...] = ()
    requires_identity: bool = False
    name: str = "foo"


class AClass:
    pass


class BClass:
    pass


FooEnum = Enum("Foo", ["a", "b"])
BarEnum = Enum("Bar", ["c", "d"])
FlagEnum = Flag("Flag", ["e", "f"])

collections = st.lists(st.sampled_from([True, BClass(), BClass]), unique=True)


@given(
    arg=st.sampled_from([None, False, AClass(), AClass, FooEnum, FooEnum.a])
    | st.integers(-5, 0),
    collection=collections | st.just(BarEnum) | st.just(FlagEnum),
    vals=collections,
    requires_identity=st.booleans(),
)
def test_check_one_of_catches_bad_inputs(arg, collection, vals, requires_identity):
    if not collection and not vals:
        assume(False)

    with pytest.raises(InvalidArgument):
        check_one_of("foo", arg, collection, *vals, requires_identity=requires_identity)


@given(...)
def test_check_one_of_supports_enum(arg: Union[FooEnum, FlagEnum]):
    assert check_one_of("foo", arg, type(arg)) is arg


@given(
    arg=st.sampled_from([None, False, AClass(), AClass, FooEnum, FooEnum.a])
    | st.integers(-5, 0),
    collection=collections,
    vals=collections,
    requires_identity=st.booleans(),
)
def test_check_one_of_passes(
    arg, collection: list, vals: list, requires_identity: bool
):
    collection.append(arg)
    assert (
        check_one_of("foo", arg, collection, *vals, requires_identity=requires_identity)
        is arg
    )
    assert (
        check_one_of("foo", arg, vals, *collection, requires_identity=requires_identity)
        is arg
    )


def test_check_one_of_raises_unsatisfiable():
    with pytest.raises(AssertionError):
        check_one_of("foo", 1, [])


is_int = partial(check_type, type_=int)
is_pos = partial(check_domain, lower=0)
check_pos_int = chain_validators(is_int, is_pos)


@given(st.integers(0, 10))
def test_chain_validators_pass_through(x: int):
    assert x == check_pos_int("arg", x)


@pytest.mark.parametrize(
    "expr, msg",
    [
        (
            lambda: check_domain("arg", 1, lower=2, incl_low=False),
            r"`arg` must satisfy `2 < arg`.  Got: `1`.",
        ),
        (
            lambda: check_domain("arg", 1, lower=2, incl_low=True),
            r"`arg` must satisfy `2 <= arg`.  Got: `1`.",
        ),
        (
            lambda: check_domain("arg", 1, lower=2, incl_low=False, lower_name="low"),
            r"`arg` must satisfy `low=2 < arg`.  Got: `1`.",
        ),
        (
            lambda: check_domain("arg", 1, lower=2, incl_low=True, lower_name="low"),
            r"`arg` must satisfy `low=2 <= arg`.  Got: `1`.",
        ),
        (
            lambda: check_domain("arg", 1, upper=0, incl_up=False),
            r"`arg` must satisfy `arg < 0`.  Got: `1`.",
        ),
        (
            lambda: check_domain("arg", 1, upper=0, incl_up=True),
            r"`arg` must satisfy `arg <= 0`.  Got: `1`.",
        ),
        (
            lambda: check_domain("arg", 1, upper=0, incl_up=False, upper_name="hi"),
            r"`arg` must satisfy `arg < hi=0`.  Got: `1`.",
        ),
        (
            lambda: check_domain("arg", 1, upper=0, incl_up=True, upper_name="hi"),
            r"`arg` must satisfy `arg <= hi=0`.  Got: `1`.",
        ),
        (
            lambda: check_type("arg", 1, str),
            r"Expected `arg` to be of type `str`. Got `1` (type: `int`).",
        ),
        (
            lambda: check_type("arg", 1, (str, bool)),
            r"Expected `arg` to be of types: `str`, `bool`. Got `1` (type: `int`).",
        ),
        (
            lambda: check_type("arg", 1, (str, bool), optional=True),
            r"Expected `arg` to be `None` or of types: `str`, `bool`. Got `1` (type: `int`).",
        ),
        (
            lambda: check_one_of("foo", 1, [], requires_identity=True),
            r"`collections` and `args` are both empty.",
        ),
        (
            lambda: check_one_of("foo", 1, [True], requires_identity=True),
            r"Expected `foo` to be: True. Got `1`.",
        ),
        (
            lambda: check_one_of("foo", 1, [], True, requires_identity=True),
            r"Expected `foo` to be: True. Got `1`.",
        ),
        (
            lambda: check_one_of("bar", 1, [True, False], requires_identity=True),
            r"Expected `bar` to be one of: False, True. Got `1`.",
        ),
        (
            lambda: check_one_of("bar", 1, [True, False], 2, requires_identity=True),
            r"Expected `bar` to be one of: 2, False, True. Got `1`.",
        ),
        (
            lambda: check_one_of("bar", 1, [], True, False, 2, requires_identity=True),
            r"Expected `bar` to be one of: 2, False, True. Got `1`.",
        ),
        (
            # error message should remove redundant names
            lambda: check_one_of(
                "bar", 1, [True], True, False, 2, requires_identity=True
            ),
            r"Expected `bar` to be one of: 2, False, True. Got `1`.",
        ),
        (
            lambda: check_one_of("bar", [1], Enum("Foo", ["a", "b"])),
            r"Expected `bar` to be one of: Foo.a, Foo.b. Got `[1]`.",
        ),
        (
            lambda: check_pos_int("foo", ["a"]),
            r"Expected `foo` to be of type `int`. Got `['a']` (type: `list`).",
        ),
        (
            lambda: check_pos_int("foo", -1),
            r"`foo` must satisfy `0 <= foo`.  Got: `-1`.",
        ),
    ],
)
def test_error_msg(expr, msg: str):
    with pytest.raises(Exception, match=re.escape(msg)):
        expr()
