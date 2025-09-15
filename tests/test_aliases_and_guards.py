from typing_extensions import TypedDict

import maite._internals.protocols.task_aliases  # noqa: F401
import maite._internals.protocols.type_guards as tg


def test_is_list_of_type():
    assert not tg.is_list_dict(23)
    assert tg.is_list_dict([{}])
    assert tg.is_list_dict([{}])


class Foo(TypedDict):
    a: int


def test_is_typed_dict():
    assert tg.is_typed_dict(Foo(a=23), Foo)
    assert not tg.is_typed_dict(23, Foo)
