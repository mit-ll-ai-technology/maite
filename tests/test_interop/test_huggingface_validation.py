import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.interop.huggingface import (
    HuggingFaceImageClassifier,
    HuggingFaceObjectDetector,
)


def everything_except(excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


@settings(max_examples=10, deadline=None)
@pytest.mark.parametrize("fn", [HuggingFaceObjectDetector, HuggingFaceImageClassifier])
@given(arg=everything_except(str))
def test_validate_init_for_model_str(fn, arg):
    with pytest.raises(InvalidArgument):
        fn(arg)


@settings(max_examples=10, deadline=None)
@pytest.mark.parametrize("fn", [HuggingFaceObjectDetector, HuggingFaceImageClassifier])
@given(model=st.text())
def test_validate_error_for_invalid_model_str(fn, model):
    with pytest.raises(InvalidArgument):
        fn(model)
