import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.interop.huggingface.configs import (
    create_huggingface_dataset_config,
    create_huggingface_model_config,
)
from jatic_toolbox.interop.huggingface.object_detection import HuggingFaceObjectDetector
from jatic_toolbox.testing.pyright import pyright_analyze


def everything_except(excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


@pytest.mark.parametrize(
    "fn",
    [
        HuggingFaceObjectDetector,
        create_huggingface_dataset_config,
        create_huggingface_model_config,
    ],
)
def test_pyright(fn):
    results = pyright_analyze(fn)
    assert results["summary"]["errorCount"] > 0


@settings(max_examples=10)
@pytest.mark.parametrize("fn", [HuggingFaceObjectDetector])
@given(arg=everything_except(str))
def test_validate_init_for_model_str(fn, arg):
    with pytest.raises(InvalidArgument):
        fn(arg)


@settings(max_examples=10)
@pytest.mark.parametrize("fn", [HuggingFaceObjectDetector])
@given(model=st.text())
def test_validate_error_for_invalid_model_str(fn, model):
    with pytest.raises(InvalidArgument):
        fn(model)
