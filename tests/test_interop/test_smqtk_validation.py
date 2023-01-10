import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.interop.smqtk.configs import create_smqtk_model_config
from jatic_toolbox.interop.smqtk.object_detection import CenterNetVisdrone
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
        CenterNetVisdrone,
        create_smqtk_model_config,
    ],
)
def test_pyright_smqtk(fn):
    results = pyright_analyze(fn)
    assert results["summary"]["errorCount"] > 0


@settings(max_examples=10)
@pytest.mark.parametrize("fn", [CenterNetVisdrone])
@given(arg=everything_except(str))
def test_validate_init_for_model_str(fn, arg):
    with pytest.raises(InvalidArgument):
        fn(arg)


@settings(max_examples=10)
@pytest.mark.parametrize("fn", [CenterNetVisdrone])
@given(model=st.text())
def test_validate_error_for_invalid_model_str(fn, model):
    with pytest.raises(InvalidArgument):
        fn(model)
