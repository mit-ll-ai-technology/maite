from pytest import Config


def test_version():
    import jatic_toolbox

    assert isinstance(jatic_toolbox.__version__, str)
    assert jatic_toolbox.__version__
    assert "unknown" not in jatic_toolbox.__version__


def test_xfail_strict(pytestconfig: Config):
    # Our test suite's xfail must be configured to strict mode
    # in order to ensure that contrapositive tests will actually
    # raise.
    assert pytestconfig.getini("xfail_strict") is True
