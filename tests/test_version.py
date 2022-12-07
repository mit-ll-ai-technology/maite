def test_version():
    import jatic_toolbox

    assert isinstance(jatic_toolbox.__version__, str)
    assert jatic_toolbox.__version__
    assert "unknown" not in jatic_toolbox.__version__
