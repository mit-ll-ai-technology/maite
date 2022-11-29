from pathlib import Path

from jatic_toolbox._internals.testing.pyright import pyright_analyze


def test_pyright_catches_errors():
    def f(x: str):
        x + 1
        x.append(2)

    results = pyright_analyze(f)
    assert results["summary"]["errorCount"] == 2


def test_pyright_scans_clean():
    def f():
        def g(x: int):
            ...

        g(1)
        g(2)

    x = pyright_analyze(f)
    assert x["summary"]["errorCount"] == 0


def test_pyright_basic():
    def f(x):
        ...

    results = pyright_analyze(f, type_checking_mode="basic")
    assert results["summary"]["errorCount"] == 0


def test_pyright_strict():
    def f(x):  # type of x is unknown, missing return annotation
        ...

    results = pyright_analyze(f, type_checking_mode="strict")
    assert results["summary"]["errorCount"] > 0


def test_python_version():
    def f(x: list[int]):  # using builtin type as generic is new to 3.9
        ...

    py38 = pyright_analyze(f, python_version="3.8")
    py39 = pyright_analyze(f, python_version="3.9")
    assert py38["summary"]["errorCount"] == 1
    assert py39["summary"]["errorCount"] == 0


def test_scan_path_to_code():
    results = pyright_analyze(Path.cwd())
    assert results["summary"]["filesAnalyzed"] > 2
