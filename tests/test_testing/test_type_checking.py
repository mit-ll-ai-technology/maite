import re
from pathlib import Path

import pytest

from jatic_toolbox.testing.pyright import pyright_analyze


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
    results = pyright_analyze(
        Path.cwd(),
        report_unnecessary_type_ignore_comment=True,
        overwrite_config_ok=True,
    )
    assert results["summary"]["filesAnalyzed"] > 2


def test_scan_path_to_code2():
    results = pyright_analyze(
        Path.cwd(),
        report_unnecessary_type_ignore_comment=False,
        overwrite_config_ok=True,
    )
    assert results["summary"]["filesAnalyzed"] > 2


def test_preamble():
    import math

    def g():
        math.acos(1)

    results = pyright_analyze(g, preamble="import math")
    assert results["summary"]["errorCount"] == 0


def test_scan_docstring_raises_on_path():
    with pytest.raises(
        ValueError, match=re.escape(r"`scan_docstring=True` can only be specified")
    ):
        pyright_analyze(Path.cwd(), scan_docstring=True)


def test_scan_docstring():
    def f():
        """
        Examples
        --------
        >>> x = 1
        >>> y = x + 'a'  # pyright should catch this
        """
        return

    results = pyright_analyze(f, scan_docstring=True)
    assert results["summary"]["errorCount"] == 1
    (message,) = [
        d["message"] for d in results["generalDiagnostics"] if d["severity"] == "error"
    ]
    assert message.startswith('Operator "+" not supported for types')
