# flake8: noqa

import re
from pathlib import Path

import pytest

from jatic_toolbox.testing.pyright import list_error_messages, pyright_analyze


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


rst_good_1 = """
    .. code-block:: python

       from pathlib import Path
   
       def print_file(x: Path) -> None:
           with x.open("r") as f: 
               print(f.read())
"""

rst_good_2 = """
    .. code-block:: pycon

       >>> from pathlib import Path
       >>>
       >>> def print_file(x: Path) -> None:
       ...     with x.open("r") as f: 
       ...         print(f.read())
"""

rst_bad_1 = """
    .. code-block:: python

       from pathlib import Path
   
       def print_file(x: int) -> None:
           with x.open("r") as f: 
               print(f.read())
"""

rst_bad_2 = """
    .. code-block:: pycon

       >>> from pathlib import Path
       >>>
       >>> def print_file(x: int) -> None:
       ...     with x.open("r") as f: 
       ...         print(f.read())
"""


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "src, expected_num_error",
    [
        (rst_good_1, 0),
        (rst_good_2, 0),
        (rst_bad_1, 1),
        (rst_bad_2, 1),
    ],
)
def test_scan_rst(src: str, expected_num_error: int):
    Path("file.rst").write_text(src)  # file will be written to a tmp dir
    results = pyright_analyze("file.rst")
    assert results["summary"]["errorCount"] == expected_num_error, list_error_messages(
        results
    )


@pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
@pytest.mark.parametrize("src, expected_num_error", [("1 + 'a'", 1), ("1 + 2", 0)])
@pytest.mark.usefixtures("cleandir")
def test_scan_ipynb(src, expected_num_error):
    import jupytext

    jupytext.write(jupytext.reads(src, fmt=".py"), "file.ipynb", fmt=".ipynb")

    results = pyright_analyze("file.ipynb")
    assert results["summary"]["errorCount"] == expected_num_error, list_error_messages(
        results
    )
