# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# flake8: noqa
import json
import re
from pathlib import Path

import pytest

from maite.testing.pyright import list_error_messages, pyright_analyze


def test_pyright_catches_errors():
    def f(x: str):
        x + 1
        x.append(2)

    results = pyright_analyze(f)
    assert results[0]["summary"]["errorCount"] == 2


def test_pyright_scans_clean():
    def f():
        def g(x: int):
            ...

        g(1)
        g(2)

    x = pyright_analyze(f)
    assert x[0]["summary"]["errorCount"] == 0


def test_pyright_basic():
    def f(x):
        ...

    results = pyright_analyze(f, type_checking_mode="basic")
    assert results[0]["summary"]["errorCount"] == 0


def test_pyright_strict():
    def f(x):  # type of x is unknown, missing return annotation
        ...

    results = pyright_analyze(f, type_checking_mode="strict")
    assert results[0]["summary"]["errorCount"] > 0


@pytest.mark.usefixtures("cleandir")
def test_python_version():
    code = """def f(x: list[int]):  # using builtin type as generic is new to 3.9
        ...
    """
    with open("f.py", "w") as file:
        file.writelines(code)

    py39 = pyright_analyze("./", python_version="3.9")
    assert py39[0]["summary"]["errorCount"] == 0
    py310 = pyright_analyze("./", python_version="3.10")
    assert py310[0]["summary"]["errorCount"] == 0
    py311 = pyright_analyze("./", python_version="3.11")
    assert py311[0]["summary"]["errorCount"] == 0


def test_scan_path_to_code():
    import maite

    results = pyright_analyze(
        Path(maite.__file__).parent, report_unnecessary_type_ignore_comment=True
    )
    assert len(results[0]["generalDiagnostics"]) >= 0


def test_preamble():
    import math

    def g():
        math.acos(1)

    results = pyright_analyze(g, preamble="import math")
    assert results[0]["summary"]["errorCount"] == 0


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
    assert results[0]["summary"]["errorCount"] == 1
    (message,) = (
        d["message"]
        for d in results[0]["generalDiagnostics"]
        if d["severity"] == "error"
    )
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
    assert (
        results[0]["summary"]["errorCount"] == expected_num_error
    ), list_error_messages(results[0])


md_good_1 = """
    blah blah

    ```python
       from pathlib import Path
   
       def print_file(x: Path) -> None:
           with x.open("r") as f: 
               print(f.read())
    ```
    ya ya
    ````
    ```python
    just an example in a literal block
    ```
    ````
    ``python
    not a block
    ``
"""

md_good_2 = """
    ```pycon

    >>> from pathlib import Path
    >>>
    >>> def print_file(x: Path) -> None:
    ...     with x.open("r") as f: 
    ...         print(f.read())
    ```
    ``pycon
    not a block
    ``
"""

md_bad_1 = """
    ```python
       from pathlib import Path
   
       def print_file(x: int) -> None:
           with x.open("r") as f: 
               print(f.read())
    ```
"""

md_bad_2 = """

    ```pycon

    >>> from pathlib import Path
    >>>
    >>> def print_file(x: int) -> None:
    ...     with x.open("r") as f: 
    ...         print(f.read())
    ```
"""


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "src, expected_num_error",
    [
        (md_good_1, 0),
        (md_good_2, 0),
        (md_bad_1, 1),
        (md_bad_2, 1),
    ],
)
def test_scan_md(src: str, expected_num_error: int):
    Path("file.md").write_text(src)  # file will be written to a tmp dir
    results = pyright_analyze("file.md")
    assert (
        results[0]["summary"]["errorCount"] == expected_num_error
    ), list_error_messages(results[0])


@pytest.mark.filterwarnings("ignore:the imp module is deprecate")
@pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
@pytest.mark.parametrize("src, expected_num_error", [("1 + 'a'", 1), ("1 + 2", 0)])
@pytest.mark.usefixtures("cleandir")
def test_scan_ipynb(src, expected_num_error):
    import jupytext

    jupytext.write(jupytext.reads(src, fmt=".py"), "file.ipynb", fmt=".ipynb")

    results = pyright_analyze("file.ipynb")
    assert (
        results[0]["summary"]["errorCount"] == expected_num_error
    ), list_error_messages(results[0])


@pytest.mark.usefixtures("cleandir")
def test_scan_doesnt_clobber_preexisting_pyright_config():
    d = Path.cwd()
    d.mkdir(exist_ok=True)

    config = d / "pyrightconfig.json"
    file_ = d / "file.py"
    pyright_target = file_

    config.write_text(json.dumps({"reportUnnecessaryTypeIgnoreComment": False}))
    expected_config = config.read_text("utf-8")
    file_.write_text("x = 1 + 1  # type: ignore")

    results = pyright_analyze(
        pyright_target,
        pyright_config={"reportUnnecessaryTypeIgnoreComment": True},
    )
    post_run_config = config.read_text("utf-8")

    assert (
        results[0]["summary"]["errorCount"] == 1
        and 'Unnecessary "# type: ignore" comment'
        in results[0]["generalDiagnostics"][0]["message"]
    )

    assert expected_config == post_run_config


@pytest.mark.parametrize("suffix", [".py", ".ipynb", ".rst", "/"])
def test_analyze_missing_file(suffix):
    with pytest.raises(FileNotFoundError, match=r"Cannot be scanned by pyright."):
        pyright_analyze(f"moo{suffix}")


def test_bad_path_to_pyright():
    def f():
        ...

    bad_path = Path("not/a/path/pyright")
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"{str(bad_path)} – doesn't exist.")
    ):
        pyright_analyze(f, path_to_pyright=Path(bad_path))


@pytest.mark.usefixtures("cleandir")
def test_unsupported_file_type():
    f = Path.cwd() / "file.txt"
    f.touch()
    with pytest.raises(ValueError, match=re.escape("File type .txt not supported")):
        pyright_analyze(f)


def test_list_error_messages():
    def f(x: int):
        return x.lower()

    results = pyright_analyze(f)[0]
    listed_errors = list_error_messages(results)
    assert len(listed_errors) == 1
    assert listed_errors[0].startswith(
        '(line start) 1: Cannot access member "lower" for type "int"'
    ) or listed_errors[0].startswith(
        '(line start) 1: Cannot access attribute "lower" for class "int"'
    )
