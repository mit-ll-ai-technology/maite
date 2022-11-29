import inspect
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict


class _Summary(TypedDict):
    filesAnalyzed: int
    errorCount: int
    warningCount: int
    informationCount: int
    timeInSec: float


class _LineInfo(TypedDict):
    line: int
    character: int


class _Range(TypedDict):
    start: _LineInfo
    end: _LineInfo


class _Diagnostic(TypedDict):
    file: str
    severity: Literal["error", "warning", "information"]
    message: str
    range: _Range
    rule: NotRequired[str]


class PyrightOutput(TypedDict):
    """The schema for the JSON output of a pyright scan"""

    version: str
    time: str
    generalDiagnostics: List[_Diagnostic]
    summary: _Summary


_found_path = shutil.which("pyright")
PYRIGHT_PATH = Path(_found_path) if _found_path else None
del _found_path


@contextmanager
def chdir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        try:
            os.chdir(tmpdirname)  # change cwd to the temp-directory
            yield Path(tmpdirname)  # yields control to the test to be run
        finally:
            os.chdir(old_dir)


def pyright_analyze(
    code_or_path,
    pyright_config: Optional[Dict[str, Any]] = None,
    path_to_pyright: Union[Path, None] = PYRIGHT_PATH,
    *,
    python_version: Optional[str] = None,
    report_unnecessary_type_ignore_comment: Optional[bool] = None,
    type_checking_mode: Optional[Literal["basic", "strict"]] = None,
) -> PyrightOutput:
    """
    Scans a Python object (e.g., a function) or file(s) using pyright and returns a JSON
    summary of the scan.

    Parameters
    ----------
    func : SourceObjectType | str | Path
        A function, module-object, class, or method to scan. Or, a path to a file or
        directory to scan.

    pyright_config : None | dict[str, Any]
        A JSON configuration for pyright's settings [1]_.

    path_to_pyright: Path
        Path to the pyright executable. Defaults to `shutil.where('pyright')` if the
        executable can be found.

    python_version: Optional[str]
        The version of Python used for this execution environment as a string in the
        format "M.m". E.g., "3.9" or "3.7"

    report_unnecessary_type_ignore_comment: Optional[bool]
        If `True` specifying `# type: ignore` for an expression that would otherwise
        not result in an error

    type_checking_mode: Optional[Literal["basic", "strict"]] = None,

    Returns
    -------
    PyrightOutput : TypedDict
        The JSON-decoded results of the scan [2]_.
            - version: str
            - time: str
            - generalDiagnostics: List[DiagnosticDict] (one entry per error/warning)
            - summary: SummaryDict

    References
    ----------
    .. [1] https://github.com/microsoft/pyright/blob/main/docs/configuration.md
    .. [2] https://github.com/microsoft/pyright/blob/main/docs/command-line.md#json-output

    Examples
    --------
    Here pyright will record an error when scan a function that attempts to add a
    string-annotated variable to an integer.

    >>> def f(x: str):
    ...     return 1 + x
    >>> pyright_analyze(f)
    {'version': '1.1.281',
     'time': '1669686515154',
     'generalDiagnostics': [{'file': 'C:\\Users\\RY26099\\AppData\\Local\\Temp\\12\\tmpcxc7erfq\\source.py',
       'severity': 'error',
       'message': 'Operator "+" not supported for types "Literal[1]" and "str"\n\xa0\xa0Operator "+" not supported for types "Literal[1]" and "str"',
       'range': {'start': {'line': 1, 'character': 11},
        'end': {'line': 1, 'character': 16}},
       'rule': 'reportGeneralTypeIssues'}],
     'summary': {'filesAnalyzed': 20,
      'errorCount': 1,
      'warningCount': 0,
      'informationCount': 0,
      'timeInSec': 0.319}}

    Whereas this function scans "clean".

    >>> def g(x: int) -> int:
    ...     return 1 + x
    >>> pyright_analyze(g)
    {'version': '1.1.281',
     'time': '1669686578833',
     'generalDiagnostics': [],
     'summary': {'filesAnalyzed': 20,
      'errorCount': 0,
      'warningCount': 0,
      'informationCount': 0,
      'timeInSec': 0.29}}

    All imports must occur within the context of the scanned-object. For example,
    consider the following

    >>> import math
    >>> def f():
    ...     math.acos(1)
    >>> pyright_analyze(f)["summary"]["errorCount"]
    1

    The import statement needs to be moved to the body of `f` in order to eliminate this
    error.

    >>> def g():
    ...     import math
    ...     math.acos(1)
    >>> pyright_analyze(f)["summary"]["errorCount"]
    0
    """
    if path_to_pyright is None:
        raise ModuleNotFoundError(
            "`pyright` was not found. It may need to be installed."
        )
    if not path_to_pyright.is_file():
        raise FileNotFoundError(
            f"`path_to_pyright – {path_to_pyright} – doesn't exist."
        )
    if not pyright_config:
        pyright_config = {}

    if python_version is not None:
        pyright_config["pythonVersion"] = python_version

    if report_unnecessary_type_ignore_comment is not None:
        pyright_config[
            "reportUnnecessaryTypeIgnoreComment"
        ] = report_unnecessary_type_ignore_comment

    if type_checking_mode is not None:
        pyright_config["typeCheckingMode"] = type_checking_mode

    source = (
        textwrap.dedent((inspect.getsource(code_or_path)))
        if not isinstance(code_or_path, (str, Path))
        else None
    )

    with chdir():
        cwd = Path.cwd()
        if source is not None:
            file_ = cwd / "source.py"
            file_.write_text(source)
        else:
            file_ = Path(code_or_path).absolute()
            assert (
                file_.exists()
            ), f"Specified path {file_} does not exist. Cannot be scanned by pyright."

        if pyright_config:
            (cwd / "pyrightconfig.json").write_text(json.dumps(pyright_config))

        proc = subprocess.run(
            [str(path_to_pyright.absolute()), str(file_.absolute()), "--outputjson"],
            cwd=file_.parent,
            encoding="utf-8",
            text=True,
            capture_output=True,
        )
        try:
            return json.loads(proc.stdout)
        except Exception:
            print(proc.stdout)
            raise
