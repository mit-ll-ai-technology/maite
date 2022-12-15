import json
import subprocess
from copy import deepcopy
from functools import _CacheInfo as CacheInfo, lru_cache
from pathlib import Path
from typing import List, Mapping, Union

from typing_extensions import Literal, NotRequired, TypedDict

from jatic_toolbox._internals.testing.pyright import (
    PYRIGHT_PATH,
    Diagnostic,
    PyrightOutput,
)
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.utils.validation import check_type


class Symbol(TypedDict):
    category: Literal[
        "class",
        "constant",
        "function",
        "method",
        "module",
        "symbol",
        "type alias",
        "variable",
    ]
    name: str
    referenceCount: int
    isExported: bool
    isTypeKnown: bool
    isTypeAmbiguous: bool
    diagnostics: List[Diagnostic]


class CompletenessSection(TypedDict):
    packageName: str
    packageRootDirectory: Path
    moduleName: str
    moduleRootDirectory: Path
    ignoreUnknownTypesFromImports: bool
    pyTypedPath: NotRequired[Path]
    exportedSymbolCounts: dict
    otherSymbolCounts: dict
    missingFunctionDocStringCount: int
    missingClassDocStringCount: int
    missingDefaultParamCount: int
    completenessScore: float
    modules: list
    symbols: List[Symbol]


class ModuleScanResults(PyrightOutput):
    """The schema for the JSON output of a type completeness scan.

    The output type of `jatic_toolbox.testing.project.ModuleScan.__call__"""

    typeCompleteness: CompletenessSection


def _pyright_type_completeness(
    module_name: str,
    *,
    path_to_pyright: Union[Path, None],
) -> ModuleScanResults:

    module_name = module_name.replace("-", "_")

    if path_to_pyright is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "`pyright` was not found. It may need to be installed."
        )
    if not path_to_pyright.is_file():
        raise FileNotFoundError(
            f"`path_to_pyright` – {path_to_pyright} – doesn't exist."
        )

    proc = subprocess.run(
        [
            str(PYRIGHT_PATH),
            "--ignoreexternal",
            "--outputjson",
            "--verifytypes",
            module_name,
        ],
        cwd=Path.cwd(),
        encoding="utf-8",
        text=True,
        capture_output=True,
    )
    try:
        out = json.loads(proc.stdout)
    except Exception as e:  # pragma: no cover
        print(proc.stdout)
        raise e

    for k in ["packageRootDirectory", "moduleRootDirectory", "pyTypedPath"]:
        if k in out:
            out[k] = Path(out[k])

    return out


class ModuleScan:
    """
    Uses pyright's type completeness scan to summarize a module's contents.

    By default, `ModuleScan`'s __call__ is cached to reduce overhead for getting
    scan results for a module multiple times. Each `ModuleScan` instance has an
    independent cache.

    Examples
    --------
    >>> from jatic_toolbox.testing.project import ModuleScan
    >>> scanner = ModuleScan()
    >>> results = scanner("jatic_toolbox")
    >>> results["summary"]
    {'filesAnalyzed': 9,
    'errorCount': 0,
    'warningCount': 0,
    'informationCount': 0,
    'timeInSec': 0.662}
    >>> results["typeCompleteness"]["packageName"]
    'jatic_toolbox'
    >>> results["typeCompleteness"]["symbols"]
    [{'category': 'class',
      'name': 'jatic_toolbox.errors.ToolBoxException',
      'referenceCount': 3,
      'isExported': True,
      'isTypeKnown': True,
      'isTypeAmbiguous': False,
      'diagnostics': []},
     {'category': 'class',
      'name': 'jatic_toolbox.errors.InternalError',
      'referenceCount': 1,
      'isExported': True,
      'isTypeKnown': True,
      'isTypeAmbiguous': False,
      'diagnostics': []},
    ...
    """

    def __init__(self) -> None:
        self._cached_scan = lru_cache(maxsize=256, typed=False)(
            _pyright_type_completeness
        )

    def __call__(
        self,
        module_name: str,
        *,
        path_to_pyright: Union[Path, None] = PYRIGHT_PATH,
        cached: bool = True,
    ) -> ModuleScanResults:
        # doc-ignore: EX01
        """
        Perform a scan on the specified module or submodule.

        Parameters
        ----------
        module_name : str
            The name of a module to scan. This must the name of a module that is
            installed in the current Python environment.

        path_to_pyright : Path, optional, keyword-only
            Path to the pyright executable (see installation instructions: [4]_).
            Defaults to `shutil.where('pyright')` if the executable can be found.

        cached : bool, optional (default=True)
            If `True`, then the result of the scan will be mediated by an LRU cache.
            I.e. subsequent identical calls will return a cached result.

        Returns
        -------
        ModuleScanResults
            A dictionary containing::

                version: str
                time: str
                generalDiagnostics: List[Diagnostic]  (empty)
                summary: Summary
                    filesAnalyzed: int
                    errorCount: int
                    warningCount: int
                    informationCount: int
                    timeInSec: float
                typeCompleteness: CompletenessSection
                    packageName: str
                    packageRootDirectory: Path
                    moduleName: str
                    moduleRootDirectory: Path
                    ignoreUnknownTypesFromImports: bool
                    pyTypedPath: NotRequired[Path]
                    exportedSymbolCounts: dict
                    otherSymbolCounts: dict
                    missingFunctionDocStringCount: int
                    missingClassDocStringCount: int
                    missingDefaultParamCount: int
                    completenessScore: float
                    modules: list
                    symbols: List[Symbol]

        Raises
        ------
        FileNotFoundError
            `module` was not found.
        """
        # numpydoc: EX01
        scan = self._cached_scan if cached else _pyright_type_completeness
        out = scan(module_name, path_to_pyright=path_to_pyright)
        _summary = out["summary"]
        if _summary["errorCount"] > 0 and _summary["filesAnalyzed"] == 0:
            raise FileNotFoundError(
                f"No files were found to analyze in associatation "
                f"with module `{module_name}`. The module may not be installed or "
                f"there may be a typo in the name."
            )

        return deepcopy(out) if cached else out

    def clear_cache(self) -> None:
        # doc-ignore: GL08
        self._cached_scan.cache_clear()

    def cache_info(self) -> CacheInfo:
        # doc-ignore: GL08
        return self._cached_scan.cache_info()


def get_public_symbols(scan: ModuleScanResults, submodule: str = "") -> List[Symbol]:
    """
    Return all public symbols (functions, classes, etc.) from a module's API.

    This function expects the results of a scan performed by
    `jatic_toolbox.testing.project.ModuleScan`, which requires that `pyright` is
    installed.

    Parameters
    ----------
    scan : ModuleScanResults
        The result of a scane performed by `ModuleScan.__call__`.

    submodule : str, optional
        If specified, only symbols from the specified submodule are included.

    Returns
    -------
    List[Symbol]
        Each symbol is a dict containing the following key-value pairs::

            category: Literal["class", "constant", "function", "method",
                                "module", "symbol", "type alias", "variable"]
            name: str
            referenceCount: int
            isExported: bool
            isTypeKnown: bool
            isTypeAmbiguous: bool
            diagnostics: List[Diagnostic]

    Examples
    --------
    Basic usage.

    >>> from jatic_toolbox.testing.project import get_public_symbols, ModuleScan
    >>> scanner = ModuleScan()
    >>> results = scanner("jatic_toolbox")
    >>> get_public_symbols(results)
    [ {'category': 'class',
    'name': 'jatic_toolbox.testing.project.ModuleScan',
    'referenceCount': 1,
    'isExported': True,
    'isTypeKnown': False,
    'isTypeAmbiguous': False,
    'diagnostics': []},
    {'category': 'class',
    'name': 'jatic_toolbox.testing.project.ModuleScanResults',
    'referenceCount': 1,
    'isExported': True,
    'isTypeKnown': False,
    'isTypeAmbiguous': False,
    'diagnostics': []},
    {'category': 'class',
    'name': 'jatic_toolbox.testing.project.Symbol',
    'referenceCount': 1,
    'isExported': True,
    'isTypeKnown': True,
    'isTypeAmbiguous': False,
    'diagnostics': []},
    ...
    ]

    Accessing symbols from the `docs` submodule.

    >>> get_public_symbols(results, submodule="jatic_toolbox.testing.docs")
    [{'category': 'type alias',
    'name': 'jatic_toolbox.testing.docs.NumpyDocErrorCode',
    'referenceCount': 1,
    'isExported': True,
    'isTypeKnown': True,
    'isTypeAmbiguous': False,
    'diagnostics': []},
    {'category': 'class',
    'name': 'jatic_toolbox.testing.docs.NumPyDocResults',
    'referenceCount': 1,
    'isExported': True,
    'isTypeKnown': True,
    'isTypeAmbiguous': False,
    'diagnostics': []},
    ...
    ]
    """
    check_type("scan", scan, Mapping)
    check_type("submodule", submodule, str)

    out = (x for x in scan["typeCompleteness"]["symbols"] if x["isExported"])
    if submodule:
        if any(not x.isidentifier() for x in submodule.split(".")):
            raise InvalidArgument(f"{submodule} is not a valid module name.")

        _out = [x for x in out if x["name"].startswith(submodule)]

        if not _out:
            raise ValueError(f"No symbols within submodule {submodule}.")
        return _out
    return list(out)
