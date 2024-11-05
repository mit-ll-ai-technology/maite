# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import json
import subprocess
from collections.abc import Collection, Generator, Mapping
from copy import deepcopy
from functools import _CacheInfo as CacheInfo, lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

from typing_extensions import NotRequired

from maite._internals.testing.pyright import PYRIGHT_PATH, Diagnostic, PyrightOutput
from maite.errors import InvalidArgument
from maite.utils.validation import check_one_of, check_type

from ..utils import is_typed_dict

Category = Literal[
    "class",
    "constant",
    "function",
    "method",
    "module",
    "symbol",
    "type alias",
    "variable",
]

CATEGORIES: frozenset[Category] = frozenset(Category.__args__)


class Symbol(TypedDict):
    category: Category
    name: str
    referenceCount: int
    isExported: bool
    isTypeKnown: bool
    isTypeAmbiguous: bool
    diagnostics: list[Diagnostic]


class SymbolCounts(TypedDict):
    withKnownType: int
    withAmbiguousType: int
    withUnknownType: int


class CompletenessSection(TypedDict):
    packageName: str
    packageRootDirectory: Path
    moduleName: str
    moduleRootDirectory: Path
    ignoreUnknownTypesFromImports: bool
    pyTypedPath: NotRequired[Path]
    exportedSymbolCounts: SymbolCounts
    otherSymbolCounts: SymbolCounts
    missingFunctionDocStringCount: int
    missingClassDocStringCount: int
    missingDefaultParamCount: int
    completenessScore: float
    modules: list[Any]
    symbols: list[Symbol]


class ModuleScanResults(PyrightOutput):
    """The schema for the JSON output of a type completeness scan.

    The output type of `maite.testing.project.ModuleScan.__call__"""

    # doc-ignore: NOQA
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

    scan_section = out["typeCompleteness"]
    for k in ["packageRootDirectory", "moduleRootDirectory", "pyTypedPath"]:
        if k in scan_section:
            scan_section[k] = Path(scan_section[k])

    return out


class ModuleScan:
    """
    Uses pyright's type completeness scan to summarize a module's contents.

    By default, `ModuleScan`'s __call__ is cached to reduce overhead for getting
    scan results for a module multiple times. Each `ModuleScan` instance has an
    independent cache.

    Methods
    -------
    __call__(module_name, *, path_to_pyright=PYRIGHT_PATH, cached=True)
        Perform a scan on the specified module or submodule.

    cache_info()
        Return a CacheInfo object with statistics about the cache of this instance.

    Examples
    --------
    >>> from maite.testing.project import ModuleScan
    >>> scanner = ModuleScan()
    >>> results = scanner("maite")
    >>> results["summary"]
    {'filesAnalyzed': ..., 'errorCount': ..., 'warningCount': ..., 'informationCount': ..., 'timeInSec': ...}
    >>> results["typeCompleteness"]["packageName"]
    'maite'
    >>> results["typeCompleteness"]["symbols"]     # will change as MAITE changes --> # doctest: +SKIP
    [{'category': 'class',
      'name': 'maite.errors.MaiteException',
      'referenceCount': 3,
      'isExported': True,
      'isTypeKnown': True,
      'isTypeAmbiguous': False,
      'diagnostics': []},
     {'category': 'class',
      'name': 'maite.errors.InternalError',
      'referenceCount': 1,
      'isExported': True,
      'isTypeKnown': True,
      'isTypeAmbiguous': False,
      'diagnostics': []},
     ...]
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
            Path to the pyright executable (see installation instructions).
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
                generalDiagnostics: list[Diagnostic]  (empty)
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
                    symbols: list[Symbol]

        Raises
        ------
        ModuleNotFoundError
            `module` was not found.
        """
        scan = self._cached_scan if cached else _pyright_type_completeness
        out = scan(module_name, path_to_pyright=path_to_pyright)
        _summary = out["summary"]
        # TODO: If a project does not have a py.typed file then no files will be analyzed.
        #       This does *not* mean that the module was not found. Instead, we should
        #       raise an error that explicitly states that the project was found be requires
        #       a py.typed file in order to be analyzed
        if _summary["errorCount"] > 0 and _summary["filesAnalyzed"] == 0:
            raise ModuleNotFoundError(
                f"No files were found to analyze in association "
                f"with module `{module_name}`. The module may not be installed or "
                f"there may be a typo in the name."
            )

        return deepcopy(out) if cached else out

    def clear_cache(self) -> None:
        # doc-ignore: GL08
        self._cached_scan.cache_clear()

    def cache_info(self) -> CacheInfo:
        """
        Access the scanner instance's cache stats.

        Returns
        -------
        CacheInfo
            A named tuple with fields:
            - hits
            - misses
            - maxsize
            - currsize
        """
        # doc-ignore: NOQA
        return self._cached_scan.cache_info()


def _is_dunder(name: str) -> bool:
    *_, name = name.split(".")
    out = (
        name.startswith("__")
        and name.endswith("__")
        and not name.startswith("___")
        and not name.endswith("___")
    )
    return out


def get_public_symbols(
    scan: ModuleScanResults, submodule: str = "", include_dunder_names: bool = False
) -> list[Symbol]:
    """
    Return all public symbols (functions, classes, etc.) from a module's API.

    This function expects the results of a scan performed by
    `maite.testing.project.ModuleScan`, which requires that `pyright` is
    installed.

    Parameters
    ----------
    scan : ModuleScanResults
        The result of a scan performed by `ModuleScan.__call__`.

    submodule : str, optional
        If specified, only symbols from the specified submodule are included.

    include_dunder_names : bool, default=False
        If `True`, then symbols like the `__init__` method of public classes will
        be included among the public symbols.

    Returns
    -------
    list[Symbol]
        Each symbol is a dict containing the following key-value pairs::

            category: Literal["class", "constant", "function", "method",
                                "module", "symbol", "type alias", "variable"]
            name: str
            referenceCount: int
            isExported: bool
            isTypeKnown: bool
            isTypeAmbiguous: bool
            diagnostics: list[Diagnostic]

    Examples
    --------
    Basic usage.

    >>> from maite.testing.project import get_public_symbols, ModuleScan
    >>> scanner = ModuleScan()
    >>> results = scanner("maite")
    >>> get_public_symbols(results)    # will change as MAITE changes --> # doctest: +SKIP
    [{'category': 'class',
      'name': 'maite.errors.MaiteException',
      'referenceCount': 3,
      'isExported': True,
      'isTypeKnown': True,
      'isTypeAmbiguous': False,
      'diagnostics': []},
     {'category': 'class',
      'name': 'maite.errors.InternalError',
      'referenceCount': 1,
      'isExported': True,
      'isTypeKnown': True,
      'isTypeAmbiguous': False,
      'diagnostics': []},
     ...]

    Accessing symbols from the `docs` submodule.

    >>> get_public_symbols(results, submodule="maite.testing.docs")
    [{'category': 'type alias', 'name': 'maite.testing.docs.NumpyDocErrorCode', 'referenceCount': 1, 'isExported': True, 'isTypeKnown': True, 'isTypeAmbiguous': False, 'diagnostics': []}, {'category': 'class', 'name': 'maite.testing.docs.NumPyDocResults', 'referenceCount': 1, 'isExported': True, 'isTypeKnown': True, 'isTypeAmbiguous': False, 'diagnostics'...
    """
    check_type("scan", scan, Mapping)
    check_type("submodule", submodule, str)
    check_type("include_dunder_names", include_dunder_names, bool)

    out = (x for x in scan["typeCompleteness"]["symbols"] if x["isExported"])
    if not include_dunder_names:
        # It is overly restrictive to only check for methods.
        # E.g. a dataclass' `__new__` method is actually
        # categorized as a function
        out = (x for x in out if not _is_dunder(x["name"]))
    if submodule:
        if any(not x.isidentifier() for x in submodule.split(".")):
            raise InvalidArgument(f"{submodule} is not a valid module name.")

        _out = [x for x in out if x["name"].startswith(submodule)]

        return _out
    return list(out)


def import_public_symbols(
    scan: ModuleScanResults,
    submodule: str = "",
    categories: Collection[Category] = frozenset(["function", "class"]),
    skip_module_not_found: Union[bool, Literal["pytest-skip"]] = True,
) -> Generator[Any, None, None]:
    """
    Import and yield all public symbols (functions, classes, etc.) from a module's API.

    This function expects the results of a scan performed by
    `maite.testing.project.ModuleScan`, which requires that `pyright` is
    installed.

    Parameters
    ----------
    scan : ModuleScanResults
        The result of a scan performed by `ModuleScan.__call__`.

    submodule : str, optional
        If specified, only symbols from the specified submodule are included.

    categories : Collection[Category], default=('function', 'class')
        The symbol categories to include.

        Valid categories are:
           - class
           - constant
           - function
           - method
           - module
           - symbol
           - type alias
           - variable

    skip_module_not_found : bool | Literal["pytest-skip"], default=True
        If `True`, symbols that cannot be imported due to a `ModuleNotFound` error
        are skipped (e.g., due to missing optional dependencies). If `False`, the
        `ModuleNotFound` error is raised.

        If `'pytest-skip'` is specified a pytest param of the symbol name – marked
        by pytest to skip – is yielded when the import raises a `ModuleNotFound` error.
        This is useful when `import_public_symbols` is used to populate a parameterized
        test, so that skipped symbols are documented.

    Yields
    ------
    Any
        The imported symbol.

    Examples
    --------
    Basic usage.

    >>> from maite.testing.project import import_public_symbols, ModuleScan
    >>> scanner = ModuleScan()
    >>> results = scanner("pyright")
    >>> list(import_public_symbols(results))[:2] # doctest: +ELLIPSIS
    [<function entrypoint at ...>, <function main at ...>]
    """

    if isinstance(skip_module_not_found, bool):
        marker = None
    elif skip_module_not_found == "pytest-skip":
        # pytest should be kept optional.
        from pytest import mark, param

        marker = mark.skip(reason="Module not found.")
    else:
        check_one_of(
            "skip_module_not_found", skip_module_not_found, [True, False, "pytest-skip"]
        )
        marker = None  # pragma: no cover
        raise Exception("unreachable")  # pragma: no cover

    for cat in categories:
        check_one_of("categories", cat, CATEGORIES)

    categories = set(categories)

    symbols = filter(
        lambda symbol: symbol["category"] in categories,
        sorted(
            get_public_symbols(scan, submodule=submodule),
            key=lambda symbol: symbol["name"],
        ),
    )

    err = (ModuleNotFoundError, AttributeError) if skip_module_not_found else ()
    cached_typeddict_names = set()

    for symbol in symbols:
        module_path, name = symbol["name"].rsplit(".", maxsplit=1)

        # TODO: probably need a more sophisticated method for importing
        # e.g. https://github.com/facebookresearch/hydra/blob/9ce67207488965431c69b2e2b8e1a2baa0ada4b8/hydra/_internal/utils.py#L614
        #
        # For example, if 'method' is included in `categories` then we can cash each
        # class-object that we import and then getattr on it when we encounter a
        # method/variable on that class. Because the symbols are sorted alphabetically
        # we are always guaranteed to encounter a class object before its members.

        if (
            symbol["category"] == "function"
            and module_path in cached_typeddict_names
            and "method" not in categories
        ):
            continue

        try:
            module = import_module(module_path)
        except err:
            if marker is not None:
                yield param(symbol["name"], marks=marker)  # type: ignore

            else:
                continue
        else:
            try:
                obj = getattr(module, name)
                if symbol["category"] == "class" and is_typed_dict(obj):
                    cached_typeddict_names.add(symbol["name"])

                yield obj
            except err:  # pragma: no cover
                # it is possible that a symbol is unreachable at runtime
                if marker is not None:
                    yield param(symbol["name"], marks=marker)  # type: ignore
                else:
                    continue
