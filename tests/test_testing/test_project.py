from pathlib import Path

import pytest

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.testing.project import (
    CompletenessSection,
    ModuleScan,
    ModuleScanResults,
    Symbol,
    SymbolCounts,
    get_public_symbols,
)
from jatic_toolbox.testing.pyright import Summary
from tests import module_scan


def test_bad_module_name():
    scanner = ModuleScan()

    with pytest.raises(
        ModuleNotFoundError,
        match=r"No files were found to analyze in association with "
        "module `NOT_VALID_MODULE`",
    ):
        scanner("NOT_VALID_MODULE")


def test_bad_path_to_pyright():
    scanner = ModuleScan()
    with pytest.raises(
        FileNotFoundError,
        match=r"`path_to_pyright` – bad_path – doesn't exist",
    ):
        scanner("jatic_toolbox", path_to_pyright=Path("bad_path"))


def test_scan_caching():
    scanner = ModuleScan()
    assert scanner.cache_info().currsize == 0
    assert scanner.cache_info().misses == 0
    assert scanner.cache_info().hits == 0

    scanner("jatic_toolbox")
    assert scanner.cache_info().currsize == 1
    assert scanner.cache_info().misses == 1
    assert scanner.cache_info().hits == 0

    scanner("jatic_toolbox")
    scanner("jatic_toolbox")
    assert scanner.cache_info().currsize == 1
    assert scanner.cache_info().misses == 1
    assert scanner.cache_info().hits == 2

    scanner.clear_cache()
    assert scanner.cache_info().currsize == 0
    assert scanner.cache_info().misses == 0
    assert scanner.cache_info().hits == 0


def test_known_scan():
    results = module_scan("jatic_toolbox")["typeCompleteness"]
    assert results["packageName"] == "jatic_toolbox"
    modules = set(v["name"] for v in results["modules"])
    # must update this if project's modules are renamed
    assert {
        "jatic_toolbox.testing.docs",
        "jatic_toolbox.testing.project",
        "jatic_toolbox.testing.pyright",
        "jatic_toolbox.testing.pytest",
        "jatic_toolbox.testing",
        "jatic_toolbox.utils.validation",
        "jatic_toolbox.utils",
        "jatic_toolbox",
    } <= modules
    assert not any(
        name.split(".")[-1].startswith("_") for name in modules
    ), "reported module is private"


@pytest.mark.parametrize("submodule", ["", "jatic_toolbox.testing.project"])
def test_public_symbols(submodule):
    symbols = get_public_symbols(
        module_scan("jatic_toolbox"), submodule=submodule, include_special_methods=False
    )
    names = set(s["name"] for s in symbols)
    assert {
        "jatic_toolbox.testing.project.ModuleScan",
        "jatic_toolbox.testing.project.get_public_symbols",
    } <= names
    for symbol in symbols:
        *_, name = symbol["name"].split(".")

        assert not name.startswith("_"), f"{symbol}: symbol is private"
        assert symbol["isExported"] is True


def test_invalid_submodule():
    results = module_scan("jatic_toolbox")
    with pytest.raises(InvalidArgument, match="11 is not a valid module name."):
        get_public_symbols(results, submodule="jatic_toolbox.11")


def test_special_method_filtering():
    dummy_results = ModuleScanResults(
        version="",
        time="",
        generalDiagnostics=[],
        summary=Summary(
            filesAnalyzed=1,
            errorCount=0,
            warningCount=0,
            informationCount=0,
            timeInSec=0.0,
        ),
        typeCompleteness=CompletenessSection(
            packageName="foo",
            packageRootDirectory=Path("/foo"),
            moduleName="foo",
            moduleRootDirectory=Path("/foo"),
            pyTypedPath=Path("."),
            ignoreUnknownTypesFromImports=False,
            missingClassDocStringCount=0,
            missingDefaultParamCount=0,
            missingFunctionDocStringCount=0,
            exportedSymbolCounts=SymbolCounts(
                withKnownType=5, withAmbiguousType=0, withUnknownType=0
            ),
            otherSymbolCounts=SymbolCounts(
                withKnownType=0, withAmbiguousType=0, withUnknownType=0
            ),
            completenessScore=100,
            modules=["foo"],
            symbols=[
                Symbol(
                    category="class",
                    name="foo.SomeClass",
                    referenceCount=1,
                    isExported=True,
                    isTypeKnown=False,
                    isTypeAmbiguous=False,
                    diagnostics=[],
                ),
                Symbol(
                    category="method",
                    name="foo.SomeClass.__init__",
                    referenceCount=1,
                    isExported=True,
                    isTypeKnown=True,
                    isTypeAmbiguous=False,
                    diagnostics=[],
                ),
                Symbol(
                    category="method",
                    name="foo.SomeClass.public_method__",
                    referenceCount=1,
                    isExported=True,
                    isTypeKnown=True,
                    isTypeAmbiguous=False,
                    diagnostics=[],
                ),
                Symbol(
                    category="method",
                    name="foo.SomeClass.___not_special___",
                    referenceCount=1,
                    isExported=True,
                    isTypeKnown=True,
                    isTypeAmbiguous=False,
                    diagnostics=[],
                ),
                Symbol(
                    category="function",
                    name="foo.__a_function__",
                    referenceCount=1,
                    isExported=True,
                    isTypeKnown=True,
                    isTypeAmbiguous=False,
                    diagnostics=[],
                ),
            ],
        ),
    )
    symbols_with_special = get_public_symbols(
        dummy_results, include_special_methods=True
    )
    symbols_without_special = get_public_symbols(
        dummy_results, include_special_methods=False
    )

    assert set(x["name"] for x in symbols_with_special) == {
        "foo.SomeClass",
        "foo.SomeClass.__init__",
        "foo.SomeClass.public_method__",
        "foo.SomeClass.___not_special___",
        "foo.__a_function__",
    }

    assert set(x["name"] for x in symbols_without_special) == {
        "foo.SomeClass",
        "foo.SomeClass.public_method__",
        "foo.SomeClass.___not_special___",
        "foo.__a_function__",
    }
