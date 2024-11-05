# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from pytest import param

from maite.errors import InvalidArgument
from maite.testing.project import (
    CompletenessSection,
    ModuleScan,
    ModuleScanResults,
    Symbol,
    SymbolCounts,
    get_public_symbols,
    import_public_symbols,
)
from maite.testing.pyright import Summary
from tests import module_scan

ParameterSet = type(param("s"))


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
        scanner("maite", path_to_pyright=Path("bad_path"))


def test_scan_caching():
    scanner = ModuleScan()
    assert scanner.cache_info().currsize == 0
    assert scanner.cache_info().misses == 0
    assert scanner.cache_info().hits == 0

    scanner("maite")
    assert scanner.cache_info().currsize == 1
    assert scanner.cache_info().misses == 1
    assert scanner.cache_info().hits == 0

    scanner("maite")
    scanner("maite")
    assert scanner.cache_info().currsize == 1
    assert scanner.cache_info().misses == 1
    assert scanner.cache_info().hits == 2

    scanner.clear_cache()
    assert scanner.cache_info().currsize == 0
    assert scanner.cache_info().misses == 0
    assert scanner.cache_info().hits == 0


def test_known_scan():
    results = module_scan("maite")["typeCompleteness"]
    assert results["packageName"] == "maite"
    modules = {v["name"] for v in results["modules"]}
    # must update this if project's modules are renamed
    assert {
        "maite.testing.docs",
        "maite.testing.project",
        "maite.testing.pyright",
        "maite.testing.pytest",
        "maite.testing",
        "maite.utils.validation",
        "maite.utils",
        "maite",
    } <= modules
    assert not any(
        name.split(".")[-1].startswith("_") for name in modules
    ), "reported module is private"


@pytest.mark.parametrize("submodule", ["", "maite.testing.project"])
def test_public_symbols(submodule):
    symbols = get_public_symbols(
        module_scan("maite"), submodule=submodule, include_dunder_names=False
    )
    names = {s["name"] for s in symbols}
    assert {
        "maite.testing.project.ModuleScan",
        "maite.testing.project.get_public_symbols",
    } <= names
    for symbol in symbols:
        *_, name = symbol["name"].split(".")

        assert not name.startswith("_"), f"{symbol}: symbol is private"
        assert symbol["isExported"] is True


def test_invalid_submodule():
    results = module_scan("maite")
    with pytest.raises(InvalidArgument, match="11 is not a valid module name."):
        get_public_symbols(results, submodule="maite.11")


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
                    category="function",
                    name="foo.SomeClass.__dataclass_method__",
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
            ],
        ),
    )
    symbols_with_special = get_public_symbols(dummy_results, include_dunder_names=True)
    symbols_without_special = get_public_symbols(
        dummy_results, include_dunder_names=False
    )

    assert {x["name"] for x in symbols_with_special} == {
        "foo.SomeClass",
        "foo.SomeClass.__init__",
        "foo.SomeClass.public_method__",
        "foo.SomeClass.___not_special___",
        "foo.SomeClass.__dataclass_method__",
    }

    assert {x["name"] for x in symbols_without_special} == {
        "foo.SomeClass",
        "foo.SomeClass.public_method__",
        "foo.SomeClass.___not_special___",
    }


def test_import_symbols():
    results = module_scan("maite_dummy.basic")
    out = list(import_public_symbols(results))

    from maite_dummy.basic.stuff import (
        AClass,
        ADataClass,
        AProtocol,
        ATypedDict,
        a_func,
    )

    expected_stuff = {a_func, AClass, ADataClass, AProtocol, ATypedDict}
    assert len(out) == len(expected_stuff)
    assert set(out) == expected_stuff


def test_import_pytest_skip():
    results = module_scan("maite_dummy.basic")
    out: list[str] = [
        x.values[0]
        for x in import_public_symbols(results, skip_module_not_found="pytest-skip")
        if isinstance(x, ParameterSet)
    ]  # type: ignore

    expected = {
        "maite_dummy.basic.needs_mygrad.func_needs_mygrad",
    }
    assert sorted(out) == sorted(expected)


def test_validate_import_public_symbols_input():
    results = module_scan("maite_dummy.basic")

    with pytest.raises(
        InvalidArgument,
        match=r"Expected `skip_module_not_found` to be one of: False, True, pytest-skip. Got `pytest-blah`.",
    ):
        list(import_public_symbols(results, skip_module_not_found="pytest-blah"))  # type: ignore
