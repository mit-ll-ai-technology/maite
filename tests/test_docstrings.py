import pytest
from pytest import param

from jatic_toolbox.testing.docs import validate_docstring
from jatic_toolbox.testing.project import ModuleScan, get_public_symbols
from jatic_toolbox.testing.pyright import list_error_messages, pyright_analyze
from jatic_toolbox.testing.pytest import cleandir
from jatic_toolbox.utils.validation import (
    chain_validators,
    check_domain,
    check_one_of,
    check_type,
)
from tests import module_scan

# Generates a string that imports all symbols from the jatic toolbox's public API.
preamble = "\n".join(
    [
        "from {} import {}".format(*x["name"].rsplit(".", maxsplit=1))
        for x in get_public_symbols(module_scan("jatic_toolbox"))
        if x["category"] in {"module", "function", "class", "type alias"}
    ]
)


@pytest.mark.parametrize(
    "func",
    [
        validate_docstring,
        chain_validators,
        check_domain,
        check_one_of,
        check_type,
        cleandir,
        ModuleScan,
        get_public_symbols,
        param(
            pyright_analyze,
            marks=pytest.mark.xfail(
                raises=AssertionError,
                strict=True,
                reason="pyright_analyze docstring intentionally includes "
                "type-check errors.",
            ),
        ),
    ],
)
def test_docstrings_scan_clean_via_pyright(func):
    results = pyright_analyze(
        func,
        scan_docstring=True,
        report_unnecessary_type_ignore_comment=True,
        preamble=preamble,
    )
    assert results["summary"]["errorCount"] == 0, list_error_messages(results)


@pytest.mark.parametrize(
    "func",
    [
        pyright_analyze,
        validate_docstring,
        chain_validators,
        check_domain,
        check_one_of,
        check_type,
        cleandir,
        ModuleScan,
        get_public_symbols,
    ],
)
def test_docstrings_adhere_to_numpydoc(func):
    results = validate_docstring(func, ignore=("SA01", "ES01"))
    assert results["error_count"] == 0, results["errors"]
