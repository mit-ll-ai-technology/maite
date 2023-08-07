import pytest

from jatic_toolbox.testing.docs import validate_docstring
from jatic_toolbox.testing.project import get_public_symbols, import_public_symbols
from jatic_toolbox.testing.pyright import list_error_messages, pyright_analyze
from tests import module_scan

# Generates a string that imports all symbols from the jatic toolbox's public API.
# TODO: Make this a convenience function exposed via our public API?
preamble = "\n".join(
    [
        "from {} import {}".format(*x["name"].rsplit(".", maxsplit=1))
        for x in get_public_symbols(module_scan("jatic_toolbox"))
        if x["category"] in {"module", "function", "class", "type alias"}
    ]
)

all_funcs_and_classes = list(
    import_public_symbols(
        module_scan("jatic_toolbox"),
        skip_module_not_found="pytest-skip",
    ),
)

PYRIGHT_SCAN_RESULTS = []
FUNCS_TO_SCAN = [
    obj
    for obj in all_funcs_and_classes
    if obj.__doc__ is not None and obj is not pyright_analyze
]
for obj, scan in zip(
    FUNCS_TO_SCAN,
    pyright_analyze(
        *FUNCS_TO_SCAN,
        scan_docstring=True,
        report_unnecessary_type_ignore_comment=True,
        preamble=preamble,
    ),
):
    PYRIGHT_SCAN_RESULTS.append([obj, scan])


@pytest.mark.parametrize("func", PYRIGHT_SCAN_RESULTS)
def test_docstrings_scan_clean_via_pyright(func):
    _, results = func
    assert results["summary"]["errorCount"] == 0, list_error_messages(results)


@pytest.mark.parametrize("obj", all_funcs_and_classes)
def test_docstrings_adhere_to_numpydoc(obj):
    results = validate_docstring(obj, ignore=("SA01", "ES01"))
    assert results["error_count"] == 0, results["errors"]
