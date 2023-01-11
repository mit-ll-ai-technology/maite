from importlib import import_module

import pytest

from jatic_toolbox.testing.docs import validate_docstring
from jatic_toolbox.testing.project import get_public_symbols
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

# List of ['jatic_toolbox.testing.pyright.list_error_messages',
#          'jatic_toolbox.testing.project.ModuleScan',
#          'jatic_toolbox.utils.validation.check_one_of`,
#           etc.]
all_functions_and_classes = sorted(
    x["name"]
    for x in get_public_symbols(module_scan("jatic_toolbox"))
    if x["category"] in {"function", "class"}
)


@pytest.mark.parametrize("obj_path", all_functions_and_classes)
def test_docstrings_scan_clean_via_pyright(obj_path: str):
    if obj_path.endswith("pyright_analyze"):
        pytest.xfail(
            reason="pyright_analyze docstring intentionally includes "
            "type-check errors.",
        )

    module_path, name = obj_path.rsplit(".", maxsplit=1)
    try:
        module = import_module(module_path)
    except ModuleNotFoundError:
        pytest.skip("Dependency not installed.")

    name = obj_path.split(".")[-1]
    obj = getattr(module, name)

    if obj.__doc__ is None:
        pytest.skip("Doesn't have docstring.")

    results = pyright_analyze(
        obj,
        scan_docstring=True,
        report_unnecessary_type_ignore_comment=True,
        preamble=preamble,
    )
    assert results["summary"]["errorCount"] == 0, list_error_messages(results)


@pytest.mark.parametrize("obj_path", all_functions_and_classes)
def test_docstrings_adhere_to_numpydoc(obj_path: str):
    module_path, name = obj_path.rsplit(".", maxsplit=1)
    try:
        module = import_module(module_path)
    except ModuleNotFoundError:
        pytest.skip("Dependency not installed.")

    obj = getattr(module, name)

    results = validate_docstring(obj, ignore=("SA01", "ES01"))
    assert results["error_count"] == 0, results["errors"]
