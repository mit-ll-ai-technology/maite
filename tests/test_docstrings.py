# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest

from maite.testing.docs import validate_docstring
from maite.testing.project import get_public_symbols, import_public_symbols
from maite.testing.pyright import list_error_messages, pyright_analyze
from tests import module_scan

# Generates a string that imports all symbols from the maite's public API.
# TODO: Make this a convenience function exposed via our public API?
preamble = "\n".join(
    [
        "from {} import {}".format(*x["name"].rsplit(".", maxsplit=1))
        for x in get_public_symbols(module_scan("maite"))
        if x["category"] in {"module", "function", "class", "type alias"}
    ]
)

all_funcs_and_classes = list(
    import_public_symbols(
        module_scan("maite"),
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


@pytest.mark.parametrize("obj, scan", PYRIGHT_SCAN_RESULTS)
def test_docstrings_scan_clean_via_pyright(obj, scan):
    if scan["summary"]["errorCount"] != 0:
        raise ValueError(
            "\n"
            + f"Pyright error in docstring for {obj}"
            + "\n"
            + "\n".join(list_error_messages(scan))
        )


@pytest.mark.parametrize("obj", all_funcs_and_classes)
def test_docstrings_adhere_to_numpydoc(obj):
    results = validate_docstring(obj, ignore=("SA01", "ES01"))

    if results["error_count"] > 0:
        errs = []
        for err_key, err_val in results["errors"].items():
            errs.append(f"{err_key}: {err_val}")
        r = "\n".join((f"{k}:{v}" for k, v in results.items() if k != "errors"))
        raise AssertionError(f"In {obj}\n{r}\n" + "\n".join(errs))
