import pytest

from jatic_toolbox.testing.pyright import list_error_messages, pyright_analyze
from jatic_toolbox.testing.pytest_fixtures import cleandir
from jatic_toolbox.utils.validation import (
    chain_validators,
    check_domain,
    check_one_of,
    check_type,
)

preamble = """from jatic_toolbox.utils.validation import (
    chain_validators,
    check_domain,
    check_one_of,
    check_type,
)
from jatic_toolbox.testing.pytest_fixtures import cleandir
"""


@pytest.mark.parametrize(
    "func",
    [
        chain_validators,
        check_domain,
        check_one_of,
        check_type,
        cleandir,
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
