# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Check that the ``all`` extra matches the union of the individual extras.

``pip install maite[all]`` must stay equivalent to installing every other extra
independently. This test parses ``pyproject.toml`` and asserts that ``all``
references exactly the union of the other optional-dependency extras, so adding a
new extra without adding it to ``all`` fails this test.
"""

import re
from pathlib import Path

import pytest

# tomllib is stdlib only on 3.11+; the check is static config, so one interpreter
# is enough. Skip (rather than depend on tomli) on 3.10.
tomllib = pytest.importorskip("tomllib")

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.skipif(not PYPROJECT.is_file(), reason="pyproject.toml not found (installed, not source tree)")
def test_all_extra_matches_union_of_extras():
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]

    assert "all" in extras, "expected an `all` extra in [project.optional-dependencies]"

    others = set(extras) - {"all"}

    # `all` entries look like `maite[mot-utils,torchmetrics,yolo-models]`; collect
    # every extra named inside them.
    referenced: set[str] = set()
    for entry in extras["all"]:
        match = re.search(r"\[([^\]]+)\]", entry)
        if match:
            referenced.update(name.strip() for name in match.group(1).split(","))

    assert referenced == others, (
        "`all` extra is out of sync with the individual extras.\n"
        f"  referenced by all: {sorted(referenced)}\n"
        f"  other extras:      {sorted(others)}\n"
        f"  missing from all:  {sorted(others - referenced)}\n"
        f"  unknown in all:    {sorted(referenced - others)}"
    )
