# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from maite.testing.pyright import pyright_analyze

# Ensure all workflow tests have no static errors using pyright

dir = Path(__file__).resolve().parent
files_to_analyze_statically = [
    f
    for f in dir.iterdir()
    if f.name.endswith(".py") and not f.name.endswith("static.py")
]

print(files_to_analyze_statically)


@pytest.mark.parametrize("fname", files_to_analyze_statically)
def test_evaluate_statically(fname):
    scan = pyright_analyze(fname)
    assert scan[0]["summary"]["errorCount"] == 0
