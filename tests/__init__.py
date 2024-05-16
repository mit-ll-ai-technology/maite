# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

from maite.testing.project import ModuleScan

module_scan = ModuleScan()

all_dummy_subpkgs = sorted(
    p.name for p in (Path(__file__).parent / "dummy_projects").glob("*") if p.is_dir()
)

assert all_dummy_subpkgs
