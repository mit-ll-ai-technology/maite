from pathlib import Path

from jatic_toolbox.testing.project import ModuleScan

module_scan = ModuleScan()

all_dummy_subpkgs = sorted(
    p.name for p in (Path(__file__).parent / "dummy_projects").glob("*") if p.is_dir()
)

assert all_dummy_subpkgs
