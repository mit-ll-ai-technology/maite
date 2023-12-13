# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


@contextmanager
def _add_to_sys_path(_dir: str):
    sys.path.insert(0, _dir)
    try:
        yield
    finally:
        sys.path.remove(_dir)


def _try_import(name: str, fullpath_to_file: str | os.PathLike[str]):
    spec = importlib.util.spec_from_file_location(name, str(fullpath_to_file))
    if spec is None or spec.loader is None:
        # pragma: no cover
        raise ImportError(
            f"Unable to load module {name} from location {fullpath_to_file}"
        )

    # spec is used to create the module spec
    module = importlib.util.module_from_spec(spec)
    # spec loader is populated or None is returned above
    spec.loader.exec_module(module)
    # Note: exec_module above populates the module namespace in-place
    return module


def _check_module_exists(name):
    return importlib.util.find_spec(name) is not None


def import_hubconf(local_dir: str | os.PathLike[str], module_name: str) -> ModuleType:
    _local_dir = Path(local_dir)
    with _add_to_sys_path(str(_local_dir)):
        hub_module = _try_import(module_name, _local_dir / module_name)

    # Note: This check does not really work as intended. Unless the import of the
    # missing dependency happens at a scope that is not evaluated on import (say inside
    # a function). We must import the hubconf module to check the dependencies var, but
    # can't do that without triggering the import error first above. The error message
    # will still be informative to user, but it is not very likely this will actually be
    # raised as the source of signal on missing dependencies.

    # We may want to consider modifying the extended hubconf spec to place this list in
    # a special location making it possible to pre-parse it before import if there is a
    # lot of user confusion around this point
    deps = getattr(hub_module, "dependencies", [])
    missing_deps = [pkg for pkg in deps if not _check_module_exists(pkg)]
    if len(missing_deps) != 0:
        raise RuntimeError(
            f"Missing dependencies for hub endpoint: {', '.join(missing_deps)}"
        )

    return hub_module
