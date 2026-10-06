# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Repo-root pytest configuration.

Excludes optional-backend ``src/`` modules from ``--doctest-modules`` collection
when their extra is not installed (they import heavy deps at module top level,
which would fail at import time). Scoped to the repo root so it covers ``src/``;
``tests/conftest.py`` handles the equivalent exclusions under ``tests/``.

Test selection uses two kinds of marker:

* **Per-extra** — extras-dependent tests carry an explicit
  ``mot_utils``/``torchmetrics``/``yolo_models`` marker (see the individual test
  modules); the tox envs select each one with ``-m <extra>``.
* **required/optional** — the ``pytest_collection_modifyitems`` hook below derives
  these from the per-extra markers so every non-``scan_docs`` item is tagged
  ``optional`` (needs an extra) or ``required`` (needs no extra), keeping
  required-feature tests marked ``@pytest.mark.required`` and optional-feature
  tests ``@pytest.mark.optional`` as required by TR-4-H-3/H-4. The bare ``py<ver>``
  tox env (no factor) runs the required tests with ``-m required``.

Because ``-m`` deselection also runs in ``pytest_collection_modifyitems``, the hook
below is registered ``tryfirst=True`` so the derived markers exist before selection
happens. Only extras-dependent tests need to be marked by hand.
"""

import pytest

from maite._internals import import_utils

collect_ignore = []

if not import_utils.is_torchmetrics_available():
    collect_ignore += [
        "src/maite/_internals/interop/metrics/torchmetrics.py",
        "src/maite/_internals/interop/metrics/torchmetrics_detection.py",
    ]

if not (
    import_utils.is_torch_available() and import_utils.is_ultralytics_available() and import_utils.is_yolov5_available()
):
    collect_ignore.append("src/maite/_internals/interop/models/yolo.py")


# Markers that flag a test as depending on an optional extra. Everything else that
# isn't the doc/pyright scan is "required" (no extra needed).
OPTIONAL_MARKERS = ("mot_utils", "torchmetrics", "yolo_models")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Derive ``required``/``optional`` from the per-extra markers.

    ``scan_docs`` (the whole-API docstring/pyright scan) covers both required and
    optional code, so it is left unmarked and excluded from both runs.
    """
    for item in items:
        if "scan_docs" in item.keywords:
            continue
        if any(marker in item.keywords for marker in OPTIONAL_MARKERS):
            item.add_marker("optional")
        else:
            item.add_marker("required")
