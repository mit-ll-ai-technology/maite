# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from importlib import import_module

import pytest
from pytest import Config

from tests import all_dummy_subpkgs


def test_version():
    import maite

    assert isinstance(maite.__version__, str)
    assert maite.__version__
    assert "unknown" not in maite.__version__


def test_xfail_strict(pytestconfig: Config):
    # Our test suite's xfail must be configured to strict mode
    # in order to ensure that contrapositive tests will actually
    # raise.
    assert pytestconfig.getini("xfail_strict") is True


@pytest.mark.parametrize("subpkg_name", all_dummy_subpkgs)
def test_dummy_projects_installed(subpkg_name: str):
    import_module(f"maite_dummy.{subpkg_name}")
