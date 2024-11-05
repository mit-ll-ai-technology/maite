# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import logging
import os
import tempfile
from collections.abc import Iterable

import pytest


@pytest.fixture()
def cleandir() -> Iterable[str]:
    """
    A pytest fixture that runs a test in a temporary directory as the current working directory.

    This is helpful for running tests that require file I/O that could pollute
    local directories. File cleanup is handled automatically.

    Yields
    ------
    tmpdirname : str
       Temporary directory name that will be removed at fixture teardown

    Examples
    --------
    This assumes that `cleandir` has been imported by the test suite's
    `conftest.py` file.

    >>> import pytest
    >>> @pytest.mark.usefixtures("cleandir")
    ... def test_writes_some_file():
    ...     from pathlib import Path
    ...     Path("dummy.txt").touch()  # file will be written to a tmp dir
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        os.chdir(tmpdirname)  # change cwd to the temp-directory
        yield tmpdirname  # yields control to the test to be run
        os.chdir(old_dir)
        logging.shutdown()
