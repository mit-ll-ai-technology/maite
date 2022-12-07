import logging
import os
import tempfile
from typing import Iterable

import pytest


@pytest.fixture()
def cleandir() -> Iterable[str]:
    """
    Run a test with temporary directory as the current working directory.

    This is helpful for running tests that require file I/O that could pollute
    local directories. File cleanup is handled automatically.

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
