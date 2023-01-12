# jatic_dummy packages

This directory is designed to house independent pip-installable projects that we can test the jatic-toolbox against. Specifically, they are meant to serve as small, static projects with known contents/problems, and we can assert that the jatic-toolbox produces the expected results when processing them.

The projects are designed to be installed under the same namespace package, called `jatic_dummy`. E.g. the `basic` package, when installed, is imported as `jatic_dummy.basic`. See [this documentation](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) (specifically the 2nd of the first 2 examples) to understand our approach to packaging namespace packages.

## Installation

**All of the jatic_dummy.X packages are automatically installed upon running the jatic-toolbox test suite**. The `tests/__init__.py` is responsible for globbing all of the dir-names in `dummy_projects` and `tests/conftest.py` is responsible for installing all of the namespace packages in the local environment if they are not already installed.

Otherwise, these packages can be installed manually. From the top-level directory of this repo (containing the `tests/`) directory, to install the `basic` package, run:

```console
$ pip install tests/dummy_projects/basic/
```

You should then be able to run, e.g., 

```python
from jatic_dummy.basic.stuff import AClass
```
