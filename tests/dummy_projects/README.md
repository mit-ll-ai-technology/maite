# maite_dummy packages

This directory is designed to house independent pip-installable projects that we can test the maite against. Specifically, they are meant to serve as small, static projects with known contents/problems, and we can assert that the maite produces the expected results when processing them.

The projects are designed to be installed under the same namespace package, called `maite_dummy`. E.g. the `basic` package, when installed, is imported as `maite_dummy.basic`. This keeps the local environment from being "polluted" with a bunch of tersely-named packages, and instead promotes a cleanly structured namespace. See [this documentation](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) (specifically the 2nd of the first 2 examples) to understand our approach to packaging namespace packages.

The basic structure of a package is (replace `<pkg_name>`):

```
dummy_projects/
|- <pkg_name>/
    |- pyproject.toml
    |- src/maite_dummy/<pkg_name>
        |- py.typed
        |- __init__.py
        |- (whatever python modules / subpackages)
```

Note the presence of the py.typed file!

With the following pyproject.toml (replace `<pkg_name>`).

```
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "maite-dummy-subpackage-<pkg_name>"
version = "0.0.1"
description = "A small example package"
requires-python = ">=3.7"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"maite_dummy.<pkg_name>" = ["py.typed"]
```



## Installation

**All of the maite_dummy.X packages are automatically installed upon running the maite test suite**. The `tests/__init__.py` is responsible for globbing all of the dir-names in `dummy_projects` and `tests/conftest.py` is responsible for installing all of the namespace packages in the local environment if they are not already installed. So simply running `pytest tests/` or `tox -e pyXX` will handle the install for you.

Otherwise, these packages can be installed manually. From the top-level directory of this repo (containing the `tests/`) directory, to install the `basic` package, run:

```console
$ pip install tests/dummy_projects/basic/
```

You should then be able to run, e.g., 

```python
from maite_dummy.basic.stuff import AClass
```
