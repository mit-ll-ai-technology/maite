# MAITE (Modular AI Trustworthy Engineering)


<p align="center">
  <a>
    <img src="https://img.shields.io/badge/python-3.8%20&#8208;%203.10-blue.svg" alt="Python version support" />
  </a>
  <a>
    <img src="https://img.shields.io/badge/coverage-100%25-green.svg" alt="Code Coverage" />
  <a href="https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#verifying-type-completeness">
    <img src="https://img.shields.io/badge/type%20completeness-100%25-green.svg" alt="Type-Completeness Score" />
  <a href="https://hypothesis.readthedocs.io/">
    <img src="https://img.shields.io/badge/hypothesis-tested-brightgreen.svg" alt="Tested with Hypothesis" />
  </a>
  </p>
  <p align="center">
    A toolbox of common types, protocols, and tooling to support T&E workflows.
  </p>
  <p align="center">
    Check out our <a href="https://jatic.pages.jatic.net/cdao/maite">documentation</a>.
  </p>
</p>

MAITE is provided to vendors as a source of common types, protocols (a.k.a structural subtypes), utilities, and tooling to be leveraged by JATIC Python projects. It is designed to streamline the development of JATIC Python projects and to ensure that end-users enjoy seamless, synergistic workflows when composing multiple JATIC capabilities. These serve to eliminate redundancies that would otherwise be shared across – and burden – the majority of JATIC projects. MAITE is designed to be a low-dependency, frequently-improved Python package that is installed by JATIC projects. The following is a brief overview of the current state of its submodules.

## Installation

To install from source, clone this repository and, from the top-level directory (in the directory containing `src/`), run:

```console
$ git clone https://gitlab.jatic.net/jatic/cdao/maite
$ cd maite
$ pip install .
```

You can install MAITE for a given release tag, e.g. `v0.3.0`, by running:

```console
$ pip install git+ssh://git@gitlab.jatic.net/jatic/cdao/maite.git@v0.3.0
```

If you would like to install an older version with the previous name jatic-toolbox for a given release tag, e.g. `v0.2.3`, run:

```console
$ pip install git+ssh://git@gitlab.jatic.net/jatic/cdao/maite.git@v0.2.3
```

## maite.protocols

Defines common types – such as an inference-mode object detector – factor to be leveraged across JATIC projects. These are specifically designed to be [protocols](https://peps.python.org/pep-0544/), which support structural subtyping. As a result developers and users can satisfy typed interfaces without having to explicitly subclass these custom types. These help to promote common interfaces across JATIC projects without introducing explicit inter-dependencies between them.

### ArrayLike

An `ArrayLike` defines a common interface for objects that can be manipulated as arrays, regardless of the specific implementation.
This allows code to be written in a more generic way, allowing it to work with different array-like objects without having to worry
about the details of the specific implementation. With an `ArrayLike` protocol, vendors can write functions and algorithms that
operate on arrays without JATIC defining the specific implementation of arrays to use.

```python
import maite.protocols as pr

# ArrayLike requires objects that implement `__array__` or `__array_interface__`.
assert not isinstance([1, 2, 3], pr.ArrayLike)

import numpy as np
np_array = np.zeros((10, 10, 3), dtype=np.uint8)
assert isinstance(np_array, pr.ArrayLike)

import torch as tr
array = tr.as_tensor(np_array)
assert isinstance(array, pr.ArrayLike)

# In the spirit of the toolbox this should pass but we currently
# do not have a proper check to support PIL images.
from PIL import Image
array = Image.fromarray(np_array)
assert not isinstance(array, pr.ArrayLike) 
```
## maite.testing

Tools that help developers create a rigorous automated test suite for their JATIC project. These include:

- The quality assurance tests that the SDP stakeholders will be running as part of the JATIC-wide CI/CD pipeline, which can be run locally by project developers.
pytest fixtures for initializing test functions with common models, datasets, and other inputs that are useful for testing machine learning code.
- Functions running static type checking tests using [pyright](https://github.com/microsoft/pyright) in a pytest test suite, including scans of both source code and example documentation code blocks.
- [Hypothesis strategies](https://hypothesis.readthedocs.io/en/latest/) for driving property-based tests of interfaces that leverage JATIC protocols.

### Pyright Static Type Checking in Code

```python
>>> def f(x: str):
...     return 1 + x
>>> pyright_analyze(f)
{'version': '1.1.281',
  'time': '1669686515154',
  'generalDiagnostics': [{'file': 'source.py',
    'severity': 'error',
    'message': 'Operator "+" not supported for types "Literal[1]" and "str"\n\xa0\xa0Operator "+" not supported for types "Literal[1]" and "str"',
    'range': {'start': {'line': 1, 'character': 11},
    'end': {'line': 1, 'character': 16}},
    'rule': 'reportGeneralTypeIssues'}],
  'summary': {'filesAnalyzed': 20,
  'errorCount': 1,
  'warningCount': 0,
  'informationCount': 0,
  'timeInSec': 0.319}}
```

## maite.interop

Wrappers and functions that JATIC protocols compatible with popular 3rd party libraries and frameworks. For example, this module can be used to wrap the object detectors provided by huggingface and timm so that they adhere to the JATIC protocols for object detectors.

```python
>>> from maite import load_dataset
>>> dataset = load_dataset(
...     provider="torchvision",
...     dataset_name="CIFAR10",
...     task="image-classification",
...     split="test",
...     root="~/data",
...     download=True
... )
```

## maite.utils

Provides:

- Functions for validating the types and values of user arguments, with explicit and consistent user-error messages, that raise maite-customized exceptions.
- Specialized PyTorch utilities to help facilitate safe and ergonomic code patterns for manipulating stateful torch objects
- Other quality assurance and convenience functions that may be widely useful across projects


## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2023 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

* Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
* SPDX-License-Identifier: MIT

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

The software/firmware is provided to you on an As-Is basis
