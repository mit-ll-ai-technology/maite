# MAITE (Modular AI Trustworthy Engineering)

<p align="center">
  <a>
    <img src="https://img.shields.io/badge/python-3.9%20&#8208;%203.11-blue.svg" alt="Python version support" />
  </a>
  <a>
    <img src="https://img.shields.io/badge/coverage-%3E90%25-green.svg" alt="Code Coverage" />
  <a href="https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#verifying-type-completeness">
    <img src="https://img.shields.io/badge/type%20completeness-100%25-green.svg" alt="Type-Completeness Score" />
  <a href="https://hypothesis.readthedocs.io/">
    <img src="https://img.shields.io/badge/hypothesis-tested-brightgreen.svg" alt="Tested with Hypothesis" />
  </a>
  </p>

  <p align="center">
    A toolbox of common types, protocols, and tooling to support AI test and evaluation workflows.
  </p>

  <p align="center">
    Check out the <a href="https://mit-ll-ai-technology.github.io/maite/">documentation</a> and
    <a href="https://github.com/mit-ll-ai-technology/maite/tree/main/examples">examples</a> for more information.
  </p>
</p>

MAITE is a library of common types, protocols (a.k.a. structural subtypes), and utilities for the test and evaluation (T&E) of supervised machine learning models. It is being developed under the [Joint AI T&E Infrastructure Capability (JATIC)](https://gitlab.jatic.net/home/) program. Its goal is to streamline the development of JATIC Python projects by ensuring seamless, synergistic workflows when working with MAITE-conforming Python packages for different T&E tasks. To this end, MAITE seeks to eliminate redundancies that would otherwise be shared across – and burden – separate efforts in machine learning test and evaluation. MAITE is designed to be a low-dependency, frequently-improved Python package that is installed by JATIC projects. The following is a brief overview of the current state of its submodules.

## Installation

### From Python Package Index (PyPI)
To install from the Python Package Index (PyPI), run:

```console
pip install maite
```

> :information_source: You can install MAITE for a given release tag, e.g. `v0.6.1`, by running:
>
>```console
>$ pip install git+ssh://git@github.com/mit-ll-ai-technology/maite.git@v0.6.1
>```

### From Source

To clone this repository and install from source, run:

```console
$ git clone https://github.com/mit-ll-ai-technology/maite
$ cd maite
$ pip install .
```

## maite.protocols

*Common types for machine learning test and evaluation*

The `protocols` subpackage defines common types – such as an inference-mode object detector – to be leveraged across JATIC projects. These are specifically designed to be [Python protocol classes](https://peps.python.org/pep-0544/), which support structural subtyping. As a result, developers and users can satisfy MAITE-typed interfaces without having to explicitly subclass. This ability helps to promote common interfaces across JATIC projects without introducing explicit inter-dependencies between them.

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

## maite.utils

*General utilities*

- Functions for validating the types and values of user arguments, with explicit and consistent user-error messages, that raise MAITE-customized exceptions.
- Specialized PyTorch utilities to help facilitate safe and ergonomic code patterns for manipulating stateful torch objects
- Other quality assurance and convenience functions that may be widely useful across projects

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2024 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

* Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
* SPDX-License-Identifier: MIT

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

The software/firmware is provided to you on an As-Is basis
