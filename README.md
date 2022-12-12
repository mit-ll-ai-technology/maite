# jatic_toolbox

![image](brand/Hydra-Zen_logo_full_filled_bkgrnd_small.png)


<p align="center">
  <a>
    <img src="https://img.shields.io/badge/python-3.7%20&#8208;%203.10-blue.svg" alt="Python version support" />
  </a>
  <a>
    <img src="https://img.shields.io/badge/coverage-100%25-green.svg" alt="Code Coverage" />
  <a href="https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#verifying-type-completeness">
    <img src="https://img.shields.io/badge/type%20completeness-100%25-green.svg" alt="Type-Completeness Score" />
  <a href="https://hypothesis.readthedocs.io/">
    <img src="https://img.shields.io/badge/hypothesis-tested-brightgreen.svg" alt="Tested with Hypothesis" />
  </a>
  </p>
</p>

## Installation

To install from source, clone this repository and, from the top-level directory (in the directory containing `src/`), run:

```console
$ pip install .
```

If you are making modifications to the source code, it is recommended to instead do an "editable" install with:

```console
$ pip install -e .
```

## Overview

`jatic_toolbox` is provided to vendors as a source of common types, protocols (a.k.a structural subtypes), utilities, and tooling to be leveraged by JATIC Python projects. It is designed to streamline the development of JATIC Python projects and to ensure that end-users enjoy seamless, synergistic workflows when composing multiple JATIC capabilities. These serve to eliminate redundancies that would otherwise be shared across – and burden –  the majority of JATIC projects. jatic_toolbox  is designed to be a low-dependency, frequently-improved Python package that is installed by JATIC projects. The following is a brief overview of the current state of its submodules.

### jatic_toolbox.protocols

Defines common types – such as an inference-mode object detector – factor to be leveraged across JATIC projects. These are specifically designed to be [protocols](https://peps.python.org/pep-0544/), which support structural subtyping. As a result developers and users can satisfy typed interfaces without having to explicitly subclass these custom types. These help to promote common interfaces across JATIC projects without introducing explicit inter-dependencies between them.


### jatic_toolbox.interop

Wrappers and functions that JATIC protocols compatible with popular 3rd party libraries and frameworks. For example, this module can be used to wrap the object detectors provided by huggingface and timm so that they adhere to the JATIC protocols for object detectors.


### jatic_toolbox.utils

Provides:

- Functions for validating the types and values of user arguments, with explicit and consistent user-error messages, that raise jatic_toolbox-customized exceptions.
- Specialized PyTorch utilities to help facilitate safe and ergonomic code patterns for manipulating stateful torch objects
- Other quality assurance and convenience functions that may be widely useful across projects

### jatic_toolbox.testing

Tools that help developers create a rigorous automated test suite for their JATIC project. These include:

- The quality assurance tests that the SDP stakeholders will be running as part of the JATIC-wide CI/CD pipeline, which can be run locally by project developers.
pytest fixtures for initializing test functions with common models, datasets, and other inputs that are useful for testing machine learning code.
- Functions running static type checking tests using [pyright](https://github.com/microsoft/pyright) in a pytest test suite, including scans of both source code and example documentation code blocks.
- [Hypothesis strategies](https://hypothesis.readthedocs.io/en/latest/) for driving property-based tests of interfaces that leverage JATIC protocols.
