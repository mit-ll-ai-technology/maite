# jatic_toolbox

`jatic_toolbox` is provided to vendors as a source of common types, protocols (a.k.a structural subtypes), utilities, and tooling to be leveraged by JATIC Python projects. It is designed to streamline the development of JATIC Python projects and to ensure that end-users enjoy seamless, synergistic workflows when composing multiple JATIC capabilities. These serve to eliminate redundancies that would otherwise be shared across – and burden –  the majority of JATIC projects. jatic_toolbox  is designed to be a low-dependency, frequently-improved Python package that is installed by JATIC projects. The following is a brief overview of the current state of its submodules.

## jatic_toolbox.protocols

Defines common types – such as an inference-mode object detector – factor to be leveraged across JATIC projects. These are specifically designed to be [protocols](https://peps.python.org/pep-0544/), which support structural subtyping. As a result developers and users can satisfy typed interfaces without having to explicitly subclass these custom types. These help to promote common interfaces across JATIC projects without introducing explicit inter-dependencies between them.


## jatic_toolbox.interop

Wrappers and functions that JATIC protocols compatible with popular 3rd party libraries and frameworks. For example, this module can be used to wrap the object detectors provided by huggingface and timm so that they adhere to the JATIC protocols for object detectors.


## jatic_toolbox.utils

Provides:

- Functions for validating the types and values of user arguments, with explicit and consistent user-error messages, that raise jatic_toolbox-customized exceptions.
- Specialized PyTorch utilities to help facilitate safe and ergonomic code patterns for manipulating stateful torch objects
- Other quality assurance and convenience functions that may be widely useful across projects

## jatic_toolbox.testing

Tools that help developers create a rigorous automated test suite for their JATIC project. These include:

- The quality assurance tests that the SDP stakeholders will be running as part of the JATIC-wide CI/CD pipeline, which can be run locally by project developers.
pytest fixtures for initializing test functions with common models, datasets, and other inputs that are useful for testing machine learning code.
- Functions running static type checking tests using [pyright](https://github.com/microsoft/pyright) in a pytest test suite, including scans of both source code and example documentation code blocks.
- [Hypothesis strategies](https://hypothesis.readthedocs.io/en/latest/) for driving property-based tests of interfaces that leverage JATIC protocols.
