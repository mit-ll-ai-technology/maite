.. meta::
   :description: The changelog for maite, including what's new.

=========
Changelog
=========

This is a record of all past maite releases and what went into them, in reverse 
chronological order.

.. _v0.9.1:

---------------------
0.9.1 - 2025-12-18
---------------------

- Change return type of `Metric.compute` from `dict[str,Any]` -> `Mapping[str,Any]` to enable `TypedDict` classes to more easily structurally subclass this return type
- Add prototype advertisement/verification mechanism of components/tasks within installed packages (`_internals/testing/project:statically_verify_exposed_component_entrypoints`)
- Implement selective dataset loading with dataset protocol subclass (`FieldwiseDataset`)
- Increase use of pytest raises (over 'xfails') for semantically correct exceptions raised in unit testing 
- Temporarily pin ipykernel dependency until upstream fix implemented (https://github.com/huggingface/xet-core/issues/526)
- Reorganize API reference section for better scalability and clarity
- Crosslink type-aliases and generics within docs
- Add prototype image segmentation protocol definitions (in `_internals`)
- Add docstrings underneath typealias definitions to give more intuitive tooltip messaging (e.g., expected shape semantics for protocols.image_classification.InputType )

.. _v0.9.0:

---------------------
0.9.0 - 2025-09-24
---------------------

- Modify `Metric.update` to accept batch metadata
- Add `augment_dataloader` MAITE task that modifies a `DataLoader` with a user-defined augmentation
- Add unit tests for `evaluate` / `predict` tasks that cover more edge cases, verifying advertised exception handling
- Handle additional YOLO outputs that broke internal doctesting in CI (for recent YOLO releases)
- Update CONTRIBUTING.md to describe using `uv` to maintain a project environment and run `tox` jobs
- Use ruff linter instead of flake8, isort, and black
- Deprecate select legacy source code
- Automatically remove extraneous metadata from Jupyter notebooks in CI

.. _v0.8.2:

---------------------
0.8.2 - 2025-07-31
---------------------

- Relax numpy \"<2\" constraint
- State support for (and test against) python 3.12
- Remove stated support for (and test against) python 3.9
- Updated torchvision object-detection tutorial with more accurate type hinting
- Mark long running test in CI/CD (permitting manual bipass)
- Use _compat.py shim to enable access to TypedDict features introduced in 3.12 (importing from typing_extensions if not available)

.. _v0.8.1:

---------------------
0.8.1 - 2025-05-14
---------------------

- Add torchmetrics object detection wrapper to `interop` submodule
- Include 'tests' directory in built source distribution to enable conda-forge testing prior to deployment

.. _v0.8.0:

---------------------
0.8.0 - 2025-04-30
---------------------

- Add torchmetrics image classification wrapper to `interop` submodule
- Add yolo object detection wrapper to `interop` submodule
- Add `evaluate_from_predictions` MAITE task that accepts precalculated model predictions to compute metrics against ground truth
- Expand `ObjectDetectionTarget` semantics to permit optionally returning scores for multiple classes
- Roll back documented requirement to have normalized (i.e. sum to 1, all greater than 0) model outputs
- Implement MAITE tasks (i.e. `evaluate`, `predict`, `evaluate_from_predictions`) as generic functions capable of using static typechecker to validate inter-component compatibility at development time
- Update sphinx doc-building dependencies
- Update the word we use to describe maite-compatible callables, preferring 'task' name to 'workflow' name with broader connotation
- Author "MAITE Vision" explainer, describing the broader goal of MAITE and defining "MAITE tasks/components" formally
- Switch package build back end to poetry; include poetry.lock file in repository

.. _v0.7.3:

---------------------
0.7.3 - 2025-02-19
---------------------

- Update GitHub CI/CD to not directly publish docs (which is now done through GitLab mirroring)
- Update GitHub CI/CD to use latest version of upload-artifact and download-artifact actions

.. _v0.7.2:

---------------------
0.7.2 - 2025-02-12
---------------------

- Add how-to guide for wrapping object detection models & datasets
- Add how-to guide for wrapping image classification models & datasets
- Add docstring examples for object detection model, dataset, metric, & augmentation
- Add docstring examples for image classification model and metric
- Add image classification tutorial
- Back MAITE ArrayLike with numpy.typing.ArrayLike
- Publish historical and latest built documentation to github (automatically)

.. _v0.7.1:

---------------------
0.7.1 - 2024-11-06
---------------------

- bugfix: correct typo github action for publishing new docs to gh-pages

.. _v0.7.0:

---------------------
0.7.0 - 2024-11-06
---------------------

- Add TypedDict `metadata` attributes to `Dataset`, `Augmentation`, `Model`, `Metric` protocols
- Add support for Python 3.11 and drop support for Python 3.8
- Automatically generate documentation from notebooks in examples
- Display progress-bar when maite `Dataset` is passed to `evaluate`
- Automatically typecheck docstring examples

.. _v0.6.1:

---------------------
0.6.1 - 2024-10-16
---------------------

- bugfix: Added upper limit to numpy dependency (\"<2\") for compatibility with torchvision
- Updated file headers/copyrights
- Added new ci job for testing pypi publishing
- Update image references in torchvision_object_detection tutorial

.. _v0.6.0:

---------------------
0.6.0 - 2024-06-14
---------------------

- Removed torch dependency (internally, input/target types are batched agnostically of task)
- Altered batch types in each task to be sequences of singleton types
- Added tqdm progress bars to evaluate/predict workflows
- Added docstring examples to Dataset and Augmentation protocols

.. _v0.5.0:

---------------------
0.5.0 - 2024-03-28
---------------------

This is a large change to protocols based on experiences of our growing userbase.
These updated protocols should enable a wider variety of use cases, but do contain backward-compatibility breaking changes.

High-level overview:

- A tutorial has been added to show `use of core protocols in object detection domain <https://github.com/mit-ll-ai-technology/maite/blob/main/examples/torchvision_object_detection/torchvision_object_detection.ipynb>`_
- An `overview of updated core protocols <https://github.com/mit-ll-ai-technology/maite/blob/main/examples/protocol_overview.ipynb>`_ has been provided 

More details:

- Class structure of core component protocols is now ML subproblem agnostic -- e.g. `Model` objects must all implement the same named methods.
- Type signatures of core component protocols are specific to ML subproblem domain, permitting subproblem specific component implementers to know more about their expected inputs/outputs.
- Within an ML subproblem, the model input type, model target type, datum metadata type (and the 3 respective batched versions of these types) are prescribed. 
- Core protocols (with the exception of `Dataset`) are only required to handle data in batched form (this may change)
- `evaluate` and `predict` functions now exist under `maite.workflows` to separate them from core protocols
- `evaluate` function takes ML components from either image processing or object detection domain (component domain compatibility enforced statically)
- `predict` function permits running model inference without evaluate
- "Interop" functionality (i.e. ability to load maite-wrapped versions of components originating from some third parties) has been rolled back temporarily as core architecture changed.

.. _v0.4.0:

---------------------
0.4.0 - 2024-01-22
---------------------

- bugfix: update maite exception naming and fix typo in import
- feature: added GitHub workflow to build and publish public documentation
- feature: added the use of DatumMetadata in tutorial basic_evaluation.ipynb


.. _v0.3.6:

---------------------
0.3.6 - 2024-01-17
---------------------
   
This release includes the following changes:

- Fix issue with loading datasets from huggingface hub (e.g., cifar10) (bugfix)
- Import dataclass from dataclasses instead of attr (bugfix)
- Remove version constraints for torchmetrics


.. _v0.3.5:

---------------------
0.3.5 - 2024-01-05
---------------------
   
This release includes two new features:

- Add dataset and datum level metadata
- Add model metadata

.. _v0.3.4:

---------------------
0.3.4 - 2023-12-21
---------------------

Publish to pypi.org

.. _v0.3.3:

---------------------
0.3.3 - 2023-12-21
---------------------

Modifying pyproject.toml to obtain better author display representation on pypi

.. _v0.3.2:

---------------------
0.3.2 - 2023-12-21
---------------------

Updated README and pyproject.toml in preparation for pypi publishing (Note: we are still publishing to test.pypi)

.. _v0.3.1:

---------------------
0.3.1 - 2023-12-21
---------------------

Updated docs

- Updated references in docs/README.md and docs/index.rst to reflect current repo name
     - Updated basic_evaluation.md tutorial
     - Implemented default parameters for object detection

- Added .github/pypi_publish.yml directory to automate publishing the repository to online packaging indices (after GitLab is mirrored to GitHub)

.. _v0.3.0:

---------------------
0.3.0 - 2023-12-20
---------------------

This release is the first release after renaming to maite, the changes are follows

- The major change is the rename jatic-toolbox to maite
- It adds headers (copyright, license) to each file
- It adds the Phase-1 protocol and base provider/hub registration system from Quansight.

.. _v0.2.0:

---------------------
0.2.0 - 2023-07-28
---------------------

This release provides a number of updates for usability and stability.


Protocols
---------

Many updates to the names of protocols, usability, and stability.
See reference documentation: https://jatic.pages.jatic.net/cdao/maite/api_reference.html

For an explanation of usage of current protocols see  (*deprecated link*) `explanation/protocols_current`.

For an overview of the future vision of protocols see  (*deprecated link*) `explanation/protocols_vision`.


Documentation
-------------

Explanations, how-tos, and tutorials have been added
to the maite documentation at https://jatic.pages.jatic.net/cdao/maite/.

Additionally, be sure to check reference documentation for examples of usage and API details.
See https://jatic.pages.jatic.net/cdao/maite/api_reference.html.


Model Inference
---------------

Previous release supported dictionary inputs to models.  This release changes the API to support
to a single or collection of arrays.  This is more consistent with model inference APIs utilized
in most deep learning frameworks.

- See image classification model definition: :class:`maite.protocols.ImageClassifier`
- See object detection model definition: :class:`maite.protocols.ObjectDetector`

Additionally, the use of pre-processors and post-processors is kept internal to model inference
rather than explicitly requiring users to manage these steps.  Integration with augmentations and
perturbations is still in development.

Stability and Usability
-----------------------

- Remove 3.7 support and update typing_extensions references
- Pins `torchmetrics < 1.0` due to breaking API changes
- Number of bug fixes and stability improvements
- Improved testing to ensure toolbox protocols and testing utilities work with minimal installation.
- Improve testing coverage for dataset and model loading
- Initial implementation of dataset, model, and metric registries. See  (*deprecated link*) `how_to/named_evaluation`.


.. _v0.1.0:

---------------------
0.1.0 - 2023-05-12
---------------------

This marks the first release of the maite.  We are not yet at a stable `v1.0.0`.  Future release will aim to improving testing and stability of the software for general use.

