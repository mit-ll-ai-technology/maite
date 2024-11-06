.. meta::
   :description: The changelog for maite, including what's new.

=========
Changelog
=========

This is a record of all past maite releases and what went into them, in reverse 
chronological order.

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

