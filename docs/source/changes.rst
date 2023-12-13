.. meta::
   :description: The changelog for maite, including what's new.

=========
Changelog
=========

This is a record of all past maite releases and what went into them, in reverse 
chronological order.

.. _v0.2.0:

---------------------
0.2.0 - 2023-07-28
---------------------

This release provides a number of updates for usability and stability.


Protocols
---------

Many updates to the names of protocols, usability, and stability.
See reference documentation: https://jatic.pages.jatic.net/cdao/maite/api_reference.html

For an explanation of usage of current protocols see :doc:`explanation/protocols_current`.

For an overview of the future vision of protocols see :doc:`explanation/protocols_vision`.


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
- Initial implementation of dataset, model, and metric registries. See :doc:`how_to/named_evaluation`.


.. _v0.1.0:

---------------------
0.1.0 - 2023-05-12
---------------------

This marks the first release of the maite.  We are not yet at a stable `v1.0.0`.  Future release will aim to improving testing and stability of the software for general use.

