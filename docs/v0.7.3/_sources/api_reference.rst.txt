.. meta::
   :description: Reference documentation for MAITE.

.. _maite-reference:

#########
Reference
#########

Encyclopedia MAITEanica.

COMING SOON! All reference documentation includes detailed Examples sections. Please scroll to the 
bottom of any given reference page to see the examples.

.. toctree::

.. currentmodule:: maite

+++++++++
Protocols
+++++++++

Data, Datasets, & DataLoaders
-----------------------------

.. autosummary::
   :template: class_no_autosummary_attrs.rst
   :toctree: generated/

   protocols.ArrayLike

   protocols.image_classification.Dataset
   protocols.image_classification.DataLoader

   protocols.object_detection.ObjectDetectionTarget
   protocols.object_detection.Dataset
   protocols.object_detection.DataLoader

Models
------

.. autosummary::
   :toctree: generated/

   protocols.image_classification.Model
   protocols.object_detection.Model

Augmentations
-------------

.. autosummary::
   :toctree: generated/

   protocols.image_classification.Augmentation
   protocols.object_detection.Augmentation

Metrics
-------

.. autosummary::
   :toctree: generated/

   protocols.image_classification.Metric
   protocols.object_detection.Metric

+++++++++
Workflows
+++++++++

.. autosummary::
   :toctree: generated/

   workflows.evaluate
   workflows.predict

+++++++++++++++++++++++++++++
Validation and Error Handling
+++++++++++++++++++++++++++++

.. currentmodule:: maite.utils.validation

.. autosummary::
   :toctree: generated/

   check_type
   check_domain
   check_one_of
   chain_validators
   

.. currentmodule:: maite.errors

.. autosummary::
   :toctree: generated/

   MaiteException
   InvalidArgument

+++++++++++++
Testing Tools
+++++++++++++

.. currentmodule:: maite.testing.docs

.. autosummary::
   :toctree: generated/

   validate_docstring
   NumpyDocErrorCode
   NumPyDocResults
   

.. currentmodule:: maite.testing.pyright

.. autosummary::
   :toctree: generated/

   pyright_analyze
   PyrightOutput
   list_error_messages


.. currentmodule:: maite.testing.pytest

.. autosummary::
   :toctree: generated/

   cleandir


.. currentmodule:: maite.testing.project

.. autosummary::
   :toctree: generated/

   ModuleScan
   get_public_symbols