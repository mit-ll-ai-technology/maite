.. meta::
   :description: Reference documentation for maite.

.. _maite-reference:

#########
Reference
#########

Encyclopedia JATICanica.

All reference documentation includes detailed Examples sections. Please scroll to the 
bottom of any given reference page to see the examples.


.. toctree::

.. currentmodule:: maite


+++++++++
Protocols
+++++++++

ArrayLike
---------

.. autosummary::
   :toctree: generated/

   protocols.ArrayLike
   protocols.SupportsArray

Data Containers
---------------

.. autosummary::
   :toctree: generated/

   protocols.HasDataBoxes
   protocols.HasDataBoxesLabels
   protocols.HasDataImage
   protocols.HasDataLabel
   protocols.HasDataObjects
   protocols.SupportsImageClassification
   protocols.SupportsObjectDetection

Dataset
-------

.. autosummary::
   :toctree: generated/

   protocols.Dataset
   protocols.VisionDataset
   protocols.ObjectDetectionDataset

Model Outputs
-------------

.. autosummary::
   :toctree: generated/

   protocols.HasLogits
   protocols.HasProbs
   protocols.HasScores
   protocols.HasDetectionLogits
   protocols.HasDetectionPredictions
   protocols.HasDetectionProbs

Models
------

.. autosummary::
   :toctree: generated/

   protocols.Model
   protocols.ImageClassifier
   protocols.ObjectDetector

Metrics
-------

.. autosummary::
   :toctree: generated/

   protocols.Metric



++++++++++++++++++++++++++++++++++++++++++
List and Load Datasets, Models and Metrics
++++++++++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   list_datasets
   load_dataset
   list_models
   load_model
   list_metrics
   load_metric

+++++++++++++++++++++++
Workflow Implementation
+++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   evaluate

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

   ToolBoxException
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