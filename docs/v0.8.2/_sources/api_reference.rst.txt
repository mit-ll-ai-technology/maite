.. meta::
   :description: Reference documentation for MAITE.

.. _maite-reference:

#########
Reference
#########

Encyclopedia MAITEanica.

.. currentmodule:: maite

protocols
---------

.. autosummary::
   :caption: protocols
   :toctree: generated/
   :template: protocol_class.rst

   protocols.ArrayLike

   protocols.image_classification.Augmentation
   protocols.image_classification.DataLoader
   protocols.image_classification.Dataset
   protocols.image_classification.Metric
   protocols.image_classification.Model

   protocols.object_detection.Augmentation
   protocols.object_detection.ObjectDetectionTarget
   protocols.object_detection.DataLoader
   protocols.object_detection.Dataset
   protocols.object_detection.Metric
   protocols.object_detection.Model

tasks
-----

.. autosummary:: 
   :caption: tasks
   :toctree: generated/
   :template: protocol_class.rst

   tasks.evaluate
   tasks.predict
   tasks.evaluate_from_predictions

interop
-------

.. autosummary::
   :caption: interop
   :toctree: generated/

   interop.metrics.torchmetrics.TMClassificationMetric
   interop.metrics.torchmetrics.TMDetectionMetric
   interop.models.yolo.YoloObjectDetector

utils
-----

.. currentmodule:: maite.utils.validation

.. autosummary::
   :toctree: generated/
   :caption: utils

   check_type
   check_domain
   check_one_of
   chain_validators

errors
------

.. currentmodule:: maite.errors

.. autosummary::
   :toctree: generated/
   :caption: errors

   MaiteException
   InvalidArgument

testing
-------

.. currentmodule:: maite.testing

.. autosummary::
   :caption: testing
   :toctree: generated/

   docs.validate_docstring
   docs.NumpyDocErrorCode
   docs.NumPyDocResults
   pyright.pyright_analyze
   pyright.PyrightOutput
   pyright.list_error_messages
   pytest.cleandir
   project.ModuleScan
   project.get_public_symbols