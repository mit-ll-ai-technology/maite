.. meta::
   :description: Reference documentation for components and primitives defined within object detection AI problem.

.. currentmodule:: maite

.. _api_object_detection:

object detection
================

We consider the object detection AI problem to encompass training models to infer bounding boxes, labels (from a predefined label set), and scores for objects in an image.

.. TODO: let primitive types (like od.InputType) reference both `ArrayLike`, and semantic expectations (which aren't checked yet)
..       then uncomment this section in both object_detection.rst (this file) and image_classification.rst (its sister in doctree)
..       see issue 853

.. _od_primitives:

.. primitives
.. ----------

.. .. autosummary::
..     :caption: primitives
..     :toctree: ../generated/

..     protocols.object_detection.InputType
..     protocols.object_detection.TargetType
..     protocols.object_detection.DatumMetadataType

.. _od_components: 

components
----------

.. autosummary::
   :caption: protocols
   :toctree: ../generated/
   :template: protocol_class.rst

   protocols.object_detection.Augmentation
   protocols.object_detection.ObjectDetectionTarget
   protocols.object_detection.DataLoader
   protocols.object_detection.Dataset
   protocols.object_detection.FieldwiseDataset
   protocols.object_detection.Metric
   protocols.object_detection.Model