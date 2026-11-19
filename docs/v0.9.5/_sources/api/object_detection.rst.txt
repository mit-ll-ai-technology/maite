.. meta::
   :description: Reference documentation for components and primitives defined within object detection AI problem.

.. currentmodule:: maite.protocols

.. _api_object_detection:

object detection
================

We consider the object detection AI problem to encompass models to infer bounding boxes, labels (from a predefined label set), and scores for objects in an image.

.. TODO: let primitive types (like od.InputType) reference both `ArrayLike`, and semantic expectations (which aren't checked yet)
..       then uncomment this section in both object_detection.rst (this file) and image_classification.rst (its sister in doctree)
..       see issue 853

.. _od_primitives:

primitives
----------

.. autosummary::
    :caption: primitives
    :toctree: ../generated/
    :template: base.rst

    object_detection.InputType
    object_detection.TargetType
    object_detection.DatumMetadataType
    object_detection.Image
    object_detection.ObjectDetectionTarget
    object_detection.DatumMetadata

.. _od_components: 

components
----------

.. autosummary::
   :caption: protocols
   :toctree: ../generated/
   :template: protocol_class.rst

   object_detection.Augmentation
   object_detection.DataLoader
   object_detection.Dataset
   object_detection.FieldwiseDataset
   object_detection.Metric
   object_detection.Model