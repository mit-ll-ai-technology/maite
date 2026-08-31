.. meta::
   :description: Reference documentation for components and primitives defined within multiobject tracking AI problem.

.. _api_multi_object_tracking:

.. currentmodule:: maite.protocols

multiobject tracking
====================

We consider the multi-object AI problem to encompass models that infer bounding boxes, labels (from a predefined set), scores, and tracks for objects in a video.

.. _mot_primitives:

.. TODO: let primitive types (like multiobject_tracking.InputType) reference both `ArrayLike`, and semantic expectations (which aren't checked yet)
..       then uncomment this section in both image_clasification.rst (this file) and object_detection.rst (its sister in doctree)
..       see issue 853

primitives
----------

.. autosummary::
    :caption: primitives
    :toctree: ../generated/
    :template: base.rst

    multiobject_tracking.InputType
    multiobject_tracking.TargetType
    multiobject_tracking.DatumMetadataType
    multiobject_tracking.VideoStream
    multiobject_tracking.MultiobjectTrackingTarget
    multiobject_tracking.DatumMetadata
    multiobject_tracking.SingleFrameObjectTrackingTarget
    multiobject_tracking.VideoFrame


.. _mot_components:

components
----------

.. autosummary::
   :toctree: ../generated/
   :caption: components
   :template: protocol_class.rst

   multiobject_tracking.Augmentation
   multiobject_tracking.DataLoader
   multiobject_tracking.Dataset
   multiobject_tracking.FieldwiseDataset
   multiobject_tracking.Metric
   multiobject_tracking.Model
