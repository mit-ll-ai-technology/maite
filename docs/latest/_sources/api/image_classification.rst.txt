.. meta::
   :description: Reference documentation for components and primitives defined within image classification AI problem.

.. _api_image_classification:

.. currentmodule:: maite

image classification
====================

We consider the image classification AI problem to encompass training models that infer which label (from a predefined set) best applies to a given image.

.. _ic_primitives:

.. TODO: let primitive types (like ic.InputType) reference both `ArrayLike`, and semantic expectations (which aren't checked yet)
..       then uncomment this section in both image_clasification.rst (this file) and object_detection.rst (its sister in doctree)
..       see issue 853

.. primitives
.. ----------

.. .. autosummary::
..     :caption: primitives
..     :toctree: ../generated/

..     protocols.image_classification.InputType
..     protocols.image_classification.TargetType
..     protocols.image_classification.DatumMetadataType

.. _ic_components:

components
----------

.. autosummary::
   :caption: components
   :toctree: ../generated/
   :template: protocol_class.rst

   protocols.image_classification.Augmentation
   protocols.image_classification.DataLoader
   protocols.image_classification.Dataset
   protocols.image_classification.FieldwiseDataset
   protocols.image_classification.Metric
   protocols.image_classification.Model