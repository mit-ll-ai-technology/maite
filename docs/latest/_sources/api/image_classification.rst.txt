.. meta::
   :description: Reference documentation for components and primitives defined within image classification AI problem.

.. _api_image_classification:

.. currentmodule:: maite.protocols

image classification
====================

We consider the image classification AI problem to encompass models that infer which label (from a predefined set) best applies to a given image.

.. _ic_primitives:

.. TODO: let primitive types (like ic.InputType) reference both `ArrayLike`, and semantic expectations (which aren't checked yet)
..       then uncomment this section in both image_clasification.rst (this file) and object_detection.rst (its sister in doctree)
..       see issue 853

primitives
----------

.. autosummary::
    :caption: primitives
    :toctree: ../generated/
    :template: base.rst

    image_classification.InputType
    image_classification.TargetType
    image_classification.DatumMetadataType
    image_classification.Image
    image_classification.ImgClassification
    image_classification.DatumMetadata

.. _ic_components:

components
----------

.. autosummary::
   :caption: components
   :toctree: ../generated/
   :template: protocol_class.rst

   image_classification.Augmentation
   image_classification.DataLoader
   image_classification.Dataset
   image_classification.FieldwiseDataset
   image_classification.Metric
   image_classification.Model