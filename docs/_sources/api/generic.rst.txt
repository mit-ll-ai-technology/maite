.. currentmodule:: maite

.. _api_generic:

generic
=======

The structural types defined within protocols.generic provide a helpful stepping stone for 
defining new AI problems. The generic structural types defined permit a specialization via concrete type parameters
that specify expectations for the lower level primitives used in the specific AI problem. 
These primitive types specify the InputType, TargetType, and DatumMetadataType along with
the structural/semantic expectations for each.

.. warning::
   The protocols.generic module is an advanced part of MAITE that 
   provides an extension point for defining new AI problems. Its use requires
   more advanced knowledge of the Python typing system. Further, it should not
   be used to type hint directly, as type hinting against a non-specialized generic
   type effectively bypasses useful type checking of those type parameters. 
   
   Use with care.

.. note::
   Shape and semantic expectations specified for a primitive (e.g., image classification specifies an `ArrayLike`
   input type with `(C, H, W)` shape semantics that is expected by `Dataset`, `DataLoader`, `Model`, and `Augmentation`) 
   are not enforceable by static type checkers. Opt-in runtime verification is being actively explored to 
   provide additional safety beyond what is possible via the Python typing spec.

.. _api_generic_primitives:

primitives
----------

.. autosummary::
   :caption: primitives
   :toctree: ../generated/
   :template: protocol_class.rst

   protocols.ArrayLike
   protocols.DatumMetadata

.. _api_generic_components:

components
----------

.. autosummary::
   :caption: components
   :toctree: ../generated/
   :template: protocol_class.rst

   protocols.generic.Augmentation
   protocols.generic.DataLoader
   protocols.generic.Dataset
   protocols.generic.Metric
   protocols.generic.Model