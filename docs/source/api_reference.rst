.. meta::
   :description: Reference documentation for jatic-toolbox.

.. _jatic-toolbox-reference:

#########
Reference
#########

Encyclopedia JATICanica.

All reference documentation includes detailed Examples sections. Please scroll to the 
bottom of any given reference page to see the examples.

.. currentmodule:: jatic_toolbox

.. autosummary::
   :toctree: generated/

   list_datasets
   load_dataset
   list_models
   load_model
   list_metrics
   load_metric
   evaluate


.. currentmodule:: jatic_toolbox.utils.validation

.. autosummary::
   :toctree: generated/

   check_type
   check_domain
   check_one_of
   chain_validators
   

.. currentmodule:: jatic_toolbox.errors

.. autosummary::
   :toctree: generated/

   ToolBoxException
   InvalidArgument

.. currentmodule:: jatic_toolbox.testing.docs

.. autosummary::
   :toctree: generated/

   validate_docstring
   NumpyDocErrorCode
   NumPyDocResults
   

.. currentmodule:: jatic_toolbox.testing.pyright

.. autosummary::
   :toctree: generated/

   pyright_analyze
   PyrightOutput
   list_error_messages


.. currentmodule:: jatic_toolbox.testing.pytest

.. autosummary::
   :toctree: generated/

   cleandir


.. currentmodule:: jatic_toolbox.testing.project

.. autosummary::
   :toctree: generated/

   ModuleScan
   get_public_symbols