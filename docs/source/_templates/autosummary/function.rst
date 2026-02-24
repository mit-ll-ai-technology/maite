{% set parts = fullname.split('.') %}  {# Split the fully qualified name #}
{{ parts[-1] | escape | underline }}

.. loaded function.rst template

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}