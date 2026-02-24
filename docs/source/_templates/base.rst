{% set parts = fullname.split('.') %}  {# Split the fully qualified name #}
{{ parts[-1] | escape | underline }}

.. loaded base.rst remplate

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
