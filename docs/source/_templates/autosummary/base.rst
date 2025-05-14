{% set parts = fullname.split('.') %}  {# Split the fully qualified name #}
{{ (parts[2:] | join('.')) | escape | underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
