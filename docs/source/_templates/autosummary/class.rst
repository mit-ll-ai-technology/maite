{% set parts = fullname.split('.') %}  {# Split the fully qualified name #}
{{ parts[-1] | escape | underline }}

.. loaded class.rst template

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{# 
(Note: This is a Jinja comment for devs, and doesn't appear in resulting rst files)
We are electing not to include autodoc introspection into templates because we are 
putting so much information into a class' docstring that is interpretable by numpydoc 
and language servers that populate hover text.
#}