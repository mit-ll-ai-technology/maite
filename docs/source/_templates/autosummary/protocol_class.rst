{{ fullname | escape | underline}}
.. module.rst template loaded
{# 
(Note: This is a Jinja comment for devs, and doesn't appear in resulting rst files)
protocol classes shouldn't have Attributes/Methods sections outside of what is manually written their class docstring
We can't rely on jinja {% if attributes %} to conditionally write 'Attributes'/'Methods' headings because 
the protocols all inherit from a base generic protocol that does have methods. Just using a separate template.
#}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}