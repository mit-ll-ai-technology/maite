.. meta::
   :description: Reference documentation for MAITE.

.. _maite-reference:

#########
Reference
#########

This page gives an overview of exposed MAITE packages and modules. These include:

* `maite.protocols` - `structural types <https://typing.python.org/en/latest/reference/protocols.html>`_ defining expectations for MAITE :ref:`components and primitives <components_tasks_primitives>` in each AI problem
* `maite.tasks` - core test & evaluation procedures operating on components and primitives
* `maite.interop` - wrappers transforming third-party objects into MAITE components

.. toctree::
   :maxdepth: 2

   api/protocols

.. toctree::
   :maxdepth: 2

   api/tasks

.. toctree::
   :maxdepth: 2

   api/interop


.. tip::
   Check out the :ref:`MAITE Vision explainer <components_tasks_primitives>` for a clear description of components, tasks, and primitives in the context of the MAITE package.

.. warning::
   Contents of `maite._internals` is considered private.
   It contains functionality that is either only used internally, or considered fundamentally more experimental.
   Users should have no expectations of stable functionality from this subpackage.
