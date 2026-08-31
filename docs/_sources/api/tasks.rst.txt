.. meta::
   :description: Reference documentation for components and primitives defined within object detection AI task.

.. currentmodule:: maite

.. _api_tasks:

tasks
=====

A procedure that consumes and produces well-defined components is termed a MAITE task. 
All below tasks are valid across AI problems and will use static type checking to validate that
all passed components are structurally compatible.

.. autosummary::
   :caption: tasks
   :toctree: generated/
   :template: protocol_class.rst

   tasks.evaluate
   tasks.predict
   tasks.evaluate_from_predictions
   tasks.augment_dataloader
