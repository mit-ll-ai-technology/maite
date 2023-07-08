.. meta::
   :description: The changelog for jatic-toolbox, including what's new.

=========
Changelog
=========

This is a record of all past jatic-toolbox releases and what went into them, in reverse 
chronological order.

.. _v0.2.0rc:

---------------------
0.2.0rc1 - 2023-07-07
---------------------

This release drops support for Python 3.7, improves protocols and usability of available provider datasets, models, and evalulate.


Release Highlights
------------------
- Simplifies interfaces to models
- Remove 3.7 support and update typing_extensions references
- Pins `torchmetrics < 1.0` due to breaking API changes
- Updates and improves protocols for usability
- `evaluate` will now pass through PIL Images when no `preprocessor` is provided
- More explicit check on availability of HuggingFace's datasets and transformers
- Interop support with hydra-zen.
- Update documentation
- fixes bug in HuggingFace `list_models`
- Fixes CI/CD errors



.. _v0.1.0:

---------------------
0.1.0 - 2023-05-12
---------------------

This marks the first release of the jatic-toolbox.  We are not yet at a stable `v1.0.0`.  Future release will aim to improving testing and stability of the software for general use.

