# Copyright 2026, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Runtime-verifiable variants of MAITE primitive types (by AI problem).

The task submodules of this package (``image_classification``, ``object_detection``,
``multiobject_tracking``) expose the ``Annotated[Base, Is[...]]`` forms of the MAITE
primitives -- the variants that carry beartype value-based validators and are meant to
be handed to ``maite.spotcheck.spotcheck_component`` for runtime verification.

These forms are NOT usable with ``isinstance`` (an ``Annotated`` alias raises
``TypeError`` under ``isinstance``). Import the plain forms from the regular
``maite.protocols.<task>`` to avoid this exception.

Each submodule requires beartype; importing one without beartype installed emits a
warning and defines nothing (mirroring ``maite.spotcheck``).
"""
