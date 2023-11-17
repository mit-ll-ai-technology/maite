# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from jatic_toolbox._internals.import_utils import is_hypothesis_available

if is_hypothesis_available():
    from jatic_toolbox._internals.testing.hypothesis import image_data
else:
    raise ImportError(
        "jatic_toolbox.testing.hypothesis requires that hypothesis be installed as a dependency."
    )  # pragma: no cover

__all__ = ["image_data"]
