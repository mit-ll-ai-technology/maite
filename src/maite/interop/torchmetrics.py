# Copyright 2025, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import warnings

from maite._internals.import_utils import is_torchmetrics_available

if is_torchmetrics_available():
    from maite._internals.interop.metrics.torchmetrics_detection import (
        TMDetectionMetric,
    )

    __all__ = ["TMDetectionMetric"]
else:
    warnings.warn(
        "The TorchMetrics wrapper requires the torchmetrics package, "
        "which can be installed with the command: `pip install maite[torchmetrics]`."
    )
