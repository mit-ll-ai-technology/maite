# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence, Union

from torch import nn

from ..base_model import BaseModel
from .typing import (
    HuggingFaceObjectDetectionPostProcessor,
    HuggingFaceProcessor,
    HuggingFaceWithDetection,
    HuggingFaceWithLogits,
)


class BaseHFModel(nn.Module, BaseModel):
    """
    Base class for HuggingFace models.

    Parameters
    ----------
    model : HuggingFaceWithLogits | HuggingFaceWithDetection
        The HuggingFace model.
    processor : HuggingFaceProcessor | None
        The HuggingFace processor.
    post_processor : HuggingFaceObjectDetectionPostProcessor | None
        The HuggingFace post processor.

    Methods
    -------
    get_labels()
        Return the labels of the model.
    """

    def __init__(
        self,
        model: Union[HuggingFaceWithLogits, HuggingFaceWithDetection],
        processor: Optional[HuggingFaceProcessor] = None,
        post_processor: Optional[HuggingFaceObjectDetectionPostProcessor] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._processor = processor
        self._post_processor = post_processor
        self._labels = [
            model.config.id2label[k] for k in sorted(model.config.id2label.keys())
        ]

    def get_labels(self) -> Sequence[str]:
        """
        Return the labels of the model.

        Returns
        -------
        Sequence[str]
            The labels of the model.

        Examples
        --------
        >>> hf_model.get_labels()
        """
        return self._labels
