# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import UserDict
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import torch as tr
from torch import Tensor

from ..protocols.type_guards import is_list_of_type, is_typed_dict
from ..protocols.typing import ArrayLike, HasDataImage, SupportsArray
from .utils import is_pil_image


class BaseModel:
    preprocessor: Optional[Callable[[Union[HasDataImage, SupportsArray]], HasDataImage]]
    parameters: Callable[[], Iterable[Tensor]]

    def _process_inputs(
        self, data: Union[HasDataImage, Sequence[ArrayLike], ArrayLike]
    ) -> Tuple[Union[tr.Tensor, Sequence[tr.Tensor]], Optional[Any]]:
        """Supports different types of inputs for a model."""

        # If single PIL image, list of one and handle below
        if is_pil_image(data):
            data = [data]

        # If single array, try to convert to tensor and stack and just return
        if not isinstance(data, (dict, UserDict, Sequence)):
            device = next(iter(self.parameters())).device
            images = tr.as_tensor(data).to(device)
            if images.ndim == 3:
                images = images.unsqueeze(0)

            return images, None

        # Extract any data out of a dictionary
        if isinstance(data, (dict, UserDict)):
            assert is_typed_dict(
                data, HasDataImage
            ), "Dictionary data must contain 'image' key."
            images = data["image"]

            if not isinstance(images, Sequence):
                images = [images]
        else:
            images = data

        assert isinstance(images, Sequence)

        # If list of PIL images, convert to tensor
        target_size = None
        if is_pil_image(images[0]):
            assert self.preprocessor is not None, "Model does not have a preprocessor."
            processed_images = self.preprocessor(images)

            images = processed_images["image"]
            device = next(iter(self.parameters())).device
            if isinstance(images, Sequence):
                assert isinstance(
                    images[0], tr.Tensor
                ), f"Invalid type {type(images[0])}"
                images = [tr.as_tensor(i).to(device) for i in images]
            else:
                assert isinstance(images, tr.Tensor)
                images = images.to(device)

            if "target_size" in processed_images:
                target_size = processed_images.get("target_size", None)

        # If list of numpy arrays, convert to tensor
        if is_list_of_type(images, ArrayLike):
            device = next(iter(self.parameters())).device
            images = [tr.as_tensor(i).to(device) for i in images]

        # For list of Tensors, try to stack into a single tensor
        if is_list_of_type(images, tr.Tensor):
            first_item = images[0]
            shape_first = first_item.shape  # type: ignore
            if all([shape_first == item.shape for item in images]):
                images = tr.stack(images)
            else:
                images = images

        assert is_list_of_type(images, tr.Tensor) or isinstance(images, tr.Tensor)
        return images, target_size
