# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from torch import Tensor
from typing_extensions import TypeAlias, TypedDict

from maite.protocols import SupportsImageClassification, VisionDataset

T_co = TypeVar("T_co", covariant=True)

HuggingFaceVisionOuput: TypeAlias = Dict[str, Any]


class Image(TypedDict):
    ...


class ClassLabel(TypedDict):
    names: Sequence[str]
    num_classes: int


@runtime_checkable
class PyTorchVisionDataset(Protocol):
    classes: List[str]

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: Any) -> Tuple[Tensor, Tensor]:
        ...


class TorchVisionDataset(VisionDataset):
    """
    Wrapper for torchvision datasets.

    Parameters
    ----------
    dataset: torchvision.datasets
        The torchvision dataset to wrap.

    Examples
    --------
    >>> from torchvision.datasets import MNIST
    >>> from maite._internals.interop.torchvision.datasets import TorchVisionDataset
    >>> dataset = TorchVisionDataset(MNIST(root="data", download=True))
    >>> len(dataset)
    """

    def __init__(
        self,
        dataset: PyTorchVisionDataset,
    ):
        """Initialize the TorchVisionDataset."""
        self._dataset = dataset
        self.features: Dict[str, Any] = {
            "image": Image(),
            "label": ClassLabel(
                names=dataset.classes, num_classes=len(dataset.classes)
            ),
        }
        self._transform = None

    def set_transform(
        self,
        transform: Callable[[SupportsImageClassification], SupportsImageClassification],
    ) -> None:
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> SupportsImageClassification:
        """
        Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The index of the item to get.

        Returns
        -------
        SupportsImageClassification
            Dictionary of `image` and `label` for each dataset.
        """
        data = self._dataset[idx]

        output = SupportsImageClassification(image=data[0], label=data[1])
        if self._transform is not None:
            output = self._transform(output)

        return output
