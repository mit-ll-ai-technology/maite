from typing import Any, Callable, Dict, Sequence, Tuple, TypeVar

from torch import Tensor
from typing_extensions import Protocol, TypeAlias, TypedDict

from jatic_toolbox.protocols import SupportsImageClassification, VisionDataset

T_co = TypeVar("T_co", covariant=True)

HuggingFaceVisionOuput: TypeAlias = Dict[str, Any]


class Image(TypedDict):
    ...


class ClassLabel(TypedDict):
    names: Sequence[str]
    num_classes: int


class PyTorchVisionDataset(Protocol):
    classes: Sequence[str]
    class_to_idx: Dict[str, int]

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
    >>> from jatic_toolbox._internals.interop.torchvision.datasets import TorchVisionDataset
    >>> dataset = TorchVisionDataset(MNIST(root="data", download=True))
    >>> len(dataset)
    """

    def __init__(
        self,
        dataset: PyTorchVisionDataset,
    ):
        """Initialize the TorchVisionDataset."""
        self.dataset = dataset
        self.features = {
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> SupportsImageClassification:
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
        data = self.dataset[idx]

        output = SupportsImageClassification(image=data[0], label=data[1])
        if self._transform is not None:
            output = self._transform(output)

        return output
