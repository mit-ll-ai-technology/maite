from typing import Any, Dict, Sequence, Tuple, TypeVar

from torch import Tensor
from typing_extensions import Protocol, TypeAlias, TypedDict

from jatic_toolbox.protocols import SupportsImageClassification, VisionDataset

T_co = TypeVar("T_co", covariant=True)

HuggingFaceVisionOuput: TypeAlias = Dict[str, Any]


class Image(TypedDict, total=False):
    ...


class ClassLabel(TypedDict, total=False):
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
        return {"image": data[0], "label": data[1]}
