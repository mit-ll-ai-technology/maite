from typing import TYPE_CHECKING, Any, List, Mapping, Sequence

from typing_extensions import TypeGuard

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ArrayLike

from ..import_utils import is_numpy_available, is_pil_available, is_torch_available
from ..protocols.typing import SupportsArray


def to_tensor_list(data: SupportsArray) -> Sequence[ArrayLike]:
    if isinstance(data, Sequence):
        if is_pil_available():
            from PIL import Image

            assert isinstance(data[0], Image.Image) or isinstance(data[0], ArrayLike)
        else:
            assert isinstance(data[0], ArrayLike)
        return data

    elif is_torch_tensor(data):
        if TYPE_CHECKING:
            from torch import Tensor

            assert isinstance(data, Tensor)
        return [x for x in data]

    elif is_numpy_array(data):
        import numpy as np

        if TYPE_CHECKING:
            assert isinstance(data, np.ndarray)
        return [np.asarray(x) for x in data]

    else:
        raise InvalidArgument(f"Unsupported JATIC data type {type(data)}.")


def is_torch_tensor(x) -> bool:
    """
    Tests if `x` is a torch tensor or not.

    Safe to call even if torch is not installed.
    """

    def _is_torch(x):
        import torch

        return isinstance(x, torch.Tensor)

    return False if not is_torch_available() else _is_torch(x)


def is_numpy_array(x) -> bool:
    """Tests if `x` is a numpy array or not."""

    def _is_numpy(x):
        import numpy as np

        return isinstance(x, np.ndarray)

    return False if not is_numpy_available() else _is_numpy(x)


def is_pil_image(x) -> TypeGuard[ArrayLike]:
    """Tests if `x` is a Image array or not."""

    def _is_pil(x):
        from PIL.Image import Image

        return isinstance(x, Image)

    return False if not is_pil_available() else _is_pil(x)


def collate_as_lists(batch: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    keys = list(batch[0].keys())
    values = []
    for k in keys:
        v = [b[k] for b in batch]
        values.append(v)

    return {k: v for k, v in zip(keys, values)}
