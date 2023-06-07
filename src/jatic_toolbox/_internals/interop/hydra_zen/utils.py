from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Callable, Type

from ..torchvision.datasets import PyTorchVisionDataset


def get_dataclass_docstring(cfg: Any, dtype: Type) -> str:
    # doc-ignore: EX01
    """
    Create a docstring for a dataclass.

    Parameters
    ----------
    cfg : Dataclass
        The dataclass to create a docstring for.
    dtype : str
        The type of dataclass.

    Returns
    -------
    str
        The docstring.
    """
    assert is_dataclass(cfg)

    params = "\n    ".join(
        [
            f"{f.name} : {f.type} = {f.default if not f.default == MISSING else 'REQUIRED'}"
            for f in fields(cfg)
        ]
    )
    doc = f"""
    Dataclass configuration object for a "{dtype.__name__}".

    To instantiate use `hydra_zen.instantiate`.

    A default of `REQUIRED` indicates that no default has been given
    and the user must provide a value.

    Parameters
    ----------
    {params}
    """

    doc += f"\n\nSee Also\n--------\n{dtype.__doc__}"

    return doc


def get_torchvision_dataset(dataset_name: str) -> Callable[..., PyTorchVisionDataset]:
    from torchvision import datasets

    return getattr(datasets, dataset_name)
