from typing import Any, Mapping

from jatic_toolbox.protocols import Dataset, SupportsImageClassification, VisionDataset

__all__ = ["HuggingFaceVisionDataset"]


class HuggingFaceVisionDataset(VisionDataset):
    """
    Wrapper for HuggingFace Dataset vision datasets.

    Assumes the dataset has features for an image and a label defined
    by the `datasets.Image` and `datasets.ClassLabel` classes.

    Examples
    --------
    >>> from jatic_toolbox.interop.huggingface.datasets import HuggingFaceVisionDataset
    >>> from datasets import load_dataset
    >>> dataset = load_dataset("mnist", split="train")
    >>> wrapped_dataset = HuggingFaceVisionDataset(dataset)
    """

    def __init__(
        self,
        dataset: Dataset[Mapping[str, Any]],
    ):
        """
        Initialize the HuggingFaceVisionDataset.

        Parameters
        ----------
        dataset : datasets.Dataset
            A dataset from HuggingFace's `datasets` library.

        Raises
        ------
        AssertionError
            If the dataset does not have an image key or a label key.
        """
        import datasets

        assert isinstance(
            dataset, datasets.Dataset
        ), "dataset must be a HuggingFace Dataset"

        self._dataset = dataset
        self.features = dataset.features

        self.image_key: str
        self.label_key: str
        for fname, f in dataset.features.items():
            if isinstance(f, datasets.ClassLabel):
                self.label_key = fname

            elif isinstance(f, datasets.Image):
                self.image_key = fname

        assert self.image_key is not None, "No image key found in dataset"
        assert self.label_key is not None, "No label key found in dataset"

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> SupportsImageClassification:
        data = self._dataset[idx]

        data_dict = SupportsImageClassification(
            image=data[self.image_key],
            label=data[self.label_key],
        )

        for k, v in data.items():
            if k not in (self.image_key, self.label_key):
                data_dict[k] = v

        return data_dict
