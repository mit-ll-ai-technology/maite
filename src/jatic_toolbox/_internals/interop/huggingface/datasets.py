from typing import Any, Callable, Mapping

from jatic_toolbox._internals.protocols import ObjectDetection, SupportsObjectDetection
from jatic_toolbox.protocols import Dataset, SupportsImageClassification, VisionDataset

__all__ = ["HuggingFaceVisionDataset"]


class HuggingFaceWrapper:
    _dataset: Dataset[Mapping[str, Any]]

    def set_transform(
        self, transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ):
        self._dataset.set_transform(transform)


class HuggingFaceVisionDataset(HuggingFaceWrapper, VisionDataset):
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

        if isinstance(dataset, datasets.DatasetDict):
            raise NotImplementedError(
                f"HuggingFaceVisionDataset does not support DatasetDicts.  Pass in one of the available datasets, {dataset.keys()}, instead."
            )

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


class HuggingFaceObjectDetectionDataset(HuggingFaceWrapper, VisionDataset):
    """Wrapper for HuggingFace Dataset object detection datasets."""

    def __init__(
        self,
        dataset: Dataset[Mapping[str, Any]],
        image_key: str = "image",
        objects_key: str = "objects",
        bbox_key: str = "bbox",
        category_key: str = "category",
    ):
        """
        Initialize the HuggingFaceVisionDataset.

        Parameters
        ----------
        dataset : datasets.Dataset
            A dataset from HuggingFace's `datasets` library.
        image_key : str
            The key for the image in the dataset.
        objects_key : str
            The key for the object containing detections in the dataset.
        bbox_key : str
            The key for the bounding box within `objects` in the dataset.
        category_key : str
            The key for the category within `objects` in the dataset.

        Examples
        --------
        >>> from jatic_toolbox.interop.huggingface.datasets import HuggingFaceObjectDetectionDataset
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("open_images", split="train")
        >>> wrapped_dataset = HuggingFaceObjectDetectionDataset(dataset)
        """

        import datasets

        if isinstance(dataset, datasets.DatasetDict):
            raise NotImplementedError(
                f"HuggingFaceVisionDataset does not support DatasetDicts.  Pass in one of the available datasets, {dataset.keys()}, instead."
            )

        assert isinstance(
            dataset, datasets.Dataset
        ), "dataset must be a HuggingFace Dataset"

        self._dataset = dataset
        self.features = dataset.features
        self.image_key = image_key
        self.objects_key = objects_key
        self.bbox_key = bbox_key
        self.category_key = category_key

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> SupportsObjectDetection:
        data = self._dataset[idx]

        image = data[self.image_key]
        objects = data[self.objects_key]

        single_example = False
        if isinstance(objects, dict):
            image = [image]
            objects = [objects]
            single_example = True

        obj_out = []
        for obj in objects:
            if isinstance(obj, dict):
                obj = [obj]

            for o in obj:
                bbox = o[self.bbox_key]
                category = o[self.category_key]
                det = ObjectDetection(bbox=bbox, label=category)
                for k, v in o.items():
                    if k not in (self.bbox_key, self.category_key):
                        det[k] = v
                obj_out.append(det)

        data_dict = SupportsObjectDetection(
            image=image[0] if single_example else image,
            objects=obj_out[0] if single_example else obj_out,
        )

        for k, v in data.items():
            if k not in (self.image_key, self.objects_key):
                data_dict[k] = v

        return data_dict
