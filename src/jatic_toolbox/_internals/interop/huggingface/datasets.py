from typing import TYPE_CHECKING, Optional

from jatic_toolbox._internals.protocols.typing import ObjectDetectionDataset
from jatic_toolbox.protocols import (
    HasDataBoxesLabels,
    SupportsImageClassification,
    SupportsObjectDetection,
    VisionDataset,
)

from .typing import HuggingFaceDataset, HuggingFaceWrapper

__all__ = ["HuggingFaceVisionDataset"]


class HuggingFaceVisionDataset(HuggingFaceWrapper, VisionDataset):
    """
    Wrapper for HuggingFace Dataset vision datasets.

    Assumes the dataset has features for an image and a label defined
    by the `datasets.Image` and `datasets.ClassLabel` classes.
    """

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        label_key: Optional[str] = None,
        image_key: Optional[str] = None,
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

        Examples
        --------
        >>> from jatic_toolbox.interop.huggingface import HuggingFaceVisionDataset
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("mnist", split="train")
        >>> wrapped_dataset = HuggingFaceVisionDataset(dataset)
        """
        import datasets

        if isinstance(dataset, datasets.DatasetDict):  # pragma: no cover
            raise NotImplementedError(
                f"HuggingFaceVisionDataset does not support DatasetDicts.  Pass in one of the available datasets, {dataset.keys()}, instead."
            )

        self._dataset = dataset

        self.image_key: Optional[str] = None
        self.label_key: Optional[str] = None
        if image_key is None or label_key is None:
            for fname, f in dataset.features.items():
                if isinstance(f, datasets.ClassLabel):
                    self.label_key = fname

                elif isinstance(f, datasets.Image):
                    self.image_key = fname

        if image_key is not None:
            self.image_key = image_key

        if label_key is not None:
            self.label_key = label_key

        assert (
            self.image_key != self.label_key
        ), f"Image key and label key are the same: {self.image_key}"
        assert (
            self.image_key in dataset.features
        ), f"Image key, {self.image_key}, not found in dataset.  Available keys: {dataset.features.keys()}"
        assert (
            self.label_key in dataset.features
        ), f"Label key, {self.label_key}, not found in dataset.  Available keys: {dataset.features.keys()}"

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> SupportsImageClassification:
        data = self._dataset[idx]

        if TYPE_CHECKING:
            # type checker doesn't recognize we already checked this above
            assert self.image_key is not None
            assert self.label_key is not None

        data_dict = SupportsImageClassification(
            image=data[self.image_key],
            label=data[self.label_key],
        )

        for k, v in data.items():
            if k not in (self.image_key, self.label_key):
                data_dict[k] = v

        return data_dict


class HuggingFaceObjectDetectionDataset(HuggingFaceWrapper, ObjectDetectionDataset):
    """Wrapper for HuggingFace Dataset object detection datasets."""

    def __init__(
        self,
        dataset: HuggingFaceDataset,
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

        assert (
            len(set([image_key, objects_key, bbox_key, category_key])) == 4
        ), "All keys must be unique"

        import datasets

        if isinstance(dataset, datasets.DatasetDict):
            raise NotImplementedError(
                f"HuggingFaceObjectDetectionDataset does not support DatasetDicts.  Pass in one of the available datasets, {dataset.keys()}, instead."
            )

        self._dataset = dataset
        self.image_key = image_key
        self.objects_key = objects_key
        self.bbox_key = bbox_key
        self.category_key = category_key

        assert (
            self.image_key in dataset.features
        ), f"No image key, {self.image_key}, found in dataset.  Available keys: {dataset.features.keys()}"
        assert (
            self.objects_key in dataset.features
        ), f"No objects key found in dataset: {self.objects_key}"

        if isinstance(dataset.features[self.objects_key], datasets.Sequence):
            assert (
                self.bbox_key in dataset.features[self.objects_key].feature
            ), "No bbox key found in dataset"
            assert (
                self.category_key in dataset.features[self.objects_key].feature
            ), "No category key found in dataset"
        elif isinstance(dataset.features[self.objects_key], (list, tuple)):
            assert (
                self.bbox_key in dataset.features[self.objects_key][0]
            ), "No bbox key found in dataset"
            assert (
                self.category_key in dataset.features[self.objects_key][0]
            ), "No category key found in dataset"

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> SupportsObjectDetection:
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
                det = HasDataBoxesLabels(boxes=bbox, labels=category)
                for k, v in o.items():
                    if k not in (self.bbox_key, self.category_key):
                        det[k] = v
                obj_out.append(det)

        data_dict: SupportsObjectDetection = {
            "image": image[0] if single_example else image,
            "objects": obj_out[0] if single_example else obj_out,
        }

        for k, v in data.items():
            if k not in (self.image_key, self.objects_key):
                data_dict[k] = v

        return data_dict
