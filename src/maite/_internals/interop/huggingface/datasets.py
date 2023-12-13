# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from maite._internals.import_utils import is_hf_datasets_available
from maite._internals.protocols.typing import ObjectDetectionDataset
from maite.errors import ToolBoxException
from maite.protocols import (
    HasDataBoxesLabels,
    SupportsImageClassification,
    SupportsObjectDetection,
    VisionDataset,
)

if is_hf_datasets_available():
    import datasets

from .typing import HuggingFaceDataset

__all__ = ["HuggingFaceVisionDataset"]


class HuggingFaceVisionDataset(VisionDataset):
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
        >>> from maite.interop.huggingface import HuggingFaceVisionDataset
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("mnist", split="train")
        >>> wrapped_dataset = HuggingFaceVisionDataset(dataset)
        """
        if isinstance(dataset, datasets.DatasetDict):  # pragma: no cover
            raise NotImplementedError(
                f"HuggingFaceVisionDataset does not support DatasetDicts.  Pass in one of the available datasets, {dataset.keys()}, instead."
            )

        self._dataset = dataset
        self._transform = None

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

        if self.image_key == self.label_key:
            raise ToolBoxException(
                f"Image key and label key are the same: {self.image_key}"
            )

        if self.image_key not in dataset.features:
            raise ToolBoxException(
                f"Image key, {self.image_key}, not found in dataset.  Available keys: {dataset.features.keys()}"
            )

        if self.label_key not in dataset.features:
            raise ToolBoxException(
                f"Label key, {self.label_key}, not found in dataset.  Available keys: {dataset.features.keys()}"
            )

    def set_transform(
        self,
        transform: Callable[[SupportsImageClassification], SupportsImageClassification],
    ) -> None:
        self._transform = transform

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

        if self._transform is not None:
            data_dict = self._transform(data_dict)

        return data_dict


class HuggingFaceObjectDetectionDataset(ObjectDetectionDataset):
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
        >>> from maite.interop.huggingface.datasets import HuggingFaceObjectDetectionDataset
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("open_images", split="train")
        >>> wrapped_dataset = HuggingFaceObjectDetectionDataset(dataset)
        """

        if len(set([image_key, objects_key, bbox_key, category_key])) != 4:
            raise ToolBoxException(
                "All keys must be unique.  Keys provided: "
                f"{image_key}, {objects_key}, {bbox_key}, {category_key}"
            )

        if isinstance(dataset, datasets.DatasetDict):
            raise NotImplementedError(
                f"HuggingFaceObjectDetectionDataset does not support DatasetDicts.  Pass in one of the available datasets, {dataset.keys()}, instead."
            )

        self._dataset = dataset
        self._transform = None
        self.image_key = image_key
        self.objects_key = objects_key
        self.bbox_key = bbox_key
        self.category_key = category_key

        if (
            self.image_key not in dataset.features
            or self.objects_key not in dataset.features
        ):
            raise ToolBoxException(
                f"Dataset does not have the expected keys: {self.image_key}, {self.objects_key}"
            )

        if isinstance(dataset.features[self.objects_key], datasets.Sequence):
            feature = dataset.features[self.objects_key].feature
        else:  # pragma: no cover
            feature = dataset.features[self.objects_key][0]

        if self.bbox_key not in feature or self.category_key not in feature:
            raise ToolBoxException(
                f"Dataset does not have the expected keys: {self.bbox_key}, {self.category_key}"
            )

    def set_transform(
        self, transform: Callable[[SupportsObjectDetection], SupportsObjectDetection]
    ) -> None:
        self._transform = transform

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

                single_detection = False
                if np.asarray(bbox).ndim == 1:  # pragma: no cover
                    single_detection = True
                    bbox = [bbox]
                    category = [category]

                det = HasDataBoxesLabels(boxes=bbox, labels=category)
                for k, v in o.items():
                    if k not in (self.bbox_key, self.category_key):
                        if single_detection:  # pragma: no cover
                            v = [v]
                        det[k] = v
                obj_out.append(det)

        data_dict: SupportsObjectDetection = {
            "image": image[0] if single_example else image,
            "objects": obj_out[0] if single_example else obj_out,
        }

        for k, v in data.items():
            if k not in (self.image_key, self.objects_key):
                data_dict[k] = v

        if self._transform is not None:
            data_dict = self._transform(data_dict)

        return data_dict
