# %% [markdown]
### Object Detection
#
# We obtain a HuggingFace model ("devonho/detr-resnet-50_finetuned_cppe5"),
# dataset (cppe5), metric and wrap each as MAITE-compliant implementers.
# We can then run the evaluate function.

# %% [markdown]
# ## Setup
# %%

# required installs include transformers, torchmetrics, pycocotools

from datasets import DatasetDict, load_dataset
from torchmetrics.detection.iou import IntersectionOverUnion
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# import albumentations

# %% [markdown]
# ## Load HuggingFace Dataset
# %%

# get cppe-5 dataset
cppe5 = load_dataset("cppe-5")
assert isinstance(
    cppe5, DatasetDict
)  # type-narrow from broad returned type of load_dataset

# %% [markdown]
# ## Load HuggingFace Model (with image processor)
# %%
# Select a particular checkpoint (for model and pre-training processing)
checkpoint = "devonho/detr-resnet-50_finetuned_cppe5"

# initialize id2label and label2id to instantiate model
categories = cppe5["train"].features["objects"].feature["category"].names

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

Cppe5_OD_Model_HF = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Load ImageProcessor
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# %% [markdown]
# ## Load augmentation
# %%
#
# #(Unused for now)
# transform = albumentations.Compose(
#     [
#         albumentations.Resize(640, 480),
#         albumentations.RandomBrightnessContrast(p=1.0),
#     ],
#     bbox_params=albumentations.BboxParams(format="pascal_voc", label_fields=["category"]),
# )

# %% [markdown]
# ## Setup for maite wrapping
# %%

# standard library imports
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

# common 3rd party imports
import numpy as np
import torch

import maite.protocols.object_detection as od

# maite-specific imports
from maite.protocols import ArrayLike
from maite.protocols.object_detection import ObjectDetectionTarget, TargetBatchType

# %% [markdown]
# ## Wrap model
# %%


# Define a dataclass to implement TargetBatchType for object detection
@dataclass
class Cppe5_ObjectDetectionTarget:
    boxes: torch.Tensor  # shape (N,4), x0, y0, x1, y1 format
    scores: torch.Tensor  # shape (N,)
    labels: torch.Tensor  # shape (N,)


class MaiteCppe5_OD:
    def __init__(self):
        ...

    def __call__(self, input: ArrayLike) -> Sequence[Cppe5_ObjectDetectionTarget]:
        # break ArrayLike into an acceptable input for HF preprocessing
        # (a sequence of np.ndarrays)
        images: List[np.ndarray] = [i for i in np.array(input)]
        orig_image_sizes = torch.Tensor([image.shape[0:2] for image in images])
        with torch.no_grad():
            inputs = image_processor(images, return_tensors="pt")
            outputs = Cppe5_OD_Model_HF(**inputs)
            results = image_processor.post_process_object_detection(
                outputs, target_sizes=orig_image_sizes, threshold=0.5
            )

        targets_out: List[Cppe5_ObjectDetectionTarget] = []
        for result in results:
            # convert boxes from xywh (coco) format to xyxy format
            # note x0,y0,x1,y1 are N-length tensors, where N is number of detections
            x0: torch.Tensor = result["boxes"][:, 0]
            y0: torch.Tensor = result["boxes"][:, 1]
            x1: torch.Tensor = x0 + result["boxes"][:, 2]
            y1: torch.Tensor = y0 + result["boxes"][:, 3]

            # return simple dataclass with required fields
            targets_out.append(
                Cppe5_ObjectDetectionTarget(
                    boxes=torch.stack([x0, y0, x1, y1], dim=1),
                    labels=result["labels"],
                    scores=result["scores"],
                )
            )

        return targets_out


# %% [markdown]
# ## Wrap dataset
# %%


class Maite_Cppe5_Dataset:
    def __init__(
        self,
        cppe5_dataset,
        partition: Literal["train", "test"] = "test",
        max_len: Optional[int] = None,
    ):
        self.partition = partition
        if max_len is not None:
            self._dataset: DatasetDict = cppe5_dataset[self.partition].select(
                [i for i in range(max_len)]
            )
        else:
            self._dataset: DatasetDict = cppe5_dataset[self.partition]

    def __getitem__(
        self, ind: int
    ) -> Tuple[ArrayLike, Cppe5_ObjectDetectionTarget, Dict[str, Any]]:
        datum = self._dataset[ind]
        objects = datum["objects"]

        # type-narrow beyond promised return signature of _dataset.__getitem__
        # We are confident these type guarantees will be maintained using this
        # particular dataset, and if they are not, we want to know via runtime
        # failures
        assert isinstance(datum, dict)
        assert isinstance(objects, dict) and not isinstance(objects, list)

        # `datum` is dictionary at runtime and guarantees no type for values.
        # This precludes operating on the values retrieved without either
        # casting or type-ignoring

        # get ArrayLike from image
        model_input = np.array(datum["image"])
        if model_input.shape[2] != 3:
            # TODO: issue warning here, no images shouldn't have 4 color channels
            model_input = model_input[:, :, :3]

        n_detections = len(objects["bbox"])

        # convert natively xywh-format boxes into xyxy-format
        boxes: np.ndarray = np.zeros((n_detections, 4))
        for i, box in enumerate(objects["bbox"]):
            x, y, w, h = box[0], box[1], box[2], box[3]
            boxes[i, :] = x, y, x + w, y + h

        labels: np.ndarray = np.array(objects["category"])
        scores: np.ndarray = np.ones(
            (n_detections,)
        )  # scores are all 1.0 since we have ground-truth

        datum_gt = Cppe5_ObjectDetectionTarget(
            boxes=torch.as_tensor(boxes),
            labels=torch.as_tensor(labels),
            scores=torch.as_tensor(scores),
        )

        # native HF format for datapoint is a dictionary, just populate datum metadata
        # with this full dictionary
        datum_metadata: dict = datum

        return (model_input, datum_gt, datum_metadata)

    def __len__(self):
        return len(self._dataset)


# %% [markdown]
# ## Wrap metrics
# %%

# TODO: either inherit from a single class to save rewriting similar methods,
# OR use a decorator that takes the raw class instance and makes it maite-
# compliant.


# create iou metric based off of torchmetrics
class IouMetric:
    def __init__(self):
        self._raw_iou_metric: Callable[
            [List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]],
            Dict[str, Any],
        ] = IntersectionOverUnion(
            box_format="xyxy",
            iou_threshold=None,
            class_metrics=False,
            respect_labels=True,
        )

    # create utility function to convert ObjectDetectionTarget_impl type to what
    # the type expected by torchmetrics IntersectionOverUnion metric
    @staticmethod
    def to_tensor_dict(tgt: ObjectDetectionTarget) -> Dict[str, torch.Tensor]:
        """
        Convert an ObjectDetectionTarget_impl into a dictionary expected internally by
        raw `update` method of raw torchmetrics method
        """
        out = {
            "boxes": torch.as_tensor(tgt.boxes),
            "scores": torch.as_tensor(tgt.scores),
            "labels": torch.as_tensor(tgt.labels),
        }

        return out

    def update(
        self,
        preds: TargetBatchType,
        targets: TargetBatchType,
    ) -> None:
        # convert to natively-typed from of preds/targets
        preds_tm = [self.to_tensor_dict(pred) for pred in preds]
        targets_tm = [self.to_tensor_dict(tgt) for tgt in targets]
        self._raw_iou_metric.update(preds_tm, targets_tm)

    def compute(self) -> Dict[str, Any]:
        return self._raw_iou_metric.compute()

    def reset(self) -> None:
        self._raw_iou_metric.reset()


# create mean average precision metric based off of torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class mAP_Metric:
    def __init__(self):
        self._raw_map_metric: Callable[
            [List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]],
            Dict[str, Any],
        ] = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=list(np.linspace(0.5, 0.95, 10)),
            rec_thresholds=None,
        )

    # create utility function to convert ObjectDetectionTarget_impl type to what
    # the type expected by torchmetrics IntersectionOverUnion metric
    @staticmethod
    def to_tensor_dict(tgt: ObjectDetectionTarget) -> Dict[str, torch.Tensor]:
        """
        Convert an ObjectDetectionTarget_impl into a dictionary expected internally by
        raw `update` method of raw torchmetrics method
        """
        out = {
            "boxes": torch.as_tensor(tgt.boxes),
            "scores": torch.as_tensor(tgt.scores),
            "labels": torch.as_tensor(tgt.labels),
        }

        return out

    def update(
        self,
        preds: TargetBatchType,
        targets: TargetBatchType,
    ) -> None:
        # convert to natively-typed from of preds/targets
        preds_tm = [self.to_tensor_dict(pred) for pred in preds]
        targets_tm = [self.to_tensor_dict(tgt) for tgt in targets]
        self._raw_map_metric.update(preds_tm, targets_tm)

    def compute(self) -> Dict[str, Any]:
        return self._raw_map_metric.compute()

    def reset(self) -> None:
        self._raw_map_metric.reset()


# %% [markdown]
# ## Wrap augmentation
# (empty for now)
# %% [markdown]
# ## Run evaluate
# %%

from maite.workflows import evaluate

MAX_LEN = 10

metric_output, preds, aug_data = evaluate(
    model=Cppe5_OD_Model_HF,
    dataset=Maite_Cppe5_Dataset(
        cppe5_dataset=cppe5, partition="train", max_len=MAX_LEN
    ),
    metric=mAP_Metric(),
    augmentation=None,
    return_augmented_data=True,
    return_preds=True,
)

from pprint import pprint as pp

pp(metric_output)

# %% [markdown]
# ## Visualize evaluate results
# %%
from random import randint

import matplotlib.patches as patches
import matplotlib.pyplot as plt

ind = randint(0, len(aug_data) - 1)


def plot_example(
    aug_data: Sequence[
        Tuple[ArrayLike, Sequence[ObjectDetectionTarget], Dict[str, Any]]
    ],
    ind,
):
    """
    Function assumes a batch_size of 1!
    """
    ex_img, ex_target, ex_md = aug_data[ind]

    fig, ax = plt.subplots()
    ax.imshow(ex_img[0])

    # Draw a red rectangle for each bounding box in ground truth lables
    for box, label in zip(ex_target[0].boxes, ex_target[0].labels):
        gt_rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        text = ax.text(box[0], box[3], id2label[label], color="red")
        ax.add_patch(gt_rect)

    # Draw a green rectangle for each bounding box in predicted labels
    for box, label in zip(preds[ind][0].boxes, preds[ind][0].labels):
        pred_rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="green",
            facecolor="none",
        )
        text = ax.text(box[0], box[3], id2label[int(label)], color="green")
        ax.add_patch(pred_rect)


[plot_example(aug_data, i) for i in range(MAX_LEN)]
