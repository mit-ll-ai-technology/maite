import typing as tp
from dataclasses import dataclass

import numpy as np
import pytest
import torch as tr
import typing_extensions as tpe
from PIL import Image
from torch.utils.data import Dataset

import jatic_toolbox
from jatic_toolbox import protocols as pr


class RandomDataset(Dataset):
    def __init__(self, data_type: str, size: int, length: int):
        if data_type == "numpy":
            self.data = tr.randn(length, 3, size, size).numpy()
        elif data_type == "tensor":
            self.data = tr.randn(length, 3, size, size)
        elif data_type == "pillow":
            self.data = [
                Image.fromarray(
                    np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                )
                for _ in range(length)
            ]

    def __getitem__(self, index) -> pr.ImageClassifierData:
        return pr.ImageClassifierData(image=self.data[index], label=0)

    def __len__(self):
        return len(self.data)


class RandomDetectionDataset(Dataset):
    def __init__(self, data_type: str, size: int, length: int):
        if data_type == "numpy":
            self.data = tr.randn(length, 3, size, size).numpy()
        elif data_type == "tensor":
            self.data = tr.randn(length, 3, size, size)
        elif data_type == "pillow":
            self.data = [
                Image.fromarray(
                    np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                )
                for _ in range(length)
            ]

        # self.data = tr.randn(length, size)

    def __getitem__(self, index):
        return dict(
            image=self.data[index],
            objects={"bbox": [[0, 0, 1, 1]], "label": [[0, 0, 0, 0]]},
        )

    def __len__(self):
        return len(self.data)


@dataclass
class VisionOutput:
    logits: tr.Tensor


@dataclass
class VisionOutputProbs:
    probs: tr.Tensor


class VisionModel(tr.nn.Module):
    def __init__(self, with_logits=False, no_dataclass=False):
        super().__init__()
        self.with_logits = with_logits
        self.no_dataclass = no_dataclass
        self.conv2d = tr.nn.Conv2d(3, 10, 1)
        self.avgpool = tr.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = tr.nn.Linear(10, 1)

    def get_labels(self):
        return [f"label_{i}" for i in range(10)]

    def forward(self, x):
        x = x["image"]
        if isinstance(x, list):
            from torchvision.transforms.functional import to_tensor

            x = tr.stack([to_tensor(i) for i in x], dim=0)

        x = self.conv2d(x)
        x = self.avgpool(x)
        x = tr.flatten(x, 1)
        x = self.linear(x)

        if self.no_dataclass:
            return x

        if self.with_logits:
            return VisionOutput(logits=x)
        else:
            return VisionOutputProbs(probs=x.softmax(-1))


@dataclass
class DetectorOutput:
    boxes: tr.Tensor
    labels: tr.Tensor
    scores: tr.Tensor


class DetectionModel(tr.nn.Module):
    def __init__(self, no_dataclass=False):
        super().__init__()
        self.no_dataclass = no_dataclass
        self.linear = tr.nn.Linear(10, 1)

    def get_labels(self):
        return [f"label_{i}" for i in range(10)]

    def forward(self, x) -> DetectorOutput:
        if self.no_dataclass:
            return x

        scores = tr.tensor([0.5])
        bbox = tr.tensor([[0, 0, 1, 1]])
        label = tr.tensor([0])
        return DetectorOutput(boxes=bbox, labels=label, scores=scores)


class Metric:
    def to(self, *args: tp.Any, **kwargs: tp.Any) -> tpe.Self:
        ...

    def compute(self) -> tp.Any:
        return 0.5

    def reset(self):
        ...

    def update(self, *args, **kwargs):
        ...


@pytest.mark.parametrize("use_progress_bar", [True, False])
@pytest.mark.parametrize("with_logits", [True, False])
@pytest.mark.parametrize("no_dataclass", [True, False])
@pytest.mark.parametrize("data_type", ["numpy", "tensor", "pillow"])
def test_evaluate(use_progress_bar, with_logits, no_dataclass, data_type):
    data = RandomDataset(data_type, 10, 10)
    model = VisionModel(with_logits=with_logits, no_dataclass=no_dataclass)
    metric = Metric()

    evaluator = jatic_toolbox.evaluate("image-classification")

    if no_dataclass:
        with pytest.raises(ValueError):
            evaluator(
                model,
                data,
                metric=dict(metric=metric),
                batch_size=4,
                device=0,
                use_progress_bar=use_progress_bar,
            )
        return

    evaluator(
        model,
        data,
        metric=dict(metric=metric),
        batch_size=4,
        device=0,
        use_progress_bar=use_progress_bar,
    )


@pytest.mark.parametrize("use_progress_bar", [True, False])
@pytest.mark.parametrize("no_dataclass", [True, False])
@pytest.mark.parametrize("data_type", ["numpy", "tensor", "pillow"])
def test_evaluate_object_detection(use_progress_bar, no_dataclass, data_type):
    # TODO: use torchvision coco instead of huggingface?
    data = RandomDetectionDataset(data_type, 10, 10)
    model = DetectionModel(no_dataclass)
    metric = Metric()

    evaluator = jatic_toolbox.evaluate("object-detection")

    if no_dataclass:
        with pytest.raises(ValueError):
            evaluator(
                model,
                data,
                metric=dict(metric=metric),
                batch_size=4,
                device=0,
                use_progress_bar=use_progress_bar,
            )
        return

    evaluator(
        model,
        data,
        metric=dict(metric=metric),
        batch_size=4,
        device=0,
        use_progress_bar=use_progress_bar,
    )
