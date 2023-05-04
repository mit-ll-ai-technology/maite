import typing as tp
from dataclasses import dataclass

import pytest
import torch as tr
import typing_extensions as tpe
from torch.utils.data import Dataset

import jatic_toolbox
from jatic_toolbox import protocols as pr


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.data = tr.randn(length, size)

    def __getitem__(self, index) -> pr.ImageClassifierData:
        return pr.ImageClassifierData(image=self.data[index], label=0)

    def __len__(self):
        return len(self.data)


class RandomDetectionDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.data = tr.randn(length, size)

    def __getitem__(self, index):
        return dict(
            image=self.data[index],
            objects=dict(bbox=[[0, 0, 1, 1]], label=[[0, 0, 0, 0]]),
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
        self.linear = tr.nn.Linear(10, 1)

    def get_labels(self):
        return [f"label_{i}" for i in range(10)]

    def forward(self, x):
        x = x["image"]
        if self.no_dataclass:
            return self.linear(x)

        if self.with_logits:
            return VisionOutput(logits=self.linear(x))
        else:
            return VisionOutputProbs(probs=self.linear(x).softmax(-1))


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
def test_evaluate(use_progress_bar, with_logits, no_dataclass):
    # TODO: use torchvision coco instead of huggingface?
    data = RandomDataset(10, 10)
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
def test_evaluate_object_detection(use_progress_bar, no_dataclass):
    # TODO: use torchvision coco instead of huggingface?
    data = RandomDetectionDataset(10, 10)
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
