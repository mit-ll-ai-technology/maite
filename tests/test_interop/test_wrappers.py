from dataclasses import dataclass, is_dataclass
from typing import Any, Iterable, Sequence, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given
from numpy.typing import NDArray
from typing_extensions import Self

from jatic_toolbox._internals.interop.huggingface.typing import (
    BatchFeature,
    HFProcessedDetection,
)
from jatic_toolbox._internals.interop.smqtk.object_detection import (
    AxisAlignedBoundingBox,
    SMQTKAxisAlignedBoxes,
)
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.interop.huggingface import (
    HuggingFaceImageClassifier,
    HuggingFaceObjectDetector,
)
from jatic_toolbox.interop.smqtk import CenterNet
from jatic_toolbox.protocols import (
    ArrayLike,
    HasDetectionLogits,
    HasLogits,
    HasObjectDetections,
)

image_strategy = hnp.arrays(float, shape=(5, 30, 30, 3), elements=st.floats(0, 1))


#
# HuggingFace-Like Models
#
@dataclass
class Output:
    pred_boxes: ArrayLike
    logits: ArrayLike


class Model:
    def __init__(self) -> None:
        super().__init__()
        self.device: Union[int, str] = 0

    def to(self, device: Union[str, int]) -> Self:
        return self

    def __call__(
        self, pixel_values: Union[ArrayLike, Sequence[ArrayLike]], **kwargs: Any
    ) -> HasDetectionLogits:
        if isinstance(pixel_values, Sequence):
            pixel_values = tr.stack([tr.as_tensor(p) for p in pixel_values], 0)
        else:
            pixel_values = tr.as_tensor(pixel_values)
        pred_boxes = tr.rand(pixel_values.shape[0], 50, 4)
        logits = tr.randn(pixel_values.shape[0], 50, 5)
        return Output(pred_boxes, logits)


class Processor:
    def __call__(
        self,
        images: Sequence[ArrayLike],
        return_tensors: Union[bool, str] = False,
        **kwargs: Any,
    ) -> BatchFeature[ArrayLike]:
        if isinstance(images, Sequence):
            pix_vals = tr.stack([tr.as_tensor(p) for p in images], 0)
        else:
            pix_vals = tr.as_tensor(images)
        return BatchFeature(pixel_values=pix_vals)

    def post_process_object_detection(
        self, outputs: HasDetectionLogits, threshold: float, target_sizes: Any
    ) -> HFProcessedDetection:
        out = []
        for logits, boxes in zip(outputs.logits, outputs.pred_boxes):
            probs = tr.softmax(logits, dim=1)
            pred_index = probs.argmax(-1)
            out.append(dict(scores=probs[pred_index], labels=pred_index, boxes=boxes))
        return out


#
# SMQTK-Like Object Detectors
#
class SMQTKTest:
    def __call__(self, img_iter: Iterable[NDArray[Any]]) -> SMQTKAxisAlignedBoxes:
        out = []
        for i in range(len(list(img_iter))):
            out_i = []
            for i in range(10):
                out_i.append((AxisAlignedBoundingBox([0, 1], [0, 1]), {0: 0.1}))
            out.append(out_i)
        return out


@pytest.mark.parametrize(
    "model, output_type",
    [
        (HuggingFaceObjectDetector(Model(), Processor()), HasObjectDetections),
        (HuggingFaceImageClassifier(Model(), Processor()), HasLogits),
        (CenterNet(SMQTKTest()), HasObjectDetections),
    ],
)
@given(image=image_strategy, to_tensor=st.booleans(), as_list=st.booleans())
def test_wrappers(model, output_type, image, to_tensor, as_list):
    if to_tensor:
        image = tr.tensor(image)

    if as_list:
        image = [i for i in image]

    out = model(image)

    assert is_dataclass(out)
    assert isinstance(out, output_type)
    for k, v in out.__dict__.items():
        assert len(v) == len(image)


def everything_except(excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


@pytest.mark.parametrize(
    "model",
    [
        HuggingFaceObjectDetector(Model(), Processor()),
        HuggingFaceImageClassifier(Model(), Processor()),
        CenterNet(SMQTKTest()),
    ],
)
@pytest.mark.parametrize(
    "T",
    [
        int,
        complex,
        float,
        str,
    ],
)
def test_wrappers_raise(model, T):
    with pytest.raises((InvalidArgument, AssertionError)):
        model(T(1.0))
