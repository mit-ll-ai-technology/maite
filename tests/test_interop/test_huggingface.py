from dataclasses import is_dataclass
from typing import TYPE_CHECKING

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import torch as tr
from hypothesis import given, settings

from jatic_toolbox.interop.huggingface import (
    HuggingFaceImageClassifier,
    HuggingFaceObjectDetector,
)
from jatic_toolbox.protocols import HasLogits, HasObjectDetections, HasProbs

image_strategy = hnp.arrays(float, shape=(5, 30, 30, 3), elements=st.integers(0, 1))


@settings(max_examples=5, deadline=None)
@given(image=image_strategy, to_tensor=st.booleans(), as_list=st.booleans())
def test_object_detector(image, to_tensor, as_list):
    hf_object_detector = HuggingFaceObjectDetector.from_pretrained(
        model="facebook/detr-resnet-50"
    )

    if to_tensor:
        image = tr.tensor(image)

    if as_list:
        image = [x for x in image]

    dets = hf_object_detector(image)

    assert is_dataclass(dets)
    assert isinstance(dets, HasObjectDetections)
    assert len(dets.boxes) == len(dets.scores)


@settings(max_examples=5, deadline=None)
@given(image=image_strategy, to_tensor=st.booleans(), as_list=st.booleans())
def test_image_classifier(image, to_tensor, as_list):
    hf_img_classifier = HuggingFaceImageClassifier.from_pretrained(
        model="microsoft/resnet-50"
    )

    if TYPE_CHECKING:
        import numpy as np

        assert isinstance(image, np.ndarray)

    if to_tensor:
        image = tr.tensor(image)

    if as_list:
        image = [x for x in image]

    output = hf_img_classifier(image)

    assert is_dataclass(output)
    assert isinstance(output, HasLogits)
    assert not isinstance(output, HasProbs)
