from dataclasses import is_dataclass

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given, settings

from jatic_toolbox.interop.huggingface import (
    HuggingFaceImageClassifier,
    HuggingFaceObjectDetector,
)
from jatic_toolbox.protocols import HasLogits, HasObjectDetections, HasProbs

image_strategy = hnp.arrays(float, shape=(30, 30, 3), elements=st.floats(0, 1))


@settings(max_examples=5, deadline=None)
@given(image=image_strategy)
@pytest.mark.parametrize("to_tensor", [True, False])
def test_object_detector(image, to_tensor):
    hf_object_detector = HuggingFaceObjectDetector.from_pretrained(
        model="facebook/detr-resnet-50"
    )

    if to_tensor:
        image = tr.tensor(image)
        dets = hf_object_detector([image])
    else:
        dets = hf_object_detector([image])

    assert is_dataclass(dets)
    assert isinstance(dets, HasObjectDetections)
    assert len(dets.boxes) == len(dets.scores)


# @settings(max_examples=5, deadline=None)
# @given(
#     scores=hnp.arrays(float, (10, 1), elements=st.floats(0, 1)),
# )
# def test_boxes(scores):
#     hf_object_detector = HuggingFaceObjectDetector(model="facebook/detr-resnet-50")
#     output = hf_object_detector._process_scores(scores)
#     assert len(output) == len(scores)
#     assert isinstance(output[0], dict)


@settings(max_examples=5, deadline=None)
@given(image=image_strategy)
@pytest.mark.parametrize("to_tensor", [True, False])
def test_image_classifier(image, to_tensor):
    hf_img_classifier = HuggingFaceImageClassifier.from_pretrained(
        model="microsoft/resnet-50"
    )

    if to_tensor:
        image = tr.tensor(image)

    output = hf_img_classifier([image])

    assert is_dataclass(output)
    assert isinstance(output, HasLogits)
    assert not isinstance(output, HasProbs)
