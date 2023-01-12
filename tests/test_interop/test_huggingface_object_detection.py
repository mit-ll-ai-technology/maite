import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given, settings

from jatic_toolbox.interop.huggingface.object_detection import (
    HuggingFaceBoundingBox,
    HuggingFaceObjectDetector,
)
from jatic_toolbox.protocols.object_detection import ObjectDetectionOutput

image_strategy = hnp.arrays(float, shape=(30, 30, 3), elements=st.floats(0, 1))


@settings(max_examples=5, deadline=None)
@given(box=hnp.arrays(float, shape=(4,), elements=st.floats(-10, 10)))
def test_huggingface_bbox(box):
    bbox = HuggingFaceBoundingBox(box)
    assert hasattr(bbox, "min_vertex")
    assert hasattr(bbox, "max_vertex")
    assert bbox.min_vertex == [box[0], box[1]]
    assert bbox.max_vertex == [box[2], box[3]]


@settings(max_examples=5, deadline=None)
@given(image=image_strategy)
@pytest.mark.parametrize("to_tensor", [True, False])
def test_object_detector(image, to_tensor):
    hf_object_detector = HuggingFaceObjectDetector(model="facebook/detr-resnet-50")

    if to_tensor:
        image = tr.tensor(image)
        with pytest.warns(UserWarning):
            dets = hf_object_detector([image])
    else:
        dets = hf_object_detector([image])

    assert isinstance(dets, list)
    assert len(dets) == 1
    assert hasattr(dets[0], "boxes")
    assert hasattr(dets[0], "scores")


@settings(max_examples=5, deadline=None)
@given(
    boxes=hnp.arrays(float, (10, 4), elements=st.floats(0, 1)),
    scores=hnp.arrays(float, (10, 1), elements=st.floats(0, 1)),
)
@pytest.mark.parametrize("to_list", [True, False])
def test_hf2jatic(boxes, scores, to_list):
    if to_list:
        boxes = boxes.tolist()
        scores = scores.tolist()

    hf_object_detector = HuggingFaceObjectDetector(model="facebook/detr-resnet-50")
    output = hf_object_detector._hfboxes_to_jatic(boxes, scores)

    assert isinstance(output, ObjectDetectionOutput)
    assert len(list(output.boxes)) == len(boxes)
    assert isinstance(list(output.boxes)[0], HuggingFaceBoundingBox)
    assert len(list(output.scores)) == len(scores)
    assert isinstance(list(output.scores)[0], dict)
