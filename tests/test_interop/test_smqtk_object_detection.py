import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given, settings

from jatic_toolbox._internals.interop.smqtk.object_detection import _MODELS
from jatic_toolbox.interop.smqtk.object_detection import CenterNetVisdrone

image_strategy = hnp.arrays(
    float, shape=(30, 30, 3), elements=st.floats(0, 1, width=32)
)


@settings(max_examples=5, deadline=None)
@pytest.mark.parametrize("models", list(_MODELS.keys()))
@given(image=image_strategy)
@pytest.mark.parametrize("to_tensor", [True, False])
def test_smqtk(models, image, to_tensor):
    object_detector = CenterNetVisdrone(models)

    if to_tensor:
        image = tr.tensor(image)
        with pytest.warns(UserWarning):
            dets = object_detector([image])
    else:
        dets = object_detector([image])

    assert isinstance(dets, list)
    assert len(dets) == 1
    assert hasattr(dets[0], "boxes")
    assert hasattr(dets[0], "scores")
