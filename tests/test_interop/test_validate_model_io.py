import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
import torch as tr
from hypothesis import given, settings
from PIL import Image

import jatic_toolbox
from jatic_toolbox import protocols as pr
from jatic_toolbox._internals.import_utils import (
    is_hf_transformers_available,
    is_torchvision_available,
)
from jatic_toolbox._internals.interop.utils import is_pil_image


@pytest.fixture(scope="session")
def hf_vision_model():
    model = jatic_toolbox.load_model(
        provider="huggingface",
        task="image-classification",
        model_name="farleyknight/mnist-digit-classification-2022-09-04",
    )
    return model.eval()  # type: ignore


@pytest.fixture(scope="session")
def tv_vision_model():
    model = jatic_toolbox.load_model(
        provider="torchvision", task="image-classification", model_name="resnet18"
    )
    return model.eval()  # type: ignore


@pytest.fixture(scope="session")
def hf_detector_model():
    model = jatic_toolbox.load_model(
        provider="huggingface",
        task="object-detection",
        model_name="hustvl/yolos-tiny",
    )
    return model.eval()  # type: ignore


@pytest.fixture(scope="session")
def tv_detector_model():
    model = jatic_toolbox.load_model(
        provider="torchvision",
        task="object-detection",
        model_name="ssdlite320_mobilenet_v3_large",
    )
    return model.eval()  # type: ignore


def _draw_data(data, img_type):
    array = data.draw(
        hnp.arrays(
            shape=(224, 224, 3),
            dtype=float,
            elements=st.floats(
                allow_infinity=False, allow_nan=False, min_value=0, max_value=1
            ),
        )
    )

    if img_type == "pillow":
        array = Image.fromarray((array * 255).astype("uint8")).convert("RGB")
        assert is_pil_image(array)
    elif img_type == "numpy":
        array = array.astype(np.float32).transpose(2, 0, 1)
        assert isinstance(array, pr.ArrayLike)
    elif img_type == "tensor":
        array = tr.tensor(array.transpose(2, 0, 1), dtype=tr.float)
        assert isinstance(array, pr.ArrayLike)
    else:
        raise ValueError(f"Unknown img_type: {img_type}")

    return array


@pytest.mark.skipif(
    not is_hf_transformers_available(),
    reason="HuggingFace transfomers is not installed.",
)
@pytest.mark.usefixtures("hf_detector_model")
@pytest.mark.parametrize(
    "img_type",
    [
        "pillow",
        "numpy",
        "tensor",
    ],
)
@pytest.mark.parametrize(
    "input_type",
    [
        None,
        "list",
        "dict",
    ],
)
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_huggingface_object_detection(hf_detector_model, data, img_type, input_type):
    array = _draw_data(data, img_type)

    with tr.no_grad():
        if input_type is None:
            output = hf_detector_model(array)
        elif input_type == "list":
            output = hf_detector_model([array])
        elif input_type == "dict":
            output = hf_detector_model({"image": array})
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        assert isinstance(output, pr.HasDetectionPredictions)


@pytest.mark.skipif(
    not is_hf_transformers_available(),
    reason="HuggingFace `transformers` is not installed.",
)
@pytest.mark.usefixtures("hf_vision_model")
@pytest.mark.parametrize(
    "img_type",
    [
        "pillow",
        "numpy",
        "tensor",
    ],
)
@pytest.mark.parametrize(
    "input_type",
    [
        None,
        "list",
        "dict",
    ],
)
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_huggingface_image_classification(hf_vision_model, data, img_type, input_type):
    array = _draw_data(data, img_type)

    with tr.no_grad():
        if input_type is None:
            output = hf_vision_model(array)
        elif input_type == "list":
            output = hf_vision_model([array])
        elif input_type == "dict":
            output = hf_vision_model({"image": array})
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    assert isinstance(output, pr.HasProbs)


@pytest.mark.skipif(
    not is_torchvision_available(), reason="TorchVision is not installed."
)
@pytest.mark.usefixtures("tv_vision_model")
@pytest.mark.parametrize(
    "img_type",
    [
        "pillow",
        "numpy",
        "tensor",
    ],
)
@pytest.mark.parametrize(
    "input_type",
    [
        None,
        "list",
        "dict",
    ],
)
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_torchvision_image_classification(tv_vision_model, data, img_type, input_type):
    array = _draw_data(data, img_type)

    with tr.no_grad():
        if input_type is None:
            output = tv_vision_model(array)
        elif input_type == "list":
            output = tv_vision_model([array])
        elif input_type == "dict":
            output = tv_vision_model({"image": array})
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    assert isinstance(output, pr.HasLogits)


@pytest.mark.skipif(
    not is_torchvision_available(), reason="TorchVision is not installed."
)
@pytest.mark.usefixtures("tv_detector_model")
@pytest.mark.parametrize(
    "img_type",
    [
        "pillow",
        "numpy",
        "tensor",
    ],
)
@pytest.mark.parametrize(
    "input_type",
    [
        None,
        "list",
        "dict",
    ],
)
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_torchvision_object_detection(tv_detector_model, data, img_type, input_type):
    array = _draw_data(data, img_type)

    with tr.no_grad():
        if input_type is None:
            output = tv_detector_model(array)
        elif input_type == "list":
            output = tv_detector_model([array])
        elif input_type == "dict":
            output = tv_detector_model({"image": array})
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        assert isinstance(output, pr.HasDetectionPredictions)
