# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

import maite
import maite.protocols as pr
from maite._internals.import_utils import (
    is_pil_available,
    is_torch_available,
    requires_hf_transformers,
    requires_torchvision,
)
from maite._internals.interop import registry
from maite._internals.interop.huggingface import api as hf_api
from maite.errors import InvalidArgument


def test_errors_list_models():
    with pytest.raises(InvalidArgument):
        maite.list_models(provider="_dummy")  # type: ignore


@pytest.mark.parametrize(
    ("task", "provider"),
    [(None, None), ("image-classification", None), ("image-classification", "dummy")],
)
def test_errors_load_model(task, provider):
    with pytest.raises(InvalidArgument):
        maite.load_model(model_name="_dummy", task=task, provider=provider)


@requires_hf_transformers
@pytest.mark.parametrize(
    ("task", "provider"),
    [(None, None), ("image-classification", None), ("image-classification", "dummy")],
)
def test_errors_hf_load_model(task, provider):
    with pytest.raises(InvalidArgument):
        hf_api.HuggingFaceAPI().load_model(model_name="_dummy", task=task)


@requires_hf_transformers
def test_load_model_from_registry(mocker):
    from ..common import huggingface as hf_common

    data = maite.list_models()
    assert all(x == y for x, y in zip(data, registry.MODEL_REGISTRY.keys()))

    feature_loader = "transformers.AutoFeatureExtractor.from_pretrained"
    model_loader = "transformers.AutoModelForImageClassification.from_pretrained"
    processor, model = hf_common.get_test_vision_model()
    mocker.patch(feature_loader, return_value=processor)
    mocker.patch(model_loader, return_value=model)
    maite.load_model(model_name="vit_for_cifar10")


@requires_hf_transformers
@pytest.mark.parametrize("task", [None, "image-classification", "bad_task"])
def test_errors_hf_load_model_from_hub(task):
    with pytest.raises(InvalidArgument):
        maite.load_model(model_name="_dummy", provider="huggingface", task=task)


@requires_torchvision
@pytest.mark.parametrize("task", [None, "image-classification", "bad_task"])
def test_errors_tv_load_model_from_hub(task):
    with pytest.raises(InvalidArgument):
        maite.load_model(model_name="_dummy", provider="torchvision", task=task)


@requires_hf_transformers
@pytest.mark.parametrize(
    "kwargs", [dict(filter_str="resnet18"), dict(model_name="resnet-50")]
)
def test_hf_list_models(kwargs):
    models = maite.list_models(provider="huggingface", **kwargs)
    assert issubclass(type(models), list)
    assert len(list(models)) > 0


@requires_hf_transformers
@pytest.mark.parametrize("task", ["image-classification", "object-detection"])
def test_hf_models(mocker, task):
    from ..common import huggingface as hf_common

    if task == "image-classification":
        processor, model = hf_common.get_test_vision_model()
        protocol = pr.ImageClassifier
        feature_loader = "transformers.AutoFeatureExtractor.from_pretrained"
        model_loader = "transformers.AutoModelForImageClassification.from_pretrained"
    else:
        processor, model = hf_common.get_test_object_detection_model()
        protocol = pr.ObjectDetector
        feature_loader = "transformers.AutoImageProcessor.from_pretrained"
        model_loader = "transformers.AutoModelForObjectDetection.from_pretrained"

    mocker.patch(feature_loader, return_value=processor)
    mocker.patch(model_loader, return_value=model)

    model = maite.load_model(
        provider="huggingface",
        task=task,
        model_name="test",
    )
    assert isinstance(model, protocol)


@requires_torchvision
@pytest.mark.parametrize(
    "task, filter_str",
    [("image-classification", "resnet18"), ("object-detection", None), (None, None)],
)
def test_tv_list_models(task, filter_str):
    models = maite.list_models(provider="torchvision", filter_str=filter_str, task=task)
    assert issubclass(type(models), list)
    assert len(list(models)) > 0


@requires_torchvision
@pytest.mark.parametrize("task", ["image-classification", "object-detection"])
def test_tv_load_models(mocker, task):
    from ..common import torchvision as tv_common

    if task == "image-classification":
        mock_model_weights, mock_model = tv_common.get_test_vision_model()
        protocol = pr.ImageClassifier
    else:
        mock_model_weights, mock_model = tv_common.get_test_object_detection_model()
        protocol = pr.ObjectDetector

    mocker.patch(
        "torchvision.models._api.get_model_weights", return_value=mock_model_weights
    )
    mocker.patch("torchvision.models._api.get_model", return_value=mock_model)

    model_out = maite.load_model(
        provider="torchvision",
        task="object-detection",
        model_name="test",
    )
    assert isinstance(model_out, protocol)


#
# Data Inputs and Model Outputs
#


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

    if img_type == "pillow" and is_pil_available():
        from PIL import Image

        from maite._internals.interop.utils import is_pil_image

        array = Image.fromarray((array * 255).astype("uint8")).convert("RGB")
        assert is_pil_image(array)
    elif img_type == "numpy":
        array = array.astype(np.float32).transpose(2, 0, 1)
        assert isinstance(array, pr.ArrayLike)
    elif img_type == "tensor" and is_torch_available():
        import torch as tr

        array = tr.tensor(array.transpose(2, 0, 1), dtype=tr.float)
        assert isinstance(array, pr.ArrayLike)
    else:
        raise ValueError(f"Unknown img_type: {img_type}")

    return array


@requires_hf_transformers
@pytest.mark.parametrize("task", ["image-classification", "object-detection"])
def test_huggingface_get_labels_sorted(task):
    from ..common import huggingface as hf_common

    if task == "image-classification":
        from maite._internals.interop.huggingface.image_classifier import (
            HuggingFaceImageClassifier,
        )

        processor, model = hf_common.get_test_vision_model()
        model = HuggingFaceImageClassifier(model, processor)  # type: ignore
    else:
        from maite._internals.interop.huggingface.object_detection import (
            HuggingFaceObjectDetector,
        )

        processor, model = hf_common.get_test_object_detection_model()
        model = HuggingFaceObjectDetector(model, processor)  # type: ignore

    assert isinstance(model, pr.Model)
    assert hasattr(model, "get_labels")
    labels = model.get_labels()

    id_map = model.model.config.id2label
    assert isinstance(id_map, dict)
    assert all(labels[k] == id_map[k] for k in id_map.keys())


@requires_torchvision
@pytest.mark.parametrize("task", ["image-classification", "object-detection"])
def test_tv_get_labels_sorted(task):
    from ..common import torchvision as tv_common

    if task == "image-classification":
        from maite._internals.interop.torchvision.torchvision import (
            TorchVisionClassifier,
        )

        weights, model = tv_common.get_test_vision_model()
        labels = weights["DEFAULT"].meta["categories"]
        model = TorchVisionClassifier(model, weights["DEFAULT"].transforms(), labels)
    else:
        from maite._internals.interop.torchvision.torchvision import (
            TorchVisionObjectDetector,
        )

        weights, model = tv_common.get_test_object_detection_model()
        labels = weights["DEFAULT"].meta["categories"]
        model = TorchVisionObjectDetector(
            model, weights["DEFAULT"].transforms(), labels
        )

    assert isinstance(model, pr.Model)
    assert hasattr(model, "get_labels")
    labels = model.get_labels()

    weights_labels = weights["DEFAULT"].meta["categories"]
    assert all(labels[k] == weights_labels[k] for k in range(len(model._labels)))  # type: ignore


@requires_hf_transformers
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
@pytest.mark.parametrize(
    "task, kwargs, model_kwargs",
    [
        ("image-classification", {}, {}),
        ("image-classification", {}, {"top_k": 100}),
        ("object-detection", {"output_as_list": False}, {"threshold": None}),
        ("object-detection", {"output_as_list": True}, {"threshold": None}),
        ("object-detection", {"output_as_list": False}, {"threshold": 0.5}),
        ("object-detection", {"output_as_list": True}, {"threshold": 0.5}),
    ],
)
def test_huggingface_inputs(task, data, img_type, input_type, model_kwargs, kwargs):
    import torch as tr

    from ..common import huggingface as hf_common

    if task == "image-classification":
        from maite._internals.interop.huggingface.image_classifier import (
            HuggingFaceImageClassifier,
        )

        processor, model = hf_common.get_test_vision_model(**kwargs)
        model = HuggingFaceImageClassifier(model, processor, **model_kwargs)  # type: ignore

        if "top_k" in model_kwargs and model_kwargs["top_k"] is not None:
            output_protocol = pr.HasScores
        else:
            output_protocol = pr.HasProbs
    else:
        from maite._internals.interop.huggingface.object_detection import (
            HuggingFaceObjectDetector,
        )

        processor, model = hf_common.get_test_object_detection_model(**kwargs)
        post_processor = getattr(processor, "post_process_object_detection", None)
        model = HuggingFaceObjectDetector(model, processor, post_processor, **model_kwargs)  # type: ignore

        if "threshold" in model_kwargs and model_kwargs["threshold"] is not None:
            output_protocol = pr.HasDetectionPredictions
        else:
            output_protocol = pr.HasDetectionLogits

    array = _draw_data(data, img_type)

    with tr.inference_mode():
        model_input = None
        if input_type is None:
            model_input = array
        elif input_type == "list":
            model_input = [array]
        elif input_type == "dict":
            model_input = pr.HasDataImage(image=array)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        output, target_sizes = model._process_inputs(model_input)
        assert isinstance(output, tr.Tensor)

        output = model(array)
        assert isinstance(output, output_protocol)


@requires_torchvision
@pytest.mark.parametrize("task", ["image-classification", "object-detection"])
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
def test_torchvision_inputs(task, data, img_type, input_type):
    import torch as tr

    from ..common import torchvision as tv_common

    if task == "image-classification":
        from maite._internals.interop.torchvision.torchvision import (
            TorchVisionClassifier,
        )

        weights, model = tv_common.get_test_vision_model()
        model = TorchVisionClassifier(model, weights["DEFAULT"].transforms())
        output_protocol = pr.HasLogits
    else:
        from maite._internals.interop.torchvision.torchvision import (
            TorchVisionObjectDetector,
        )

        weights, model = tv_common.get_test_object_detection_model()
        model = TorchVisionObjectDetector(model, weights["DEFAULT"].transforms())
        output_protocol = pr.HasDetectionPredictions

    array = _draw_data(data, img_type)

    with tr.inference_mode():
        model_input = None
        if input_type is None:
            model_input = array
        elif input_type == "list":
            model_input = [array]
        elif input_type == "dict":
            model_input = pr.HasDataImage(image=array)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        output, target_sizes = model._process_inputs(model_input)
        assert isinstance(output, tr.Tensor)

        output = model(array)
        assert isinstance(output, output_protocol)
