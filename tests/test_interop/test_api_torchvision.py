from unittest import mock

import pytest
import torch as tr
import torchvision.models
from hypothesis import given, strategies as st

import jatic_toolbox
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.interop.torchvision import (
    TorchVisionClassifier,
    TorchVisionDataset,
    TorchVisionObjectDetector,
)
from jatic_toolbox.protocols import HasDetectionScorePredictions, HasLogits
from jatic_toolbox.testing.hypothesis import image_data
from tests.common.torchvision import (
    get_test_object_detection_model,
    get_test_vision_dataset,
    get_test_vision_model,
)


def test_tv_list_datasets():
    datasets = jatic_toolbox.list_datasets(provider="torchvision")
    assert issubclass(type(datasets), list)
    assert len(datasets) == 70


def test_tv_list_models():
    models = jatic_toolbox.list_models(provider="torchvision", filter_str="resnet18")
    assert issubclass(type(models), list)
    assert len(models) == 1


@given(task=st.text(min_size=1))
def test_tv_load_dataset_unsupported_task(task):
    with pytest.raises(ValueError):
        jatic_toolbox.load_dataset(
            provider="torchvision", task=task, dataset_name="test"
        )


@given(task=st.text(min_size=1))
def test_tv_load_model_unsupported_task(task):
    with pytest.raises(ValueError):
        jatic_toolbox.load_model(provider="torchvision", task=task, model_name="test")


@given(use_split=st.booleans(), has_split=st.booleans(), has_train=st.booleans())
def test_tv_load_image_classification_dataset(use_split, has_split, has_train):
    # Create a mock dataset object
    mock_dataset = get_test_vision_dataset(has_split, has_train)

    # Patch the load_dataset function to return the mock dataset object
    with mock.patch.object(
        jatic_toolbox._internals.interop.torchvision.api,
        "_get_torchvision_dataset",
        return_value=mock_dataset,
    ):
        kw = {}

        if use_split:
            kw = {"split": "train"}

            if not has_split and not has_train:
                with pytest.raises(TypeError):
                    jatic_toolbox.load_dataset(
                        provider="torchvision",
                        task="image-classification",
                        dataset_name="test",
                        **kw,
                    )
                return

        dataset = jatic_toolbox.load_dataset(
            provider="torchvision",
            task="image-classification",
            dataset_name="test",
            **kw,
        )
        assert type(dataset) == TorchVisionDataset
        assert len(dataset) == len(dataset._dataset)

        def transform(x):
            x["foo"] = [1]
            return x

        dataset.set_transform(transform)

        temp = dataset[0]
        assert type(temp) == dict
        assert "image" in temp
        assert "label" in temp
        assert "foo" in temp


@pytest.mark.parametrize(
    "task, loader",
    [
        (TorchVisionClassifier, get_test_vision_model),
        (TorchVisionObjectDetector, get_test_object_detection_model),
    ],
)
@given(data=st.composite(image_data)(), image_as_dict=st.booleans())
def test_tv_vision_processors(task, loader, data, image_as_dict):
    weights, model = loader()

    with pytest.raises(InvalidArgument):
        hf_model = task(model)
        hf_model.preprocessor([dict(image=data)])

    hf_model = task(model, weights["DEFAULT"].transforms())

    if image_as_dict:
        features = hf_model.preprocessor([dict(image=data)])
    else:
        features = [hf_model.preprocessor([data])]
    assert len(features) == 1
    assert isinstance(features[0], dict)
    assert "image" in features[0]

    output = hf_model({"image": [data] * 10})
    assert isinstance(output, (HasLogits, HasDetectionScorePredictions))


@pytest.mark.parametrize("image_as_dict", [None, "image", "pixel_values", "foo"])
def test_tv_load_vision_model(image_as_dict):
    mock_model_weights, mock_model = get_test_vision_model()

    with mock.patch.object(
        torchvision.models._api, "get_model_weights", return_value=mock_model_weights
    ), mock.patch.object(torchvision.models._api, "get_model", return_value=mock_model):
        model_out = jatic_toolbox.load_model(
            provider="torchvision",
            task="image-classification",
            model_name="test",
        )
        assert isinstance(model_out, TorchVisionClassifier)

        data = tr.randn(1, 3, 10, 10)
        if image_as_dict is not None:
            data = {image_as_dict: data}

        if image_as_dict == "foo":
            with pytest.raises(InvalidArgument):
                out = model_out(data)
            return

        out = model_out(data)
        assert isinstance(out, HasLogits)


@pytest.mark.parametrize("image_as_dict", [None, "image", "pixel_values", "foo"])
def test_tv_load_object_detection_model(image_as_dict):
    mock_model_weights, mock_model = get_test_object_detection_model()

    with mock.patch.object(
        torchvision.models._api, "get_model_weights", return_value=mock_model_weights
    ), mock.patch.object(torchvision.models._api, "get_model", return_value=mock_model):
        model_out = jatic_toolbox.load_model(
            provider="torchvision",
            task="object-detection",
            model_name="test",
        )
        assert isinstance(model_out, TorchVisionObjectDetector)

        data = tr.randn(1, 3, 10, 10)

        if image_as_dict is not None:
            data = {image_as_dict: data}

        if image_as_dict == "foo":
            with pytest.raises(InvalidArgument):
                out = model_out(data)
            return

        out = model_out(data)

        assert isinstance(out, HasDetectionScorePredictions)
