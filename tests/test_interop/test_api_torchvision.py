from unittest import mock

import pytest
import torch as tr
import torchvision.models
from hypothesis import given, strategies as st

import jatic_toolbox
from jatic_toolbox._internals.interop.torchvision.datasets import TorchVisionDataset
from jatic_toolbox.interop.torchvision import (
    TorchVisionClassifier,
    TorchVisionObjectDetector,
)
from jatic_toolbox.protocols import HasLogits, HasObjectDetections
from tests.common.torchvision import (
    get_test_object_detection_model,
    get_test_vision_dataset,
    get_test_vision_model,
)


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


@given(with_processor=st.booleans())
def test_tv_load_vision_model(with_processor):
    mock_model_weights, mock_model = get_test_vision_model()

    with mock.patch.object(
        torchvision.models._api, "get_model_weights", return_value=mock_model_weights
    ), mock.patch.object(torchvision.models._api, "get_model", return_value=mock_model):
        model_out = jatic_toolbox.load_model(
            provider="torchvision",
            task="image-classification",
            model_name="test",
            with_processor=with_processor,
        )
        assert isinstance(model_out, TorchVisionClassifier)

        out = model_out(tr.randn(1, 3, 10, 10))
        assert isinstance(out, HasLogits)


@given(
    with_processor=st.booleans(),
)
def test_tv_load_object_detection_model(with_processor):
    mock_model_weights, mock_model = get_test_object_detection_model()

    with mock.patch.object(
        torchvision.models._api, "get_model_weights", return_value=mock_model_weights
    ), mock.patch.object(torchvision.models._api, "get_model", return_value=mock_model):
        model_out = jatic_toolbox.load_model(
            provider="torchvision",
            task="object-detection",
            model_name="test",
            with_processor=with_processor,
        )
        assert isinstance(model_out, TorchVisionObjectDetector)

        out = model_out(tr.randn(1, 3, 10, 10))
        assert isinstance(out, HasObjectDetections)
