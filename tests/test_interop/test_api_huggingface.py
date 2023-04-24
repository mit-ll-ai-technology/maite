from unittest import mock

import datasets
import pytest
import torch as tr
from hypothesis import given, strategies as st
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
)

import jatic_toolbox
from jatic_toolbox._internals.interop.huggingface.datasets import (
    HuggingFaceObjectDetectionDataset,
    HuggingFaceVisionDataset,
)
from jatic_toolbox._internals.interop.huggingface.image_classifier import (
    HuggingFaceImageClassifier,
)
from jatic_toolbox._internals.interop.huggingface.object_detection import (
    HuggingFaceObjectDetector,
)
from jatic_toolbox.protocols import HasLogits, HasObjectDetections
from tests.common.huggingface import (
    get_test_detection_dataset,
    get_test_object_detection_model,
    get_test_vision_dataset,
    get_test_vision_model,
)


@given(task=st.text(min_size=1))
def test_hf_load_dataset_unsupported_task(task):
    with pytest.raises(ValueError):
        jatic_toolbox.load_dataset(
            provider="huggingface", task=task, dataset_name="test"
        )


@given(task=st.text(min_size=1))
def test_hf_load_model_unsupported_task(task):
    with pytest.raises(ValueError):
        jatic_toolbox.load_model(provider="huggingface", task=task, model_name="test")


def test_hf_load_dataset_raises_warning(mocker):
    a_dataset = get_test_vision_dataset(image_key="image", label_key="label")
    mock_dataset = dict(train=a_dataset)
    mocker.patch.object(datasets, "load_dataset", return_value=mock_dataset)

    with pytest.warns(UserWarning):
        jatic_toolbox.load_dataset(
            provider="huggingface", task=None, dataset_name="test"
        )

    with pytest.warns(UserWarning):
        jatic_toolbox.load_dataset(
            provider="huggingface", task="image-classification", dataset_name="test"
        )


@pytest.mark.parametrize(
    "image_key, label_key",
    [(None, "label"), ("image", None), (None, None), ("image", "image")],
)
def test_hf_load_dataset_unsupported_vision_keys(image_key, label_key):
    # Create a mock dataset object
    mock_dataset = get_test_vision_dataset(image_key=image_key, label_key=label_key)

    # Patch the load_dataset function to return the mock dataset object
    with mock.patch.object(datasets, "load_dataset", return_value=mock_dataset):
        with pytest.raises(AssertionError):
            jatic_toolbox.load_dataset(
                provider="huggingface",
                task="image-classification",
                dataset_name="test",
                image_key=image_key,
                label_key=label_key,
            )


@pytest.mark.parametrize(
    "image_key, object_key, bbox_key, category_key",
    [
        (None, "objects", "bbox", "category"),
        ("image", None, "bbox", "category"),
        ("image", "objects", None, "category"),
        ("image", "objects", "bbox", None),
        (None, None, None, None),
        ("image", "image", "bbox", "category"),
    ],
)
def test_hf_load_dataset_unsupported_detection_keys(
    image_key, object_key, bbox_key, category_key
):
    # Create a mock dataset object
    mock_dataset = get_test_detection_dataset(
        image_key=image_key,
        object_key=object_key,
        bbox_key=bbox_key,
        category_key=category_key,
    )

    with mock.patch.object(datasets, "load_dataset", return_value=mock_dataset):
        with pytest.raises(AssertionError):
            jatic_toolbox.load_dataset(
                provider="huggingface",
                task="object-detection",
                dataset_name="test",
                image_key=image_key,
                objects_key=object_key,
                bbox_key=bbox_key,
                category_key=category_key,
            )


@given(
    image_key=st.text(min_size=1),
    label_key=st.text(min_size=1),
    use_keys=st.booleans(),
)
def test_hf_load_image_classification_dataset(image_key, label_key, use_keys):
    # Create a mock dataset object
    mock_dataset = get_test_vision_dataset(image_key=image_key, label_key=label_key)

    # Patch the load_dataset function to return the mock dataset object
    with mock.patch.object(datasets, "load_dataset", return_value=mock_dataset):
        # Call the load_dataset function and assert that it returns the mocked dataset
        key_kwargs = (
            {"image_key": image_key, "label_key": label_key} if use_keys else {}
        )

        if len(set([image_key, label_key])) != 2:
            with pytest.raises(AssertionError):
                jatic_toolbox.load_dataset(
                    provider="huggingface",
                    task="image-classification",
                    dataset_name="test",
                    **key_kwargs,
                )
            return

        dataset = jatic_toolbox.load_dataset(
            provider="huggingface",
            task="image-classification",
            dataset_name="test",
            **key_kwargs,
        )
        assert type(dataset) == HuggingFaceVisionDataset
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


@given(
    image_key=st.text(min_size=1),
    object_key=st.text(min_size=1),
    bbox_key=st.text(min_size=1),
    category_key=st.text(min_size=1),
    objects_as_list=st.booleans(),
)
def test_hf_load_object_detection_dataset(
    image_key, object_key, bbox_key, category_key, objects_as_list
):
    # Create a mock dataset object
    mock_dataset = get_test_detection_dataset(
        image_key=image_key,
        object_key=object_key,
        bbox_key=bbox_key,
        category_key=category_key,
        objects_as_list=objects_as_list,
    )

    with mock.patch.object(datasets, "load_dataset", return_value=mock_dataset):
        if len(set([image_key, object_key, bbox_key, category_key])) != 4:
            with pytest.raises(AssertionError):
                jatic_toolbox.load_dataset(
                    provider="huggingface",
                    task="object-detection",
                    dataset_name="test",
                    image_key=image_key,
                    objects_key=object_key,
                    bbox_key=bbox_key,
                    category_key=category_key,
                )
            return

        # Call the load_dataset function and assert that it returns the mocked dataset
        dataset = jatic_toolbox.load_dataset(
            provider="huggingface",
            task="object-detection",
            dataset_name="test",
            image_key=image_key,
            objects_key=object_key,
            bbox_key=bbox_key,
            category_key=category_key,
        )
        assert type(dataset) == HuggingFaceObjectDetectionDataset
        assert len(dataset) == len(dataset._dataset)

        temp = dataset[0]
        assert type(temp) == dict
        assert "image" in temp
        assert "objects" in temp
        assert "label" in temp["objects"]
        assert "bbox" in temp["objects"]


@given(with_processor=st.booleans())
def test_hf_load_vision_model(with_processor):
    processor, model = get_test_vision_model()

    with mock.patch.object(
        AutoFeatureExtractor, "from_pretrained", return_value=processor
    ), mock.patch.object(
        AutoModelForImageClassification, "from_pretrained", return_value=model
    ):
        model_out = jatic_toolbox.load_model(
            provider="huggingface",
            task="image-classification",
            model_name="test",
            with_processor=with_processor,
        )
        assert isinstance(model_out, HuggingFaceImageClassifier)

        out = model_out(tr.randn(1, 3, 10, 10))
        assert isinstance(out, HasLogits)


@given(
    with_processor=st.booleans(),
    with_post_processor=st.booleans(),
    output_as_list=st.booleans(),
)
def test_hf_load_object_detection_model(
    with_processor, with_post_processor, output_as_list
):
    processor, model = get_test_object_detection_model(
        with_post_processor, output_as_list
    )

    with mock.patch.object(
        AutoImageProcessor, "from_pretrained", return_value=processor
    ), mock.patch.object(
        AutoModelForObjectDetection, "from_pretrained", return_value=model
    ):
        model_out = jatic_toolbox.load_model(
            provider="huggingface",
            task="object-detection",
            model_name="test",
            with_processor=with_processor,
            with_post_processor=with_post_processor,
        )
        assert isinstance(model_out, HuggingFaceObjectDetector)

        out = model_out(tr.randn(1, 3, 10, 10))
        assert isinstance(out, HasObjectDetections)
