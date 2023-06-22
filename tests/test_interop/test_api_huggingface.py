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
from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.interop.huggingface import (
    HuggingFaceImageClassifier,
    HuggingFaceObjectDetectionDataset,
    HuggingFaceObjectDetector,
    HuggingFaceVisionDataset,
)
from jatic_toolbox.protocols import (
    HasDetectionLogits,
    HasDetectionPredictions,
    HasLogits,
    HasProbs,
)
from jatic_toolbox.testing.hypothesis import image_data

from ..common.huggingface import (
    get_test_detection_dataset,
    get_test_object_detection_model,
    get_test_vision_dataset,
    get_test_vision_model,
)


def test_hf_list_datasets():
    datasets = jatic_toolbox.list_datasets(provider="huggingface", dataset_name="cifar")
    assert issubclass(type(datasets), list)
    assert len(list(datasets)) == 2


def test_hf_list_models():
    models = jatic_toolbox.list_models(provider="huggingface", filter_str="resnet18")
    assert issubclass(type(models), list)
    assert len(list(models)) == 3


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
        assert "labels" in temp["objects"]
        assert "boxes" in temp["objects"]


@pytest.mark.parametrize(
    "task, loader",
    [
        (HuggingFaceImageClassifier, get_test_vision_model),
        (HuggingFaceObjectDetector, get_test_object_detection_model),
    ],
)
@given(data=st.composite(image_data)(), image_as_dict=st.booleans())
def test_hf_vision_processors(task, loader, data, image_as_dict):
    processor, model = loader()

    if hasattr(processor, "post_process_object_detection"):
        hf_model = task(model, processor, processor.post_process_object_detection)
    else:
        hf_model = task(model, processor)

    if image_as_dict:
        pre_data = [{"image": data}]
    else:
        pre_data = [data]

    features = hf_model.preprocessor(pre_data)

    if image_as_dict:
        assert len(features) == 1
        assert isinstance(features[0], dict)
        assert "image" in features[0]
    else:
        assert isinstance(features, dict)
        assert "image" in features

    output = hf_model({"image": [data] * 10})
    assert isinstance(output, HasLogits)

    output = hf_model.post_processor(output)
    assert isinstance(output, (HasProbs, HasDetectionPredictions))


@pytest.mark.parametrize("image_as_dict", [None, "image", "pixel_values", "foo"])
@pytest.mark.parametrize("top_k", [None, 2, 100])
def test_hf_load_vision_model(top_k, image_as_dict):
    if top_k == 0:
        top_k = None

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
            top_k=top_k,
        )
        assert isinstance(model_out, HuggingFaceImageClassifier)

        data = tr.randn(1, 3, 10, 10)
        if image_as_dict is not None:
            data = {image_as_dict: data}

        if image_as_dict is not None and image_as_dict not in ["image"]:
            with pytest.raises(InvalidArgument):
                out = model_out(data)
            return

        out = model_out(data)
        assert isinstance(out, HasLogits)

        out = model_out.post_processor(out)
        assert isinstance(out, HasProbs)


@given(
    output_as_list=st.booleans(),
    threshold=st.floats(min_value=0, max_value=1),
)
@pytest.mark.parametrize("image_as_dict", [None, "image", "pixel_values", "foo"])
def test_hf_load_object_detection_model(output_as_list, threshold, image_as_dict):
    processor, model = get_test_object_detection_model(output_as_list)

    with mock.patch.object(
        AutoImageProcessor, "from_pretrained", return_value=processor
    ), mock.patch.object(
        AutoModelForObjectDetection, "from_pretrained", return_value=model
    ):
        model_out = jatic_toolbox.load_model(
            provider="huggingface",
            task="object-detection",
            model_name="test",
            threshold=threshold,
        )
        assert isinstance(model_out, HuggingFaceObjectDetector)

        data = tr.randn(1, 3, 10, 10)
        if image_as_dict is not None:
            data = {image_as_dict: data}

        if image_as_dict is not None and image_as_dict not in ["image"]:
            with pytest.raises(InvalidArgument):
                out = model_out(data)
            return

        out = model_out(data)
        assert isinstance(out, HasDetectionLogits)

        out = model_out.post_processor(out)
        assert isinstance(out, HasDetectionPredictions)
