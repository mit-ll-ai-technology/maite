# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest

import maite
import maite.protocols as pr
from maite._internals.import_utils import requires_hf_datasets, requires_torchvision
from maite._internals.interop import registry
from maite._internals.interop.huggingface import api as hf_api
from maite._internals.interop.torchvision import api as tv_api
from maite.errors import InvalidArgument, ToolBoxException


@pytest.mark.parametrize(
    ("task", "provider"),
    [(None, None), ("image-classification", None), ("image-classification", "dummy")],
)
def test_errors_load_dataset(task, provider):
    with pytest.raises(InvalidArgument):
        maite.load_dataset(dataset_name="_dummy", task=task, provider=provider)


@requires_hf_datasets
@requires_torchvision
def test_load_dataset_from_registry(mocker):
    from ..common import huggingface as hf_common

    data = maite.list_datasets()
    assert all(x == y for x, y in zip(data, registry.DATASET_REGISTRY.keys()))

    mock_dataset = hf_common.get_test_vision_dataset()
    mocker.patch("datasets.load_dataset", return_value=mock_dataset)
    maite.load_dataset(dataset_name="cifar10-test")


@requires_hf_datasets
def test_errors_hf_load_dataset_from_hub():
    with pytest.raises(FileNotFoundError):
        maite.load_dataset(
            dataset_name="_dummy", provider="huggingface", task="image-classification"
        )

    with pytest.raises(InvalidArgument):
        maite.load_dataset(dataset_name="_dummy", provider="huggingface")

    with pytest.raises(InvalidArgument):
        maite.load_dataset(
            provider="huggingface",
            task="bad-task",  # type: ignore
            dataset_name="test",
        )

    with pytest.raises(InvalidArgument):
        hf_api.HuggingFaceAPI().load_dataset(
            task=None,  # type: ignore
            dataset_name="test",
        )


@requires_torchvision
def test_errors_tv_load_dataset_from_hub():
    with pytest.raises(InvalidArgument):
        maite.load_dataset(
            dataset_name="_dummy", provider="torchvision", task="image-classification"
        )

    with pytest.raises(InvalidArgument):
        maite.load_dataset(dataset_name="_dummy", provider="torchvision")

    with pytest.raises(InvalidArgument):
        tv_api.TorchVisionAPI().load_dataset(
            task="bad-task",  # type: ignore
            dataset_name="test",
        )


@requires_hf_datasets
@pytest.mark.parametrize("task", ["image-classification", "object-detection"])
def test_hf_datasets(mocker, task):
    from ..common import huggingface as hf_common

    data = maite.list_datasets(provider="huggingface", dataset_name="cats_vs_dogs")
    assert len(list(data)) == 1

    if task == "image-classification":
        mock_dataset = hf_common.get_test_vision_dataset()
    else:
        mock_dataset = hf_common.get_test_detection_dataset()

    mocker.patch("datasets.load_dataset", return_value=mock_dataset)
    dataset = maite.load_dataset(
        provider="huggingface",
        task=task,
        dataset_name="test",
    )
    assert len(dataset) >= 1
    assert isinstance(dataset, pr.Dataset)

    example = dataset[0]

    if task == "image-classification":
        assert pr.is_typed_dict(example, pr.SupportsImageClassification)
    else:
        assert pr.is_typed_dict(example, pr.SupportsObjectDetection)


@requires_torchvision
def test_tv_list_datasets():
    data = maite.list_datasets(provider="torchvision", dataset_name="MNIST")
    assert len(list(data)) == 5


@requires_torchvision
@pytest.mark.parametrize("task", ["image-classification", "object-detection"])
@pytest.mark.parametrize("has_split", [True, False])
@pytest.mark.parametrize("has_train", [True, False])
def test_tv_datasets(mocker, task, has_split, has_train):
    from ..common import torchvision as tv_common

    mock_dataset = tv_common.get_test_vision_dataset(
        has_split=has_split, has_train=has_train
    )
    mocker.patch.object(tv_api, "_get_torchvision_dataset", return_value=mock_dataset)

    if task == "object-detection":
        with pytest.raises(ToolBoxException):
            maite.load_dataset(
                provider="torchvision",
                task=task,
                dataset_name="test",
            )
    else:
        dataset = maite.load_dataset(
            provider="torchvision",
            task="image-classification",
            dataset_name="test",
            split="train",
        )
        assert len(dataset) == 4
        assert isinstance(dataset, pr.Dataset)

        assert hasattr(dataset, "set_transform")
        dataset.set_transform(lambda x: x)  # type: ignore
        example = dataset[0]
        assert pr.is_typed_dict(example, pr.SupportsImageClassification)


@requires_hf_datasets
@pytest.mark.parametrize(
    "image_key, label_key",
    [(None, "label"), ("image", None), (None, None), ("image", "image")],
)
def test_hf_load_dataset_unsupported_vision_keys(mocker, image_key, label_key):
    from ..common import huggingface as hf_common

    # Create a mock dataset object
    mock_dataset = hf_common.get_test_vision_dataset(
        image_key=image_key, label_key=label_key
    )

    # Patch the load_dataset function to return the mock dataset object
    # with mock.patch.object(datasets, "load_dataset", return_value=mock_dataset):
    mocker.patch("datasets.load_dataset", return_value=mock_dataset)

    with pytest.raises(ToolBoxException):
        maite.load_dataset(
            provider="huggingface",
            task="image-classification",
            dataset_name="test",
            image_key=image_key,
            label_key=label_key,
        )


@requires_hf_datasets
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
    mocker, image_key, object_key, bbox_key, category_key
):
    from ..common import huggingface as hf_common

    # Create a mock dataset object
    mock_dataset = hf_common.get_test_detection_dataset(
        image_key=image_key,
        object_key=object_key,
        bbox_key=bbox_key,
        category_key=category_key,
    )

    mocker.patch("datasets.load_dataset", return_value=mock_dataset)

    with pytest.raises(ToolBoxException):
        maite.load_dataset(
            provider="huggingface",
            task="object-detection",
            dataset_name="test",
            image_key=image_key,
            objects_key=object_key,
            bbox_key=bbox_key,
            category_key=category_key,
        )


@requires_hf_datasets
def test_hf_transforms(mocker):
    from ..common import huggingface as hf_common

    for mock_dataset, task in [
        (hf_common.get_test_vision_dataset(), "image-classification"),
        (hf_common.get_test_detection_dataset(), "object-detection"),
    ]:
        mocker.patch("datasets.load_dataset", return_value=mock_dataset)

        dataset = maite.load_dataset(
            provider="huggingface",
            task=task,
            dataset_name="test",
        )
        assert len(dataset) >= 1
        assert isinstance(dataset, pr.Dataset)

        assert hasattr(dataset, "set_transform")

        example_pre = dataset[0]
        assert "new_field" not in example_pre

        def add_field(x):
            x.update(new_field=0)
            return x

        dataset.set_transform(add_field)  # type: ignore

        example_post = dataset[0]

        assert "new_field" in example_post


@requires_torchvision
def test_tv_transforms(mocker):
    from ..common import torchvision as tv_common

    mock_dataset = tv_common.get_test_vision_dataset()
    mocker.patch.object(tv_api, "_get_torchvision_dataset", return_value=mock_dataset)

    dataset = maite.load_dataset(
        provider="torchvision",
        task="image-classification",
        dataset_name="test",
    )
    assert len(dataset) >= 1
    assert isinstance(dataset, pr.Dataset)

    assert hasattr(dataset, "set_transform")

    example_pre = dataset[0]
    assert "new_field" not in example_pre

    def add_field(x):
        x.update(new_field=0)
        return x

    dataset.set_transform(add_field)  # type: ignore

    example_post = dataset[0]

    assert "new_field" in example_post
