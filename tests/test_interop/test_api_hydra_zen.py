from hydra_zen.typing import Builds

from jatic_toolbox import get_dataset_builder, get_metric_builder, get_model_builder

# TODO: Add more tests


def test_api_hydra_zen_metric():
    config = get_metric_builder(provider="torchmetrics", metric_name="Accuracy")
    assert isinstance(config, Builds)

    config = get_metric_builder(provider="torcheval", metric_name="MulticlassAccuracy")
    assert isinstance(config, Builds)


def test_api_hydra_zen_dataset():
    config = get_dataset_builder(
        provider="huggingface",
        dataset_name="fashion_mnist",
        task="image-classification",
        split="test",
    )
    assert isinstance(config, Builds)

    config = get_dataset_builder(
        provider="torchvision", dataset_name="MNIST", task="image-classification"
    )
    assert isinstance(config, Builds)


def test_api_hydra_zen_model():
    config = get_model_builder(
        provider="torchvision", model_name="Resnet18", task="image-classification"
    )
    assert isinstance(config, Builds)

    config = get_model_builder(
        provider="huggingface",
        model_name="Methmani/ImageClassification_fashion-mnist",
        task="image-classification",
    )
    assert isinstance(config, Builds)
