DATASET_REGISTRY = {
    "cifar10-test": dict(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="test",
    ),
    "fashion_mnist-test": dict(
        provider="huggingface",
        dataset_name="fashion_mnist",
        task="image-classification",
        split="test",
    ),
    "coco-val": dict(
        provider="huggingface",
        dataset_name="detection-datasets/coco",
        task="object-detection",
        split="val",
    ),
}

MODEL_REGISTRY = {
    "vit_for_cifar10": dict(
        provider="huggingface",
        task="image-classification",
        model_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    ),
    "fasterrcnn_resnet50_fpn": dict(
        provider="torchvision",
        model_name="fasterrcnn_resnet50_fpn",
        task="object-detection",
    ),
}

METRIC_REGISTRY = {
    "multiclass_accuracy": dict(
        provider="torchmetrics",
        metric_name="Accuracy",
        task="multiclass",
    ),
    "mean_average_precision": dict(
        provider="torchmetrics",
        metric_name="MeanAveragePrecision",
    ),
    "classification_report": dict(
        provider="torchmetrics",
        metric_name="ClassificationReport",
    ),
}
