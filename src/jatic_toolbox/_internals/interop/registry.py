from ..import_utils import is_hydra_zen_available

JATIC_REGISTRY = None

if is_hydra_zen_available():
    import hydra_zen

    import jatic_toolbox

    JATIC_REGISTRY = hydra_zen.ZenStore(name="jatic_registry", overwrite_ok=True)

    #
    # Datasets
    #
    JATIC_REGISTRY(
        hydra_zen.builds(
            jatic_toolbox.get_dataset_builder,
            provider="huggingface",
            dataset_name="cifar10",
            task="image-classification",
            split="test",
        ),
        group="dataset",
        name="cifar10",
    )

    JATIC_REGISTRY(
        hydra_zen.builds(
            jatic_toolbox.get_dataset_builder,
            provider="huggingface",
            dataset_name="fashion_mnist",
            task="image-classification",
            split="test",
        ),
        group="dataset",
        name="fashion_mnist",
    )

    JATIC_REGISTRY(
        hydra_zen.builds(
            jatic_toolbox.get_dataset_builder,
            provider="huggingface",
            dataset_name="detection-datasets/coco",
            task="object-detection",
            split="val",
        ),
        group="dataset",
        name="coco",
    )

    #
    # Models
    #
    JATIC_REGISTRY(
        hydra_zen.builds(
            jatic_toolbox.get_model_builder,
            provider="huggingface",
            task="image-classification",
            model_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        ),
        group="model",
        name="vit_cifar10",
    )

    JATIC_REGISTRY(
        hydra_zen.builds(
            jatic_toolbox.get_model_builder,
            provider="torchvision",
            model_name="fasterrcnn_resnet50_fpn",
            task="object-detection",
        ),
        group="model",
        name="fasterrcnn_resnet50_fpn",
    )

    #
    # Metrics
    #
    JATIC_REGISTRY(
        hydra_zen.builds(
            jatic_toolbox.get_metric_builder,
            provider="torchmetrics",
            metric_name="Accuracy",
            task="multiclass",
        ),
        group="metric",
        name="multiclass_accuracy",
    )

    JATIC_REGISTRY(
        hydra_zen.builds(
            jatic_toolbox.get_metric_builder,
            provider="torchmetrics",
            metric_name="MeanAveragePrecision",
        ),
        group="metric",
        name="mean_average_precision",
    )

    def classification_report_builder(num_classes: int, average: str = "none"):
        return hydra_zen.builds(
            dict,
            accuracy=hydra_zen.builds(
                jatic_toolbox.load_metric,
                provider="torchmetrics",
                metric_name="Accuracy",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
            f1score=hydra_zen.builds(
                jatic_toolbox.load_metric,
                provider="torchmetrics",
                metric_name="F1Score",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
            precision=hydra_zen.builds(
                jatic_toolbox.load_metric,
                provider="torchmetrics",
                metric_name="Precision",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
            recall=hydra_zen.builds(
                jatic_toolbox.load_metric,
                provider="torchmetrics",
                metric_name="Recall",
                task="multiclass",
                num_classes=num_classes,
                average=average,
            ),
        )

    JATIC_REGISTRY(
        hydra_zen.builds(
            classification_report_builder,
            num_classes=10,
            average="none",
        ),
        group="metric",
        name="classification_report",
    )
