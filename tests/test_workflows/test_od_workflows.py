# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from maite._internals.protocols import generic as gen
from maite.workflows import evaluate, predict


def test_simple_component_structural(
    od_simple_augmentation,
    od_simple_dataset,
    od_simple_dataloader,
    od_simple_model,
    od_simple_metric,
):
    # verify types pass isinstance checks

    assert isinstance(
        od_simple_augmentation, gen.Augmentation
    ), "augmentation structural check fail"
    assert isinstance(od_simple_metric, gen.Metric), "metric structural check fail"
    assert isinstance(od_simple_dataset, gen.Dataset), "dataset structural check fail"
    assert isinstance(
        od_simple_dataloader, gen.DataLoader
    ), "dataloader structural check fail"
    assert isinstance(od_simple_model, gen.Model), "model structural check fail"


def test_simple_od_evaluate(
    od_simple_augmentation,
    od_simple_dataset,
    od_simple_dataloader,
    od_simple_model,
    od_simple_metric,
):
    # Run types through "evaluate" workflow

    evaluate(
        model=od_simple_model,
        dataloader=od_simple_dataloader,
        metric=od_simple_metric,
        augmentation=od_simple_augmentation,
    )

    evaluate(
        model=od_simple_model,
        dataset=od_simple_dataset,
        metric=od_simple_metric,
        augmentation=od_simple_augmentation,
    )


def test_simple_od_predict(
    od_simple_augmentation,
    od_simple_dataset,
    od_simple_dataloader,
    od_simple_model,
    od_simple_metric,
):
    # Run types through "predict" workflow

    predict(
        model=od_simple_model,
        dataloader=od_simple_dataloader,
        augmentation=od_simple_augmentation,
    )

    predict(
        model=od_simple_model,
        dataset=od_simple_dataset,
        augmentation=od_simple_augmentation,
    )
