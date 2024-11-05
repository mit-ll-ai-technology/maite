# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from maite._internals.protocols import generic as gen
from maite.workflows import evaluate, predict


def test_simple_ic_structural(
    ic_simple_augmentation,
    ic_simple_dataset,
    ic_simple_dataloader,
    ic_simple_model,
    ic_simple_metric,
):
    # verify types pass isinstance checks

    assert isinstance(
        ic_simple_augmentation, gen.Augmentation
    ), "augmentation structural check fail"
    assert isinstance(ic_simple_metric, gen.Metric), "metric structural check fail"
    assert isinstance(ic_simple_dataset, gen.Dataset), "dataset structural check fail"
    assert isinstance(
        ic_simple_dataloader, gen.DataLoader
    ), "dataloader structural check fail"
    assert isinstance(ic_simple_model, gen.Model), "model structural check fail"


def test_simple_ic_evaluate(
    ic_simple_augmentation,
    ic_simple_dataset,
    ic_simple_dataloader,
    ic_simple_model,
    ic_simple_metric,
):
    # Run types through "evaluate" workflow

    evaluate(
        model=ic_simple_model,
        dataloader=ic_simple_dataloader,
        metric=ic_simple_metric,
        augmentation=ic_simple_augmentation,
    )

    evaluate(
        model=ic_simple_model,
        dataset=ic_simple_dataset,
        metric=ic_simple_metric,
        augmentation=ic_simple_augmentation,
    )


def test_simple_ic_predict(
    ic_simple_augmentation,
    ic_simple_dataset,
    ic_simple_dataloader,
    ic_simple_model,
    ic_simple_metric,
):
    # Run types through "predict" workflow

    predict(
        model=ic_simple_model,
        dataloader=ic_simple_dataloader,
        augmentation=ic_simple_augmentation,
    )

    predict(
        model=ic_simple_model,
        dataset=ic_simple_dataset,
        augmentation=ic_simple_augmentation,
    )
