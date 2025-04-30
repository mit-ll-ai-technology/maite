# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from maite._internals.protocols import generic as gen
from maite.tasks import evaluate, predict
from tests.component_impls import (
    ic_simple_component_impls as ici,
    od_simple_component_impls as odi,
)


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


def test_simple_ic_evaluate():
    # Run types through "evaluate" task

    ic_simple_model = ici.ModelImpl()
    ic_simple_dataset = ici.DatasetImpl()
    ic_simple_dataloader = ici.DataLoaderImpl(dataset=ici.DatasetImpl())
    ic_simple_metric = ici.MetricImpl()
    ic_simple_augmentation = ici.AugmentationImpl()

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
    # Run types through "predict" task

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


def test_simple_od_structural(
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
    # Run types through "evaluate" task

    od_simple_model = odi.ModelImpl()
    od_simple_dataset = odi.DatasetImpl()
    od_simple_dataloader = odi.DataLoaderImpl(dataset=odi.DatasetImpl())
    od_simple_metric = odi.MetricImpl()
    od_simple_augmentation = odi.AugmentationImpl()

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
):
    # Run types through "predict" task

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


@pytest.mark.xfail
def test_simple_bad_evaluate(
    ic_simple_augmentation,
    ic_simple_dataset,
    od_simple_dataloader,
    od_simple_model,
    od_simple_metric,
):
    # Run types through "evaluate" task

    evaluate(
        model=od_simple_model,
        dataloader=od_simple_dataloader,
        metric=od_simple_metric,
        augmentation=ic_simple_augmentation,
    )

    evaluate(
        model=od_simple_model,
        dataset=ic_simple_dataset,
        metric=od_simple_metric,
        augmentation=ic_simple_augmentation,
    )


@pytest.mark.xfail
def test_simple_bad_predict(
    ic_simple_augmentation,
    ic_simple_dataset,
    od_simple_dataloader,
    od_simple_model,
):
    # Run types through "predict" task

    predict(
        model=od_simple_model,
        dataloader=od_simple_dataloader,
        augmentation=ic_simple_augmentation,
    )

    predict(
        model=od_simple_model,
        dataset=ic_simple_dataset,
        augmentation=ic_simple_augmentation,
    )


def test_ic_predict_return_data_flag(
    ic_mock_dataset, ic_mock_model, ic_mock_augmentation
):
    # Test `return_augmented_data` flag
    for use_aug in [True, False]:
        for return_augmented_data in [True, False]:
            preds, data = predict(
                model=ic_mock_model,
                dataset=ic_mock_dataset,
                augmentation=ic_mock_augmentation if use_aug else None,
                batch_size=1,
                return_augmented_data=return_augmented_data,
            )

            # Verify num returned predictions
            n = len(ic_mock_dataset)
            assert (
                len(preds) == n
            ), "should return same number of predictions as dataset length when batch_size is 1"

            # Verify num returned data points
            expected_data_len = n if return_augmented_data else 0
            assert (
                len(data) == expected_data_len
            ), f"should return {expected_data_len} data points when `return_augmented_data` is {return_augmented_data} and batch_size is 1"

            # Verify returned data has augmentation applied
            if use_aug and return_augmented_data:
                i = 3
                xb, yb, mdb = data[i]  # get ith batch
                x = xb[0]  # get first (only) element out of size-1 batch
                x = np.asarray(x)  # bridge
                expected_value = (i + 1) % 10
                assert (
                    x[0][0][0] == expected_value
                ), f"mock augmentation should bump first value in data point {i} from {i} to {expected_value}"

            # Verify model applied to correct data (original or augmented)
            i = 3
            y = preds[i][0]  # get ith prediction out of ith size-1 batch
            y = np.asarray(y)  # bridge
            expected_class = (i + 1) % 10 if use_aug else i % 10
            assert (
                y[expected_class] == 1
            ), f"mock model should predict class {expected_class} for {'augmented' if use_aug else 'original'} data point {i}"
