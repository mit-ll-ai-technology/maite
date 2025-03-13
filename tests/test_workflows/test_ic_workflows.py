# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import numpy as np

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
