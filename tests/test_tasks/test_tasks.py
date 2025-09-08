# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Literal, Sequence

import numpy as np
import pytest

import maite.protocols.image_classification as ic
from maite._internals.protocols import generic as gen
from maite._internals.tasks.generic import _SimpleDataLoader
from maite.tasks import augment_dataloader, evaluate, predict
from tests.component_impls import (
    ic_simple_component_impls as ici,
)
from tests.component_impls import (
    od_simple_component_impls as odi,
)


def test_ic_augment_dataloader(
    ic_mock_dataset: ic.Dataset,
    ic_mock_augmentation: ic.Augmentation,
):
    ic_mock_dataloader = _SimpleDataLoader(ic_mock_dataset, 2)

    dataloader = augment_dataloader(
        augmentation=ic_mock_augmentation, dataloader=ic_mock_dataloader
    )
    batches = list(dataloader)
    inputs = [item for batch in batches for item in batch[0]]

    assert len(ic_mock_dataset) == len(inputs), (
        "size dataloader should match dataset size"
    )

    i = 3
    x = inputs[i]
    x = np.asarray(x)  # bridge
    expected_value = (i + 1) % 10
    assert x[0][0][0] == expected_value, (
        f"mock augmentation should bump first value in data point {i} from {i} to {expected_value}"
    )


def test_simple_ic_structural(
    ic_simple_augmentation,
    ic_simple_dataset,
    ic_simple_dataloader,
    ic_simple_model,
    ic_simple_metric,
):
    # verify types pass isinstance checks

    assert isinstance(ic_simple_augmentation, gen.Augmentation), (
        "augmentation structural check fail"
    )
    assert isinstance(ic_simple_metric, gen.Metric), "metric structural check fail"
    assert isinstance(ic_simple_dataset, gen.Dataset), "dataset structural check fail"
    assert isinstance(ic_simple_dataloader, gen.DataLoader), (
        "dataloader structural check fail"
    )
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

    assert isinstance(od_simple_augmentation, gen.Augmentation), (
        "augmentation structural check fail"
    )
    assert isinstance(od_simple_metric, gen.Metric), "metric structural check fail"
    assert isinstance(od_simple_dataset, gen.Dataset), "dataset structural check fail"
    assert isinstance(od_simple_dataloader, gen.DataLoader), (
        "dataloader structural check fail"
    )
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
            assert len(preds) == n, (
                "should return same number of predictions as dataset length when batch_size is 1"
            )

            # Verify num returned data points
            expected_data_len = n if return_augmented_data else 0
            assert len(data) == expected_data_len, (
                f"should return {expected_data_len} data point batches when `return_augmented_data` is {return_augmented_data} and batch_size is 1"
            )

            # Verify returned data has augmentation applied
            if use_aug and return_augmented_data:
                i = 3
                xb, yb, mdb = data[i]  # get ith batch
                x = xb[0]  # get first (only) element out of size-1 batch
                x = np.asarray(x)  # bridge
                expected_value = (i + 1) % 10
                assert x[0][0][0] == expected_value, (
                    f"mock augmentation should bump first value in data point {i} from {i} to {expected_value}"
                )

            # Verify model applied to correct data (original or augmented)
            i = 3
            y = preds[i][0]  # get ith prediction out of ith size-1 batch
            y = np.asarray(y)  # bridge
            expected_class = (i + 1) % 10 if use_aug else i % 10
            assert y[expected_class] == 1, (
                f"mock model should predict class {expected_class} for {'augmented' if use_aug else 'original'} data point {i}"
            )


def check_num_returned_predictions(
    dataset_size: int, preds: Sequence[Sequence[ic.TargetType]]
) -> None:
    num_data_points = sum([len(batch) for batch in preds])
    assert num_data_points == dataset_size, (
        f"should return {dataset_size} predictions given size-{dataset_size} dataset"
    )


def check_num_returned_batches(
    dataset_size: int,
    batch_size: int,
    return_type: Literal["augmented_data", "preds"],
    batches: Sequence[Sequence[Any]],
):
    expected_num_batches = dataset_size // batch_size + (
        1 if dataset_size % batch_size > 0 else 0
    )
    assert len(batches) == expected_num_batches, (
        f"should return {expected_num_batches} {return_type} batches for a size-{dataset_size} dataset when batch_size is {batch_size}"
    )


def check_augmented_predictions(preds: Sequence[Sequence[ic.TargetType]]):
    # Behavior of mock dataset:
    # - data point i input is shape-(3, 32, 32) array with all values equal to i
    # - data point i target is i % 10
    # - e.g.,
    # - data point 3: x[0, 0, 0] is 3 (as well as all other elements); target is 3
    # - data point 12: x[0, 0, 0] is 12 (as well as all other elements); target is 2

    # Behavior of mock model:
    # - predicted class is int(x[0, 0, 0]) % 10
    # - e.g.,
    # - (unperturbed) data point 3: x[0, 0, 0] is 3 and model correctly predicts class 3
    # - (unperturbed) data point 12: x[0, 0, 0] is 12 and model correctly predicts class 2

    # Behavior of mock augmentation:
    # - if data point x is "odd", i.e., if int(x[0, 0, 0]) % 2 is 1, then add 1 to x[0, 0, 0]
    # - e.g.,
    # - data point 3: x[0, 0, 0] would change from 3 to 4; mock model would incorrectly predict class 4 for point 3
    # - data point 12: x[0, 0, 0] would remain 12; mock model still correctly predict class 2 for point 12

    i = 0  # instance number
    for batch in preds:
        for pred in batch:
            aug = (
                1 if i % 2 == 1 else 0
            )  # augmentation bumps "odd" data points (i.e., with int(x[0, 0, 0]) % 2 == 1)

            expected_class = (
                i + aug
            ) % 10  # mock model predicts class as x[0, 0, 0] % 10

            y = np.asarray(pred)

            assert y[expected_class] == 1, (
                f"mock model should predict class {expected_class} for augmented data point {i}"
            )

            i += 1


def check_returned_data_augmented(
    aug_data: Sequence[
        tuple[
            Sequence[ic.InputType],
            Sequence[ic.TargetType],
            Sequence[ic.DatumMetadataType],
        ]
    ],
):
    i = 0  # instance number
    for batch in aug_data:
        xb, _, _ = batch
        for x_aug in xb:
            x_aug = np.asarray(x_aug)

            delta = (
                1 if i % 2 == 1 else 0
            )  # augmentation bumps "odd" data points (i.e., with int(x[0, 0, 0]) % 2 == 1)
            expected = i + delta

            assert x_aug[0, 0, 0] == expected, (
                f"mock augmentation should bump first value in data point {i} from {i} to {expected}"
            )

            i += 1


def check_returned_data_unchanged(
    aug_data: Sequence[
        tuple[
            Sequence[ic.InputType],
            Sequence[ic.TargetType],
            Sequence[ic.DatumMetadataType],
        ]
    ],
):
    i = 0  # instance number
    for batch in aug_data:
        xb, _, _ = batch
        for x_aug in xb:
            x_aug = np.asarray(x_aug)
            expected = i
            assert x_aug[0, 0, 0] == expected, (
                f"first value in data point {i} should be {i} since no augmentation applied"
            )

            i += 1


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_evaluate_full(
    ic_mock_dataset, ic_mock_model, ic_mock_augmentation, ic_accuracy_metric, batch_size
):
    # Test "full" use of `evaluate`
    result, preds, aug_data = evaluate(
        model=ic_mock_model,
        dataset=ic_mock_dataset,
        augmentation=ic_mock_augmentation,
        metric=ic_accuracy_metric,
        batch_size=batch_size,
        return_augmented_data=True,
        return_preds=True,
    )

    # Verify metric result reflects wrong predictions for all "odd" instances
    # - evidence that: dataset iterated over, augmentation applied, metric used
    assert result["accuracy"] == 0.5

    # Verify preditions are returned and "correct" (i.e., as expected)
    dataset_size = len(ic_mock_dataset)
    check_num_returned_predictions(dataset_size, preds)
    check_num_returned_batches(dataset_size, batch_size, "preds", preds)
    check_augmented_predictions(preds)

    # Verify augmented data returned and "correct" (i.e., as expected)
    check_num_returned_batches(dataset_size, batch_size, "augmented_data", aug_data)
    check_returned_data_augmented(aug_data)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_evaluate_dataloader(
    ic_mock_dataset, ic_mock_model, ic_mock_augmentation, ic_accuracy_metric, batch_size
):
    from maite._internals.tasks.generic import _SimpleDataLoader

    dataloader: ic.DataLoader = _SimpleDataLoader[
        ic.InputType, ic.TargetType, ic.DatumMetadataType
    ](dataset=ic_mock_dataset, batch_size=batch_size)

    result, preds, aug_data = evaluate(
        model=ic_mock_model,
        dataloader=dataloader,
        augmentation=ic_mock_augmentation,
        metric=ic_accuracy_metric,
        return_augmented_data=True,
        return_preds=True,
    )

    # Verify metric result reflects wrong predictions for all "odd" instances
    # - evidence that: dataset iterated over, augmentation applied, metric used
    assert result["accuracy"] == 0.5

    # Verify preditions returned and "correct" (i.e., as expected)
    dataset_size = len(ic_mock_dataset)
    check_num_returned_predictions(dataset_size, preds)
    check_num_returned_batches(dataset_size, batch_size, "preds", preds)
    check_augmented_predictions(preds)

    # Verify augmented data returned and "correct" (i.e., as expected)
    check_num_returned_batches(dataset_size, batch_size, "augmented_data", aug_data)
    check_returned_data_augmented(aug_data)


def test_evaluate_no_augmentation(ic_mock_dataset, ic_mock_model, ic_accuracy_metric):
    batch_size = 2

    result, preds, aug_data = evaluate(
        model=ic_mock_model,
        dataset=ic_mock_dataset,
        metric=ic_accuracy_metric,
        batch_size=batch_size,
        return_augmented_data=True,
        return_preds=True,
    )

    # Verify metric result reflects all correct predictions
    # - evidence that: dataset iterated over, augmentation applied, metric used
    assert result["accuracy"] == 1.0

    # Verify preditions shape
    dataset_size = len(ic_mock_dataset)
    check_num_returned_predictions(dataset_size, preds)
    check_num_returned_batches(dataset_size, batch_size, "preds", preds)

    # Verify augmented data returned and unchanged (since not really augmented)
    check_num_returned_batches(dataset_size, batch_size, "augmented_data", aug_data)
    check_returned_data_unchanged(aug_data)


def test_evaluate_false_return_flags(
    ic_mock_dataset, ic_mock_model, ic_accuracy_metric
):
    result, preds, aug_data = evaluate(
        model=ic_mock_model,
        dataset=ic_mock_dataset,
        metric=ic_accuracy_metric,
        batch_size=2,
        return_augmented_data=False,
        return_preds=False,
    )

    # Verify metric result reflects all correct predictions
    # - evidence that: dataset iterated over, augmentation applied, metric used
    assert result["accuracy"] == 1.0

    # Verify no returned data
    assert len(preds) == 0, (
        f"preds should have length 0 when return_preds is False; was {len(preds)}"
    )
    assert len(aug_data) == 0, (
        f"aug_data should have length 0 when return_augmented_data is False; was {len(aug_data)}"
    )


def test_evaluate_dataset_and_dataloader(
    ic_mock_dataset, ic_mock_model, ic_accuracy_metric
):
    from maite._internals.tasks.generic import _SimpleDataLoader

    dataloader_batch_size = 2
    dataloader: ic.DataLoader = _SimpleDataLoader[
        ic.InputType, ic.TargetType, ic.DatumMetadataType
    ](dataset=ic_mock_dataset, batch_size=dataloader_batch_size)

    dataset_batch_size = 3
    with pytest.raises(Exception):
        _, _, _ = evaluate(
            model=ic_mock_model,
            dataset=ic_mock_dataset,
            batch_size=dataset_batch_size,
            dataloader=dataloader,
            metric=ic_accuracy_metric,
            return_augmented_data=False,
            return_preds=False,
        )


@pytest.mark.skip(
    reason="evaluate currently allows ignored batch_size when dataloader provided"
)
def test_evaluate_dataloader_and_batch_size(
    ic_mock_dataset, ic_mock_model, ic_accuracy_metric
):
    from maite._internals.tasks.generic import _SimpleDataLoader

    dataloader_batch_size = 2
    dataloader: ic.DataLoader = _SimpleDataLoader[
        ic.InputType, ic.TargetType, ic.DatumMetadataType
    ](dataset=ic_mock_dataset, batch_size=dataloader_batch_size)

    dataset_batch_size = 3
    with pytest.raises(Exception):
        _, _, _ = evaluate(
            model=ic_mock_model,
            dataloader=dataloader,
            batch_size=dataset_batch_size,
            metric=ic_accuracy_metric,
            return_augmented_data=False,
            return_preds=False,
        )


def test_evaluate_no_model(ic_mock_dataset, ic_accuracy_metric):
    with pytest.raises(Exception):
        _, _, _ = evaluate(
            # NOTE: ignoring type checking so can call evaluate incorrectly
            model=None,  # type: ignore
            dataset=ic_mock_dataset,
            metric=ic_accuracy_metric,
        )


@pytest.mark.skip(reason="evaluate currently allows providing no metric")
def test_evaluate_no_metric(ic_mock_model, ic_mock_dataset):
    with pytest.raises(Exception):
        _, _, _ = evaluate(model=ic_mock_model, dataset=ic_mock_dataset, metric=None)


def test_evaluate_no_dataset_or_dataloader(ic_mock_model, ic_accuracy_metric):
    with pytest.raises(Exception):
        _, _, _ = evaluate(model=ic_mock_model, metric=ic_accuracy_metric)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_predict_full(ic_mock_dataset, ic_mock_model, ic_mock_augmentation, batch_size):
    # Test "full" use of `precict`
    preds, aug_data = predict(
        model=ic_mock_model,
        dataset=ic_mock_dataset,
        augmentation=ic_mock_augmentation,
        batch_size=batch_size,
        return_augmented_data=True,
    )

    # Verify preditions returned and "correct" (i.e., as expected)
    dataset_size = len(ic_mock_dataset)
    check_num_returned_predictions(dataset_size, preds)
    check_num_returned_batches(dataset_size, batch_size, "preds", preds)
    check_augmented_predictions(preds)

    # Verify augmented data returned and "correct" (i.e., as expected)
    check_num_returned_batches(dataset_size, batch_size, "augmented_data", aug_data)
    check_returned_data_augmented(aug_data)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_predict_dataloader(
    ic_mock_dataset, ic_mock_model, ic_mock_augmentation, batch_size
):
    from maite._internals.tasks.generic import _SimpleDataLoader

    dataloader: ic.DataLoader = _SimpleDataLoader[
        ic.InputType, ic.TargetType, ic.DatumMetadataType
    ](dataset=ic_mock_dataset, batch_size=batch_size)

    preds, aug_data = predict(
        model=ic_mock_model,
        dataloader=dataloader,
        augmentation=ic_mock_augmentation,
        return_augmented_data=True,
    )

    # Verify preditions returned and "correct" (i.e., as expected)
    dataset_size = len(ic_mock_dataset)
    check_num_returned_predictions(dataset_size, preds)
    check_num_returned_batches(dataset_size, batch_size, "preds", preds)
    check_augmented_predictions(preds)

    # Verify augmented data returned and "correct" (i.e., as expected)
    check_num_returned_batches(dataset_size, batch_size, "augmented_data", aug_data)
    check_returned_data_augmented(aug_data)


def test_predict_no_augmentation(ic_mock_dataset, ic_mock_model):
    batch_size = 2

    preds, aug_data = predict(
        model=ic_mock_model,
        dataset=ic_mock_dataset,
        batch_size=batch_size,
        return_augmented_data=True,
    )

    # Verify preditions shape
    dataset_size = len(ic_mock_dataset)
    check_num_returned_predictions(dataset_size, preds)
    check_num_returned_batches(dataset_size, batch_size, "preds", preds)

    # Verify augmented data returned and unchanged (since not really augmented)
    check_num_returned_batches(dataset_size, batch_size, "augmented_data", aug_data)
    check_returned_data_unchanged(aug_data)


def test_predict_no_return_data(ic_mock_dataset, ic_mock_model):
    batch_size = 2

    preds, aug_data = predict(
        model=ic_mock_model,
        dataset=ic_mock_dataset,
        batch_size=batch_size,
        return_augmented_data=False,
    )

    # Verify preditions shape
    dataset_size = len(ic_mock_dataset)
    check_num_returned_predictions(dataset_size, preds)
    check_num_returned_batches(dataset_size, batch_size, "preds", preds)

    # Verify no returned data
    assert len(aug_data) == 0, (
        f"aug_data should have length 0 when return_augmented_data is False; was {len(aug_data)}"
    )


def test_predict_dataset_and_dataloader(ic_mock_dataset, ic_mock_model):
    from maite._internals.tasks.generic import _SimpleDataLoader

    dataloader_batch_size = 2
    dataloader: ic.DataLoader = _SimpleDataLoader[
        ic.InputType, ic.TargetType, ic.DatumMetadataType
    ](dataset=ic_mock_dataset, batch_size=dataloader_batch_size)

    dataset_batch_size = 3
    with pytest.raises(Exception):
        _, _ = predict(
            model=ic_mock_model,
            dataset=ic_mock_dataset,
            batch_size=dataset_batch_size,
            dataloader=dataloader,
            return_augmented_data=False,
        )


@pytest.mark.skip(
    reason="predict currently allows ignored batch_size when dataloader provided"
)
def test_predict_dataloader_and_batch_size(ic_mock_dataset, ic_mock_model):
    from maite._internals.tasks.generic import _SimpleDataLoader

    dataloader_batch_size = 2
    dataloader: ic.DataLoader = _SimpleDataLoader[
        ic.InputType, ic.TargetType, ic.DatumMetadataType
    ](dataset=ic_mock_dataset, batch_size=dataloader_batch_size)

    dataset_batch_size = 3
    with pytest.raises(Exception):
        _, _ = predict(
            model=ic_mock_model,
            dataloader=dataloader,
            batch_size=dataset_batch_size,
            return_augmented_data=False,
        )


def test_predict_no_model(ic_mock_dataset):
    with pytest.raises(Exception):
        _, _ = predict(
            # NOTE: ignoring type checking so can call evaluate incorrectly
            model=None,  # type: ignore
            dataset=ic_mock_dataset,
        )


def test_predict_no_dataset_or_dataloader(ic_mock_model):
    with pytest.raises(Exception):
        _, _ = predict(model=ic_mock_model)
