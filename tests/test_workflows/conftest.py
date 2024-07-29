# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
# SPDX-License-Identifier: MIT


# define fixtures for use by any module in this directory

import pytest

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
from tests.component_impls import (
    ic_simple_component_impls as ici,
    od_simple_component_impls as odi,
)


# --- ic component fixtures ---
@pytest.fixture(scope="module")
def ic_simple_dataset() -> ic.Dataset:
    return ici.DatasetImpl()


@pytest.fixture(scope="module")
def ic_simple_dataloader(ic_simple_dataset) -> ic.DataLoader:
    return ici.DataLoaderImpl(dataset=ic_simple_dataset)


@pytest.fixture(scope="module")
def ic_simple_augmentation() -> ic.Augmentation:
    return ici.AugmentationImpl()


@pytest.fixture(scope="module")
def ic_simple_model() -> ic.Model:
    return ici.ModelImpl()


@pytest.fixture(scope="module")
def ic_simple_metric() -> ic.Metric:
    return ici.MetricImpl()


# --- od component fixtures ---
@pytest.fixture(scope="module")
def od_simple_dataset() -> od.Dataset:
    return odi.DatasetImpl()


@pytest.fixture(scope="module")
def od_simple_dataloader(od_simple_dataset) -> od.DataLoader:
    return odi.DataLoaderImpl(dataset=od_simple_dataset)


@pytest.fixture(scope="module")
def od_simple_augmentation() -> od.Augmentation:
    return odi.AugmentationImpl()


@pytest.fixture(scope="module")
def od_simple_model() -> od.Model:
    return odi.ModelImpl()


@pytest.fixture(scope="module")
def od_simple_metric() -> od.Metric:
    return odi.MetricImpl()
