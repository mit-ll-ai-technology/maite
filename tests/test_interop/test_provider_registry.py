# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

import itertools

import pytest

from maite._internals.interop.provider_registry import (
    create_provider,
    get_provider_type,
    list_providers,
    register_provider,
)
from maite._internals.protocols.generic import Dataset, Metric, Model
from maite._internals.protocols.task_aliases import TaskName
from maite.errors import InternalError, InvalidArgument
from maite.protocols import DatasetMetadata, MetricMetadata, ModelMetadata


class AModel:
    def __init__(self):
        self.metadata = ModelMetadata(id="a_model")

    def __call__(self, arg: list):
        return arg


class AMetric:
    def init(self):
        self.pred = 0
        self._count = 0
        self.metadata = MetricMetadata(id="a_metric")

    def reset(self):
        pass

    def update(self, pred, target):
        self.pred = pred
        self.target = target
        self._count += 1

    def compute(self):
        return {"metrics": self.pred / self._count}


class ADataset:
    def __init__(self, data):
        self.data = data
        self.metadata = DatasetMetadata(id="a_dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class BaseProvider:
    def __init__(self, model_names):
        self._names = model_names

    def help(self, name):
        return name


class FullModelProvider(BaseProvider):
    def list_models(
        self,
        *,
        filter_str: str | list[str] | None = None,
        model_name: str | None = None,
        task: TaskName | None = None,
    ):
        return [m for m in self._names]

    def load_model(self, model_name: str, task: TaskName, **kwargs) -> Model:
        return AModel()


class FullMetricProvider(BaseProvider):
    def list_metrics(self):
        return [m for m in self._names]

    def load_metric(self, metric_name: str) -> Metric:
        return AMetric()


class FullDatasetProvider(BaseProvider):
    def list_datasets(self):
        return [m for m in self._names]

    def load_dataset(
        self,
        *,
        dataset_name: str,
        task: TaskName,
        split: str | None = None,
    ) -> Dataset:
        return ADataset([x for x in range(len(dataset_name))])


class UniqueNameGen:
    __name__ = "UniqueNameGen"

    def __init__(self):
        self.counter = itertools.count(0)

    def __call__(self, prefix):
        return f"{prefix}_{next(self.counter)}"


_name_gen_instance = UniqueNameGen()


@pytest.fixture
def name_gen():
    return _name_gen_instance


def test_register_provider(name_gen):
    # Preamble, capture "native providers"
    native_providers = set(list_providers())
    # Part 1 - test registration

    # no error
    register_provider(FullModelProvider, key=name_gen("provider"))

    # no protocol - Error
    with pytest.raises(InvalidArgument):
        register_provider(BaseProvider, key=name_gen("provider"))  # type: ignore
    # no key - Error
    with pytest.raises(InvalidArgument):
        register_provider(FullModelProvider)

    # define key at class scope
    class ProviderWithKey(FullModelProvider):
        PROVIDER_REGISTRY_KEY = name_gen("provider")

    # Good
    register_provider(ProviderWithKey)

    # Re-register -> Warning, but okay
    with pytest.warns(
        Warning, match=f"under name {ProviderWithKey.PROVIDER_REGISTRY_KEY}"
    ):
        register_provider(ProviderWithKey)

    new_registered_providers = set(list_providers()) - native_providers
    # we performed 3 successful registrations above, but the final one overwrites an existing one,
    # so there should be 2 non-native providers registered
    assert len(new_registered_providers) == 2


def test_list_providers(name_gen):
    # generate 6 names
    names = [name_gen("provider") for _ in range(6)]
    ds_provider = names[0]
    met_provider = names[1]
    model_provider = names[2]
    ds_met_provider = names[3]
    ds_model_provider = names[4]
    any_provider = names[5]

    # register some providers
    register_provider(FullDatasetProvider, key=ds_provider)
    register_provider(FullMetricProvider, key=met_provider)
    register_provider(FullModelProvider, key=model_provider)

    # create some types satisfying multiple provider protocols
    class DSAndMetricProvider(FullDatasetProvider, FullMetricProvider):
        pass

    class DSAndModelProvider(FullDatasetProvider, FullModelProvider):
        pass

    class DSMetricModelProvider(
        FullDatasetProvider, FullMetricProvider, FullModelProvider
    ):
        pass

    register_provider(DSAndMetricProvider, key=ds_met_provider)
    register_provider(DSAndModelProvider, key=ds_model_provider)
    register_provider(DSMetricModelProvider, key=any_provider)

    bare_list = list_providers()
    # all registered names should be in the unfiltered list
    assert all(n in bare_list for n in names)
    # limit to model
    model_prov_list = list_providers(enforce_protocol="model")
    assert all(
        n in model_prov_list for n in (model_provider, ds_model_provider, any_provider)
    )
    assert all(
        n not in model_prov_list for n in (ds_provider, met_provider, ds_met_provider)
    )
    # limit to dataset
    ds_prov_list = list_providers(enforce_protocol="dataset")
    assert all(
        n in ds_prov_list
        for n in (ds_provider, ds_met_provider, ds_model_provider, any_provider)
    )
    assert all(n not in ds_prov_list for n in (met_provider, model_provider))
    # limit to metric
    met_prov_list = list_providers(enforce_protocol="metric")
    assert all(
        n in met_prov_list for n in (met_provider, ds_met_provider, any_provider)
    )
    assert all(
        n not in met_prov_list for n in (ds_provider, model_provider, ds_model_provider)
    )


def test_get_provider_type(name_gen):
    p1_name = name_gen("provider")
    register_provider(FullDatasetProvider, key=p1_name)

    p1_type = get_provider_type(p1_name)
    assert p1_type is FullDatasetProvider

    # raises when it can't match the key
    with pytest.raises(InvalidArgument):
        get_provider_type(name_gen("provider"))

    # raises when the key does not match the protocol provided
    with pytest.raises(InvalidArgument):
        get_provider_type(p1_name, enforce_protocol="model")


def test_create_provider(name_gen):
    # generate and register a provider with kwargs for __init__
    provider_name = name_gen("provider")

    @register_provider(key=provider_name)
    class ProviderWithKwargInit(FullDatasetProvider):
        def __init__(self, names, *, a=None, b=None, c=None):
            super().__init__(names)
            self.a = a
            self.b = b
            self.c = c

    # generate some args/kwargs
    names_arg = [name_gen("name") for _ in range(4)]
    provider_kwargs = {"a": "foo", "b": "bar", "c": "baz"}
    # find *and* instantiate the provider type
    my_provider = create_provider(
        provider_name, provider_args=(names_arg,), provider_kwargs=provider_kwargs
    )
    assert isinstance(my_provider, ProviderWithKwargInit)
    assert my_provider.a == provider_kwargs["a"]
    assert my_provider.b == provider_kwargs["b"]
    assert my_provider.c == provider_kwargs["c"]
    provider_datasets = my_provider.list_datasets()
    assert all(name in provider_datasets for name in names_arg)


def test_registry_instantiate_error():
    from maite._internals.interop.provider_registry import _ProviderRegistry

    with pytest.raises(InternalError):
        _ = _ProviderRegistry()
