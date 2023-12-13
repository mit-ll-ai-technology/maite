# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any

import pytest

from maite._internals.interop.artifact_hub.deduction import (
    make_entrypoint_deduction_filter,
)
from maite._internals.interop.artifact_hub.module_utils import import_hubconf
from maite._internals.interop.artifact_hub.registry import HubEndpointRegistry
from maite._internals.testing.pyright import chdir
from maite.errors import InternalError, InvalidArgument
from maite.interop.provider import ArtifactHubProvider
from maite.protocols import (
    Dataset,
    ImageClassifier,
    Metric,
    Model,
    ObjectDetectionDataset,
    ObjectDetector,
    VisionDataset,
)


def test_register_endpoint():
    from maite._internals.interop.artifact_hub.registry import HubEndpointRegistry

    class FakeEndpoint:
        def __init__(self, path):
            self.path = path

        def get_cache_or_reload(self, *args, **kwargs):
            return self.path

        def update_options(self, **options):
            return self

    # using free_function form
    ArtifactHubProvider.register_endpoint(FakeEndpoint, spec_tag="FOO")

    assert "FOO" in HubEndpointRegistry.registered_endpoints

    # using decorator form
    @ArtifactHubProvider.register_endpoint(spec_tag="FOO2")
    class Endpoint2(FakeEndpoint):
        pass

    assert "FOO2" in HubEndpointRegistry.registered_endpoints

    # bad registration (no spec)
    with pytest.raises(InvalidArgument):
        ArtifactHubProvider.register_endpoint(FakeEndpoint)

    # warning on re-register
    with pytest.warns(Warning, match="under spec tag FOO2"):
        ArtifactHubProvider.register_endpoint(FakeEndpoint, spec_tag="FOO2")

    # The above case should overwrite the type matched with the spec_tag
    assert HubEndpointRegistry.registered_endpoints["FOO2"] is FakeEndpoint
    assert HubEndpointRegistry.registered_endpoints["FOO2"] is not Endpoint2

    # extract spec_tag from class body
    class WithTag(FakeEndpoint):
        ENDPOINT_REGISTRY_TAG = "BAR"

    ArtifactHubProvider.register_endpoint(WithTag)

    # and as a decorator
    @ArtifactHubProvider.register_endpoint
    class WithTagDecorated(FakeEndpoint):
        ENDPOINT_REGISTRY_TAG = "BAR2"

    class NotAnEndpoint:
        pass

    with pytest.raises(InvalidArgument):
        ArtifactHubProvider.register_endpoint(NotAnEndpoint, spec_tag="bad_endpoint")  # type: ignore


def test_artifact_hub_api():
    @ArtifactHubProvider.register_endpoint(spec_tag="testing")
    class TemporaryDummyEndpoint:
        def __init__(self, path):
            self.path = Path(path)
            if not self.path.exists():
                self.path.mkdir(parents=True)
            hc = self.path / "hubconf.py"
            hc.write_text(
                "\n".join(
                    [
                        x.strip()
                        for x in """
                          from maite.protocols import Model, Dataset, Metric
                          def model() -> Model: ...
                          def dataset(split) -> Dataset:...
                          def metric() -> Metric:...
                          """.split(
                            "\n"
                        )
                    ]
                )
            )
            self.options = {"foo": "foo", "bar": "bar"}

        def update_options(self, **options):
            self.options.update(**options)
            return self

        def get_cache_or_reload(self, *args, **options):
            return self.path

    assert "testing" in ArtifactHubProvider.list_registered_endpoints()

    # attempt to create from bad spec
    assert "bad_endpoint" not in ArtifactHubProvider.list_registered_endpoints()
    with pytest.raises(InvalidArgument):

        @ArtifactHubProvider.register_endpoint(spec_tag="bad_endpoint")  # type: ignore
        class NotAnEndpoint:
            pass

    # attempt to create from missing spec

    with pytest.raises(InvalidArgument):
        provider = ArtifactHubProvider(spec_path=f"bad_endpoint::{Path.cwd()}")

    with chdir() as temp_dir:
        provider = ArtifactHubProvider(spec_path=f"testing::{temp_dir}")

        provider.update_endpoint_options(foo="bar", bar="foo")
        assert provider.endpoint.options["foo"] == "bar"  # type: ignore
        assert provider.endpoint.options["bar"] == "foo"  # type: ignore

        assert provider.list_datasets() == ["dataset"]
        assert provider.load_dataset(dataset_name="dataset") is None
        # no such ep
        with pytest.raises(InvalidArgument):
            provider.load_dataset(dataset_name="does-not-exist")
        # not a dataset ep
        with pytest.raises(InvalidArgument):
            provider.load_dataset(dataset_name="metric")

        assert provider.list_metrics() == ["metric"]
        assert provider.load_metric(metric_name="metric") is None
        # no such ep
        # no such ep
        with pytest.raises(InvalidArgument):
            provider.load_metric(metric_name="does-not-exist")
        # not a dataset ep
        with pytest.raises(InvalidArgument):
            provider.load_metric(metric_name="model")

        assert provider.list_models() == ["model"]
        assert provider.load_model(model_name="model") is None
        # no such ep
        with pytest.raises(InvalidArgument):
            provider.load_model(model_name="does-not-exist")
        # not a model ep
        with pytest.raises(InvalidArgument):
            provider.load_model(model_name="dataset")

        # check help, when a function has no docstring __doc__ is none
        assert provider.help("model") == ""


def test_module_utils():
    good_hubconf = """
def function():
    return "function_called"
"""

    bad_hubconf = """
import this_module_does_not_exist
def function():
    return this_module_does_not_exist.function()
"""

    missing_deps_hubconf = """
dependencies = ["this_module_does_not_exist", "maite"]
"""

    with chdir() as local_dir:
        good_hubconf_file = Path(local_dir) / "hubconf.py"
        good_hubconf_file.write_text(good_hubconf)
        module = import_hubconf(local_dir, "hubconf.py")
        assert module.function() == "function_called"

    with chdir() as local_dir:
        bad_hubconf_file = Path(local_dir) / "hubconf.py"
        bad_hubconf_file.write_text(bad_hubconf)
        with pytest.raises(ImportError):
            module = import_hubconf(local_dir, "hubconf.py")

    with chdir() as local_dir:
        missing_deps_hubconf_file = Path(local_dir) / "hubconf.py"
        missing_deps_hubconf_file.write_text(missing_deps_hubconf)
        with pytest.raises(RuntimeError):
            module = import_hubconf(local_dir, "hubconf.py")


def test_entrypoint_deduction():
    class MocHubConfNamespace:
        def __init__(self, **kwargs):
            for k in kwargs:
                setattr(self, k, kwargs[k])

    class MyMetric:
        def init(self):
            self.data = 0
            self._count = 0

        def reset(self):
            pass

        def update(self, data):
            self.data += data
            self._count += 1

        def compute(self):
            return self.data / self._count

        def to(self, *args, **kwargs):
            return self

    # generate some entrypoints to fill our module-namespace like object
    def build_namespace():
        def returns_something() -> Any:
            pass

        def returns_model() -> Model:  # type: ignore
            pass

        def returns_dataset() -> Dataset:  # type: ignore
            pass

        def returns_metric() -> Metric:  # type: ignore
            pass

        def returns_object_detection_model() -> ObjectDetector:  # type: ignore
            pass

        def returns_object_detection_dataset() -> ObjectDetectionDataset:  # type: ignore
            pass

        def returns_vision_dataset() -> VisionDataset:  # type: ignore
            pass

        def returns_image_classifier() -> ImageClassifier:  # type: ignore
            pass

        def returns_my_metric() -> MyMetric:  # type: ignore
            pass

        return MocHubConfNamespace(
            returns_something=returns_something,
            returns_model=returns_model,
            returns_metric=returns_metric,
            returns_dataset=returns_dataset,
            returns_object_detection_model=returns_object_detection_model,
            returns_object_detection_dataset=returns_object_detection_dataset,
            returns_vision_dataset=returns_vision_dataset,
            returns_image_classifier=returns_image_classifier,
            returns_my_metric=returns_my_metric,
            not_callable="info",
        )

    module = build_namespace()
    # simple filter (requires_deduction by default)
    simple_filter = make_entrypoint_deduction_filter()
    # The unannotated ep does not pass
    assert not simple_filter(getattr(module, "returns_something"))
    # the Un-callable ep does not pass
    assert not simple_filter(getattr(module, "not_callable"))
    # others should pass
    assert simple_filter(getattr(module, "returns_dataset"))
    assert simple_filter(getattr(module, "returns_vision_dataset"))

    # 100% permissive filter any callable should pass
    any_callable_filter = make_entrypoint_deduction_filter(require_deduction=False)
    assert any_callable_filter(getattr(module, "returns_something"))
    assert not any_callable_filter(getattr(module, "not_callable"))

    # type filter
    model_filter = make_entrypoint_deduction_filter(filter_type="model")
    assert model_filter(getattr(module, "returns_model"))
    assert model_filter(getattr(module, "returns_object_detection_model"))
    assert model_filter(getattr(module, "returns_image_classifier"))
    assert not model_filter(getattr(module, "returns_object_detection_dataset"))

    # task filter
    object_detection_filter = make_entrypoint_deduction_filter(
        filter_task="object-detection"
    )
    # TODO: Protocol is underspecified isinstance(Model, ObjectDetector) and isinstance(ObjectDetector, Model) are both true
    # only the second should be true (all object detectors are models, but not all models are object Detectors)
    # assert not object_detection_filter(getattr(module, "returns_model"))
    assert object_detection_filter(getattr(module, "returns_object_detection_model"))
    assert not object_detection_filter(getattr(module, "returns_metric"))
    # TODO: Protocol is underspecified, see above but for ImageClassifiers
    # assert not object_detection_filter(getattr(module, "returns_image_classifier"))
    assert object_detection_filter(getattr(module, "returns_object_detection_dataset"))

    # name match filter
    name_match_filter = make_entrypoint_deduction_filter(name="returns_my_metric")
    assert name_match_filter(getattr(module, "returns_my_metric"))
    for name in ("returns_dataset", "returns_model", "returns_metric"):
        assert not name_match_filter(getattr(module, name))

    # regex filter
    contains_dataset_filter = make_entrypoint_deduction_filter(
        filter_regex=".*_dataset"
    )
    assert contains_dataset_filter(getattr(module, "returns_dataset"))
    assert contains_dataset_filter(getattr(module, "returns_object_detection_dataset"))
    assert not contains_dataset_filter(
        getattr(module, "returns_object_detection_model")
    )

    # name startswith, passing single str, list with one str, list with multiple str
    for arg in ("returns_my", ["returns_my"], ["returns_my_", "returns_my"]):
        name_startswith_filter = make_entrypoint_deduction_filter(filter_str=arg)
        assert name_startswith_filter(getattr(module, "returns_my_metric"))
        assert not name_startswith_filter(getattr(module, "returns_dataset"))

    # image_classifier
    img_class_filter = make_entrypoint_deduction_filter(
        filter_task="image-classification"
    )
    assert img_class_filter(getattr(module, "returns_image_classifier"))
    assert img_class_filter(getattr(module, "returns_vision_dataset"))
    assert img_class_filter(getattr(module, "returns_object_detection_model"))
    assert not img_class_filter(getattr(module, "returns_my_metric"))

    # errors
    with pytest.raises(InvalidArgument):
        # can't specify task without requiring deduction
        make_entrypoint_deduction_filter(
            require_deduction=False, filter_task="image-classification"
        )

    with pytest.raises(InvalidArgument):
        # can't specify type without requiring deduction
        make_entrypoint_deduction_filter(require_deduction=False, filter_type="model")

    with pytest.raises(InvalidArgument):
        # bad task
        make_entrypoint_deduction_filter(filter_task="a-fake-task")  # type: ignore

    with pytest.raises(InvalidArgument):
        # bad type
        make_entrypoint_deduction_filter(filter_type="unknown-type")  # type: ignore


def test_registry_instantiate_issue():
    with pytest.raises(InternalError):
        _ = HubEndpointRegistry()
