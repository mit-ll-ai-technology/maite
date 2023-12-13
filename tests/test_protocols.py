# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import itertools
from pathlib import Path

import pytest

import maite.protocols as pr
from maite._internals.import_utils import is_numpy_available, is_torch_available
from maite._internals.testing.pyright import chdir
from maite.testing.pyright import pyright_analyze


def define_model(input_type, output_type):
    return f"""
from maite.protocols import Model
from typing import List, Tuple
{input_type} as InputType
{output_type} as OutputType

def f(x: Model[[InputType], OutputType]):
    ...

class ModelNoLabel:
    def __call__(self, x: InputType) -> OutputType:
        ...

class GoodModel:
    def __call__(self, x: InputType) -> OutputType:
        ...
    def get_labels(self) -> List[str]:
        ...

class BadInputModel:
    def __call__(self, x: InputType, y: str) -> OutputType:
        ...
    def get_labels(self) -> List[str]:
        ...

class BadOutputModel:
    def __call__(self, x: InputType) -> Tuple[OutputType, str]:
        ...
    def get_labels(self) -> List[str]:
        ...


f(ModelNoLabel())
f(BadInputModel())
f(BadOutputModel())
f(GoodModel())
"""


def save_models():
    inputs = [
        "from builtins import int",
        "from maite.protocols import ArrayLike",
        "from maite.protocols import SupportsArray",
    ]
    outputs = [
        "from builtins import int",
        "from maite.protocols import HasProbs",
        "from maite.protocols import HasLogits",
    ]

    with chdir():
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            Path(f"f_{i}.py").write_text(define_model(input, output))

        fs = [f for f in Path.cwd().iterdir() if f.name.endswith(".py")]
        results = pyright_analyze(*fs)
        return zip(inputs, outputs, results)


@pytest.mark.parametrize("input, output, result", save_models())
def test_model(input, output, result):
    assert result["summary"]["errorCount"] == 3


@pytest.mark.parametrize("protocol", [pr.Model, pr.ImageClassifier, pr.ObjectDetector])
def test_model_isinstance(protocol):
    class A:
        ...

    class B:
        def get_labels(self):
            ...

    assert not isinstance(A, protocol)
    assert isinstance(B, protocol)


def define_metrics(input_type, output_type):
    return f"""
from maite.protocols import Model
from typing import List, Tuple
{input_type} as InputType
{output_type} as OutputType

def f(x: Metrics[[InputType], OutputType]):
    ...

class MetricNone:
    ...

class BadMetric:
    def reset(self): ...
    def update(self, x: OutputType): ...
    def compute(self) -> InputType: ...
    def to(self, device): ...

class GoodMetric:
    def reset(self): ...
    def update(self, x: InputType): ...
    def compute(self) -> OutputType: ...
    def to(self, device): ...

f(MetricNone())
f(Badmetric())
f(GoodMetric())
    """


def save_metrics():
    inputs = [
        "from builtins import int",
        "from maite.protocols import ArrayLike",
        "from maite.protocols import SupportsArray",
    ]
    outputs = [
        "from builtins import int",
        "from maite.protocols import HasProbs",
        "from maite.protocols import HasLogits",
    ]

    with chdir():
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            Path(f"f_{i}.py").write_text(define_metrics(input, output))

        fs = [f for f in Path.cwd().iterdir() if f.name.endswith(".py")]
        results = pyright_analyze(*fs)
        return zip(inputs, outputs, results)


@pytest.mark.parametrize("input, output, result", save_metrics())
def test_metrics(input, output, result):
    assert result["summary"]["errorCount"] == 2


def test_metric_isinstance():
    class A:
        ...

    class B:
        def reset(self):
            ...

        def update(self, x: int):
            ...

        def compute(self) -> str:
            ...

        def to(self, device):
            ...

    assert not isinstance(A, pr.Metric)
    assert isinstance(B, pr.Metric)


def define_provider(
    has_help: bool,
    has_list_model: bool,
    has_load_model: bool,
    has_list_dataset: bool,
    has_load_dataset: bool,
    has_list_metric: bool,
    has_load_metric: bool,
):
    _help_def = "def help(self, name: str) -> str: ..."
    _list_model_def = "def list_models(self, filter_str: str | List[str] | None = None, model_name: str | None = None, task: str | None = None, **kwargs: Any) -> Iterable[Any]: ..."
    _load_model_def = "def load_model(self, model_name: str, task: TaskName | None = None, **kwargs: Any) -> Model: ... "
    _list_dataset_def = "def list_datasets(self, **kwargs: Any) -> Iterable[Any]: ..."
    _load_dataset_def = "def load_dataset(self, dataset_name: str, task: TaskName | None = None, split: str | None = None, **kwargs: Any) -> Dataset: ...  "
    _list_metric_def = "def list_metrics(self, **kwargs: Any) -> Iterable[Any]: ..."
    _load_metric_def = (
        "def load_metric(self, metric_name: str, **kwargs: Any) -> Metric: ..."
    )

    body = {
        "help_def": _help_def if has_help else "",
        "list_model_def": _list_model_def if has_list_model else "",
        "load_model_def": _load_model_def if has_load_model else "",
        "list_dataset_def": _list_dataset_def if has_list_dataset else "",
        "load_dataset_def": _load_dataset_def if has_load_dataset else "",
        "list_metric_def": _list_metric_def if has_list_metric else "",
        "load_metric_def": _load_metric_def if has_load_metric else "",
    }
    # NOTE: edge-case when has_x are all false, we must generate a valid class body
    if all(len(v) == 0 for v in body.values()):
        body["help_def"] = "pass"

    return """
from maite.protocols import MetricProvider, ModelProvider, DatasetProvider, Model, Dataset, Metric, TaskName
from typing import List, Iterable, Any, Type, TypeAlias

def expects_dataset_provider(provider: DatasetProvider):
    ...

def expects_metric_provider(provider: MetricProvider):
    ...

def expects_model_provider(provider: ModelProvider):
    ...

def expects_any_provider(provider: ModelProvider | DatasetProvider | MetricProvider):
    ...

def expects_any_provider_type(provider_type: Type[ModelProvider] | Type[DatasetProvider] | Type[MetricProvider]):
    ...

AnyProvider: TypeAlias = ModelProvider | DatasetProvider | MetricProvider

def expects_any_provider_type_alias(provider_type: Type[AnyProvider]):
    ...

class MyProvider:
    {help_def}

    {list_metric_def}

    {load_metric_def}

    {list_dataset_def}

    {load_dataset_def}

    {list_model_def}

    {load_model_def}


expects_dataset_provider(MyProvider())
expects_metric_provider(MyProvider())
expects_model_provider(MyProvider())
expects_any_provider(MyProvider())
expects_any_provider_type(MyProvider)
expects_any_provider_type_alias(MyProvider)

    """.format(
        **body
    )


def save_providers():
    cases = {}
    with chdir():
        # repeat 7 for has_help + has_x_y (x: list/load, y: model,dataset,metric)
        for i, has_methods in enumerate(itertools.product([True, False], repeat=7)):
            contents = define_provider(*has_methods)
            Path(f"f_{i}.py").write_text(contents)
            cases[f"f_{i}.py"] = has_methods

        fs = [f for f in Path.cwd().iterdir() if f.name.endswith(".py")]
        results = pyright_analyze(*fs)
        for f, r in zip(fs, results):
            has_flags = cases[f.name]
            cases[f.name] = (has_flags, r)
        return cases.values()


@pytest.mark.parametrize("has_methods, result", save_providers())
def test_providers(has_methods, result):
    # each use of `MyProvider` will generate a type error for each of the three uses that is not okay
    has_help, *rest = has_methods
    model_ok = all((has_help, *rest[0:2]))
    dataset_ok = all((has_help, *rest[2:4]))
    metric_ok = all((has_help, *rest[4:]))

    expected_error_count = 0

    if not model_ok:
        expected_error_count += 1
    if not dataset_ok:
        expected_error_count += 1
    if not metric_ok:
        expected_error_count += 1

    if not any([model_ok, metric_ok, dataset_ok]):
        # 3 checks for any protocol will fail of none are satisfied
        expected_error_count += 3

    assert result["summary"]["errorCount"] == expected_error_count


#
# TODO: Phase out tests below for batch-like
#


def test_arraylike():
    def func():
        from typing import List, Tuple

        from maite.protocols import ArrayLike

        def f(x: ArrayLike):
            ...

        # pass
        class MyArray:
            def __array__(self) -> List[int]:
                ...

            @property
            def shape(self) -> Tuple[int, ...]:
                ...

        f(MyArray())

        # fails
        # fmt: off
        def _int() -> int: ...
        def _complex() -> complex: ...
        def _float() -> float: ...
        def _str() -> str: ...
        def _bytes() -> bytes: ...
        def _list_int() -> List[int]: ...
        def _list_complex() -> List[complex]: ...
        def _list_float() -> List[float]: ...
        def _list_str() -> List[str]: ...
        def _list_bytes() -> List[bytes]: ...
        # fmt: om

        f(_int())
        f(_complex())
        f(_float())
        f(_str())
        f(_bytes)
        f(_list_int())
        f(_list_complex())
        f(_list_float())
        f(_list_str())
        f(_list_bytes)

        class MyBadArray:
            def not_array(self) -> List[int]:
                ...

            @property
            def shape(self) -> Tuple[int, ...]:
                ...

        f(MyBadArray())

    x = pyright_analyze(func)
    print(x)
    assert x[0]["summary"]["errorCount"] == 11


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not installed.")
def test_torch_arraylike():
    def func():
        from torch import Tensor

        from maite.protocols import ArrayLike

        def f(x: ArrayLike):
            ...

        def _tensor() -> Tensor:
            ...

        f(_tensor())

    x = pyright_analyze(func)
    assert x[0]["summary"]["errorCount"] == 0


@pytest.mark.skipif(not is_numpy_available(), reason="NumPy is not installed.")
def test_numpy_arraylike():
    def func():
        import numpy as np

        from maite.protocols import ArrayLike

        def f(x: ArrayLike):
            ...

        def _numpy() -> np.ndarray:
            ...

        f(_numpy())

    x = pyright_analyze(func)
    assert x[0]["summary"]["errorCount"] == 0


def test_classifier_workflow():
    def func():
        from maite.protocols import (
            DataLoader,
            ImageClassifier,
            Metric,
            SupportsImageClassification,
            VisionDataLoader,
        )

        def f(
            dataloader: DataLoader[SupportsImageClassification],
            model: ImageClassifier,
            metric: Metric,
        ):
            metric.reset()
            for batch in dataloader:
                output = model(batch["image"])
                metric.update(output, batch)
            metric.compute()

        def dl() -> VisionDataLoader:
            ...

        def model() -> ImageClassifier:
            ...

        def metric() -> Metric:
            ...

        f(dl(), model(), metric())

    x = pyright_analyze(func)
    assert x[0]["summary"]["errorCount"] == 0, x["generalDiagnostics"]


def test_object_detector_workflow():
    def func():
        from maite.protocols import (
            DataLoader,
            Metric,
            ObjectDetectionDataLoader,
            ObjectDetector,
            SupportsObjectDetection,
        )

        def f(
            dataloader: DataLoader[SupportsObjectDetection],
            model: ObjectDetector,
            metric: Metric,
        ):
            metric.reset()
            for batch in dataloader:
                output = model(batch["image"])
                metric.update(output, batch)
            metric.compute()

        def dl() -> ObjectDetectionDataLoader:
            ...

        def model() -> ObjectDetector:
            ...

        def metric() -> Metric:
            ...

        f(dl(), model(), metric())

    x = pyright_analyze(func)
    assert x[0]["summary"]["errorCount"] == 0
