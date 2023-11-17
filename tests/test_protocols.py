# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

import jatic_toolbox.protocols as pr
from jatic_toolbox._internals.import_utils import is_numpy_available, is_torch_available
from jatic_toolbox._internals.testing.pyright import chdir
from jatic_toolbox.testing.pyright import pyright_analyze


def define_model(input_type, output_type):
    return f"""
from jatic_toolbox.protocols import Model
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
        "from jatic_toolbox.protocols import ArrayLike",
        "from jatic_toolbox.protocols import SupportsArray",
    ]
    outputs = [
        "from builtins import int",
        "from jatic_toolbox.protocols import HasProbs",
        "from jatic_toolbox.protocols import HasLogits",
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
from jatic_toolbox.protocols import Model
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
        "from jatic_toolbox.protocols import ArrayLike",
        "from jatic_toolbox.protocols import SupportsArray",
    ]
    outputs = [
        "from builtins import int",
        "from jatic_toolbox.protocols import HasProbs",
        "from jatic_toolbox.protocols import HasLogits",
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


#
# TODO: Phase out tests below for batch-like
#


def test_arraylike():
    def func():
        from typing import List, Tuple

        from jatic_toolbox.protocols import ArrayLike

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

        from jatic_toolbox.protocols import ArrayLike

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

        from jatic_toolbox.protocols import ArrayLike

        def f(x: ArrayLike):
            ...

        def _numpy() -> np.ndarray:
            ...

        f(_numpy())

    x = pyright_analyze(func)
    assert x[0]["summary"]["errorCount"] == 0


def test_classifier_workflow():
    def func():
        from jatic_toolbox.protocols import (
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
        from jatic_toolbox.protocols import (
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
