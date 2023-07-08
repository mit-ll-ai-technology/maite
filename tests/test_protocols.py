import pytest

from jatic_toolbox._internals.import_utils import is_numpy_available, is_torch_available
from jatic_toolbox.testing.pyright import pyright_analyze


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
    assert x["summary"]["errorCount"] == 11


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
    assert x["summary"]["errorCount"] == 0


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
    assert x["summary"]["errorCount"] == 0


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
    assert x["summary"]["errorCount"] == 0, x["generalDiagnostics"]


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
    assert x["summary"]["errorCount"] == 0
