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


def test_typed_collection():
    def func():
        from jatic_toolbox.protocols import TypedCollection

        # on it's own it's just a simple type
        def f(data: TypedCollection[float]):
            ...

        # failing type check
        f("string")
        f(["string"])
        f(("string",))
        f(dict(x="string"))

        f(1 + 1j)
        f([1 + 1j])
        f((1 + 1j,))
        f(dict(x=1 + 1j))

        # passing
        f(1.0)
        f([1.0])
        f((1.0,))
        f(dict(x=1.0))

        def f2(*inputs: TypedCollection[float]):
            ...

        # passes
        f2(1.0, 1.0)
        f2([1.0], dict(x=1.0))
        f2((1.0,), [1.0, 2.0], 1.0)
        f2(dict(x=1.0))
        f2(dict(x=[1.0, 2.0]))

        # fails
        f2(1.0, "string")
        f2(1.0, ["string"])
        f2([1.0], ("string",))
        f2(1.0, dict(x="string"))

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 12


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not installed.")
def test_torch_typed_collection():
    def func():
        from typing import Dict, Sequence

        from torch import Tensor

        from jatic_toolbox.protocols import TypedCollection

        def f(x: TypedCollection[Tensor]):
            ...

        # fmt: off
        def _tensor() -> Tensor: ...
        def _seq() -> Sequence[Tensor]: ...
        def _dict() -> Dict[str, Tensor]: ...
        # fmt: on

        f(_tensor())
        f(_seq())
        f(_dict())

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 0


@pytest.mark.skipif(not is_numpy_available(), reason="NumPy is not installed.")
def test_numpy_typed_collection():
    def func():
        from typing import Dict, Sequence

        import numpy as np

        from jatic_toolbox.protocols import TypedCollection

        def f(x: TypedCollection[np.ndarray]):
            ...

        # fmt: off
        def _numpy() -> np.ndarray: ...
        def _seq() -> Sequence[np.ndarray]: ...
        def _dict() -> Dict[str, np.ndarray]: ...
        # fmt: on

        f(_numpy())
        f(_seq())
        f(_dict())

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 0


def test_augmention():
    def func():
        from typing import Optional

        from jatic_toolbox.protocols import Augmentation, TypedCollection

        def augmenter(f: Augmentation[float]):
            ...

        def aug1(
            *inputs: TypedCollection[float], rng: Optional[int] = 0
        ) -> TypedCollection[float]:
            ...

        def aug2(*inputs: float, rng: Optional[int] = 0) -> TypedCollection[float]:
            ...

        def aug3(
            inputs: TypedCollection[float], rng: Optional[int] = 0
        ) -> TypedCollection[float]:
            ...

        def aug4(*inputs: TypedCollection[float]) -> TypedCollection[float]:
            ...

        augmenter(aug1)
        augmenter(aug2)
        augmenter(aug3)
        augmenter(aug4)

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 3


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not installed.")
def test_torch_augmentation():
    def func():
        from typing import Any, Optional

        from torch import Tensor

        from jatic_toolbox.protocols import Augmentation, TypedCollection

        def augmenter(f: Augmentation[Tensor]):
            ...

        def _tensor(
            *inputs: TypedCollection[Tensor], rng: Optional[Any] = 0
        ) -> TypedCollection[Tensor]:
            ...

        augmenter(_tensor)

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 0


@pytest.mark.skipif(not is_numpy_available(), reason="NumPy is not installed.")
def test_numpy_augmentation():
    def func():
        from typing import Any, Optional

        import numpy as np

        from jatic_toolbox.protocols import Augmentation, TypedCollection

        def augmenter(f: Augmentation[np.ndarray]):
            ...

        def _numpy(
            *inputs: TypedCollection[np.ndarray], rng: Optional[Any] = 0
        ) -> TypedCollection[np.ndarray]:
            ...

        augmenter(_numpy)

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 0


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not installed.")
def test_torch_classifier_workflow():
    def func():
        from typing import Dict, Union

        from torch import Tensor

        from jatic_toolbox.protocols import (
            Classifier,
            DataLoader,
            Metric,
            SupportsClassification,
            TorchClassifier,
            TorchDataLoader,
            TorchMetric,
        )

        def compute_metrics(
            model: Classifier[Tensor],
            dataloader: DataLoader[SupportsClassification[Tensor]],
            metric: Metric[Union[Tensor, Dict[str, Tensor]]],
        ):
            ...

        def torch_dl() -> TorchDataLoader:
            ...

        def torch_model() -> TorchClassifier:
            ...

        def torch_metric() -> TorchMetric:
            ...

        model = torch_model()
        dl = torch_dl()
        metric = torch_metric()

        compute_metrics(model, dl, metric)

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 0


@pytest.mark.skipif(not is_numpy_available(), reason="NumPy is not installed.")
def test_numpy_classifier_workflow():
    def func():
        from typing import Dict, Union

        from numpy import ndarray

        from jatic_toolbox.protocols import (
            Classifier,
            DataLoader,
            Metric,
            NumPyClassifier,
            NumPyDataLoader,
            NumPyMetric,
            SupportsClassification,
        )

        def compute_metrics(
            model: Classifier[ndarray],
            dataloader: DataLoader[SupportsClassification[ndarray]],
            metric: Metric[Union[ndarray, Dict[str, ndarray]]],
        ):
            ...

        def numpy_dl() -> NumPyDataLoader:
            ...

        def numpy_model() -> NumPyClassifier:
            ...

        def numpy_metric() -> NumPyMetric:
            ...

        model = numpy_model()
        dl = numpy_dl()
        metric = numpy_metric()

        compute_metrics(model, dl, metric)

    x = pyright_analyze(func)
    assert x["summary"]["errorCount"] == 0
