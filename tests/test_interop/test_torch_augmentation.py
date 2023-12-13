# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import random
from typing import Any, Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
import torch as tr
from hypothesis import given, settings

from maite.errors import InvalidArgument
from maite.interop.augmentation import AugmentationWrapper


def to_numpy(x):
    return x


def to_torch(x):
    return tr.as_tensor(x)


image_strategy = hnp.arrays(float, shape=(3, 30, 20), elements=st.floats(0, 1)).map(
    to_numpy
)


def random_shift(data, **kwargs):
    if isinstance(data, np.ndarray):
        return np.roll(data, 1, axis=-1)
    elif isinstance(data, tr.Tensor):
        return tr.roll(data, 1, dims=-1)


class RandomCrop:
    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__()
        self.output_size = size

    def _get_params(self, flat_inputs):
        assert len(flat_inputs) > 0
        h, w = flat_inputs[0].shape[-2:]
        th, tw = self.output_size

        dw = 0
        if tw < w:
            dw = random.randint(0, w - tw)

        dh = 0
        if th < h:
            dh = random.randint(0, h - th)

        return dict(
            left=dw, right=dw + tw, top=dh, bottom=dh + th, noise=random.uniform(0, 1)
        )

    def _transform(self, inpt: Any, params):
        return (
            inpt[
                ..., params["top"] : params["bottom"], params["left"] : params["right"]
            ]
            + params["noise"]
        )

    def __call__(self, data, **kwargs: Any) -> Any:
        params = self._get_params(data)
        return self._transform(data, params)


class NumpyRandomCrop(RandomCrop):
    def _get_params(self, flat_inputs):
        assert len(flat_inputs) > 0
        h, w = flat_inputs[0].shape[-2:]
        th, tw = self.output_size

        dw = 0
        if tw < w:
            dw = np.random.randint(0, w - tw)

        dh = 0
        if th < h:
            dh = np.random.randint(0, h - th)

        return dict(
            left=dw, right=dw + tw, top=dh, bottom=dh + th, noise=random.uniform(0, 1)
        )


@given(image=image_strategy)
def test_rng(image):
    xform = AugmentationWrapper(RandomCrop((8, 19)))
    with pytest.raises(InvalidArgument):
        xform(image, rng="1")  # pyright: ignore [reportGeneralTypeIssues]

    with pytest.raises(InvalidArgument):
        xform(image, rng=1.1)  # pyright: ignore [reportGeneralTypeIssues]


@given(image=image_strategy)
@pytest.mark.parametrize("to_type", [to_numpy, to_torch])
@pytest.mark.parametrize(
    "transform",
    [
        random_shift,
        RandomCrop(size=(8, 9)),
    ],
)
def test_augmentor(image, to_type, transform):
    data = to_type(image)
    xform = AugmentationWrapper(transform)
    out = xform(data)
    assert isinstance(out, type(data))


@settings(deadline=None)
@given(image=image_strategy)
@pytest.mark.parametrize(
    "transform",
    [
        RandomCrop(size=(8, 9)),
        NumpyRandomCrop(size=(8, 9)),
    ],
)
def test_reproducible(image, transform):
    xform = AugmentationWrapper(transform)

    out = xform(image, rng=1)
    assert isinstance(out, np.ndarray)

    out2 = xform(image, rng=1)
    assert isinstance(out2, np.ndarray)

    out3 = xform(image, rng=22)
    assert isinstance(out3, np.ndarray)

    np.testing.assert_allclose(out, out2)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, out, out3)


@settings(deadline=None)
@given(image=image_strategy)
@pytest.mark.parametrize(
    "collection",
    [
        (None, None),
        (None, dict),
        (None, list),
        (None, tuple),
        (dict, dict),
        (dict, list),
        (dict, tuple),
        (list, list),
        (list, tuple),
        (tuple, tuple),
    ],
)
@pytest.mark.parametrize(
    "transform",
    [
        RandomCrop(size=(8, 9)),
    ],
)
def test_nested_inputs(image, collection, transform):
    xform = AugmentationWrapper(transform)

    inputs = []
    for t in collection:
        if t is None:
            inputs.append(image)
        elif t == dict:
            inputs.append(dict(data=image))
        elif t == list:
            inputs.append([image])
        elif t == tuple:
            inputs.append((image,))
        else:
            raise ValueError(f"Unknown type {t}")

    out = xform(inputs)
    assert isinstance(out, list)
    assert len(out) == len(inputs)
    for o, i in zip(out, inputs):
        assert type(o) == type(i)
