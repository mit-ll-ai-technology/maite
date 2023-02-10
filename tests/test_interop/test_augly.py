import random

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
import torch as tr
from augly import image as augly_image
from hypothesis import given, settings
from PIL import Image

from jatic_toolbox.interop.augly import Augly
from jatic_toolbox.protocols import Augmentation


def to_numpy(x):
    return x.astype("uint8")


image_strategy = hnp.arrays(float, shape=(30, 30, 3), elements=st.floats(0, 1)).map(
    to_numpy
)


def to_pil(x):
    return Image.fromarray(x.astype("uint8")).convert("RGBA")


def test_type():
    assert isinstance(Augly, Augmentation)


@given(image=image_strategy)
def test_rng(image):
    xform = Augly(augly_image.RandomAspectRatio())
    with pytest.raises(ValueError):
        xform(image, rng="1")

    with pytest.raises(ValueError):
        xform(image, rng=1.1)


@settings(max_examples=5, deadline=None)
@given(image=image_strategy)
@pytest.mark.parametrize("to_type", [to_numpy, tr.from_numpy, to_pil])
@pytest.mark.parametrize(
    "transform, aug_kwargs",
    [
        (
            augly_image.RandomAspectRatio(),
            dict(),
        ),
        (
            augly_image.meme_format,
            dict(
                caption_height=75,
                meme_bg_color=(0, 0, 0),
                text_color=(255, 255, 255),
            ),
        ),
    ],
)
def test_object_detector(image, to_type, transform, aug_kwargs):
    # process image
    image *= 255
    image = to_type(image)

    xform = Augly(transform, **aug_kwargs)

    if isinstance(image, tr.Tensor):
        with pytest.raises(AssertionError):
            out = xform(image)
    else:
        out = xform(image)
        assert isinstance(out, type(image))


@given(image=image_strategy)
@pytest.mark.parametrize(
    "transform, aug_kwargs",
    [
        (
            augly_image.RandomAspectRatio(),
            dict(),
        ),
    ],
)
def test_reproducible(image, transform, aug_kwargs):
    xform = Augly(transform, **aug_kwargs)

    out = xform(image, rng=1)
    out2 = xform(image, rng=1)
    out3 = xform(image, rng=22)
    out4 = xform(image, rng=random.Random(1))

    assert isinstance(out, type(image))
    np.testing.assert_allclose(out, out2)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, out, out3)
    np.testing.assert_allclose(out, out4)


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
    "transform, aug_kwargs",
    [
        (
            augly_image.RandomAspectRatio(),
            dict(),
        ),
        (
            augly_image.RandomBlur(),
            dict(),
        ),
    ],
)
def test_nested_inputs(image, collection, transform, aug_kwargs):
    xform = Augly(transform, **aug_kwargs)

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

    out = xform(*inputs)
    assert isinstance(out, tuple)
    for o, i in zip(out, inputs):
        assert type(o) == type(i)
