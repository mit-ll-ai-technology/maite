# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any

import hypothesis.strategies as st
import numpy as np
from hypothesis.strategies import DrawFn
from numpy.typing import NDArray


# Define a strategy for generating random image data
def image_data(
    draw: DrawFn, channels: int = 3, max_width: int = 10, max_height: int = 10
) -> NDArray[Any]:
    """
    Create a strategy for generating random image data.

    Parameters
    ----------
    draw : Callable
        The Hypothesis draw function.
    channels : int, optional
        The number of channels in the image, by default 3.
    max_width : int, optional
        The maximum width of the image, by default 10.
    max_height : int, optional
        The maximum height of the image, by default 10.

    Returns
    -------
    np.ndarray
        A random image array.

    Examples
    --------
    >>> import hypothesis.strategies as st
    >>> from maite.testing.hypothesis import image_data
    >>> st.composite(image_data)().example()    # nondeterministic --> # doctest: +SKIP
    array([[[  0,  10,  20]]])
    """
    # Generate a random width and height for the image
    width = draw(st.integers(min_value=1, max_value=max_width))
    height = draw(st.integers(min_value=1, max_value=max_height))

    # Generate random pixel values for the image
    pixels = draw(
        st.lists(
            elements=st.integers(min_value=0, max_value=255),
            min_size=width * height * channels,
            max_size=width * height * channels,
        )
    )

    # Reshape the pixel values into an image array
    image_array = np.array(pixels).reshape((height, width, channels))

    return image_array
