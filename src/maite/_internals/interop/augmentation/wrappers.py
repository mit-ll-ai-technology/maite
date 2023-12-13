# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import random
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Protocol, TypeVar, cast

import numpy as np
import torch as tr
from torch.utils._pytree import tree_flatten, tree_unflatten

from maite._internals.import_utils import is_numpy_available, is_torch_available
from maite.errors import InvalidArgument

__all__ = ["AugmentationWrapper"]

T = TypeVar("T")


class AugFun(Protocol[T]):
    def __call__(self, data: T, **kwargs: Any) -> T:
        ...


class AugmentationWrapper(Generic[T]):
    def __init__(self, aug_fun: AugFun[T], **kwargs: Any):
        """
        Return a MAITE Augmentation for transform.

        Parameters
        ----------
        aug_fun : Callable[..., T]
            An augmentation callable to transform data.

        **kwargs : Any
            Optional arguments for `aug_fun`.

        Examples
        --------
        First create a random image using NumPy:

        >>> import numpy
        >>> img_np = numpy.random.rand(100,100,3) * 255

        Now get the MAITE Augmentation for a given an augmentation.

        >>> from augly.images import RandomAspectRatio
        >>> xform = AugmentationWrapper(RandomAspectRatio)

        Lastly, generate the augmentation:

        >> aug_img_np = xform(img_np)
        """
        self._aug_fun = aug_fun
        self._kwargs = kwargs

    def __call__(
        self,
        inputs: T,
        rng: Optional[int] = None,
    ) -> T:
        """
        Pipeline for augmentating data.

        Parameters
        ----------
        inputs : T
            Inputs to augment.

        rng : Optional[int] = None
            For reproducibility.

        Returns
        -------
        T
            All augmented inputs in the same format is
            the inputs.

        Raises
        ------
        InvalidArgument
            Raises if the `rng` keyword is not `int`.

        Examples
        --------
        First lets define a random image to work with:

        >>> import numpy
        >>> from PIL import Image
        >>> img_np = numpy.random.rand(100,100,3) * 255
        >>> img_pil = Image.fromarray(imarray.astype('uint8')).convert('RGBA')

        Next load load an augmentation using Augley [1]_.  We define
        the augmentation for PIL and NumPy arrays.

        >>> from augly.images import RandomAspectRatio
        >>> xform = AugmentationWrapper(RandomAspectRatio())

        We can execute on a single image:

        >>> out_pil = xform(img_pil)
        >>> assert len(out_pil) == 1
        >>> assert isinstance(out_pil, Image.Image)

        >>> out_np = xfrom(img_np)
        >>> assert len(out_np) == 1
        >>> assert isinstance(out_np, np.ndarray)

        The pipeline supports a "PyTree" of inputs of the
        same type, e.g.,

        >>> out_np = xform(img_np, dict(np_image=img_np))
        >>> assert len(out_np) == 2
        >>> assert isinstance(out_np[0], np.ndarray)
        >>> assert isinstance(out_np[1], dict)
        >>> assert isinstance(out_np[1]["np_image"], np.ndarray)

        Lastly, we can support reproducible augmentations:

        >>> import random
        >>> out = xform(img_pil, rng=1)
        >>> out2 = xform(img_pil, rng=1)
        >>> out3 = xform(img_pil, rng=2)
        >>> assert np.all(out[0]==out2[0])
        >>> assert np.all(out[0]!=out3[0])
        """

        if rng is not None:
            if isinstance(rng, int):
                if is_torch_available():
                    tr.manual_seed(rng)

                if is_numpy_available():
                    np.random.seed(rng)

                random.seed(rng)
            else:
                raise InvalidArgument("Only `int` is supported for RNG right now.")

        data, tree_spec = tree_flatten(inputs)
        if TYPE_CHECKING:
            data = cast(List[T], data)

        data_aug = [self._aug_fun(d, **self._kwargs) for d in data]

        return tree_unflatten(data_aug, tree_spec)
