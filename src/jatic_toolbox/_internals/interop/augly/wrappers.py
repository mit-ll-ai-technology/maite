import random
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from augly.image import aug_np_wrapper
from numpy.typing import NDArray
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing_extensions import TypeAlias

from jatic_toolbox.protocols import TypedCollection

__all__ = ["Augly"]

Audio: TypeAlias = NDArray
Image: TypeAlias = PILImage
URL: TypeAlias = Union[str, List[str]]
T: TypeAlias = Union[Image, Audio, URL, Tensor]


class Augly:
    def __init__(self, aug_fun: Callable[..., T], **kwargs: Any):
        """
        Return a JATIC Augmentation for an AugLy [1]_ transform.

        Parameters
        ----------
        aug_fun : Callable[..., T]
            An AugLy callable to transform data.

        **kwargs : Any
            Optional arguments for `aug_fun`.

        References
        ----------
        [1] 'Augly data augmentation library <https://github.com/facebookresearch/AugLy>`_

        Examples
        --------
        First create a random image using NumPy:

        >>> import numpy
        >>> img_np = numpy.random.rand(100,100,3) * 255

        Now get the JATIC Augmentation for a given Augly
        transformation:

        >>> from augly.images import RandomAspectRatio
        >>> xform = Augly(RandomAspectRatio)

        Lastly, generate the augmentation:

        >> aug_img_np = xform(img_np)
        """
        self.aug_fun = aug_fun
        self.kwargs = kwargs

    def __call__(
        self,
        *inputs: TypedCollection[T],
        rng: Optional[Union[int, random.Random]] = None,
    ) -> Union[TypedCollection[T], Tuple[TypedCollection[T], ...]]:
        """
        Pipeline for augmentating data with Augly [1]_.

        Parameters
        -----------
        *inputs : TypedCollection[T]
            Inputs to augment.

        rng : Optional[RandomState] = None
            For reproducibility.

        Returns
        -------
        TypedCollection[T]
            All augmented inputs in the same format is
            the inputs.

        Raises
        ------
        InvalidArgument
            Raises if the `rng` keyword is not either `random.Random` or `int`.

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
        >>> xform = Augly(RandomAspectRatio())

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

        Lastly, we can support reproducible Augly transforms:

        >>> import random
        >>> out = xform(img_pil, rng=1)
        >>> out2 = xform(img_pil, rng=1)
        >>> out3 = xform(img_pil, rng=2)
        >>> out4 = xform(img_pil, rng=random.Random(1))
        >>> assert np.all(out[0]==out2[0])
        >>> assert np.all(out[0]!=out3[0])
        >>> assert np.all(out[0]==out4[0])
        """
        if rng is not None:
            if isinstance(rng, int):
                random.seed(rng)
            elif isinstance(rng, random.Random):
                random.setstate(rng.getstate())
            else:
                raise ValueError(
                    f"rng must be int or `random.Random` object not {type(rng)}"
                )

        if isinstance(rng, int):
            random.seed(rng)
        elif isinstance(rng, random.Random):
            random.setstate(rng.getstate())

        data, tree_spec = tree_flatten(inputs)

        if isinstance(data[0], np.ndarray):
            data_aug = [
                aug_np_wrapper(image=d, aug_function=self.aug_fun, **self.kwargs) for d in data  # type: ignore
            ]
        else:
            data_aug = [self.aug_fun(d, **self.kwargs) for d in data]

        outputs = tree_unflatten(data_aug, tree_spec)
        if len(inputs) == 1:
            return outputs[0]
        return outputs
