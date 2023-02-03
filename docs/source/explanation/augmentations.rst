.. meta::
   :description: Augmentation Protocol.


========================================
Explanation of the Augmentation Protocol
========================================

A Python :py:class:`typing.Protocol` is a way to define a common interface for a group of related classes. 
In this case, we can define a :py:class:`typing.Protocol` for JATIC augmentation libraries that take in an array
or any nested collection of arrays (e.g., `dict`, `list`, `tuple`, `dict`) comprising of the same type of arra (e.g. `np.ndarray`).
Additionally, since most augmentations rely on random numbers, a user must be able to optionally pass in a random number
generator object capable of reproducing results.  

First, lets show the Protocol defined in :func:`jatic_toolbox.protocols.augmentation.Augmentation`:


.. code-block:: python
    :caption: Augmentation Protocol

    class Augmentation(Protocol):
        def __call__(
            self, *inputs: PyTree[T], rng: Optional[RandomNumberGenerator] = None
        ) -> PyTree[T]:
            """
            Applies an agumentation to each item in the input and returns a corresponding container of augmented items.

            Inputs can be arrays or nested data structures of data collections (e.g., list, tuple, dict).

            Parameters
            ----------
            *inputs : PyTree[T]
                Any arbitrary structure of nested Python containers, e.g., list of image arrays.
                All types comprising the tree must be the same.

            rng : RandomNumberGenerator | None (default: None)
                An optional random number generator for reproducibility.

            Returns
            -------
            PyTree[T]
                A corresponding collection of transformed objects.
            """
            ...

The `Augmentation` Protocol defines a callable object that takes in a `PyTree` (described in more detail below): 
an array, list of arrays, dictionaries of arrays, or tuple of arrays. Addiation, an optional random number generator (`rng`)
can be provided to support reproducibility. The protocol specifies that the `__call__` method should return
the augmented data of the same type of type as the input.


Any class that wants to conform to this protocol must implement this `__call__` method. Once you have defined this protocol, 
you can use it as a type hint for the objects in your augmentation library and to check that any user defined class conform
to the protocol.

PyTree
------

In this explanation we will implement a `RandomCrop` augmentation for images.  Before implementing the augmentation we should
introduce a very concept of `PyTree`. There are multiple variations of this concept developed for:

- `PyTorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/_pytree.py>`_ 
- `JAX PyTrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_
- `Optimized PyTrees <https://github.com/metaopt/optree>`_

The JAX documentation provides a very good overview of the concept.  For the rest of this document we will limit our use
case to the PyTorch implementation. Note, none of the implementation above require special tensors or containers to function. 
The two functions most relevant to augmentations is :obj:`torch.utils._pytree.tree_flatten` and
:obj:`torch.utils._pytree.tree_unflatten`. Here's an example using simple lists:

.. code-block:: python
    :caption: PyTree Example With Simple Lists

    from torch.utils._pytree import tree_flatten, tree_unflatten

    flat_inputs, tree_spec = tree_flatten(([1, 2, 3], (4, 5, 6), dict(a=[7,8,9])))
    # flat_inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # add 10 to each value
    augment = [x + 10 for x in flat_inputs]

    # move back to original
    aug_inputs = tree_unflatten(augment, tree_spec)
    # aug_inputs = ([11, 12, 13], (14, 15, 16), {"a": [17, 18, 19])


And another example using NumPy arrays:

.. code-block:: python
    :caption: PyTree Example Numpy Arrays

    from torch.utils._pytree import tree_flatten, tree_unflatten
    import numpy as np

    x = np.zeros((2, 2))
    y = np.ones((2, 2))

    flat_inputs, tree_spec = tree_flatten((x, dict(a=y)))
    # [array([[0., 0.],
    #        [0., 0.]]),
    # array([[1., 1.],
    #        [1., 1.]])]

    # add 10 to each value
    augment = [x + 10 for x in flat_inputs]

    # move back to original
    aug_inputs = tree_unflatten(augment, tree_spec)
    # (array([[10., 10.],
    #        [10., 10.]]),
    # {'a': array([[11., 11.],
    #             [11., 11.]])})


It should be clear how this is useful for common ML tasks in object detection and segmentation where one often requires similar augmentations to be
performed on the input and the target variable.  Similarly for sequences of data that may require the same augmentation across all sequences.


Example
-------

Now lets implement an augmentation that implements our Protocol.

.. code-block:: python
    :caption: Random Crop Augmentation

    from numpy.random import Generator, default_rng
    from torch.utils._pytree import tree_flatten, tree_unflatten

    import numpy as np


    class RandomCrop(Augmentation):
        def __init__(self, size: Tuple[int, int]):
            """
            Randomly crop the last two dimension of an array (e.g., height and width of image).

            Parameters
            ----------
            size: Tuple[int, int]
                The desired dimensions of the last two dimensions of the array.
            """
            super().__init__()
            self.output_size = size

        def _get_params(self, flat_inputs: List[ArrayLike], rng: Generator) -> Dict[str, int]:
            """
            Calculate the parameters of the random crop to be applied for all inputs.

            Parameters
            ----------
            flat_inputs: List[ArrayLike]
                A set of inputs 
            
            rng: Generator
            """
            assert len(flat_inputs) > 0
            h, w = np.asarray(flat_inputs[0]).shape
            th, tw = self.output_size

            dw = 0
            if tw < w:
                dw = rng.integers(0, w - tw)

            dh = 0
            if th < h:
                dh = rng.integers(0, h - th)

            return dict(bottom=dh, top=dh + th, left=dw, right=dw + tw)

        def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            """Apply the augmentation."""
            return inpt[..., params["bottom"] : params["top"], params["left"] : params["right"]]

        def __call__(self, *inputs: List[ArrayLike], rng: Optional[Generator] = None) -> List[ArrayLike]:
            if rng is None:
                rng = default_rng()

            # flatten the inputs
            flat_inputs, spec = tree_flatten(inputs)

            # calculate the parameters for cropping all inputs
            params = self._get_params(flat_inputs, rng=rng)

            # apply the augmentation
            flat_outputs = [self._transform(inpt, params) for inpt in flat_inputs]

            # return augmented objects in original format
            return tree_unflatten(flat_outputs, spec)

Here are a few examples of executing `RandomCrop`:

.. code-block:: python
    :caption: Examples Executing RandomCrop

    import numpy as np
    from numpy.random import default_rng

    random_crop = RandomCrop((2, 1))

    # Example 1: Simple Array
    x = np.arange(16).reshape(4, 4)
    xout, = random_crop(x, rng=default_rng(0))
    print("x:", xout.shape)
    # prints "x: (2, 1)"

    # Example 2: Multiple Arrays
    y = np.arange(100, 120).reshape(1, 5, 4)
    xout, yout = random_crop(x, y, rng=default_rng(0))
    print("y:", yout.shape)
    # prints "y: (1, 2, 1)"

    # Example 2: Multiple Arrays
    xy, = random_crop([x, y], rng=default_rng(0))

    # Example 3: List of Array, Tuple, and Dict
    z = dict(val=np.arange(1000, 1032).reshape(1, 2, 4, 4))
    xyz, = random_crop([x, y, z], rng=default_rng(0))
    print("z:", xyz[2]["val"].shape)
    # prints "z: (1, 2, 2, 1)"