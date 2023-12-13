==========================================
Wrap a Custom Image Classification Dataset
==========================================

The MAITE defines a protocol for image classification datasets,
:class:`~maite.protocols.VisionDataset`, that standardizes the dataset interface for use with
other MAITE methods.
Many JATIC users may come to the MAITE with
their own custom dataset that they wish to use with our tools,
but does not initially conform to our protocol.

In this How-To, we will walk through the process of wrapping a
custom image classification dataset to conform to JATIC's
:class:`~maite.protocols.VisionDataset` protocol.

After familiarizing ourselves with the :class:`~maite.protocols.VisionDataset` protocol,
we will:

1. Start with a custom image classification dataset that does not conform to :class:`~maite.protocols.VisionDataset`
2. Create a custom wrapper for our dataset so it conforms to :class:`~maite.protocols.VisionDataset`
3. Verify that a wrapped dataset does indeed conform to the JATIC API


0. The :class:`~maite.protocols.VisionDataset` protocol
===============================================================

A dataset conforming to JATIC's :class:`~maite.protocols.VisionDataset` should contain the following two methods:

- ``__len__``, which returns an integer value representing the number of data samples in the dataset
- ``__getitem__``, which takes in an `int` index, and returns an output that is of type :class:`~maite.protocols.SupportsImageClassification`

:class:`~maite.protocols.SupportsImageClassification` is a custom JATIC data type that is also of type
:class:`~maite.protocols.HasDataImage` and :class:`~maite.protocols.HasDataLabel`.
It is a :py:class:`~typing.TypedDict`
that contains an "image" of type :class:`~maite.protocols.SupportsArray` and a "label" of type ``int``,
a sequence of ``int``, or :class:`~maite.protocols.SupportsArray`. And :class:`~maite.protocols.SupportsArray` covers
many common array-like structures, such as
`PyTorch Tensor <https://pytorch.org/docs/stable/tensors.html>`__
and `NumPy NDArray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
(as well as sequences of these data structures).

Next, we'll walk through an example of wrapping a custom dataset
to conform to this protocol.


1. Starting with a custom image classification dataset
======================================================

We start by assuming the user already has a custom class defined for their dataset.

For example, here we define a generic dataset that leverages Torchvision's
`ImageFolder <https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder>`__ class:

.. code:: python

    from torchvision.datasets import ImageFolder

    # Example custom dataset using Torchvision's ImageFolder class
    class CustomDataset(ImageFolder):
        def __init__(self, root: str):
            super().__init__(root)

        def __getitem__(self, index: int):
            image, label = super().__getitem__(index)
            return (image, label) # does not currently conform to JATIC's SupportsImageClassification

The use of ``ImageFolder`` in this example dataset is meant for illustrative purposes only,
as it assumes the dataset is organized using a specific folder structure.

Currently, ``__getitem__`` returns a ``Tuple``, however to conform to the JATIC API, we need it to
return an output of the form :class:`~maite.protocols.SupportsImageClassification`.

2. Creating a custom dataset wrapper
====================================

To conform to :class:`~maite.protocols.VisionDataset`,
create a wrapper for your custom dataset which includes ``__len__`` and ``__getitem__`` methods,
with ``__getitem__`` returning an output of type :class:`~maite.protocols.SupportsImageClassification`.

For example:

.. code:: python

    from torch import Tensor
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import ToTensor
    from maite.protocols import SupportsImageClassification

    class JaticDatasetWrapper():
        def __init__(self, custom_dataset: ImageFolder):
            self.custom_dataset = custom_dataset
            self.transform = ToTensor()
        
        def __len__(self) -> int:
            return len(self.custom_dataset)
        
        def __getitem__(self, index: int) -> SupportsImageClassification:
            data = self.custom_dataset[index]
            image: Tensor = self.transform(data[0])
            label: int = data[1]
            return {"image": image, "label": label}

In this example, we used Torchvision's ``ToTensor`` transform to ensure that
our image was converted to a ``Tensor``, which is of type :class:`~maite.protocols.SupportsArray`.
However, there are other data types that also conform to :class:`~maite.protocols.SupportsArray`
(e.g., ``numpy.ndarray``)
that could have been used as well, depending on the needs of the user.
Additionally, a user may wish to apply other
transforms to their data before returning the dictionary output,
which is allowed as long as the final "image" and "label" adhere to
the data types specified by :class:`~maite.protocols.HasDataImage`
and :class:`~maite.protocols.HasDataLabel`, respectively.

Note that we have also included type annotations in our dataset wrapper,
which will make it easier to perform static type checking
later on to ensure our dataset properly adheres to the JATIC protocols.
While type annotations are not required, we encourage users to include them
for enhanced documentation and safegaurding.

You are now ready to instantiate your JATIC-conforming dataset as follows:

.. code:: python

    custom_dataset = CustomDataset(root="<...>")
    jatic_dataset = JaticDatasetWrapper(custom_dataset)


3. Verifying the wrapped dataset conforms to JATIC protocols
============================================================

You can verify that your dataset does indeed conform to
the JATIC protocols by running the following through a
static type checker:

.. code:: python

    from maite.protocols import VisionDataset

    def f(dataset: VisionDataset):
            ...

    f(jatic_dataset) # should pass

Here, we create an empty method ``f()`` to test whether an input to the function
adheres to the :class:`~maite.protocols.VisionDataset` protocol, according
to its type annotations.

Additionally, you can perform an instance check on your wrapped
dataset to verify that it contains the required
methods for the :class:`~maite.protocols.VisionDataset` protocol,
and leverage :func:`~maite.protocols.is_typed_dict`, a JATIC helper method,
to verify that the output of a dataset is indeed a typed dictionary with the required keys:

.. code:: python

    import maite.protocols as pr

    assert isinstance(jatic_dataset, pr.VisionDataset) # should pass
    assert is_typed_dict(jatic_dataset[0], pr.SupportsImageClassification) # should pass
    assert is_typed_dict(jatic_dataset[0], pr.HasDataImage) # should pass
    assert is_typed_dict(jatic_dataset[0], pr.HasDataLabel) # should pass

We have now walked through the process of wrapping a custom image classification
dataset for use with the MAITE. Your dataset is now ready for use with
other tools in the JATIC ecosystem, such as running an evaluation using the 
maite's :func:`maite.evaluate` method.
