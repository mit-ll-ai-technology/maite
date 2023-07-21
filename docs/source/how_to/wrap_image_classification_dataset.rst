=================================================
How to wrap a custom image classification dataset
=================================================

The JATIC Toolbox defines a protocol for image classification datasets,
:class:`~jatic_toolbox.protocols.VisionDataset`, that standardizes the dataset interface for use with
other toolbox methods.
The toolbox also provides APIs for listing and loading datasets from common providers,
including `HuggingFace <https://huggingface.co/docs/datasets/index>`__ and
`Torchvision <https://pytorch.org/vision/stable/datasets.html>`__,
that automatically adhere to this protocol.
However, many JATIC users may come to the toolbox with
their own custom dataset that they wish to use with our tools.

In this How-To, we will walk through the process of wrapping a
custom image classification dataset to conform to JATIC's
:class:`~jatic_toolbox.protocols.VisionDataset` protocol:

1. First, we will familiarize ourselves with the :class:`~jatic_toolbox.protocols.VisionDataset` protocol
2. Next, we will define a custom image classification dataset
3. Then, we will wrap our dataset to conform to :class:`~jatic_toolbox.protocols.VisionDataset`
4. Additionally, we will show an alternative approach leveraging JATIC's ``TorchVisionDataset`` wrapper
5. Finally, we will verify that our wrapped dataset does indeed conform our API


1. The :class:`~jatic_toolbox.protocols.VisionDataset` protocol
===============================================================

Similar to PyTorch's `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__ class,
a dataset conforming to JATIC's :class:`~jatic_toolbox.protocols.VisionDataset` should contain both a
``__len__`` and ``__getitem__`` method.
The dataset class should have the following structure:

.. code:: python

    from typing import Any
    from jatic_toolbox.protocols import SupportsImageClassification

    class VisionDataset():
        def __len__(self) -> int:
            ...

        def __getitem__(self, index: Any) -> SupportsImageClassification
            ...

Thus, ``__len__`` must return an integer value representing the number of
data samples in the dataset, and
``__getitem__`` must return an output that is of type :class:`~jatic_toolbox.protocols.SupportsImageClassification`.

:class:`~jatic_toolbox.protocols.SupportsImageClassification` is a custom JATIC data type that is also of type
:class:`~jatic_toolbox.protocols.HasDataImage` and :class:`jatic_toolbox.protocols.HasDataLabel`.
It is a :py:class:`typing.TypedDict`
that contains an "image" of type :class:`~jatic_toolbox.protocols.SupportsArray` and a "label" of type ``int``,
a sequence of ``int``, or :class:`~jatic_toolbox.protocols.SupportsArray`. And :class:`~jatic_toolbox.protocols.SupportsArray` covers
many common array-like structures, such as
`PyTorch Tensors <https://pytorch.org/docs/stable/tensors.html>`__
and `Numpy arrays <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`__
(as well as sequences of these data structures).

Next, we'll walk through two example approaches to wrapping a custom dataset
that conforms to these protocols.


2. Define a custom image classification dataset
===============================================

Define a class for your dataset that contains a ``__len__`` and ``__getitem__`` method.

For example, here we define a dataset that leverages Torchvision's
`ImageFolder <https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder>`__ class:

.. code:: python

    from typing import Any, Tuple
    from torchvision.datasets import ImageFolder

    # Custom dataset using Torchvision's ImageFolder class
    class CustomDataset(ImageFolder):
        def __init__(self, root: str):
            super().__init__(root)

        def __len__(self) -> int:
            return super().__len__()

        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            return super().__getitem__(index)

Note that the use of ``ImageFolder`` is meant for illustrative purposes only,
as it assumes your dataset is organized using a specific folder structure.
Feel free to define your dataset using any approach of your choosing, as long
as it contains the necessary methods.

Currently, our ``__getitem__`` method returns a ``Tuple``, however we need it to
return an output of the form :class:`~jatic_toolbox.protocols.SupportsImageClassification`.

3. Wrap dataset to conform to :class:`~jatic_toolbox.protocols.VisionDataset`
=============================================================================

To conform to :class:`~jatic_toolbox.protocols.VisionDataset`,
modify the output of ``__getitem__`` to return an output of type :class:`~jatic_toolbox.protocols.SupportsImageClassification`.

For example:

.. code:: python

    from torch import Tensor
    from torchvision.transforms import ToTensor
    from jatic_toolbox.protocols import SupportsImageClassification

    class JaticDataset(CustomDataset):
        def __getitem__(self, index: int) -> SupportsImageClassification:
            data = super().__getitem__(index)
            transform = ToTensor()
            image: Tensor = transform(data[0])
            label: int = data[1]
            return SupportsImageClassification(image=image, label=label)

Note that here, we used Torchvision's ``ToTensor`` transform to ensure that
our image was converted to a ``Tensor``, which is of type :class:`~jatic_toolbox.protocols.SupportsArray`.
However, there are other data types that also conform to :class:`~jatic_toolbox.protocols.SupportsArray`
(e.g., ``numpy.array``)
that could have been used as well, depending on the needs of the user.
Additionally, a user may wish to apply other
transforms to their data before returning the dictionary output,
which is allowed as long as the final "image" and "label" adhere to
the data types specified by :class:`~jatic_toolbox.protocols.HasDataImage`
and :class:`~jatic_toolbox.protocols.HasDataLabel`, respectively.

You are now ready to instantiate your JATIC-conforming dataset:

.. code:: python

    jatic_dataset = JaticDataset(
        root=<...> # path to data directory
    )

4. Alternative approach leveraging JATIC's ``TorchVisionDataset`` wrapper
=========================================================================

In our previous example, our custom dataset inherited from Torchvision's
``ImageFolder`` class. Rather than modifying our dataset's ``__getitem__`` method
directly, we also could have leveraged ``jatic_toolbox.interop.TorchVisionDataset``,
which can be used to convert any Torchvision dataset into a JATIC :class:`~jatic_toolbox.protocols.VisionDataset`.

For example, first instantiate a dataset that conforms to Torchvision's dataset API:

.. code:: python
    
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import ToTensor

    custom_dataset = ImageFolder(
        root=<...>, # path to data directory
        transform=ToTensor()
    )

Again, we are using ``ImageFolder`` for illustrative purposes, but you can
use other Torchvision dataset classes to instantiate a custom dataset instance as well.

Next, utilize JATIC's ``TorchVisionDataset`` to wrap the custom dataset:

.. code:: python

    from jatic_toolbox.interop.torchvision import TorchVisionDataset

    jatic_dataset = TorchVisionDataset(custom_datatset)

This wrapped dataset will now automatically produce outputs that are of the
form :class:`~jatic_toolbox.protocols.SupportsImageClassification`.


5. Verify the wrapped dataset conforms to JATIC protocols
=========================================================

You can now verify that your dataset does indeed conform to
the JATIC protocols:

.. code:: python

    from typing import TYPE_CHECKING
    from jatic_toolbox.protocols import (
        HasDataImage,
        HasDataLabel,
        is_typed_dict,
        SupportsArray,
        SupportsImageClassification,
        VisionDataset
    )

    if TYPE_CHECKING:
        def f(dataset: VisionDataset):
            ...

        f(jatic_dataset)
        
    assert is_typed_dict(jatic_dataset[0], SupportsImageClassification)
    assert isinstance(jatic_dataset[0], HasDataImage)
    assert isinstance(jatic_dataset[0], HasDataLabel)
    assert isinstance(jatic_dataset[0]["image"], SupportsArray)

We have now walked through the process of wrapping a custom image classification
dataset for use with the JATIC toolbox. Your dataset is now ready for use with
other tools in the JATIC ecosystem, such as running an evaluation using the 
toolbox's :func:`jatic_toolbox.evaluate` method.
