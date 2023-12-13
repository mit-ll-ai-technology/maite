.. meta::
   :description: Setting up Image Classifiers.

==================================
Wrap an Image Classification Model
==================================

The MAITE provides a powerful and flexible interface for models
of many shapes, purposes, and provider to interact in a uniform way with
other disparate machine learning objects. Using JATIC wrapped models, we
can run useful functions like :class:`maite.evaluate`, which allows us to seamlessly
evaluate our model using metrics defined from a handful of metric
provider libraries.

In this how-to, you will learn: how to wrap a Pytorch model that you
have created yourself to be compatible with JATIC.  This model's inputs can be extremely
flexible, though should in general take in an ArrayLike image representation
and output predictions in the form of logits, confidence percentages, or other
statistic.  Specifically, outputs can be in the form of HasLogits, HasScores, or HasProbs. 
This demo will show how to:

- Wrapping data outputs
- Defining necessary functions
- Instantiation
- Getting predictions

We assume that you have a JATIC wrapped dataset for the purposes of this
how-to.  Following are the specific imports needed for this demo.

.. code:: python

    from typing import Sequence, TYPE_CHECKING, Callable
    
    from maite import evaluate, load_metric
    from maite.protocols import (
        ImageClassifier,
        HasLogits,
        ArrayLike,
        SupportsArray,
        SupportsImageClassification
    )

Defining Model Outputs
======================

Not only are JATIC models themselves standardized, but also their
outputs. This means we should create a wrapper for our model's outputs.
We will create a :py:func:`~dataclasses.dataclass` called MNISTOutputs. It will simply define
the logits that is a type that implements :class:`~maite.protocols.SupportsArray`.
Though this specific implementation is for logits, a very similar wrapper would be created
for either :class:`~maite.protocols.HasProbs` like outputs, and :class:`~maite.protocols.HasScores` like outputs, depending on the output
of the pytorch model.  The test at the bottom of the code block ensures that our wrapper class
matches the format JATIC expects from our output.

.. code:: python

    from dataclasses import dataclass

    @dataclass
    class MNISTOutput:
        logits: SupportsArray

    if TYPE_CHECKING:
        import torch as tr

        output = MNISTOutput(logits=tr.rand(1, 10))
        assert isinstance(output, HasLogits)


Wrapping Image Classification Models
====================================

With our model output defined, we will now define the model itself.
Image classification must implement the :class:`~maite.protocols.ImageClassifier` protocol.  
That is, it must be a callable function that take in a data object that supports the 
:class:`~maite.protocols.SupportsArray` protocol while also implementing the 
`get_labels` method.  This ``__call__`` function will process images through the model, 
outputting predictions on that data.  In our case, it will output logits.  The ``get_labels``
method returns strings associated with each class that can be output, giving themselves
human readable tags.

Methods
-------

``__call__`` needs the image data provided as an argument, and will
return the predictions the model provides on that data. To ensure JATIC
compliance, the input data should be typed as :class:`~maite.protocols.SupportsArray`.
The function will return the logits prediction output, so we will tag our output with HasLogits.

``get_labels`` doesn't take any arguments, and simply returns a list of the
classes that are predictable. The output is thus marked as a
``Sequence[str]``.

.. code:: python

    class MNISTModel():
        def __init__(self, model: Callable[[SupportsArray], SupportsArray]):
            self.model = model
        
        def __call__(self, data: SupportsArray) -> HasLogits:
            output = self.model(data)
            return MNISTOutput(output)
        
        def get_labels(self) -> Sequence[str]:
            return [
                'car',
                'harbor',
                'helicopter',
                'oil_gas_field',
                'parking_lot',
                'plane',
                'runway_mark',
                'ship',
                'stadium',
                'storage_tank'
            ]


Wrapping our pytorch model is now as simple as calling our wrapper class
with the model itself.  The tests at the bottom of the code block ensures
the input to the model match JATIC expectations, and shows that our wrapped
model conforms to the JATIC ImageClassifier standard.

.. code:: python

    jatic_model = MNISTModel(model)

    if TYPE_CHECKING:
        import torch as tr

        def f(model: ImageClassifier):
            ...

        f(jatic_model)


        data = tr.rand(3, 10, 10)

        assert isinstance(data, ArrayLike)
        assert isinstance(data, SupportsArray)
        assert isinstance(jatic_model, ImageClassifier)

This is the end of this demonstration.  With a wrapped JATIC image classification model,
we can now explore the many other features of the JATIC library, including evaluating the
model against a number of metrics from several different libraries.