.. meta::
   :description: Outline of vision of toolbox protocols.

.. attention::

    Note that this document is a work in progress and is subject to change.

    The code in this document requires:

    .. code-block:: console

        $ pip install python >= 3.11
        $ pip install pyright >= 1.1.3
        $ pip install typing-extensions >= 4.7
        $ pip install phantom-tensors >= 0.2

===========================
Vision of Toolbox Protocols
===========================

This document outlines the vision for the toolbox protocols and their usages.  For documentation on the current
implementation please refer to this :doc:`page <protocols_current>`.
These protocols are designed to define the inputs and outputs of various functions in the toolbox.
They are intended for use with static type checkers like mypy and pyright.

The toolbox protocols aim to help answer the following questions:

1. What types of data are provided by a dataset? (e.g., single image, batch of images, sequence of images)
2. What are the expected inputs and outputs of a given model? (e.g., batch of channel-first images as input, predictions as output)
3. What types of data are needed to compute metrics? (e.g., model outputs, data objects)

By explicitly defining the types of inputs and outputs, these protocols help users avoid common
errors and make debugging easier. They also facilitate the integration of new models, datasets,
and metrics, making it easier to focus on important innovations while improving the quality of
testing and evaluation.

Below is an notional workflow for a user, the goal is to make this workflow easy to use,
well understood, and robust to swapping out models, datasets, and metrics.

.. code-block:: python

    # Types capable of helping developers understand the inputs and outputs
    # of provided models and requirements for a desired workflow

    ModelType = Model["<input type>", "<output type>"]
    DatasetType = Dataset["<data object type>"]
    MetricType = Metric["<model output type>", "<data object type>"]

    def vision_workflow(
        model: ModelType,
        dataset: DatasetType,
        metric: MetricType,
    ) -> "<metric output type>":
        # dataset is a well understood iterator
        for data in dataset:
            # data object is well understoond
            # e.g, we might expect all datasets
            # to contain the following attribute
            # and method
            data_value1 = data.some_data_value1
            data_value2 = data.get_data_value2()           

            # model inputs are well understood
            model_output = model(data["image"])

            # we might expect a model to have the
            # following method
            model_value = model.get_model_value()

            # metric inputs are well understood
            metric.update(model_output, data)

        # metric output is well understood
        metric_output = metric.compute()
        return metric_output


.. dropdown:: Python Imports

    .. code-block:: python

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import (TYPE_CHECKING, Any, Callable, Generic, Iterable, Iterator,
                            Literal, Mapping, NewType, Protocol, Sequence, Type,
                            TypeAlias, TypeVar, runtime_checkable)

        import numpy as np
        import torch as tr
        from phantom_tensors import parse
        from phantom_tensors.array import SupportsArray as ArrayLike
        from phantom_tensors.words import Batch, Channel, Height, Width
        from typing_extensions import TypedDict

.. dropdown:: Generic Types

    .. code-block:: python

        T = TypeVar("T")
        T1 = TypeVar("T1")
        T2 = TypeVar("T2")
        T_co = TypeVar("T_co", covariant=True)
        T_cont = TypeVar("T_cont", contravariant=True)


Array Objects
-------------

The following protocol, `ArrayLike`, covers various array-like objects used in the toolbox,
such as numpy arrays, torch tensors, and other similar objects. `ArrayLike` objects can be
manipulated into arrays of a desired type (e.g., numpy array, torch tensor) and are expected to
have a `shape` attribute that is a tuple of integers.

.. code-block:: python

    from typing_extensions import TypeVarTuple, Unpack

    Shape = TypeVarTuple("Shape")

    @runtime_checkable
    class ArrayLike(Protocol[Unpack[Shape]]):
        def __array__(self) -> Any:
            ...

        @property
        def shape(self) -> tuple[Unpack[Shape]]:
            ...

.. dropdown:: ArrayLike Examples

    The following examples demonstrate the usage of ArrayLike objects:

    **Runtime Validation**

    .. code-block:: python

        assert isinstance(tr.rand(10), ArrayLike)
        assert isinstance(np.zeros(10), ArrayLike)

    **Static Type Checking**

    .. code-block:: python

        if TYPE_CHECKING:

            def supports_array(x: ArrayLike):
                ...

            supports_array(tr.rand(10))
            supports_array(np.zeros(10))

            # fails because python lists do not implement the ArrayLike protocol
            supports_array([1, 2, 3])

Explicitly Typed Arrays
-----------------------

Explicitly typed arrays are useful for defining the expected shape of arrays 
(e.g., channel-first images, batched images) and for specifying explicit types
for model inputs and outputs. To define explicitly typed arrays, we can take
advantage of variadic shapes (as defined in `PEP 646 <https://peps.python.org/pep-0646>`_)). 


This is necessary because the shape of the array is not known at runtime.
Although it may complicate a user's workflow, it supports the vision for the
toolbox and ensures well-understood interfaces.

The following dimension types are used to define the shape of the array:

.. code-block:: python

    from phantom_tensors.words import Batch, Channel, Height, Width

    Category = NewType("Category", int)
    ImageRGB = ArrayLike[Height, Width, Literal[3]]
    ImageChannelLast = ArrayLike[Height, Width, Channel]
    ImageChannelFirst = ArrayLike[Channel, Height, Width]
    BatchImageChannelLast = ArrayLike[Batch, Height, Width, Channel]
    BatchImageChannelFirst = ArrayLike[Batch, Channel, Height, Width]
    Label: TypeAlias = int
    BatchedLabel = ArrayLike[Batch]
    Predictions = NewType("Predictions", int)


.. admonition:: Tip

    We take advantage of `phantom-tensors <https://github.com/rsokl/phantom-tensors>`_  packages
    `parse` function to properly cast arrays to the desired shape.


.. dropdown:: Explicitly Typed Array Examples

    The following examples demonstrate the usage of explicitly typed arrays:

    **Runtime Validation**

    .. code-block:: python

        np_image_last = parse((np.zeros((10, 10, 3)), ImageChannelLast))
        tr_image_first = parse((tr.zeros((3, 10, 10)), ImageChannelFirst))
        false_tr_image_first = parse((tr.zeros((10, 10, 3)), ImageChannelFirst))

        assert isinstance(np_image_last, ArrayLike)
        assert isinstance(tr_image_first, ArrayLike)


    **Static Type Checking**

    .. code-block:: python

        if TYPE_CHECKING:
            # A function that supports a single channel first image
            def supports_chw(x: ImageChannelFirst):
                ...

            # fails because type checker is unaware of the shape dimensions
            supports_chw(tr.rand(3, 10, 10))
            supports_chw(np_image_last)
            supports_chw(tr_image_first)

            # be careful though!!
            supports_chw(false_tr_image_first)


Data Objects
------------

Data objects are assumed to be mappings that contain all the necessary data for
computing model predictions and metrics. For vision tasks, a data object must
have an "image" key. For metrics like accuracy, a data object must have a "label" key.
Other tasks, like object detection, may require additional keys.

**Images and Labels**

.. code-block:: python

    class HasLabel(TypedDict, Generic[T]):
        label: T


    class HasImage(TypedDict, Generic[T]):
        image: T

**Object Detections**

.. code-block:: python

    BoxDim = NewType("BoxDim", int)


    @runtime_checkable
    class Boxes(Protocol[Unpack[Shape]]):
        def __array__(self) -> Any:
            ...

        @property
        def shape(self) -> tuple[Unpack[Shape]]:
            ...

        # TODO: convert output type isn't helpful
        def convert(self, format: str) -> ArrayLike:
            ...


    Tbox = TypeVar("Tbox", bound=Boxes)


    class HasBoxes(TypedDict, Generic[Unpack[Shape]]):
        boxes: Boxes[Unpack[Shape]]


    HasVisionBoxes: TypeAlias = HasBoxes[BoxDim]


    class HasVisionDetections(TypedDict, Generic[Unpack[Shape]]):
        box: Boxes[Unpack[Shape]]
        label: Label

**Task Support**

.. code-block:: python

    class SupportsVisionClassification(TypedDict, Generic[T1, T2]):
        image: T1
        label: T2


    class SupportsVisionObjectDetecton(TypedDict, Generic[T1, T2]):
        image: T1
        objects: Sequence[HasVisionDetections[T2]]


Here we define the protocols for Datasets and DataLoaders and make a distinction between
vision tasks that only require an image and those that require both an image and a label.

.. code-block:: python

    class _Dataset(Protocol[T]):
        def __getitem__(self, idx: Any) -> T:
            ...

        def set_transform(self, transform: Callable[[T], T]):
            ...


    @runtime_checkable
    class ClassLabel(Protocol):
        names: list[str]
        num_classes: int | None  # none to support huggingface feature

        def str2int(self, values: str | Iterable) -> int | Iterable:
            ...

        def int2str(self, values: int | Iterable) -> str | Iterable:
            ...


    class VisionFeatures(TypedDict):
        label: ClassLabel


    @runtime_checkable
    class VisionDataset(_Dataset[HasImage[T]], Protocol[T]):
        features: VisionFeatures


    @runtime_checkable
    class VisionClassificationDataset(
        _Dataset[SupportsVisionClassification[T1, T2]], Protocol[T1, T2]
    ):
        features: VisionFeatures


    @runtime_checkable
    class VisionDataLoader(Protocol[T]):
        def __iter__(self) -> Iterator[HasImage[T]]:
            ...


    @runtime_checkable
    class VisionClassificationDataLoader(Protocol[T1, T2]):
        def __iter__(self) -> Iterator[SupportsVisionClassification[T1, T2]]:
            ...


.. dropdown:: How to Validate TypedDict Protocols

    The following code can be used to validate the above TypedDict protocols.
    Since `isinstance` cannot be used with TypedDicts, a custom function,
    `is_typed_dict`, is used instead.

    .. code-block:: python

        Td = TypeVar("Td", bound=TypedDict)

        def is_typed_dict(object: Any, target: Type[Td]) -> bool:
            if not isinstance(object, dict):
                return False

            k_obj = set(object.keys())
            ks = set(target.__annotations__.keys())

            if hasattr(target, "__total__") and target.__total__:
                return all(k in k_obj for k in ks)
            else:
                return any(k in k_obj for k in ks)

.. dropdown:: Data Object Examples

    The following examples demonstrate the usage of data objects:

    **Runtime Validation**

    .. code-block:: python

        from datasets.features import ClassLabel as ImplClassLabel  # noqa: E402

        assert is_typed_dict({"image": tr_image_first}, HasImage)

        class TestDataset:
            def __init__(self):
                self.features = VisionFeatures(
                    label=ImplClassLabel(names=["a", "b"], num_classes=2)
                )

            def __getitem__(self, idx: Any) -> HasImage[ImageChannelFirst]:
                return HasImage(image=tr_image_first)

            def set_transform(
                self,
                transform: Callable[[HasImage[ImageChannelFirst]], HasImage[ImageChannelFirst]],
            ):
                ...

            def __len__(self) -> int:
                return 10


        class TestDataLoader:
            def __iter__(self) -> Iterator[HasImage[ImageChannelFirst]]:
                for _ in range(10):
                    yield HasImage(image=tr_image_first)


        dataset = TestDataset()
        assert hasattr(dataset, "features")
        assert "label" in dataset.features
        example_label = dataset.features["label"].str2int("a")
        assert example_label == 0

        example_data = dataset[0]
        image_from_example = example_data["image"]

        assert is_typed_dict(example_data, HasImage)
        assert image_from_example.shape[0] == 3


        dataloader = TestDataLoader()
        example_dl = next(iter(dataloader))
        image_from_example_dl = example_dl["image"]

        assert is_typed_dict(example_dl, HasImage)
        assert image_from_example_dl.shape[0] == 3

    **Static Type Checking**

    .. code-block:: python

        if TYPE_CHECKING:

            def supports_has_image(x: HasImage[ImageChannelFirst]):
                ...

            # fails because the type checker is unaware of the shape dimensions
            supports_has_image({"image": tr.zeros(3, 10, 10)})

            # passes because it's properly casted
            supports_has_image({"image": image_from_example})

            def supports_dataset(x: VisionDataset[ImageChannelFirst]):
                ...

            supports_dataset(dataset)

            def supports_dataloader(x: VisionDataLoader[ImageChannelFirst]):
                ...

            supports_dataloader(dataloader)

Model Objects
-------------

Models are assumed to be callable and return an object with attributes required for metric computation. Here
we define a protocol that explicitly requires defining the expected inputs and outputs of a model.  This explicit
definition will make it easier to check that the model is being used correctly.

.. code-block:: python

    @runtime_checkable
    class HasProbs(Protocol):
        probs: ArrayLike[Batch, Category]


    @runtime_checkable
    class HasPredictions(Protocol):
        scores: Sequence[ArrayLike]
        labels: Sequence[ArrayLike]


    @runtime_checkable
    class Model(Protocol[T_cont, T_co]):
        def __call__(self, data: T_cont) -> T_co:
            ...

        def get_labels(self) -> list[str]:
            ...

    
    # an example of explicitly defining a model type with input and output types
    MyVisionModel: TypeAlias = Model[BatchImageChannelFirst, HasPredictions]


.. dropdown:: Model Object Examples

    The following examples demonstrate the usage of model objects:

    **Runtime Validation**

    .. code-block:: python

        class TestModel:
            def __call__(self, data: BatchImageChannelFirst) -> HasPredictions:
                ...

            def get_labels(self) -> list[str]:
                ...


        model = TestModel()
        assert isinstance(model, Model)

    **Static Type Checking**

    .. code-block:: python

        if TYPE_CHECKING:

            def supports_model(x: Model[BatchImageChannelFirst, HasPredictions]):
                ...

            # passes
            supports_model(model)

            # fails because image is the wrong shape
            model(np_image_last)

            # passes because it's properly casted with the correct shape
            batched_np_image_first = parse((np.zeros((5, 3, 10, 10)), BatchImageChannelFirst))
            model(batched_np_image_first)

Metric Protocol
---------------

The Metric protocol supports any type of distributed metric computation. A metric is assumed to be stateful and have a `reset`
method to clear the state. The `update` method is called for each batch of data, and the `compute` method is called
at the end of the evaluation loop to return the final metric value.

.. code-block:: python

    T1_cont = TypeVar("T1_cont", contravariant=True, bound=HasProbs | HasPredictions)
    T2_cont = TypeVar("T2_cont", contravariant=True, bound=Mapping[str, Any])


    @runtime_checkable
    class Metric(Protocol[T1_cont, T2_cont]):
        def reset(self) -> None:
            ...

        def update(self, model_output: T1_cont, target: T2_cont) -> None:
            ...

        def compute(self) -> Mapping[str, Any]:
            ...

.. dropdown:: Metric Object Examples

    The following examples demonstrate the usage of metric objects:

    **Runtime Validation**

    .. code-block:: python

        class TestMetric:
            def reset(self) -> None:
                ...

            def update(
                self, model_output: HasPredictions, target: Mapping[str, ArrayLike]
            ) -> None:
                ...

            def compute(self) -> Mapping[str, Any]:
                ...


        metric = TestMetric()
        assert isinstance(metric, Metric)

    **Static Type Checking**

    .. code-block:: python

        if TYPE_CHECKING:

            def supports_metric(x: Metric[HasPredictions, Mapping[str, ArrayLike]]):
                ...

            supports_metric(metric)  # passes

            def supports_metric_2(x: Metric[HasProbs, Mapping[str, ArrayLike]]):
                ...

            # fails
            supports_metric_2(metric)


            def supports_metric_3(x: Metric[HasPredictions, str]):
                ...

            # fails
            supports_metric_3(metric)


Evaluation Function
-------------------

Here we show an example evalulation workflow in the sampe spirit of what we desired to achieve
at the beginning of this explanation. The evaluation function takes a model, data, and metric
and returns a dictionary of metric values.  The advantage of this workflow is that the code 
inputs and outputs are well understood and can be easily tested.  Below we see:

- The model requires a batch of channel first images and returns an object with a "probs" attribute.
- The data is an iterable of batches of channel first images and labels.
- The metric requires an object with a "probs" attribute and a batch of data with a "label" key.

.. code-block:: python

        class EvalDataLoader(Protocol[T_co]):
            def __iter__(self) -> Iterator[T_co]:
                ...


        def evaluate(
            model: Model[BatchImageChannelFirst, HasProbs],
            data: EvalDataLoader[
                SupportsVisionClassification[BatchImageChannelFirst, BatchedLabel]
            ],
            metric: Metric[HasProbs, HasLabel[BatchedLabel]],
        ) -> Mapping[str, Any]:
            metric.reset()
            for batch in data:
                image = batch["image"]
                output = model(image)

                # validation
                assert isinstance(output, HasProbs)
                assert "label" in batch

                metric.update(output, batch)

            metric_output = metric.compute()
            return metric_output


Implementation Examples
-----------------------

The following examples demonstrate the implementation of model, metric, and evaluation function objects:

.. dropdown:: Dataset Implementation

    .. code-block:: python

        class ImplVisionDataLoader:
            def __init__(self):
                self.images = [tr.rand(5, 3, 32, 32) for _ in range(10)]
                self.labels = [tr.randint(0, 10, (5,)) for _ in range(10)]

            def __iter__(
                self,
            ) -> Iterator[SupportsVisionClassification[BatchImageChannelFirst, BatchedLabel]]:
                for image, label in zip(self.images, self.labels):
                    image, label = parse((image, BatchImageChannelFirst), (label, BatchedLabel))
                    yield SupportsVisionClassification(image=image, label=label)


.. dropdown:: Model Implementation

    .. code-block:: python

        @dataclass
        class ImplModelOutput:
            probs: ArrayLike[Batch, Category]
            labels: BatchedLabel


        class ImplModel:
            def __call__(self, data: BatchImageChannelFirst) -> HasProbs:
                assert isinstance(data, tr.Tensor) or isinstance(data, Sequence)
                probs, labels = parse(
                    (tr.rand(len(data), 10), ArrayLike[Batch, Category]),
                    (tr.arange(len(data)), BatchedLabel),
                )
                return ImplModelOutput(probs, labels)

            def get_labels(self) -> list[str]:
                return [str(i) for i in range(10)]


.. dropdown:: Metric Implementation

    .. code-block:: python

        class AccuracyMetric:
            def __init__(self):
                from torcheval.metrics.classification import MulticlassAccuracy

                self._metric = MulticlassAccuracy()

            def reset(self) -> None:
                self._metric.reset()

            def update(self, model_output: HasProbs, target: HasLabel[BatchedLabel]) -> None:
                probs = model_output.probs
                targets = target["label"]
                assert isinstance(probs, tr.Tensor)
                assert isinstance(targets, tr.Tensor)
                self._metric.update(probs, targets)

            def compute(self) -> Mapping[str, Any]:
                accuracy = self._metric.compute()
                return {"MulticlassAccuracy": accuracy.item()}

Now we can evaluate the model on the dataset.

.. code-block:: python

    dl = ImplVisionDataLoader()
    it = iter(dl)
    example_image = next(it)
    assert isinstance(example_image, dict)
    assert "image" in example_image and "label" in example_image
    assert isinstance(example_image["image"], tr.Tensor)

    model = ImplModel()
    assert isinstance(model, Model)

    metric = AccuracyMetric()
    assert isinstance(metric, Metric)

    output = evaluate(model, dl, metric)  # no type checking issues
    assert isinstance(output, dict)
    assert isinstance(output["MulticlassAccuracy"], float)

