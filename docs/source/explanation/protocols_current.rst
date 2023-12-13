.. meta::
   :description: Current implementation of toolbox protocols.

.. attention::

   Protocols are still under development, any feedback on expected behavior, use cases,
   and desired functionality is welcome. Please open an issue on the 
   `Toolbox Repo <https://gitlab.jatic.net/jatic/cdao/maite/-/issues>`_.

===========================================
Current Implementation of Toolbox Protocols
===========================================

This explanation outlines the current implementation of the protocols used in the toolbox. For documentation
on the vision of the protocols, see :doc:`Vision of Toolbox Protocols <protocols_vision>`.  

The protocols described in this document aim to assist developers in defining interfaces for their Python code,
ensuring compatibility with the toolbox. When creating specific interfaces for others to use in workflows,
developers are encouraged to adopt these protocols. However, it is advisable for developers to avoid
incorporating the protocols into their own code to prevent unnecessary complexity and to prioritize static
type checking with interfaces. Minimal validation should be emphasized, especially when speed is not a primary concern.

The toolbox protocols aim to help define and self-document code to make it easier to understand and use across different
tasks. They also help to ensure that code is compatible with the toolbox, making it easier to integrate new models,
datasets, and metrics. The protocols are designed to be flexible and extensible, allowing developers to define
interfaces that are specific to their needs while still being compatible with the toolbox.

An example workflow defined purely with toolbox protocols is shown below. This workflow explicitly defines the
inputs to the workflow along with the requirements on the interfaces of objects such as models, datasets, and metrics.

.. code-block:: python

   def my_workflow(
      data: DataLoader[SupportsImageClassification],
      model: Model[[SupportsArray], HasProbs],
      metric: Metric[[HasProbs, HasDataLabel], float]
   ):
      metric.reset()  # <-- expects this method
      for batch in data:
         # batch is a dictionary with keys "image" and "label"
         image: SupportsArray = batch["image"]

         # model_output is a dictionary with keys "probs" and "logits"
         model_output: HasProbs = model(image)

         # metric.update expects a dictionary with "label" and
         # and a named attributed with "probs"
         metric.update(model_output, batch) 

      return metric.compute()  # <-- expects this method


By explicitly defining the types of inputs and outputs, these protocols help users avoid common
errors and make debugging easier. They also facilitate the integration of new models, datasets,
and metrics, making it easier to focus on important innovations while improving the quality of
testing and evaluation.

.. dropdown:: Python Imports

    .. code-block:: python

      from __future__ import annotations

      from dataclasses import dataclass
      from typing import runtime_checkable, Protocol


      import maite.protocols as pr
      import numpy as np
      import torch as tr
      from maite.testing.pyright import pyright_analyze
      from PIL import Image


Array Objects
-------------

An :class:`~maite.protocols.ArrayLike` defines a common interface for objects that can be manipulated as arrays, regardless of the specific implementation.
This allows code to be written in a more generic way, allowing it to work with different array-like objects without having to
worry about the details of the specific implementation. With an :class:`~maite.protocols.ArrayLike` protocol, developers can write functions and algorithms
that operate on arrays without defining the specific implementation of arrays to use. 

This will improve code readability and maintainability, as well as make it easier to switch to different array implementations
if needed. For example, developers can write functions that takes an :class:`~maite.protocols.ArrayLike` object as input and perform some mathematical
operation on the elements. This function would work with any object that satisfies the :class:`~maite.protocols.ArrayLike` protocol,
such as a numpy `ndarray`, a PyTorch `tensor`, or a custom object that implements the same methods and attributes as
the :class:`~maite.protocols.ArrayLike` protocol. In addition, an :class:`~maite.protocols.ArrayLike` protocol is useful for providing type hints and improving code safety,
as it can be used in conjunction with a static type checker to ensure that the correct types of objects are being passed
as arguments. This can help catch errors before they cause problems at runtime.

.. code-block:: python

   @runtime_checkable
   class ArrayLike(Protocol):
      def __array__(self) -> Any:
         ...

.. dropdown:: Validation

   .. code-block:: python

      # ArrayLike requires objects that implement `__array__` or `__array_interface__`.
      assert not isinstance([1, 2, 3], pr.ArrayLike)

      np_array = np.zeros((10, 10), dtype=np.uint8)
      assert isinstance(np_array, pr.ArrayLike)
      assert isinstance(tr.as_tensor(np_array), pr.ArrayLike)

      # Pillow images do not implement `__array__` and therefore
      # do not technically pass typing check.
      # However, they can be converted to numpy arrays and pass the check.
      from PIL import Image
      array = Image.fromarray(np_array)
      assert not isinstance(array, pr.ArrayLike)

      assert isinstance(np.asarray(array), pr.ArrayLike)

.. dropdown:: Type Checking

   .. admonition:: Note
   
      The use of `pyright_analyze` requires the `pyright` package to be installed
      and all imports and code to be within a function. See :class:`maite.testing.pyright.pyright_analyze`
      for more details.

   .. code-block:: python

      from maite.testing.pyright import pyright_analyze

      def test_array_like():
         import maite.protocols as pr
         import numpy as np
         import torch as tr
         from PIL import Image
         
         def array_like(x: pr.ArrayLike):
            ...

         np_array = np.zeros((10, 10, 3), dtype=np.uint8)
         array_like(np_array)  # passes pyright
         array_like([np_array])  # does not pass pyright 

         array_like(tr.as_tensor(np_array))  # passes pyright
         array_like([tr.as_tensor(np_array)])  # does not pass pyright 

         # Pillow images do not implement `__array__` and therefore
         # do not technically pass typing check.
         # However, they can be converted to numpy arrays and pass the check.
         array = Image.fromarray(np_array)
         array_like(array)  # does not pass pyright
         array_like([array])  # does not pass pyright

         # convert array to numpy array works though
         assert array_like(np.asarray(array))
         assert array_like([np.asarray(array)])

      out = pyright_analyze(test_array_like)[0]
      assert out["summary"]["errorCount"] == 5, out["summary"]["errorCount"]


Data Objects
------------

Data objects are assumed to be mappings that contain all the necessary data for
computing model predictions and metrics. For vision tasks, a data object must
have an "image" key. For metrics like accuracy, a data object must have a "label" key.

**Data Containers**

.. code-block:: python

   SupportsArray: TypeAlias = Union[ArrayLike, Sequence[ArrayLike]]

   class HasDataImage(TypedDict):
      image: SupportsArray


   class HasDataLabel(TypedDict):
      label: Union[int, SupportsArray, Sequence[int]]


   class HasDataBoxes(TypedDict):
      boxes: SupportsArray


   class HasDataBoxesLabels(HasDataBoxes):
      labels: Union[Sequence[int], SupportsArray]


   class HasDataObjects(TypedDict):
      objects: Union[HasDataBoxesLabels, Sequence[HasDataBoxesLabels]]


**Task Support**

.. code-block:: python

   class SupportsImageClassification(HasDataImage, HasDataLabel):
      ...


   class SupportsObjectDetection(HasDataImage, HasDataObjects):
      ...

**Dataset**

.. code-block:: python

   @runtime_checkable
   class Dataset(Protocol[T_co]):
      def __len__(self) -> int:
         ...

      def __getitem__(self, index: Any) -> T_co:
         ...


   VisionDataset: TypeAlias = Dataset[SupportsImageClassification]
   ObjectDetectionDataset: TypeAlias = Dataset[SupportsObjectDetection]


.. dropdown:: Validation

   .. code-block:: python

      array = tr.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
      assert not pr.is_typed_dict(array, pr.HasDataImage)
      assert not pr.is_typed_dict({"not_image": array}, pr.HasDataImage)
      assert pr.is_typed_dict({"image": array}, pr.HasDataImage)

      from PIL import Image
      array = Image.fromarray(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
      assert pr.is_typed_dict({"image": array}, pr.HasDataImage) 

.. dropdown:: Type Checking

   .. admonition:: Note
   
      The use of `pyright_analyze` requires the `pyright` package to be installed
      and all imports and code to be within a function. See :class:`maite.testing.pyright.pyright_analyze`
      for more details.

   .. code-block:: python

      from maite.testing.pyright import pyright_analyze

      def test_supports_image():
         from typing import cast
         import maite.protocols as pr
         import torch as tr

         def supports_image(x: pr.HasDataImage):
            ...

         array = tr.zeros(3, 10, 10)
         supports_image(array)  # does not pass pyright
         supports_image({"not_image": array})  # does not pass pyright
         supports_image({"image": array})  # passes pyright

      results = pyright_analyze(test_supports_image)[0]
      assert results["summary"]["errorCount"] == 3, results["summary"]["errorCount"]


Model Objects
-------------

Models are assumed to be callable and return an object with attributes required for metric computation.

For image classification tasks, a model output must either have a "probs" vector across all categories
or a "predictions" output containing "scores" and "labels" attributes.

**Model Outputs**

.. code-block:: python

   class HasLabel(Protocol):
      label: SupportsArray


   class HasBoxes(Protocol):
      boxes: SupportsArray


   class HasLogits(Protocol):
      logits: SupportsArray


   class HasProbs(Protocol):
      probs: SupportsArray


   class HasScores(Protocol):
      scores: SupportsArray
      labels: SupportsArray


   class HasDetectionLogits(HasBoxes, HasLogits, Protocol): ...
   class HasDetectionProbs(HasProbs, HasBoxes, Protocol): ...
   class HasDetectionPredictions(HasBoxes, HasScores, Protocol):  ...

**Models**

.. code-block:: python

   class Model(Protocol[P, T]):
      __call__: Callable[P, T]
      def get_labels(self) -> Sequence[str]: ...

   ImageClassifier = Model[[SupportsArray], Union[HasLogits, HasProbs, HasScores]]
   ObjectDetector = Model[[SupportsArray], Union[HasDetectionLogits, HasDetectionProbs, HasDetectionPredictions]]


.. dropdown:: Validation

   .. code-block:: python

      from dataclasses import dataclass
      import maite.protocols as pr
      import torch as tr

      @dataclass
      class DummyOutputTensor:
         probs: tr.Tensor

      @dataclass
      class DummyOutput:
         probs: pr.SupportsArray

      import torch as tr
      array = tr.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

      assert not isinstance(array, pr.HasProbs)
      assert not isinstance({"probs": array}, pr.HasProbs) 
      assert isinstance(DummyOutputTensor(array), pr.HasProbs)
      assert isinstance(DummyOutput(array), pr.HasProbs)


.. dropdown:: Type Checking

   .. admonition:: Note
   
      The use of `pyright_analyze` requires the `pyright` package to be installed
      and all imports and code to be within a function. See :class:`maite.testing.pyright.pyright_analyze`
      for more details.

   .. code-block:: python

      from maite.testing.pyright import pyright_analyze

      def test_supports_probs():
         from dataclasses import dataclass
         import maite.protocols as pr
         import torch as tr

         @dataclass
         class DummyOutputTensor:
            probs: tr.Tensor

         @dataclass
         class DummyOutput:
            probs: pr.SupportsArray

         def supports_probs(x: pr.HasProbs):
            ...

         array = tr.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
         supports_probs(array)  # does not pass pyright
         supports_probs({"probs": array})  # does not pass pyright (needs to be named attribute)

         # In the spirit of the toolbox this should pass.
         # Future work will be to support this.
         supports_probs(DummyOutputTensor(array))  # does not pass pyright

         supports_probs(DummyOutput(array))  # passes pyright

      assert pyright_analyze(test_supports_probs)[0]["summary"]["errorCount"] == 3

Metric Protocol
---------------

The Metric protocol supports any type of distributed metric computation.
A metric is assumed to be stateful and have a `reset` method to clear the state.
The `update` method is called for each batch of data, and the `compute` method is called
at the end of the evaluation loop to return the final metric value.

.. code-block:: python

   class Metric(Protocol[P, T]):
      reset: Callable[[], None]
      update: Callable[[P], None]
      compute: Callable[[], T]
      to: Callable[..., Self]


.. dropdown:: Metric Object Examples

    The following examples demonstrate the usage of metric objects:

    .. code-block:: python

        class TestMetric:
            def reset(self) -> None:
                ...

            def update(self, probs: ArrayLike, label: ArrayLike) -> None:
                ...

            def compute(self) -> float:
                ...


        metric = TestMetric()
        assert isinstance(metric, pr.Metric)

        if TYPE_CHECKING:

            def supports_metric(x: pr.Metric[[ArrayLike, ArrayLike], float]) -> None:
                ...

            supports_metric(metric)  # passes


