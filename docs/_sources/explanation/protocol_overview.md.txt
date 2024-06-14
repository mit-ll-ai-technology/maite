# Overview of MAITE Protocols

MAITE provides protocols for the following AI components:

* models
* datasets
* dataloaders
* augmentations
* metrics

 MAITE protocols specify expected interfaces of these components (i.e, a minimal set of required attributes, methods, and method type signatures) to promote interoperability in test and evaluation (T&E). This enables the creation of higher-level workflows (e.g., an `evaluate` utility) that can interact with any components that conform to the protocols.

## 1 Concept: Bridging ArrayLikes

MAITE defines a protocol called `ArrayLike` (inspired by NumPy's [interoperability approach](https://numpy.org/devdocs/user/basics.interoperability.html)) that helps components that natively use different flavors of tensors (e.g., NumPy ndarray, PyTorch Tensor, JAX ndarray) work together.

In this example, the functions "type narrow" from `ArrayLike` to the type they want to work with internally. Note that this doesn't necessarily require a conversion depending on the actual input type.


```python
import numpy as np
import torch

from maite.protocols import ArrayLike

def my_numpy_fn(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x)
    # ...
    return arr

def my_torch_fn(x: ArrayLike) -> torch.Tensor:
    tensor = torch.as_tensor(x)
    # ...
    return tensor

# can apply NumPy function to PyTorch Tensor
np_out = my_numpy_fn(torch.rand(2, 3))

# can apply PyTorch function to NumPy array
torch_out = my_torch_fn(np.random.rand(2, 3))

# note: no performance hit from conversion when all `ArrayLike`s are from same library
# or when can share the same underlying memory
torch_out = my_torch_fn(torch.rand(2, 3))
```

By using bridging, we MAITE can permit implementers of the protocol to internally interact with their own types while exposing a more open interface to other MAITE-compliant components.

## 2 Data Types

MAITE represents an *individual* data item as a tuple of:

* input (i.e., image),
* target (i.e., label), and
* metadata (at the datum level)

and a *batch* of data items as a tuple of:

* input batches,
* target batches, and
* metadata batches.

MAITE provides versions of `Model`, `Dataset`, `DataLoader`, `Augmentation`, and `Metric` protocols that correspond to different machine learning tasks (e.g. image classification, object detection) by parameterizing protocol interfaces on the particular input, target, and metadata types associated with that task.

### 2.1 Image Classification

For image classification with `Cl` image classes, we have:

```python
InputType: TypeAlias = ArrayLike  # shape-(C, H, W) tensor with single image
TargetType: TypeAlias = ArrayLike  # shape-(Cl) tensor of one-hot encoded true class or predicted probabilities
DatumMetadataType: TypeAlias = Dict[str, Any]

InputBatchType: TypeAlias = Sequence[ArrayLike]  # element shape-(C, H, W) tensor of N images
TargetBatchType: TypeAlias = Sequence[ArrayLike]  # element shape-(Cl,)
DatumMetadataBatchType: TypeAlias = Sequence[DatumMetadataType]
```

Notes:
* `TargetType` is used for both ground truth (coming from a dataset) and predictions (output from a model). So for a problem with 4 classes,
  * true label of class 2 would be one-hot encoded as `[0, 0, 1, 0]`
  * prediction from a model would be a vector of pseudo-probabilities, e.g., `[0.1, 0.0, 0.7, 0.2]`
* `InputType` and `InputBatchType` are shown with shapes following PyTorch channels-first convention

These type aliases along with the versions of the various component protocols that use these types can be imported from `maite.protocols.image_classification` (if necessary):


```python
# import protocol classes
from maite.protocols.image_classification import (
    Dataset,
    DataLoader,
    Model,
    Augmentation,
    Metric
)

# import type aliases
from maite.protocols.image_classification import (
    InputType,
    TargetType,
    DatumMetadataType,
    InputBatchType,
    TargetBatchType,
    DatumMetadataBatchType
)
```

Alternatively, image classification components and types can be accessed via the module directly:


```python
import maite.protocols.image_classification as ic

# model: ic.Model = load_model(...)
```

### 2.2 Object Detection

For object detection with `D_i` detections in an image `i`, we have:

```python
class ObjectDetectionTarget(Protocol):
    @property 
    def boxes(self) -> ArrayLike: ...  # shape-(D_i, 4) tensor of bounding boxes w/format X0, Y0, X1, Y1

    @property
    def labels(self) -> ArrayLike: ... # shape-(D_i) tensor of labels for each box

    @property
    def scores(self) -> ArrayLike: ... # shape-(D_i) tensor of scores for each box (e.g., probabilities)

InputType: TypeAlias = ArrayLike  # shape-(C, H, W) tensor with single image
TargetType: TypeAlias = ObjectDetectionTarget
DatumMetadataType: TypeAlias = Dict[str, Any]

InputBatchType: TypeAlias = Sequence[ArrayLike]  # sequence of N ArrayLikes each of shape (C, H, W)
TargetBatchType: TypeAlias = Sequence[TargetType]   # sequence of object detection "target" objects
DatumMetadataBatchType: TypeAlias = Sequence[DatumMetadataType]
```

Notes:
* `ObjectDetectionTarget` contains a single label and score per box
* `InputType` and `InputBatchType` are shown with shapes following PyTorch channels-first convention

## 3 Models

All models implement a `__call__` method that takes the `InputBatchType` and produces the `TargetBatchType` appropriate for the given machine learning task.


```python
import maite.protocols.image_classification as ic
print(ic.Model.__doc__)
```

    
        A model protocol for the image classification ML subproblem.
    
        Implementers must provide a `__call__` method that operates on a batch of model
        inputs (as `Sequence[ArrayLike]) and returns a batch of model targets (as
        `Sequence[ArrayLike]`)
    
        Methods
        -------
    
        __call__(input_batch: Sequence[ArrayLike]) -> Sequence[ArrayLike]
            Make a model prediction for inputs in input batch. Input batch is expected to
            be `Sequence[ArrayLike]` with each element of shape `(C, H, W)`.
        



```python
import maite.protocols.object_detection as od
print(od.Model.__doc__)
```

    
        A model protocol for the object detection ML subproblem.
    
        Implementers must provide a `__call__` method that operates on a batch of model inputs
        (as `Sequence[ArrayLike]`s) and returns a batch of model targets (as
        `Sequence[ObjectDetectionTarget]`)
    
        Methods
        -------
    
        __call__(input_batch: Sequence[ArrayLike]) -> Sequence[ObjectDetectionTarget]
            Make a model prediction for inputs in input batch. Elements of input batch
            are expected in the shape `(C, H, W)`.
        


## 4 Datasets and DataLoaders

`Dataset`s provide access to single data items and `DataLoader`s  provide access to batches of data with the input, target, and metadata types corresponding to the given machine learning task.


```python
print(ic.Dataset.__doc__)
```

    
        A dataset protocol for image classification ML subproblem providing datum-level
        data access.
    
        Implementers must provide index lookup (via `__getitem__(ind: int)` method) and
        support `len` (via `__len__()` method). Data elements looked up this way correspond
        to individual examples (as opposed to batches).
    
        Indexing into or iterating over the an image_classification dataset returns a
        `Tuple` of types `ArrayLike`, `ArrayLike`, and `Dict[str,Any]`.
        These correspond to the model input type, model target type, and datum-level
        metadata, respectively.
    
        Methods
        -------
    
        __getitem__(ind: int) -> Tuple[ArrayLike, ArrayLike, Dict[str, Any]]
            Provide map-style access to dataset elements. Returned tuple elements
            correspond to model input type, model target type, and datum-specific metadata type,
            respectively.
    
        __len__() -> int
            Return the number of data elements in the dataset.
    
        Examples
        --------
    
        We create a dummy set of data and use it to create a class that implements
        this lightweight dataset protocol:
    
        >>> import numpy as np
        >>> from typing import List, Dict, Any, Tuple
        >>> from maite.protocols import ArrayLike
    
        Assume we have 5 classes, 10 datapoints, and 10 target labels, and that we want
        to simply have an integer 'id' field in each datapoint's metadata:
    
        >>> N_CLASSES: int = 5
        >>> N_DATUM: int = 10
        >>> images: List[np.ndarray] = [np.random.rand(3, 32, 16) for _ in range(N_DATUM)]
        >>> targets: np.ndarray = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, N_DATUM)]
        >>> metadata: List[Dict] = [{'id': i} for i in range(N_DATUM)]
    
        Constructing a compliant dataset just involves a simple wrapper that fetches
        individual datapoints, where a datapoint is a single image, target, metadata 3-tuple.
    
        >>> class ImageDataset:
        ...     def __init__(self,
        ...                  images: List[np.ndarray],
        ...                  targets: np.ndarray,
        ...                  metadata: List[Dict[str, Any]]):
        ...         self.images = images
        ...         self.targets = targets
        ...         self.metadata = metadata
        ...     def __len__(self) -> int:
        ...         return len(images)
        ...     def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        ...         return self.images[ind], self.targets[ind], self.metadata[ind]
    
        We can instantiate this class and typehint it as an image_classification.Dataset.
        By using typehinting, we permit a static typechecker to verify protocol compliance.
    
        >>> from maite.protocols import image_classification as ic
        >>> dataset: ic.Dataset = ImageDataset(images, targets, metadata)
    
        Note that when writing a Dataset implementer, return types may be narrower than the
        return types promised by the protocol (np.ndarray is a subtype of ArrayLike), but
        the argument types must be at least as general as the argument types promised by the
        protocol.
        



```python
print(ic.DataLoader.__doc__)
```

    
        A dataloader protocol for the image classification ML subproblem providing
        batch-level data access.
    
        Implementers must provide an iterable object (returning an iterator via the
        `__iter__` method) that yields tuples containing batches of data. These tuples
        contain types `Sequence[ArrayLike]` (elements of shape `(C, H, W)`),
        `Sequence[ArrayLike]` (elements shape `(Cl, )`), and `Sequence[Dict[str, Any]]`,
        which correspond to model input batch, model target type batch, and a datum metadata batch.
    
        Note: Unlike Dataset, this protocol does not require indexing support, only iterating.
    
        Methods
        -------
    
        __iter__ -> Iterator[tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[Dict[str, Any]]]]
            Return an iterator over batches of data, where each batch contains a tuple of
            of model input batch (as `Sequence[ArrayLike]`), model target batch (as
            `Sequence[ArrayLike]`), and batched datum-level metadata
            (as `Sequence[Dict[str,Any]]`), respectively.
    
        



```python
print(od.DataLoader.__doc__)
```

    
        A dataloader protocol for the object detection ML subproblem providing
        batch-level data access.
    
        Implementers must provide an iterable object (returning an iterator via the
        `__iter__` method) that yields tuples containing batches of data. These tuples
        contain types `Sequence[ArrayLike]` (elements of shape `(C, H, W)`),
        `Sequence[ObjectDetectionTarget]`, and `Sequence[Dict[str, Any]]`,
        which correspond to model input batch, model target type batch, and a datum metadata batch.
    
        Note: Unlike Dataset, this protocol does not require indexing support, only iterating.
    
    
        Methods
        -------
    
        __iter__ -> Iterator[tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[Dict[str, Any]]]]
            Return an iterator over batches of data, where each batch contains a tuple of
            of model input batch (as `Sequence[ArrayLike]`), model target batch (as
            `Sequence[ObjectDetectionTarget]`), and batched datum-level metadata
            (as `Sequence[Dict[str,Any]]`), respectively.
    
        


## 5 Augmentations

`Augmentation`s take in and return a batch of data with the `InputBatchType`, `TargetBatchType`, and `DatumMetadataBatchType` types corresponding to the given machine learning task.

Augmentations can access the datum-level metadata associated with each data item to potentially tailor the augmentation to individual items. Augmentations can also associate new datum-level metadata with each data item, e.g., documenting aspects of the actual change that was applied (e.g., the actual rotation angle sampled from a range of possible angles).


```python
print(ic.Augmentation.__doc__)
```

    
        An augmentation protocol for the image classification subproblem.
    
        An augmentation is expected to take a batch of data and return a modified version of
        that batch. Implementers must provide a single method that takes and returns a
        labeled data batch, where a labeled data batch is represented by a tuple of types
        `Sequence[ArrayLike]` (with elements of shape `(C, H, W)`), `Sequence[ArrayLike]`
        (with elements of shape `(Cl, )`), and `Sequence[Dict[str,Any]]`. These correspond
        to the model input batch type, model target batch type, and datum-level metadata
        batch type, respectively.
    
        Methods
        -------
    
        __call__(datum: Tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[dict[str, Any]]]) ->          Tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[dict[str, Any]]])
            Return a modified version of original data batch. A data batch is represented
            by a tuple of model input batch (as `Sequence[ArrayLike]` with elements of shape
            `(C, H, W)`), model target batch (as an `Sequence[ArrayLike]` of shape `(N, Cl)`),
            and batch metadata (as `Sequence[Dict[str, Any]]`), respectively.
    
        Examples
        --------
    
        We can write an implementer of the augmentation class as either a function or a class.
        The only requirement is that the object provide a __call__ method that takes objects
        at least as general as the types promised in the protocol signature and return types
        at least as specific.
    
        >>> import copy
        >>> import numpy as np
        >>> from typing import Dict, Any, Tuple, Sequence
        >>> from maite.protocols import ArrayLike
        >>>
        >>> class ImageAugmentation:
        ...     def __call__(
        ...         self,
        ...         data_batch: Tuple[Sequence[ArrayLike], Sequence[ArrayLike], Sequence[Dict[str, Any]]]
        ...     ) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[Dict[str, Any]]]:
        ...         inputs, targets, mds = data_batch
        ...         # We copy data passed into the constructor to avoid mutating original inputs
        ...         # By using np.ndarray constructor, the static type-checker will let us treat
        ...         # generic ArrayLike as a more narrow return type
        ...         inputs_aug = [copy.copy(np.array(input)) for input in inputs]
        ...         targets_aug = [copy.copy(np.array(target)) for target in targets]
        ...         mds_aug = copy.deepcopy(mds)  # deepcopy in case of nested structure
        ...         # Modify inputs_aug, targets_aug, or mds_aug as needed
        ...         # In this example, we just add a new metadata field
        ...         for i, md in enumerate(mds_aug):
        ...             md['new_key'] = i
        ...         return inputs_aug, targets_aug, mds_aug
    
        We can typehint an instance of the above class as an Augmentation in the
        image_classification domain:
    
        >>> from maite.protocols import image_classification as ic
        >>> im_aug: ic.Augmentation = ImageAugmentation()
    
        



```python
print(od.Augmentation.__doc__)
```

    
        An augmentation protocol for the object detection subproblem.
    
        An augmentation is expected to take a batch of data and return a modified version of
        that batch. Implementers must provide a single method that takes and returns a
        labeled data batch, where a labeled data batch is represented by a tuple of types
        `Sequence[ArrayLike]`, `Sequence[ObjectDetectionTarget]`, and `Sequence[Dict[str,Any]]`.
        These correspond to the model input batch type, model target batch type, and datum-level
        metadata batch type, respectively.
    
        Methods
        -------
    
        __call__(datum: Tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[dict[str, Any]]]) ->          Tuple[Sequence[ArrayLike], Sequence[ObjectDetectionTarget], Sequence[dict[str, Any]]]
            Return a modified version of original data batch. A data batch is represented
            by a tuple of model input batch (as `Sequence ArrayLike` with elements of shape
            `(C, H, W)`), model target batch (as `Sequence[ObjectDetectionTarget]`), and
            batch metadata (as `Sequence[Dict[str,Any]]`), respectively.
        


## 6 Metrics

The `Metric` protocol is inspired by the design of existing libraries like Torchmetrics and Torcheval. The `update` method operates on batches of predictions and truth labels by either caching them for later computation of the metric (via `compute`) or updating sufficient statistics in an online fashion.


```python
print(ic.Metric.__doc__)
```

    
        A metric protocol for the image classification ML subproblem.
    
        A metric in this sense is expected to measure the level of agreement between model
        predictions and ground-truth labels.
    
        Methods
        -------
    
        update(preds: Sequence[ArrayLike], targets: Sequence[ArrayLike]) -> None
            Add predictions and targets to metric's cache for later calculation. Both
            preds and targets are expected to be sequences with elements of shape `(Cl,)`.
    
        compute() -> Dict[str, Any]
            Compute metric value(s) for currently cached predictions and targets, returned as
            a dictionary.
    
        reset() -> None
            Clear contents of current metric's cache of predictions and targets.
        



```python
print(od.Metric.__doc__)
```

    
        A metric protocol for the object detection ML subproblem.
    
         A metric in this sense is expected to measure the level of agreement between model
         predictions and ground-truth labels.
    
         Methods
         -------
    
         update(preds: Sequence[ObjectDetectionTarget], targets: Sequence[ObjectDetectionTarget]) -> None
             Add predictions and targets to metric's cache for later calculation.
    
         compute() -> Dict[str, Any]
             Compute metric value(s) for currently cached predictions and targets, returned as
             a dictionary.
    
         reset() -> None
             Clear contents of current metric's cache of predictions and targets.
        


## 7 Workflows

MAITE provides high-level utilities for common workflows such as `evaluate` and `predict`. They can be called with either `Dataset`s or `DataLoader`s, and with optional `Augmentation`.

The `evaluate` function can optionally return the model predictions and (potentially-augmented) data batches used during inference.

The `predict` function returns the model predictions and (potentially-augmented) data batches used during inference, essentially calling `evaluate` with a dummy metric.


```python
from maite.workflows import evaluate, predict
# we can also import from object_detection module
# where the function call signature is the same
```


```python
print(evaluate.__doc__)
```

    
        Evaluate a model's performance on data according to some metric with optional augmentation.
    
        Some data source (either a dataloader or a dataset) must be provided
        or an InvalidArgument exception is raised.
    
        Parameters
        ----------
        model : SomeModel
            Maite Model object.
    
        metric : Optional[SomeMetric], (default=None)
            Compatible maite Metric.
    
        dataloader : Optional[SomeDataloader], (default=None)
            Compatible maite dataloader.
    
        dataset : Optional[SomeDataset], (default=None)
            Compatible maite dataset.
    
        batch_size : int, (default=1)
            Batch size for use with dataset (ignored if dataset=None).
    
        augmentation : Optional[SomeAugmentation], (default=None)
            Compatible maite augmentation.
    
        return_augmented_data : bool, (default=False)
            Set to True to return post-augmentation data as a function output.
    
        return_preds : bool, (default=False)
            Set to True to return raw predictions as a function output.
    
        Returns
        -------
        Tuple[Dict[str, Any], Sequence[TargetType], Sequence[Tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]]]
            Tuple of returned metric value, sequence of model predictions, and
            sequence of data batch tuples fed to the model during inference. The actual
            types represented by InputBatchType, TargetBatchType, and DatumMetadataBatchType will vary
            by the domain of the components provided as input arguments (e.g. image
            classification or object detection.)
            Note that the second and third return arguments will be empty if
            return_augmented_data is False or return_preds is False, respectively.
        



```python
print(predict.__doc__)
```

    
        Make predictions for a given model & data source with optional augmentation.
    
        Some data source (either a dataloader or a dataset) must be provided
        or an InvalidArgument exception is raised.
    
        Parameters
        ----------
        model : SomeModel
            Maite Model object.
    
        dataloader : Optional[SomeDataloader], (default=None)
            Compatible maite dataloader.
    
        dataset : Optional[SomeDataset], (default=None)
            Compatible maite dataset.
    
        batch_size : int, (default=1)
            Batch size for use with dataset (ignored if dataset=None).
    
        augmentation : Optional[SomeAugmentation], (default=None)
            Compatible maite augmentation.
    
        Returns
        -------
        Tuple[Sequence[SomeTargetBatchType], Sequence[Tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
            A tuple of the predictions (as a sequence of batches) and a sequence
            of tuples containing the information associated with each batch.
        

