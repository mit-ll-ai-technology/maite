.. _define-new-ai-problem:

================================
Define a New AI Problem
================================

This guide shows how to extend MAITE by defining a new AI problem type. MAITE uses a **plugin-inspired architecture** where types are defined for primitives and generic component protocols are specialized with those types. No subclassing is required.

.. contents::
   :local:
   :depth: 2

Overview: The Plugin Architecture
==================================

MAITE's extensibility is based on a simple pattern:

1. **Define primitives** (or reuse existing ones)
2. **Create type aliases** for InputType, TargetType, DatumMetadataType
3. **Specialize generic protocols** by parameterizing them with those types
4. **Document everything** in numpydoc-formatted docstrings

.. figure:: ../_static/images/plugin_architecture_flow.svg
   :width: 700
   :align: center
   :alt: Flow diagram showing how to define a new AI problem

   **Plugin Architecture Flow**: Start with primitives, create type aliases, specialize protocols.

.. note::
   **Why Structural Types?**

   MAITE often opts to use structural subtypes rather than nominal subtypes to permit
   less coupling. Structural subtypes like Python protocol classes and TypedDicts can be
   subclassed explicitly, but do not need to be. Their purpose is defining expected type
   signatures that implementers must satisfy in a way that is checkable by a static
   typechecker. Classes with the right attributes and methods automatically satisfy the
   protocol without explicit subclassing.

Step 1: Identify Primitives
============================

Primitives are the basic data types an AI problem uses:

- **InputType**: What the model takes as input
- **TargetType**: What the model predicts (or ground truth labels)
- **DatumMetadataType**: Metadata associated with each data point

Reusing Existing Primitives
----------------------------

**Key principle**: Reuse primitives across AI problems when possible.

For example, both ``image_classification`` and ``object_detection`` use the same ``Image`` primitive::

    # Both use this
    Image: TypeAlias = ArrayLike  # (C, H, W) shape, where ArrayLike is from maite.protocols

**When to reuse:**

- ✅ Input is an image → use :py:class:`~maite.protocols.ArrayLike`
- ✅ Need image-level metadata → use :py:class:`~maite.protocols.DatumMetadata`
- ✅ Classification labels → reuse label type from ``image_classification``

**When to define new:**

- ❌ New data modality (e.g., audio, text, point clouds)
- ❌ Complex structured output (e.g., segmentation masks, object detections)
- ❌ Domain-specific metadata

.. figure:: ../_static/images/primitive_reuse.svg
   :width: 600
   :align: center
   :alt: Diagram showing Image primitive used in multiple AI problems

   **Primitive Reuse**: The ``Image`` primitive is shared across image classification,
   object detection, and semantic segmentation.

Example: Text Classification Primitives
-----------------------------------------

Let's define primitives for a new text classification problem::

    # src/maite/_internals/protocols/text_classification.py

    from __future__ import annotations
    from typing import Protocol, runtime_checkable
    from typing_extensions import TypeAlias
    from maite.protocols import ArrayLike, DatumMetadata

    # New primitive: Text is just a string
    Text: TypeAlias = str
    """Semantic alias for text input.

    Use `Text` when referring to raw text strings that serve as model input.
    """

    # New primitive: Text classification target
    @runtime_checkable
    class TextClassificationTarget(Protocol):
        """
        A text classification target protocol.

        Represents classification labels for text, supporting both
        one-hot encoding and class probabilities.

        Attributes
        ----------
        labels : ArrayLike
            Class labels with shape ``(N_CLASSES,)``
        """
        @property
        def labels(self) -> ArrayLike:
            ...

    # Reuse existing metadata
    DatumMetadataType: TypeAlias = DatumMetadata
    """Role alias for datum-level metadata in text-classification protocol signatures.

    Use `DatumMetadataType` in generic protocol contexts where metadata appears as a
    type argument. Currently equivalent to `DatumMetadata`.
    """

Step 2: Define Type Aliases
============================

Type aliases separate inherent types from the roles they play in an AI problem.

Semantic vs Role Aliases
-------------------------

MAITE uses two kinds of type aliases:

1. **Semantic aliases**: Give domain meaning to inherent types::

    # ArrayLike is inherent type (object coercible to numpy array)
    # Image gives it semantic meaning
    Image: TypeAlias = ArrayLike
    """Semantic alias for a single image datum.

    Use `Image` when you want to emphasize domain meaning ("this value is an image"),
    rather than protocol position. Expected shape semantics are `(C, H, W)`.
    """

2. **Role aliases**: Define positions in protocol signatures::

    # Image is semantic type
    # InputType defines its role in protocols
    InputType: TypeAlias = Image
    """Role alias for model/dataset input in the text-classification protocol family.

    Use `InputType` in generic protocol contexts where the type parameter represents
    "input position". Currently equivalent to `Text`.
    """

    TargetType: TypeAlias = TextClassificationTarget
    """Role alias for model/dataset target in the text-classification protocol family.

    Use `TargetType` in generic protocol contexts where the type parameter represents
    "target position". Currently equivalent to `TextClassificationTarget`.
    """

    DatumMetadataType: TypeAlias = DatumMetadata
    """Role alias for datum-level metadata in text-classification protocol signatures.

    Use `DatumMetadataType` in generic protocol contexts where metadata appears as a
    type argument. Currently equivalent to `DatumMetadata`.
    """

The distinction:

- :py:class:`~maite.protocols.ArrayLike` is what the object technically **is** (inherent type)
- ``Image`` is what the object **means** in the domain (semantic alias)
- ``InputType`` is what **role** the object plays in protocols (role alias)

Role aliases are used when specializing generic protocols::

    class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
        # InputType occupies the "input position" in the generic protocol
        ...

Step 3: Specialize Generic Protocols
======================================

Type aliases specialize the generic component protocols:

Dataset Protocol
----------------

::

    from maite._internals.protocols import generic as gen

    class Dataset(
        gen.Dataset[InputType, TargetType, DatumMetadataType],
        Protocol
    ):
        """
        A dataset protocol for text classification AI problem providing
        datum-level data access.

        Implementers must provide index lookup (via ``__getitem__(ind: int)`` method)
        and support ``len`` (via ``__len__()`` method). Data elements looked up
        this way correspond to individual examples (as opposed to batches).

        Indexing into or iterating over a text classification dataset returns a
        ``tuple`` of types ``str``, ``TextClassificationTarget``, and ``DatumMetadata``.
        These correspond to the model input type, model target type, and datum-level
        metadata, respectively.

        Methods
        -------
        __getitem__(ind: int) -> tuple[str, TextClassificationTarget, DatumMetadata]
            Provide map-style access to dataset elements. Returned tuple elements
            correspond to model input type, model target type, and datum-specific
            metadata, respectively.

        __len__() -> int
            Return the number of data elements in the dataset.

        Attributes
        ----------
        metadata : DatasetMetadata
            A typed dictionary containing at least an 'id' field of type str

        Examples
        --------

        Create a simple text classification dataset:

        >>> from maite.protocols import DatasetMetadata, DatumMetadata
        >>> from maite.protocols import text_classification as tc
        >>> import numpy as np

        >>> class SimpleTextDataset:
        ...     metadata: DatasetMetadata = {"id": "simple_text_dataset"}
        ...
        ...     def __init__(self, texts: list[str], labels: list[int]):
        ...         self.texts = texts
        ...         self.labels = labels
        ...
        ...     def __len__(self) -> int:
        ...         return len(self.texts)
        ...
        ...     def __getitem__(self, idx: int) -> tuple[str, tc.TargetType, DatumMetadata]:
        ...         text = self.texts[idx]
        ...         # Convert label to one-hot
        ...         label_array = np.zeros(10)
        ...         label_array[self.labels[idx]] = 1
        ...         target = MyTarget(labels=label_array)
        ...         metadata: DatumMetadata = {"id": str(idx)}
        ...         return text, target, metadata

        >>> # Type hint enables static type checking
        >>> dataset: tc.Dataset = SimpleTextDataset(["hello", "world"], [0, 1])
        """
        ...

Model Protocol
--------------

::

    class Model(gen.Model[InputType, TargetType], Protocol):
        """
        A model protocol for the text classification AI problem.

        Implementers must provide a ``__call__`` method that operates on a batch
        of model inputs (as ``Sequence[str]``) and returns a batch of model targets
        (as ``Sequence[TextClassificationTarget]``).

        Methods
        -------
        __call__(input_batch: Sequence[str]) -> Sequence[TextClassificationTarget]
            Make a model prediction for inputs in input batch.

        Attributes
        ----------
        metadata : ModelMetadata
            A typed dictionary containing at least an 'id' field of type str

        Examples
        --------

        Create a simple text classification model:

        >>> from maite.protocols import ModelMetadata
        >>> from maite.protocols import text_classification as tc
        >>> from typing import Sequence
        >>> import numpy as np

        >>> class SimpleTextModel:
        ...     metadata: ModelMetadata = {"id": "simple_classifier"}
        ...
        ...     def __call__(self, batch: Sequence[str]) -> Sequence[tc.TargetType]:
        ...         # Dummy model: predict based on text length
        ...         predictions = []
        ...         for text in batch:
        ...             probs = np.random.random(10)
        ...             probs /= probs.sum()
        ...             predictions.append(MyTarget(labels=probs))
        ...         return predictions

        >>> model: tc.Model = SimpleTextModel()
        """
        ...

.. tip::
   **Follow the Pattern**

   Look at ``image_classification.py`` and ``object_detection.py`` as templates.
   They follow the same pattern:

   1. Import generic protocols
   2. Define/reuse primitives
   3. Create type aliases
   4. Specialize each protocol (Dataset, DataLoader, Model, Metric, Augmentation)

Step 4: Document with Numpydoc
================================

MAITE uses **class-level** numpydoc docstrings for proper Sphinx rendering.

Key Requirements
----------------

✅ **Class-level docstrings**: Put docstring on the class, not individual methods

✅ **Numpydoc format**: Use proper sections (Parameters, Returns, Attributes, Examples)

✅ **Shape semantics**: Document array shapes in prose (typechecker can't verify)

✅ **Examples**: Include working examples that can be doctested

Example with All Sections::

    class MyProtocol(Protocol):
        """
        One-line summary.

        More detailed description of what this protocol represents and when
        to use it. Explain the role it plays in the AI problem.

        Methods
        -------
        method_name(arg1: Type1, arg2: Type2) -> ReturnType
            Brief description of what this method does.

        Attributes
        ----------
        attr_name : AttrType
            Description of the attribute and its purpose.

        Notes
        -----
        Document any semantic requirements that can't be enforced by the
        type checker. For example:

        - Expected array shapes (e.g., "(C, H, W)")
        - Coordinate system conventions (e.g., "x0, y0, x1, y1 format")
        - Value ranges (e.g., "probabilities must sum to 1.0")
        - Ordering requirements (e.g., "boxes should be sorted by score")

        Examples
        --------

        Always include at least one working example:

        >>> from maite.protocols import my_module as mm
        >>>
        >>> class MyImplementation:
        ...     # Implementation here
        ...     pass
        >>>
        >>> obj: mm.MyProtocol = MyImplementation()
        """

Step 5: Verify Type Completeness
==================================

MAITE maintains 100% type completeness. After defining protocols, verify::

    # Run pyright with verifytypes
    $ pyright --verifytypes maite

This checks:

- All public APIs have type annotations
- All generic type parameters are specified
- Return types are properly annotated
- No use of ``Any`` without justification

.. code-block:: python

   # ✅ Good: Fully typed
   def my_function(x: ArrayLike) -> ArrayLike:
       return x

   # ❌ Bad: Missing return type
   def my_function(x: ArrayLike):
       return x

   # ❌ Bad: Using Any
   def my_function(x: Any) -> Any:
       return x

Step 6: Add to API Reference
==============================

Finally, add the new AI problem to the API documentation:

1. Create ``docs/source/api/custom_problem.rst`` following the pattern of existing AI problem API files (see ``api/object_detection.rst`` as a reference)

2. Add the new file to ``docs/source/api/protocols.rst`` in the toctree

3. Update ``src/maite/protocols/__init__.py`` to export the types

See existing AI problems (object detection, image classification) for complete examples of API documentation structure.

Complete Example: Semantic Segmentation
========================================

Here's a complete minimal example for semantic segmentation:

.. code-block:: python

   # src/maite/_internals/protocols/semantic_segmentation.py

   from __future__ import annotations
   from typing import Protocol
   from typing_extensions import TypeAlias
   from maite._internals.protocols import generic as gen
   from maite.protocols import ArrayLike, DatumMetadata

   # Reuse existing primitives
   Image: TypeAlias = ArrayLike
   """Semantic alias for a single image datum.

   Use `Image` when you want to emphasize domain meaning ("this value is an image"),
   rather than protocol position. Expected shape semantics are `(C, H, W)`.
   """

   # New primitive: segmentation mask
   SegmentationMask: TypeAlias = ArrayLike
   """Semantic alias for a segmentation mask.

   Use `SegmentationMask` when referring to per-pixel class labels. Expected shape
   semantics are `(H, W)` where each pixel value is an integer class ID.
   """

   # Role aliases
   InputType: TypeAlias = Image
   """Role alias for model/dataset input in the semantic-segmentation protocol family.

   Use `InputType` in generic protocol contexts where the type parameter represents
   "input position". Currently equivalent to `Image`.
   """

   TargetType: TypeAlias = SegmentationMask
   """Role alias for model/dataset target in the semantic-segmentation protocol family.

   Use `TargetType` in generic protocol contexts where the type parameter represents
   "target position". Currently equivalent to `SegmentationMask`.
   """

   DatumMetadataType: TypeAlias = DatumMetadata
   """Role alias for datum-level metadata in semantic-segmentation protocol signatures.

   Use `DatumMetadataType` in generic protocol contexts where metadata appears as a
   type argument. Currently equivalent to `DatumMetadata`.
   """

   class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
       """
       A dataset protocol for semantic segmentation.

       Returns tuples of (image, mask, metadata) where:
       - image: ArrayLike of shape (C, H, W)
       - mask: ArrayLike of shape (H, W) with integer class labels
       - metadata: DatumMetadata dict with 'id' field

       Examples
       --------
       >>> import numpy as np
       >>> from maite.protocols import DatasetMetadata, DatumMetadata
       >>> from maite.protocols import semantic_segmentation as ss
       >>>
       >>> class MySegDataset:
       ...     metadata: DatasetMetadata = {"id": "seg_dataset"}
       ...
       ...     def __init__(self):
       ...         self.images = [np.random.rand(3, 100, 100) for _ in range(10)]
       ...         self.masks = [np.random.randint(0, 5, (100, 100)) for _ in range(10)]
       ...
       ...     def __len__(self) -> int:
       ...         return len(self.images)
       ...
       ...     def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
       ...         return self.images[idx], self.masks[idx], {"id": str(idx)}
       >>>
       >>> dataset: ss.Dataset = MySegDataset()
       """
       ...

Best Practices
==============

1. **Reuse Primitives**

   Check existing AI problems first. If they use similar data, reuse their primitives.

2. **Structural Types**

   Define primitives as protocols when they have structure (like ``ObjectDetectionTarget``).
   Use simple type aliases when they're just wrappers (like ``Image = :py:class:`~maite.protocols.ArrayLike```).

3. **Shape Documentation**

   Always document expected array shapes in docstrings (typechecker can't verify these).

4. **Examples in Docstrings**

   Include at least one working example for each protocol. Make them runnable with doctest.

5. **Class-Level Docs**

   Put numpydoc docstrings on the class, not on individual protocol methods.

6. **Consistency**

   Follow the patterns in existing AI problems (object_detection, image_classification).

Troubleshooting
===============

Protocol Not Recognized
-----------------------

**Problem**: ``MyClass`` should satisfy protocol but typechecker says it doesn't.

**Solution**: Check that:

- Method signatures match exactly (including argument names for non-positional args)
- All required attributes are present
- Attributes are properties if protocol defines them as properties
- Used ``@runtime_checkable`` decorator on protocol

Type Parameter Mismatch
-----------------------

**Problem**: ``Generic[T1, T2, T3]`` doesn't match specialization.

**Solution**: Count your type parameters:

- Dataset, DataLoader, Augmentation: 3 parameters (Input, Target, Metadata)
- Model: 2 parameters (Input, Target)
- Metric: 2 parameters (Target, Metadata)

Shape Mismatches
----------------

**Problem**: Arrays have wrong shape at runtime but typechecker doesn't catch it.

**Solution**: Shape semantics aren't enforced by typechecker. Document them clearly:

.. code-block:: python

   """
   Parameters
   ----------
   image : ArrayLike
       Input image with shape ``(C, H, W)`` where C is channels,
       H is height, and W is width. **Must** follow this shape.
   """

Further Reading
===============

- :ref:`api_generic` - Generic protocol documentation
- :ref:`maite-vision` - MAITE's vision for interoperability
- :ref:`how_to/static_typing` - Static typing guide
- Python typing spec: https://typing.readthedocs.io/en/latest/spec/protocol.html

.. rubric:: Footnotes

