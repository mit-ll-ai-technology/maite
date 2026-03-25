.. _maite_users:

========================================
MAITE User Types
========================================

The four architectural layers of MAITE (described in :ref:`maite_layered_architecture`.) correspond to four distinct user types, each operating at different level of abstraction. [#practical_layer_spanning_caveat]_
These tiers represent a gradient from rich problem context (at the application level) to broad interoperability context (at the architectural level).

Component/Task Integrator (Level 3)
-----------------------------------

Developers integrating components and tasks operate at the application level with rich problem context.
They work with specific datasets, concrete model architectures, and particular application requirements, understanding the nuances of their specific problem domain (e.g., pedestrian detection in autonomous vehicles, medical image classification).

**Activities:**

- Compose existing components (datasets, models, metrics) and/or tasks
- Work in the context of a predefined AI problem type

**Example:**

.. code-block:: python

   from maite.tasks import evaluate

   # Use existing components
   dataset = load_coco_dataset()
   model = load_pretrained_yolo()
   metric = MyCOCOMetric()

   # Run generic task
   results = evaluate(model, dataset, metric)

Component/Task Implementer (Level 2)
-------------------------------------

Developers implementing components or tasks relevant to a specific AI problem (e.g., see component protocols in `maite.protocols.object_detection`) focus on satisfying structural and semantic requirements to provide new T&E capabilities.
These developers implement objects that are useful across many applications within a given AI problem.

**Activities:**

- Implements components or tasks within a predefined AI problem type
- Wraps third-party libraries to satisfy MAITE protocols

**Example:**

.. code-block:: python

   from maite.protocols import object_detection as od

   class CustomYOLOModel:
       """Satisfies od.Model protocol through structural compatibility."""

       def __call__(
           self,
           inputs: Sequence[od.Image]
       ) -> Sequence[od.ObjectDetectionTarget]:
           # Implementation transforms images to detections
           ...

       @property
       def metadata(self) -> od.ModelMetadata:
           return {"id": "custom_yolo_v8"}

Component implementers satisfy existing protocols without modifying MAITE's core source code.

AI Problem Author (Level 1)
----------------------------

AI problem authors work to identifying common problem structure across many specific applications (e.g., all speech-to-text applications, all multi-object tracking applications) and to codify that structure in the form of AI-problem specific primitive types.
They can leverage broad understanding of a given AI problem to ensure their definition is useful for developers implementing or integrating components or tasks in the context of that AI problem.

**Activities:**

- Define three primitive types relevant to the chosen AI problem and their behavioral expectations (using 'semantic type aliases' and 'role type aliases' when helpful)
- Specialize generic component protocols (from `maite.generic`) with problem-specific types, documenting AI-problem specific context in component protocol docstrings

**Example:**

.. code-block:: python

   # Define primitives using documented semantic aliases
   Audio: TypeAlias = ArrayLike
   """Semantic alias for a single audio datum.

   Use `Audio` when you want to emphasize domain meaning ("this value is audio").
   Expected shape semantics are `(C, T)` where C is channels and T is time samples.
   Values represent audio amplitude, typically normalized to [-1.0, 1.0].
   """

   Transcript: TypeAlias = str
   """Semantic alias for a text transcription target/prediction.

   Use `Transcript` when referring to text transcriptions directly.
   Values represent natural language text output from speech recognition.
   """

   # Create role aliases
   InputType: TypeAlias = Audio
   """Role alias for model/dataset input in the speech-to-text protocol family.

   Use `InputType` in generic protocol contexts where the type parameter
   represents "input position". Currently equivalent to Audio.
   """

   TargetType: TypeAlias = Transcript
   """Role alias for model/dataset target in the speech-to-text protocol family.

   Use `TargetType` in generic protocol contexts where the type parameter
   represents "target position". Currently equivalent to Transcript.
   """

   DatumMetadataType: TypeAlias = DatumMetadata
   """Role alias for datum-level metadata in speech-to-text protocol signatures.

   Use `DatumMetadataType` in generic protocol contexts where metadata appears
   as a type argument.
   """

   # Specialize protocols
   class Dataset(gen.Dataset[InputType, TargetType, DatumMetadataType], Protocol):
       """Speech-to-text dataset protocol"""
       ...

Like component & task implementers, AI problem authors extend MAITE to new problem domains without modifying MAITE's core source code.

MAITE Architect (Level 0)
--------------------------

MAITE architects operate at the most broadly applicable and abstract level.
They identify patterns observed across many AI problem types (object detection, text classification, speech recognition, etc.) and design generic abstractions that can be specialized to form AI problem definitions.
They have minimal specific problem context but deep understanding of structural patterns common to supervised learning problems.

**Activities:**

- Design and maintain the fundamental roles and relationships between components using broad context from many specific AI problem domains
- Define generic component protocols
- Ensure protocols support common use cases across multiple AI problems
- Consider architectural-level trade-offs between modularity, extensibility, simplicity, safety, and reproducibility

**Example:**

.. code-block:: python

   # Core protocol design
   class Dataset(Protocol, Generic[InputT, TargetT, MetaT]):
       """
       Generic dataset protocol.

       Type parameters enable specialization to specific AI problems
       while maintaining consistent interface expectations.
       """
       def __getitem__(self, idx: int) -> tuple[InputT, TargetT, MetaT]: ...
       def __len__(self) -> int: ...
       metadata: DatasetMetadata

MAITE architects work on the core framework, considering needs across many AI problems.

Footnotes
=========

.. [#practical_layer_spanning_caveat] In practice, specific users may very easily work across multiple layers, but considering more narrowly-defined user categories is helpful for explanation.
