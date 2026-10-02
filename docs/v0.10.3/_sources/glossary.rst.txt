.. glossary::

Glossary
========

Core MAITE Terms
-----------------

AI problem
    Defined by choosing concrete types for the three primitives (input type, target type, and metadata type)
    and specifying behavioral expectations; examples include :ref:`image classification <api_image_classification>`,
    :ref:`object detection <api_object_detection>`, and :ref:`multi-object tracking <api_multi_object_tracking>`;
    see :ref:`MAITE Layered Architecture <maite_layered_architecture>` for more details

Augmentation
    A MAITE component that takes a batch of data as input and returns a potentially modified batch of data as
    output; see the `Augmentation <./generated/maite.protocols.generic.Augmentation.html>`_ API for more details

component
    Implementer of a MAITE-defined Python protocol class (``DataLoader``, ``Dataset``, ``Augmentation``,
    ``Model``, or ``Metric``) that follows prescribed semantics; see :ref:`Vision for Interoperability in AI Test
    and Evaluation <maite-vision>` for more details

DataLoader
    A MAITE component that provides batch-level data access via an iterator; see the
    `DataLoader <./generated/maite.protocols.generic.DataLoader.html>`_ API for more details

Dataset
    A MAITE component that provides datum-level data access via index-based lookup; see the
    `Dataset <./generated/maite.protocols.generic.Dataset.html>`_ API for more details

datum
    An individual data item that's a tuple of input, target (output), and metadata

Metric
    A MAITE component that computes some measure of "agreement" between model predictions and ground-truth labels;
    see the `Metric <./generated/maite.protocols.generic.Metric.html>`_ API for more details

Model
    A MAITE component that takes a batch of inputs and produces a batch of outputs, with types appropriate to the
    particular AI problem; see the `Model <./generated/maite.protocols.generic.Model.html>`_ API for more details

primitive
    Object with class and semantics of a member variable type, argument type, or return type of a MAITE-defined
    Python protocol class; see :ref:`Vision for Interoperability in AI Test and Evaluation <maite-vision>`
    for more details

task
    A Python callable that accepts only arguments typed as MAITE components or MAITE primitives, and returns
    MAITE components, MAITE primitives, and/or Python objects of built-in/broadly-accepted types with
    well-documented semantics; see :ref:`Vision for Interoperability in AI Test and Evaluation <maite-vision>`
    for more details

wrapper
    A Python class that implements a MAITE component protocol by translating to and from a native component

Typing Concepts
---------------

ArrayLike
    A protocol type representing objects that can be coerced to numpy arrays; the foundational inherent type for
    most MAITE primitives including images, bounding boxes, and model outputs; see
    :py:class:`~maite.protocols.ArrayLike` for details

batch
    A collection of multiple data items (datums) processed together; DataLoader MAITE-defined protocol classes
    yield batches, while Dataset MAITE-defined protocol classes provide individual datums that are collected into
    batches; batches are the fundamental unit for Model, Augmentation, and Metric MAITE-defined protocol classes

inherent type
    The actual Python type (e.g., ``ArrayLike``, ``TypedDict``) before domain-specific aliasing; the runtime type
    that best fits from the Python language; the starting point in MAITE's three-layer type alias system
    (inherent type → semantic alias → role alias); see :ref:`maite_layered_architecture` for more details

role alias
    Type alias specifying which semantic type occupies which position in generic protocols (e.g., ``InputType``,
    ``TargetType``, ``DatumMetadataType``); the final layer in MAITE's type alias system
    (inherent type → semantic alias → role alias); makes protocol signatures both generic and self-documenting;
    see :ref:`maite_layered_architecture` for more details

semantic alias
    Type alias that captures domain meaning (e.g., ``Image``, ``BoundingBox``) with behavioral expectations
    documented in its docstring; the middle layer in MAITE's type alias system
    (inherent type → semantic alias → role alias); separates what a type technically is from what it means in the
    domain; see :ref:`maite_layered_architecture` for more details

static type checker
    A tool (e.g., ``Pyright``, ``mypy``) that analyzes code for type compatibility at development time without
    executing it; MAITE recommends ``Pyright`` for verifying protocol compliance

structural subtyping
    Type compatibility determined by matching attributes, methods, and type signatures rather than nominal
    inheritance; Python ``Protocol`` classes and ``TypedDict`` classes use structural subtyping to enable
    plug-and-play component substitution without requiring explicit inheritance; see the Python documentation on
    `protocols <https://typing.python.org/en/latest/spec/protocol.html>`_ for more information

Ecosystem and General Terms
----------------------------

interoperability
    The ability of components from different libraries to work together seamlessly; MAITE's primary design
    objective is enabling broad interoperability across the JATIC ecosystem through standardized interfaces

JATIC
    Joint AI Test Infrastructure Capability; the broader ecosystem of AI test and evaluation Python libraries
    that MAITE serves by providing common interfaces and standards

protocol
    A Python structural type used to provide strict, consistent, and machine-readable definition of MAITE
    components that specify minimum expected attribute names, attribute types, method names, and method type
    signatures; see the Python documentation on `protocols <https://typing.python.org/en/latest/spec/protocol.html>`_
    for more information

test and evaluation (T&E)
    The process of evaluating the performance of an AI model under various conditions (that hopefully match/mimic
    the deployment environment as closely as possible)
