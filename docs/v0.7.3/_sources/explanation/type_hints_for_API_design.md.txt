# A Primer on Python Typing: Relevant Language Features, Methods, and Tools for the T&E Framework
> Ryan Soklaski, Lincoln Laboratory (2022)

In 2015, Python 3.5 introduced support for writing type annotated code. It's proposal was [spearheaded by engineers at Dropbox](https://dropbox.tech/application/our-journey-to-type-checking-4-million-lines-of-python), who wanted to be able to statically analyze their 4 million line Python code base for bugs. Since then, Python's typing support has become the one of the language's most actively-developed and rapidly-evolving features. Accordingly, the majority of popular third-party libraries have refactored their projects to incorporate – and even ship – type annotations[^pandas]. That being said, because Python is well-known for being dynamically typed, the details and merits of writing statically typed Python code are, understandably, obscure to many.

[^pandas]: As of writing this, version 1.5.0 of pandas – the wildly-popular data analysis library – was just released, and one of its primary new features is [adding improved type annotations](https://pandas.pydata.org/docs/whatsnew/v1.5.0.html#pandas-stubs)

Thus the goal of this document is to provide:
   - a basic explanation of what it means to write statically typed Python code
   - an overview of the tools that make type annotations powerful
   - justification for why this language feature is critical for our project to leverage
   - motivation for adopting specific typing features and methods in the T&E framework
   - demonstration of what this would look like for the T&E framework

> **Following along:**
> The code-snippets in this document are designed to be runnable/parsable. The easiest way to follow along is to copy these snippets into scripts in VSCode with the Pylance extension installed. Under settings specify `Python > Analysis: Type Checking Mode -> basic`. This  will enable pyright's basic static type checking features: demonstrated incompatible types will appear as red squiggles, and you can mouse over variables to see what the statically-inferred types are.
> To get a feel for mypy and pyright (and their differences) try writing/running snippets in the browser-based [mypy-playground](https://mypy-play.net/?mypy=latest&python=3.10) and [pyright playground](https://pyright-playground.decorator-factory.su/).

## A quick introduction to writing statically typed Python code

Writing a type-annotated Python function simply means that we annotate the function's signature to describe the types of its input(s) and output(s).
The following `count_vowels` function is annotated to indicate that its input is a string type and its output is an integer.

```python
def count_vowels(x: str) -> int:  # `: str` and `-> int` are the annotations
    return sum(1 for char in x if char in set("aeiouAEIOU"))
```

In the parlance of Python these annotations are referred to as type-hints because *they are not enforced at runtime by the Python interpreter*. This is a critical point: **type annotations do not add runtime checking/enforcements of types** without the help of separate 3rd party tools. On their own, type annotations are effectively a form of documentation[^hints_as_docs]. That being said, we will see that there is an ecosystem of excellent 3rd party tools – static and runtime type checkers, and data parsers – that make Python's typing features invaluable.

[^hints_as_docs]: This alone mak
es type annotations worthwhile! In effect, type annotations are the only language-enforced standard for documentation in Python; they are concise, expressive, and more widely understood (by both users and IDEs) than any other means of documenting interfaces in one's code.

Before we move on, let's look at a few example code snippets to familiarize ourselves a bit more with Python's type annotation syntax and what they can express.

```python
from typing import Callable, List, Mapping, Optional, TypeVar, Union, Protocol, Literal

# The following are type annotations that could be included, e.g., in a
# function' signature

# Either an integer or a float
Union[int, float]

# Either a boolean or `None`
Optional[bool]

# Either the string 'cat' or the string 'dog'
Literal["cat", "dog"]

# Any object supporting the call syntax – f(...) – that accepts three
# integer-values inputs and returns a string
Callable[[int, int, int], str]

# A list of an arbitrary number of strings
List[str]

# A mapping, such as a dictionary, whose keys are strings
# and whose values are floats
Mapping[str, float]

# A type named Person that has two attributes: `name` (str) and `age` (int)
# and a "greeting" method, which accepts no inputs and returns a string
class Person:
    name: str
    age: int

    def greeting(self) -> str:
      ...

# A function that accepts a single input and returns an output of the
# same type
T = TypeVar("T")
def type_preserving_func(x: T) -> T:
  ...

# The following describes *any* type that exposes the method: 
# `<obj>.open(file_path: str)`
# Note: this is a protocol, which enables a feature known as "structural subtyping".
# This will be an important feature that we discuss later
class Openable(Protocol):
    def open(self, file_path: str):
        ...
```

For more examples of type annotations, please refer to [this cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

## Tools that make type annotations worthwhile

There are three main types of tools that make type annotations far more powerful than serving as mere documentation: [static type checkers](#static-type-checkers), [runtime type checkers](#runtime-type-checkers), and [data parsers/validators](#parsers).

### Static Type Checkers

A static type checker is able to scan one's code as a static collection of files and analyze it for errors based solely on its type annotations. Consider the following example:

```python
# contents of example.py
from typing import Iterable

def get_data_registry() -> dict[str, int]:
    ...

def process_data(x: Iterable[int]) -> int:
    data_total = sum(x)
    return data_total

def run_app():
    registry = get_data_registry()
    process_data(registry.keys())  # <-- static type checker flags error here!
```

In the above, we made a mistake where we passed to `process_data` (which expects an iterable of integers) `registry.keys()` (which is an iterable of strings, not integers). A static type checker will be able to process this file *without actually running any of the code* and raise an error when it sees this discrepancy. So, instead of having to run our code, either via a unit test or in earnest (and then having to read through the stack trace when our program crashes), we can instead run a static type checker to catch this bug in a matter of seconds.

[mypy](https://mypy.readthedocs.io/en/stable/) and [pyright](https://mypy.readthedocs.io/en/stable/) are the two most prominent static type checkers for Python[^mypy_pyright]. Both of these have a command line interface that makes them simple to incorporate into automated test suites. This is what it looks like to run pyright on the above file:

[^mypy_pyright]: mypy is the defacto standard static type checker for Python, as it was the first of its kind and was developed, in part, by the creator of Python himself. pyright is developed by Microsoft and is the core technology that undergirds their VSCode Python extension.

```console
$ pyright example.py 
pyright 1.1.266
/home/rsokl/rai/example.py
  /home/rsokl/rai/example.py:11:16 - error: Argument of type "dict_keys[str, int]" cannot be assigned to parameter "x" of type "Iterable[int]" in function "process_data"
    TypeVar "_T_co@Iterable" is covariant
      "str" is incompatible with "int" (reportGeneralTypeIssues)
1 error, 0 warnings, 0 information 
Completed in 0.578sec
```

An error message is provided that clearly indicates the typing annotation error, as well as a general message indicating that removing the type annotations would not remedy the problem. 
It is also worthwhile to note that these static type checkers are designed to make do with *partially* typed code as well; one does not need to fully type-annotate their code to reap the benefits of static type checkers. That being said, mileage between the different checkers vary: pyright is particularly good at supporting "gradual typing" (it was designed with this feature in mind). See the following example:

```python
# Demonstrating pyright's ability to infer types through un-annotated functions
def make_int() -> int: ...
def add(x, y): return x + y  # note: not annotated!

x, y = make_int(), make_int()
z = add(x, y)

reveal_type(z)  # pyright reveals: int, mypy reveals: Any
```

> A quick note about the above example. The function `reveal_type` is not actually a "real" function, nor is it imported from any module; the above code would crash were we to actually run it. Its sole purpose is to elicit from static type checkers the type that they have resolved for the specified object. This function is understood by all static type checkers.

Popular Python IDEs like PyCharm and VSCode (via Pylance) are built around static type checkers; if you copy the above code into one of these editors, you will see a red squiggle appear under `registry.keys()`. The static type checker is constantly being run in real time by these IDEs so that you *catch this bug the moment you write it*. This capability tightens the feedback loop for developers working this code: you can be more confident when refactoring a given function, knowing that the static type checker will scan the rest of the code base for any incompatibilities that the change introduces.

In total, **static type checkers transform type annotations from being mere documentation to being *verified* documentation**: any code pattern that contradicts your annotated interface will be flagged by the type checker.

### Runtime type checkers

Runtime type checkers are libraries that typically provide function decorators and other higher-order objects that wrap your code with type-checking boilerplate code. Unlike static type checkers, these will actually *enforce* your type annotations at runtime. Consider the following example using the runtime type checker [beartype](https://github.com/beartype/beartype):


```python
from beartype import beartype

@beartype
def process_age(age: int) -> int:
    return age
```


```pycon
>>> process_age("hello")
BeartypeCallHintParamViolation: @beartyped process_age() parameter x='hello' violates type hint class 'int', as 'hello' not instance of int.
```

Recall that, without the `@beartype` decorator, `process_age('hello')` would happily return `'hello'` regardless of the function's annotations. Instead, the `@beartype` decorator consumes the `process_age` function, processes its signature, generates type-checking code according to its type annotations, and returns a new version of `process_age` that begins with the type-checking code on the input it receives, and ends with code for checking the type of the return value.

[beartype](https://github.com/beartype/beartype), [pydantic](https://pydantic-docs.helpmanual.io/) and [typeguard](https://github.com/agronholm/typeguard) are the most popular runtime type checkers for Python. Note that these are only able to catch bugs in the form of mismatched interfaces/types when you are actually running your code. They are good at preventing silent or severe errors from occurring[^beartype_overhead].

[^beartype_overhead]: These also add additional runtime overhead to each function call.

Thus, **runtime type checkers transform annotations from being documentation to being *enforced contracts***. 

### Parsers

A parser takes in data, validates at runtime that said data adheres to some requirements or schema, and then returns the data *as a more specific type* – one that reflects that the data has been validated[^parse]. You then write your library's interface[^interface] in terms of these more specific – or narrowed – types, so that those functions need not be responsible for performing the same validation in multiple places. This helps us organize software systems into a parsing phase, where the bulk of all validation and error-handling occurs, and an execution phase, where we are working with types that "prove" that our functions will not fail due to malformed data. Libraries like [pydantic](https://pydantic-docs.helpmanual.io/) and [phantom-types](https://github.com/antonagestam/phantom-types) provide useful types and parsing capabilities towards this end.

[^interface]: A library's interface is comprised of the set of symbols – modules, functions, classes, and other objects – that it exposes. Refer to [this document](https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#library-interface) for the rules that type checkers follow to determine which symbols comprise a library's interface.

We will see concrete examples of this parsing paradigm in practice later.

[^parse]: In effect, the difference between a validator and a parser is all in the return type. Validators convey no additional type information, whereas a parser preserves knowledge – in the form of a statically defined type – of the validation process. I encountered this definition of a parser from the excellent article [Parse, Don't Validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/), which goes on to say:
    > What is a parser? Really, a parser is just a function that consumes less-structured input and produces more-structured output. By its very nature, a parser is a partial function — some values in the domain do not correspond to any value in the range — so all parsers must have some notion of failure [...] Under this flexible definition, parsers are an incredibly powerful tool: they allow discharging checks on input up-front, right on the boundary between a program and the outside world, and once those checks have been performed, they never need to be checked again!


[^popular_parsers]: Although we used `phantom-types` here, the most popular parsing library for Python is . 


Thus, **type annotations used in conjunction with parsers can help make illegal states unrepresentable**.



## Justifying Our Emphasis on Typing in the T&E Framework

Here, we present some compelling anecdotes to show that industry leaders have "voted with their feet", so to speak, to demonstrate that Python's typing features are especially important and effective for developing machine learning frameworks like ours. We also summarize, concretely, how type annotations will benefit the T&E framework.

It may not be well known that Microsoft's static type checker for Python, pyright, was born out of an acquisition of a machine learning company[^eric_sig]. This included the company's code base, which amounted to hundreds of files of un-annotated Python code. Eric Traut, the lead of this team at Microsoft, considered the process of incrementally adding static types to the project a top priority; without type annotations and a capable static type checker, reasoning about, maintaining, and refactoring the code base, was not feasible (especially for new developers). Thus Eric developed pyright[^traut], which is now the foundation of VSCode's Python extension, and evolved the ML code base to eventually adopt thorough type annotations throughout. That Microsoft was willing to invest this level of effort, *just to have better type checking capabilities* (they could have just used mypy!), speaks volumes of the importance of this language feature.

[^eric_sig]: This is according to an oral history provided by Eric Traut (creator of pyright) at a typing-sig meeting.

[^traut]: According to Eric, he perused mypy and the other type checkers and decided that they were insufficient. So.. he just wrote his own. Within a year, Microsoft completely refactored VSCode's Python extension around pyright, where it now supports ~4 million users. This guy is the former lead of the Window's core team and the Skype app team... so yeah.. he could grind me to dust with his brain.

This isn't the only place where it has become clear that Python's typing features are highly valued by machine learning framework engineers. Recently, Python added a new language feature described in [PEP 646](https://peps.python.org/pep-0646/), which  was proposed by *a joint team of Google and Meta engineers* in service of writing more expressive type-annotated TensorFlow and PyTorch code, respectively. There are also popular projects like DeepMind's [tensor annotations](https://github.com/deepmind/tensor_annotations), [torchtyping](https://github.com/patrick-kidger/torchtyping), and [jaxtyping](https://github.com/google/jaxtyping), which are all concerned with being able to provide improved annotations for common machine learning library interfaces. 

How do the above anecdotes map onto our needs? We are going to need to provide and communicate our APIs to end-users who will need to understand how to interact with various aspects of our framework — be it an object detection model registry, the entry point for an inference engine, or merely importing and using a metric function. We need to document these interfaces in a way that is precise, concise, expressive, and consistent. As indicated by various industry leaders, there is also great value to persisting this level and quality of detail throughout the *internals* of the framework as well; this will lead to a framework that is more robust and trustworthy, and that is easier to maintain and to refactor as the field of machine learning inevitably evolves.

By adopting type annotations in a careful and principled way, the T&E framework team – as well as our collaborators and users – will be able to:

- concisely and expressively document our interfaces using the same language-standardized form that industry leaders in this field have adopted (and have invested substantial resources into)
- *statically verify* the correctness of our interfaces in an automated way (and in real-time within IDEs), via static type checkers [^types_as_tests]
- use runtime type-checkers to enforce critical interfaces in the framework as *contracts*
- and leverage data parsing techniques to stratify our framework's components into validation and execution phases, so that, in the execution phase, *illegal states become unrepresentable* (or, at least, hard to represent) 

In total, these capabilities will help our team's effort *scale effectively* as we develop, maintain, and adapt the T&E framework. Furthermore, this will help us to provide users with a first-class experience in terms of the quality, simplicity, and reliability of our APIs. The next section will help to demonstrate specific typing features that will empower us to achieve these outcomes.

[^types_as_tests]: Type annotations are a natural means of quality assurance in a code base, they reduce the need for exhaustive and brittle integration tests without sacrificing effective code coverage (via static type checking). Instead of having to 



## Motivating the Adoption of Specific Typing Features and Methods in the T&E Framework

The following are qualities that the framework's type annotations and broader design must satisfy:
- They must serve as *legible* documentation[^legible] that is consistent across all of our APIs. Understanding the type annotations of one function should make it easier to understand our other functions at a glance, and it should be clear when and how a user should parse (validate) their data before passing it to a given function.
- Wherever possible, users should be able to satisfy our typed interfaces using standard-library types (e.g. `int` or `dict`), through common third party data types (e.g. `torch.Tensor`), or through *structural subtypes*. Users should *never* have to inherit from our types to satisfy our interfaces, unless they are explicitly inheriting complex functionality that they ought not implement themselves.
- We should leverage types that convey knowledge/description of data validation processes (e.g. `NonEmptyTensor` would signal that tensor has been proven to contain no size-0 dimensions); refining data to appropriately "narrowed" types as early as is feasible helps to eliminate ad-hoc and repetitive validation checks throughout our code and our users' code. 
- Annotations leading to false positives in static type checkers are unacceptable. Faulty alarms get turned off and ignored.
- Annotations leading to false negatives in static type checkers are only acceptable in obscure edge cases, where runtime checks are *guaranteed* to catch the error
- Annotations leading to either false positive or false negatives during runtime type checking are unacceptable

[^legible]: It can become tempting to be overly pedantic with annotations. For example could rewrite
    
    ```python
    def count_vowels(text: str) -> int: ...
    ```
    as..
    
    ```python
    def count_vowels(text: Iterable[Hashable]) -> int: ...
    ```
    but the latter might be hard to understand for users. If the latter is really a superset of the former, then, by definition, we could always refactor towards it later – as-needed – without breaking compatibility elsewhere.


Some or all of these may also be fruitful recommendations for us to suggest to vendors. The hope here is that the T&E *ecosystem* manifests in such a way that its packages are: broadly consistent, judicious in their interdependencies, and are generally easy to understand and use. Because the vendors' experience levels with Python, machine learning APIs, typing, and other relevant topics may vary quite broadly, it is especially helpful to spell out and motivate the following goals in a clear way.

The remainder of this section will elaborate on these goals.

### On using annotations to write legible documentation

> Goal: Annotations must serve as *legible* documentation that is consistent across all of our APIs. Understanding the type annotations of one function should make it easier to understand our other functions at a glance, and it should be clear when and how a user should parse (validate) their data before passing it to a given function.

There are some recent Python typing features that help to make type annotations more legible and less intimidating[^besties]. These are very simple and yet are quite effective. [PEP 585](https://peps.python.org/pep-0585/) enables one to rewrite annotations for various collections using built-in types

[^besties]: These lessons build off of the recommendations of the official [typing best practices](https://typing.readthedocs.io/en/latest/source/best_practices.html)  documentation, and include considerations that are particular to tensor-centric code bases.

```python
# before PEP 585
import typing

def f(ages: typing.List[int], records: typing.Dict[str, int]): ...
```
```python
# after PEP 585
from __future__ import annotations  # required for Python < 3.9

def f(ages: list[int], records: dict[str, int]): ...
```

Similarly, [PEP 604](https://peps.python.org/pep-0604/) enables unions to be represented more succinctly

```python
# before PEP 604
def f(x: typing.Union[int, str]): ...
```
```python
# after PEP 604
from __future__ import annotations  # required for Python < 3.10

def f(x: int | str): ...
```

[PEP 613](https://peps.python.org/pep-0613/) adds explicit type aliases via `TypeAlias`, which enable one to write type annotations that have complex/lengthy implementations, but short and descriptive representations. Let's use an explicit type alias to summarize an "array-like" type (a type that can be converted to a numpy array).

```python
from typing import Sequence
from typing_extensions import TypeAlias
import torch as tr

Scalars: TypeAlias = int | float | complex
# Supports array-likes from 0D to 2D structures
ArrayLike: TypeAlias = Scalars | Sequence[Scalars] | Sequence[Sequence[Scalars]] 

def to_tensor(x: ArrayLike) -> tr.Tensor:
    ...

to_tensor(0)  # static type checker: OK
to_tensor([1, []])  # static type checker: ERROR!
to_tensor([1, 1])  # static type checker: OK
to_tensor([[2+1j, 3+0j], [1-8j, 2+10j]])  # static type checker: OK
```


One of the biggest documentation challenges for tensor-heavy code bases is to convey what the shape requirements are on an input tensor for any given function or model. Does a function work on any tensor? An image-like tensor? A batch of image-like tensors? Until recently, this could only be addressed by describing tensor-shape requirements in a documentation string. This is still called for, but it would be better if this information could be conveyed in a more standardized and tool-accessible way.

Along these lines, [PEP 646](https://peps.python.org/pep-0646/) was proposed by engineers from Google and Meta who work on machine learning libraries and frameworks. This lead to the introduction of variadic generics in Python's typing system, which makes it feasible to write types for arrays/tensors of arbitrary shapes. There are plenty of technical details to read here, but let's consider an example.

In the following example, we use a variadic generic type to re-represent the PyTorch `Tensor` so that we can statically represent tensor-shape information as part of the type. We'll see that this enables us to concisely document how tensors containing images, batches of images, videos, and time-series data have their memory laid out (e.g. `channel x height x width` for an image). We will also see that static type-checkers can leverage this type information to catch incompatibilities for us.


```python
from typing_extensions import TypeVarTuple, Unpack, TypeAlias
from typing import Generic, Any
import torch
from typing import NewType

Shape = TypeVarTuple("Shape")

# A PyTorch tensor with additional shape type information
# This is a so-called "variadic generic": the Shape type variable can vary in length/contents
class Tensor(Generic[Unpack[Shape]], torch.Tensor):
    ...

# Declaring descriptive aliases for common array dimensions
Height: TypeAlias = int
Width: TypeAlias = int
Channel: TypeAlias = int
Time: TypeAlias = int
Batch: TypeAlias = int

# Some representative utility functions for loading tensor data
def load_time_series(path: str) -> Tensor[Time]: ...
def load_image(path: str) -> Tensor[Channel, Height, Width]: ...
def load_video(path: str) -> Tensor[Time, Channel, Height, Width]: ...

# Some functions working with tensors..
# Stack multiple Tensors along a leading "Batch" dimension
def stack(*arrs: Tensor[Unpack[Shape]]) -> Tensor[Batch, Unpack[Shape]]: ...

# Get the resolution, HxW, from any shape-(..., H, W) tensor
def get_img_resolution(img: Tensor[Unpack[tuple[Any, ...]], Height, Width]) -> tuple[Height, Width]: ...

list_of_images = [load_image(p) for p in ["a.png", "b.png"]]  # list[Tensor[Channel, Height, Width]]
img_tensor = stack(*list_of_images)  # Tensor[Batch, Channel, Height, Width]
img_res = get_img_resolution(img_tensor)  # Tuple[Height, Width]

list_of_videos = [load_video(p) for p in ["a.mp4", "b.mp4"]]  # list[Tensor[Time, Channel, Height, Width]]
video_tensor = stack(*list_of_videos)  # Tensor[Batch, Time, Channel, Height, Width]
video_res = get_img_resolution(video_tensor)  # Tuple[Height, Width]

time_series = load_time_series("data.pkl")  # Tensor[Time]
# attempting to get the resolution of a shape-(Time,) tensor...
get_img_resolution(time_series)  # static type check: error!
```

Consider how opaque the above code would become if we were to replace all of the tensor annotations with a bare `Tensor`! Prior to this PEP, we would have needed to make a special type to represent each of `Tensor[Time]`, `Tensor[Channel, Height, Width]`, etc. Otherwise, these critical details can only be conveyed via documentation strings.

That all being said, this is a very new language feature and must be handled with care. As of writing this pyright supports it, [but mypy does not yet](https://github.com/python/mypy/issues/12840). Furthermore, these shape-annotations are not natively propagated by PyTorch operations:

```python
img_tensor = load_image("img.png")  # type-checker sees: Tensor[Channel, Height, Width]
img_tensor = img_tensor * 2  # type-checker sees: Tensor
```

Until PyTorch (or any other array-based math library) adds shape-specific support to their annotations, which is blocked by the fact that shape-arithmetic cannot yet be expressed in Python, this shape information cannot be relied on to provide correctness checks on the mathematical operations being performed. Instead, they should only be relied upon at high-level interfaces – between significant nodes of processing (data loaders, models, etc.). E.g., one could annotate that a model requires a `Tensor[Batch, Channel, Height, Width]` as its input – this is highly valuable information – but the implementation of the forward pass itself can only be validated against the information provided by a bare `Tensor` type.

Finally, while interfaces annotated with shape-typed tensors serve as excellent documentation for users, it can be hard for said users to actually construct data that satisfy those types (at a static type-checking level). E.g. none of PyTorch's functions can be used to specifically create a `Tensor[Height, Width]`-typed output. Thus we would want to provide simple convenience functions that enable users to parse bare tensors into corresponding shape-annotated tensors (more on this in the section on type-narrowing).

In summary, the added ability to express variadic types, as introduced by [PEP 646](https://peps.python.org/pep-0646/), enables us to convey critical information about tensor shapes at the type level. That being said, they must be used judiciously and with restraint, as informed by what these *aren't* capable of and by the ways in which they can make our framework less accessible to our users.


### Typed interfaces should be informative, inspire good design, and be easy to satisfy

>  Goal: Wherever possible, users should be able to satisfy our typed interfaces using standard-library types (e.g. `int` or `dict`), through common third party data types (e.g. `torch.Tensor`), or through *structural subtypes*. Users should *never* have to inherit from our types to satisfy our interfaces, unless they are explicitly inheriting complex functionality that they ought not implement themselves.

There is a lot to unpack here. We will attempt to do so in terms of an example that is of immediate concern to the T&E framework. First, we will present a nominal design for the API of a simplified object detector. Next we will highlight the subtle-but-critical shortcomings of the design. Finally, we will motivate the adoption of **structural subtyping** (via **protocols**) in this scenario, and will demonstrate how this greatly improves the quality of our API (so as to achieve the above-stated goal).

Consider the following (simplified) API for an object detector that is shipped by `our_library`:

```python
import abc
from typing import Sequence, Any, Dict, Tuple
from typing_extensions import TypeAlias
from torch import Tensor

from our_library import BoundingBox

ClassScores: TypeAlias = Dict[Any, float]

class OurDetectorAPI(abc.ABC):
    @abc.abstractmethod
    def detect(self, img: Tensor) -> Sequence[Tuple[BoundingBox, ClassScores]]: 
        raise NotImplemented() 
```

where `BoundingBox` is a class that we have created that has all sorts of convenient functions for working with bounding boxes. It might look something like:

```python
class BoundingBox:
    def __init__(self, left: float, top: float, right: float, bottom: float):
        # check that bbox coords satisfy, e.g., left <= right
        # use bbox coords to construct vertices
        ...
    def compute_box_area(self) -> float: ...
    def get_intersection(self, other_box: "BoundingBox") -> "BoundingBox": ...
```

Naturally, we would have code elsewhere in our library that can leverage any such detector; one such function might look like this:

```python
def measure_detector_precision_and_recall(model: OurDetectorAPI) -> float:
    if not isinstance(model, OurDetectorAPI):
        raise TypeError("You've gotta be one of us!")
    
    data = load_data()
    detections = model.detect(data)
    ...
```

Suppose that this is the API that we propose to third parties. If they abide by this, then we assure them that their object detectors will plug-and-play with our framework. At first glance, this may all appear to be perfectly reasonable. Indeed, the interface is informative to the end user: it tells us that a detector consumes a image described by a PyTorch tensor and returns a series of bounding-box & score detection-pairs. However, this design fails in that **it is not easy to satisfy, nor does it inspire good design practices**. Let's see why this is.

Far and away the most pressing issue with this design is that *it unnecessarily requires third parties to install our library as a dependency* (and thus incur all of our dependencies as well). It is impossible for them to implement a satisfactory detector – e.g., one that can be passed to `measure_detector_precision_and_recall` –  without inheriting from `our_library.OurDetectorAPI` and without reporting their detected bounding boxes in terms of `out_library.BoundingBox` instances. This is despite the fact that they may not need any of the functionality that is provided by these classes! This adds needless complexity: e.g., now a docker image of their library needs to include our library and its dependency.

Next, third parties are required to include `our_library.OurDetectorAPI` explicitly in their class hierarchy, even though its sole purpose is documentation and the enforcement of structure[^forced_structure]. This greatly hinders users from being able to understand our API at a glance. Instead, those implementing the API will need to check the details of the `OurDetectorAPI` to see if there is some important functionality being inherited, and to understand what, specifically, they have to implement, and if their functionality interacts with any "statefulness" in the detector.

[^forced_structure]: And this enforced structure is wholly unnecessary! Why should we care if the object detector has a method that is specifically named `detect`? What if they already have a `detect` method that satisfies a different API? We will see that it is better to request that a detector is simply a *callable*.

For those with pre-existing detectors, this means that they either have to eat the complexity of multiple inheritance, or, preferably, they would write an "adapter" class that encapsulates their detector and exposes our interface:


```python
from our_library import BoundingBox, OurDetectorAPI
from their_library import TheirDetector

class SadCompatShim(OurDetectorAPI):
    def __init__(self, actual_detector: TheirDetector):
        self.det = actual_detector

    # this is the best-case scenario
    def detect(self, img: Tensor) -> Sequence[Tuple[BoundingBox, ClassScores]]:
        their_bboxes, their_scores = self.det.their_detection_method(img)
        our_bboxes = [BoundingBox(*bbox) for bbox in their_bboxes]
        return list(zip(our_bboxes, their_scores))
```
[^gross]

[^gross]: This implementation may *look* simple enough, but there much more complexity afoot than one might expect! There is a whole new class that the third party must worry about: where in *their* API to they make it available? If the have pickled versions of `TheirDetector`, how do they reanimate those via `SadSappyCompatShim`? If they have tools for converting, say, huggingface detectors to `TheirDetector`, do they now have to duplicate those tools work with our API?

When all is said and done, the author of `SadCompatShim` might think: "Wow, I added `their_library` to my dependencies and read through the source code of `OurDetectorAPI` just for that? All of that, just to implement a glorified function". Indeed, this is just the world's most expensive function definition.

How do we improve upon this? How do we achieve the requirements specified at the outset of this subsection?

There is a powerful technique that we can use to ameliorate *all* of these issues: we can refactor our API in terms of protocol types. **A protocol type is a type whose subtypes do not have to inherit from it, rather, they merely need to implement the protocol**. That is, a subtype of a protocol is simply anything that implements the same *structure* as the protocol. As such, this concept is referred to as structural subtyping (a.k.a ducktyping). This feature was formally introduced by Python 3.8, via [PEP 544](https://peps.python.org/pep-0544/)[^backport].

[^backport]: This typing feature and all of the other features highlighted in this document have been made backwards compatible with Python 3.7+ via the `typing_extensions` package.

Let's use our current example to understand what structural subtyping is all about. The following is a redesign of our simple object detection API, which leverages protocols

```python
from typing import Any, Dict, Sequence, Tuple, Protocol, runtime_checkable

from torch import Tensor
from typing_extensions import TypeAlias, runtime_checkable

ClassScores: TypeAlias = Dict[Any, float]

@runtime_checkable
class BoundingBox(Protocol):
    left: float
    top: float
    right: float
    bottom: float

@runtime_checkable  # <-- enables `isinstance` checks to look for necessary structure [1]
class OurDetectorAPI(Protocol):
    def __call__(self, img: Tensor) -> Sequence[Tuple[BoundingBox, ClassScores]]:
        ...

def measure_detector_precision_and_recall(model: OurDetectorAPI) -> float:
    if not isinstance(model, OurDetectorAPI):  # <-- [1]: I.e, this still works!
        raise TypeError("You've gotta be one of us!")
    
    data = load_data()
    detections = model(data)
    ...
```

Because `BoundingBox` inherits from `Protocol`, *any* object that exposes `float`-valued attributes named `left`, `top`, `right`, and `bottom` is an instance of `BoundingBox`. Similarly, `OurDetectorAPI` is satisfied by *anything* that supports the call syntax – `obj(img)` – and whose signature is (a superset of) `(img: Tensor) -> Sequence[tuple[BoundingBox, dict[Any, float]]]`.

See that we have retained all of the important structure here: we are still telling a 3rd party exactly how we expect them to represent a detector, a bounding box, and classification scores. At the same time, we have also **eschewed all of the unnecessary dependencies and complexities of the old APIs**: 
- Third parties need not install additional libraries to satisfy this API
- Our detector API doesn't depend on any statefulness, so our user sees that we don't expect anything more complicated than a pure function when we ask for a detector.
- Bounding boxes are simple to represent using standard data types (e.g., named tuples), and are trivial to understand

To hammer home the elegance that we have achieved here, let's see what it looks like for us to implement the most trivial detector – one that always returns *no detections* – in the old API:

```python
# Implementing an "empty" detector in the old API
from our_library import OurDetectorAPI  # <- our library must be installed

class EmptyDetector(OurDetectorAPI):
    def detect(self, img):
        return []

empty_detector = EmptyDetector()
```

... and in the new API:

```python
# Implementing an "empty" detector in the new API
empty_detector = lambda img: []
```
[^check_it_out]

[^check_it_out]: Even static type checkers will agree that `lambda img: []` does indeed satisfy the `OurDetectorAPI` protocol! 

It isn't an overstatement to claim that our introduction of protocols into this API is transformational. The adoption of structural subtyping might be the most important, but subtle, recommendation made in this entire document. It substantially reduces the cognitive load and technical debt incurred by the user (i.e. you can understand our detector API at a glance, and you can implement it from the comfort of your own library), and it also encourages careful design decisions from API authors ("Do you *really* need this to be a strict abstract base class? What functionality is actually being inherited?").

Lastly, it is perfectly fine for a library's internal detector class to have additional bells & whistles, like a method that a configuration manager can hook into. That being said, those bells & whistles shouldn't be part of the the documented (annotated) interface unless they are actually required. It is easy to create and compose multiple protocols for this sort of scenario:

```python
@runtime_checkable
class Configurable(Protocol):
    def ___special_config_interface__(self) -> dict[str, Any]: ...

class ConfigurableDetector(OurDetectorAPI, Configurable, Protocol): ...

def orchestrate_detector(model: ConfigurableDetector): ...
```

Now `orchestrate_detector` can demand: "you have to be a detector-like callable that also has our expected config-management hooks, *if* you want to be auto-registered by our framework". This is an incremental increase in complexity that users only opt into if they actually want to use that functionality. Inquiring users can see that their elegant pure-function detector doesn't work with this part of the API – static type checkers will tell them this straight away:

```python
empty_detector = lambda x: []
orchestrate_detector(empty_detector)  # static type checker: error
```


### Validate early in your program and use narrow types to prove that you did so

> Goal: We should leverage types that convey knowledge/description of data validation processes (e.g. `NonNanTensor` would signal that tensor has been proven to contain no `NaN` elements); refining data to appropriately "narrowed" types as early as is feasible helps to eliminate ad-hoc and repetitive validation checks throughout our code and our users' code. 

This section is heavily inspired by Alexis King's excellent blog post ["Parse, Don't Validate"](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/), and attempts to convey how the article's titular recommendation can be pursued in Python. 

Consider the following code

```python
from torch import Tensor
import torch.nn as nn
from typing import Iterable

def load_data() -> Tensor: ...
def load_model() -> nn.Module: ...

def measure_data_distr(img_batch: Tensor):
    if not batch.ndim == 4 or not batch.shape[1] == 3:
        raise TypeError("Not image batch-like")
    # <actual functionality here>

def compute_accuracy(img_batch: Tensor, model: nn.Module):
    if not batch.ndim == 4 or not batch.shape[1] == 3:
        raise TypeError("not batch-like")
    # <actual functionality here>

def compute_calibration(img_batch: Tensor, models: Iterable[nn.Module]):
    if not batch.ndim == 4 or not batch.shape[1] == 3:
        raise TypeError("Not image batch-like")
    # <actual functionality here>

if __name__ == "__main__":
    tensor = load_data()
    model = load_model()

    measure_data_distr(tensor)
    compute_accuracy(tensor, model)
    compute_calibration(tensor, [model])
```

The above code has several issues[^shotgun]. It violates the DRY principle (Don't Repeat Yourself): if we ever change the conditions that we check for a batch (e.g. maybe we add a dtype check), then we will need to change the check across all functions. Furthermore, the `Tensor` annotation is too vague for users and type-checkers alike: neither can predict, at a glance, that passing in a 1D tensor will cause the code to crash. Lastly, we pay a runtime cost, which may be menial in this case but can become considerable if we need to perform rigorous checks on the tensor (e.g. ensure that the data is free of NaNs).

[^shotgun]: In fact, code like this can often devolve into an anti-pattern known as [shotgun parsing](http://langsec.org/papers/langsec-cwes-secdev2016.pdf).

That all being said, the above code is a dime a dozen in the world of Python-based machine learning. Heck, at least it has *some* annotations and validation. So what can be done to improve this sort of code?  

This is where the recommendation "parse, don't validate" can pay off. First, we must reiterate here what we mean by "parsing". We mean that we are taking some object/data that is described by some broad type (e.g., `tuple[Any, ...]`) and validating some predicate about it (e.g., that the tuple is non-empty and contains only strings), and then annotate that object/data with a more precise type (e.g., `NonEmpty[tuple[str, ...]]`), which serves as a "proof" that the object/data has passed through this validation procedure. We can then pass this downstream to functions that require `NonEmpty[tuple[str, ...]]` as their input types. By designing our code this way, we can strive to localize all of our validation and error-handling to one place – the parsing stage of our program – and proceed to execute using data that need not be re-validated.

Let's turn this last paragraph into code[^phantom]

[^phantom]: The [phantom-types](https://github.com/antonagestam/phantom-types) library provides types with predicates, which can be used to check a value's type *and* to perform validation on the value itself at runtime. This makes short work of the sort of parsing example that we lay out.:

```python
from typing import cast

# 0. Define types that describe specific validated states that your
#    library depends on across multiple interfaces
from our_library.narrow_types import NonEmpty

# Returns unstructured/unvalidated data
def stream_data() -> tuple[str, ...]: ...

# Create functions that can validate that the data satisfies
# specific properties and ascribe to the validated data a new 
# type, which serves as proof of validation
def parse_stream(stream: tuple[str, ...]) -> NonEmpty[tuple[str, ...]]:
    if not isinstance(stream, tuple): raise TypeError("not tuple")
    if not stream: raise TypeError("is empty")
    if not all(isinstance(item, str) for item in stream): raise TypeError("not strings")
    
    # The sole purpose of this line is for infroming the static type checker.
    # We don't actually use the `NonEmpty` type to change our data
    # at all!
    proven_data = cast(NonEmpty[tuple[str, ...]], stream)  
    return proven_data

# Design downstream functions to require this narrowed type, which can only
# be obtained by going through the parsing process. Now functions operate
# safely without having to re-validate the data at every turn, and their
# requirements are now explicitly documented via annotations
def consumer1(data: NonEmpty[tuple[str, ...]]): ...
def consumer2(data: NonEmpty[tuple[str, ...]]): ...
def consumer3(data: NonEmpty[tuple[str, ...]]): ...

if __name__ == "__main__":
    # 1. Start with unstructured data
    data = stream_data()  # type checker sees: tuple[str, ...]
    
    # Attempting to pass `data` to, e.g., `consumer1` would produce
    # a static type checking error.

    # 2. Enter "parsing" phase of program, where we validate the data
    #    and ascribe a narrowed type to the data.
    #    This is where we handle and log errors.
    try:
        # input: tuple[str, ...]  (less-structured)
        parsed_data = parse_stream(data)
        # output: NonEmpty[tuple[str, ...]]  (more-structured)
    except TypeError:
        # log error
        # cue graceful recovery
        ...

    # 3. Enter "execution" phase of program: the 'illegal state' of having empty data
    #    here is impossible, assuming we rely faithfully on our type checker, because
    #    we are working with data that has been "proven" to be valid
    consumer1(parsed_data)  # type checker: OK
    consumer2(parsed_data)  # type checker: OK
    consumer3(parsed_data)  # type checker: OK
```

See that, unlike the first example that we considered in this section, this code is free of repetition, it is explicit, and it is highly legible. If someone were to want to leverage `consumer1` in isolation, they would immediately see that they need to feed it a non-empty tuple of data.

#### Type narrowing

One thing that may be surprising about the above example is that we don't actually create an instance of the `NonEmpty` type anywhere! It's sole purpose is to convey information to the static type checker. Thus, at the crux of this parsing technique is the notion of **type-narrowing**. According to [mypy's documentation](https://mypy.readthedocs.io/en/latest/type_narrowing.html), "Type narrowing is when you convince a type checker that a broader type is actually more specific". And this is precisely what we have done: our call to `typing.cast` is where we convinced the type checker that our data is `NonEmpty[...]`. Otherwise, our code never relies on any functionality from `NonEmpty` at runtime. Hopefully, this revelation makes the process of parsing less daunting: we can literally just make up a type whose main purpose is its specificity in the eyes of (human) readers and static type checkers. Its utility comes from its consistent use across our interfaces.

There are many ways that one can perform type-narrowing. Here are some examples:

```python
x: Any   # starting with: x can be Any type

# narrow x via isinstance:
if isinstance(x, int):
    # type checker narrows x to `int` here
    ...
elif isinstance(x, str):
    # type checker narrows x to `str` here
    ...

y: int | list[int]  # starting with: y can be an int or list of ints

# narrow y via assert
assert isinstance(y, list)
y  # type checker narows y to list[int]  (it is impossible for y to be an int here at runtime)

# via casting
from typing import cast

z: Any  # starting with: x can be Any type

# Warning: you can lie to the type checker using `cast`.
# `cast` doesn't do any processing at runtime
out = cast(tuple[int, int], z)  # type checker sees `out` as `tuple[int, int]`
```

One can also define [type guards](https://mypy.readthedocs.io/en/latest/type_narrowing.html#user-defined-type-guards) to facilitate nice type narrowing in control-flow contexts:

```python
from typing import Any

from typing_extensions import TypeGuard

class NonNegativeInt(int):
    ...

# this is our type-guard, which can narrow int -> NonNegativeInt
def is_non_negative_int(x: int) -> TypeGuard[NonNegativeInt]: 
    return 0 < x

def process_age(x: NonNegativeInt): ...

def main(x: int):
    if is_non_negative_int(x):
        # x is narrowed to NonNegativeInt
        process_age(x)
    else:
        # x is an int here
        # log error
        ...
```

Armed with our newfound knowledge of type narrowing, let's rewrite the original example from this subsection in a way that uses the "parsing" paradigm.

Let's create a subtype of `torch.Tensor`, called `TensorBCHW`, which is designed to represent a batch of images of shape `Batch x Channel x Height x Width`. This is the narrow type that we will use in our annotations to signal that a PyTorch tensor has been validated as being "batch-like". Note that we will never actually create an instance of `TensorBCHW`[^fancy_shapes].

[^fancy_shapes]: As described in a previous session, we could also leverage variadic type variables to describe tensors with various shapes, without having to create unique types for each one. This is laid out in [PEP-646](https://peps.python.org/pep-0646/).


```python
from torch import Tensor
import torch.nn as nn
from typing import cast, Iterable
from typing_extensions import TypeGuard


class TensorBCHW(Tensor):
    """Signals that a PyTorch tensor has been validated to
    be shaped like a batch of images: (B, C, H, W)"""
    ...

def load_data() -> Tensor: ...
def load_model() -> nn.Module: ...


def is_batch_of_images(
    x: Tensor, expected_channel_size: int
) -> TypeGuard[TensorBCHW]:
    return isinstance(x, Tensor) and x.ndim == 4 and x.shape[1] == expected_channel_size
        

def measure_data_distr(batch: TensorBCHW): ...

def compute_accuracy(batch: TensorBCHW, model: nn.Module): ...

def compute_calibration(batch: TensorBCHW, models: Iterable[nn.Module]): ...


if __name__ == "__main__":
    model = load_model()
    tensor = load_data()
    
    # type checker sees tensor as: Tensor
    if not is_batch_of_images(tensor, expected_channel_size=3):
        raise TypeError("not a batch!")
    # type checker sees tensor as: TensorBCHW


    # static type-checker ensures input is `TensorBCHW`
    measure_data_distr(tensor)
    compute_accuracy(tensor, model)
    compute_calibration(tensor, [model])
```

We see that, once again, adopting this "parsing" mindset not only consolidates the code's validation logic, but it also incorporates static type information that conveys "proof" that the data we are passing to our functions has already been validated. This code is more efficient and less repetitive than before, plus it is easier for both the developer and the user to reason about. A user looking at our `compute_accuracy` function will see immediately that it does not accept any-ol' `Tensor`, but that we expect them to first parse their data using our `parse_tensor_as_batch` function.

It must be noted that the interface we expose to the user, via `load_data`, is still simple to understand and to satisfy. Type narrowing mainly helps *us* to write clean, consistent, and non-repetitive internal code. That being said, if users want to use our internal code, they will need to buy into our belief system – parse, don't validate – and learn how to use our `is_batch_of_images` function to narrow their types before using our functions[^whelp].

[^whelp]: Or.. they are just coding in a Jupyter notebook and have no clue what these annotation things even are. In which case they can pass whatever they want into our function. This is Python, man. Do what you want.

### Preventing type checkers from being noisy or unreliable

> Goal: Make explicit certain hard requirements that our type annotations must satisfy, so that our types provide us with far more signal than noise. And, reflect on ways in which these requirements might limit the typing language features that we adopt as well the particular type checkers that we leverage.

Annotating a code base is can reveal a Goldilocks problem: annotations that are overly specific can cause your users to experience false alarms, which leads to [alarm fatigue](https://en.wikipedia.org/wiki/Alarm_fatigue) (they'll turn the alarm off!). On the other hand, annotations that are too broad fail to protect users from writing buggy code. And, in both scenarios, these annotations fail their primary purpose: to serve as reliable, verified documentation. There are a few rules that we can establish to help ensure that type our annotations are "just right".

- Annotations leading to false positives in static type checkers are unacceptable (faulty alarms get turned off and are ignored). 
- Annotations leading to false negatives in static type checkers are acceptable in edge cases that are arduous or impossible to annotate. Here, runtime checks must be put into place to catch the error. 
- Annotations leading to either false positive or false negatives during *runtime* type checking are unacceptable (these are just good ol' fashioned bugs!)

mypy's [common issues](https://mypy.readthedocs.io/en/stable/common_issues.html) page is a fantastic resource to help us put these rules in practice.

Ultimately, these rules – in conjunction with the other goals specified throughout this article – are meant to help calibrate our priorities so that our leveraging types does far more good than harm. For developers, we want annotations to make our code to be easier for us to maintain, refactor, and debug, but we do not want to spend too much time fussing with them. For users, we want our annotations to accelerate their ability to "grok"[^grokit] our APIs at a glance, to make our code safe and robust for them to use, and to tighten the feedback loop when they are using our APIs incorrectly (ideally, they would immediately see a red squiggle upon writing the bug).

[^grokit]: grok (`/ɡräk/`): understand (something) intuitively or by empathy.


### Picking a static type checker

> A final, boring, and unfortunate section of this otherwise fun-filled article.

A challenge with maintaining typed Python code is: there are multiple type checkers, which have different capabilities, different design priorities, and different user bases. Depending on how sophisticated one's annotations are, it can be hard to appease all of the type checkers simultaneously. E.g. your code might scan as "all-clean" in pyright and have error reports in mypy. But fixing the mypy errors, might make pyright complain.. and so begins the world's most esoteric game of wack-a-mole.

Accordingly, there can be advantages to having a development team standardize on a single type checker when it comes to scanning their internal code base. That being said, it is preferable to design one's publicly-facing API to be compatible with both mypy and pyright, while only standardizing on one for full internal scans. This will also help increase the likelihood that, e.g., PyCharm's type checker will also be compatible with the public API (and thus PyCharm users will have a high-quality experience using your library).

The following are some of the pros, cons, and other pertinent details of mypy and pyright.

#### mypy

**Pros**: mypy is the defacto standard type checker for Python. It is well-known, and, among projects that do use a type checker, there is a good chance that they are using mypy. It has a plugin system that permits added flexibility, so that projects can achieve some things that are otherwise impossible in Python's typing system.

**Cons**: mypy can be slow on the uptake when implementing new features (a particularly infamous case: it took [4+ years for mypy to add provisional support for recursive types](https://github.com/python/mypy/issues/731)). It still does not support variadic generics, which we saw could be quite useful for us to describe tensor-typed interfaces. Furthermore, mypy is not used by default within any of the major IDEs; one must specifically opt-in to installing third party extensions, which are not necessarily maintained by the mypy team. 


**Who Uses It**: Most projects that run a static type checker as part of their testing processes.

#### pyright

**Pros**: pyright is very fast to incorporate the latest Python typing features, often times out-pacing mypy by months or sometimes even years. Its developers are quick to fix bugs and make improvements: they typically release a new version of pyright each week. Because pyright is used under the hood by VSCode's Python extension, it is by far the most widely-used type-checker for Python[^pywhat]. A major benefit of this is that it is trivial for VSCode users to start using pyright, and to get instant feedback from it as they are writing code.

[^pywhat]: That being said, 99.9% of its users don't know what pyright is nor that they are even using a type checker.

**Cons**: pyright does not support any sort of plugin system, so it is not possible for libraries to express some of the same dynamism in pyright as they can in mypy. Despite being used, implicitly, by so many VSCode users, pyright does not have the same market share of the Python typing community as does mypy. 


**Who Uses It**: Nearly all VSCode Python users, and developers who want to hone the experience of VSCode users. 

## Additional resources

- [mypy's documentation](https://mypy.readthedocs.io/en/stable/index.html): a great resource for learning about typing concepts as well as the concrete details of Python's typing features and its shortcomings.
- [pyright's typing guidance for Python libraries](https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#typing-guidance-for-python-libraries): provides an in-depth description of what it means for a library to be "type complete". Also provides useful recommendations of best practices.
- [The `typing` module's documentation](https://typing.readthedocs.io/en/latest/): includes best practices and recommendations for writing tests for annotations.
