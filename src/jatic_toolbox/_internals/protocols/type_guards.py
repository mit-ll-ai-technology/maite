from typing import Any, Dict, List, Type, TypeVar

from typing_extensions import TypedDict, TypeGuard

from .typing import ModelWithPostProcessor, ModelWithPreProcessor

K = TypeVar("K")
T = TypeVar("T")
Td = TypeVar("Td", bound=TypedDict)


def has_preprocessor(obj: Any) -> TypeGuard[ModelWithPreProcessor]:
    """
    Check if object has preprocessor attribute.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    TypeGuard[Preprocessor]
        True if object has preprocessor attribute.

    Examples
    --------
    >>> from jatic_toolbox.protocols import Preprocessor, ImageClassifierData
    >>> class Foo:
    ...     def preprocessor(self) -> Preprocessor[ImageClassifierData]:
    ...         ...
    >>> has_preprocessor(Foo())
    True
    """
    return hasattr(obj, "preprocessor")


def has_post_processor(obj: Any) -> TypeGuard[ModelWithPostProcessor]:
    """
    Check if object has post_processor attribute.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    TypeGuard[PostProcessor]
        True if object has post_processor attribute.

    Examples
    --------
    >>> from jatic_toolbox._internals.protocols.typing import PostProcessor
    >>> class Foo:
    ...     def post_processor(self) -> PostProcessor:
    ...         ...
    >>> has_post_processor(Foo())
    True
    """
    return hasattr(obj, "post_processor")


def is_list_dict(d: Any, guard: Type[T] = Dict[Any, Any]) -> TypeGuard[List[T]]:
    """
    Check if object is a list of dictionaries.

    Parameters
    ----------
    d : Any
        The object to check.
    guard : Type[T]
        The type guard of the dictionaries. Defaults to dict.

    Returns
    -------
    TypeGuard[List[T]]
        True if object is a list of dictionaries.

    Examples
    --------
    >>> is_list_dict([{"a": 1}, {"b": 2}])
    True
    """
    return isinstance(d, (list, tuple)) and isinstance(d[0], dict)


def is_typed_dict(object: Any, target: Type[Td]) -> TypeGuard[Td]:
    """
    Check if object is a typed dictionary.

    Parameters
    ----------
    object : Any
        The object to check.

    target : Type[T]
        The type of the dictionary.

    Returns
    -------
    TypeGuard[T]
        True if object is a typed dictionary.

    Examples
    --------
    >>> from typing_extensions import TypedDict
    >>> class Foo(TypedDict):
    ...     a: int
    >>> is_typed_dict({"a": 1}, Foo)
    True
    """
    if not isinstance(object, dict):
        return False

    k_obj = set(object.keys())
    ks = set(target.__annotations__.keys())

    if hasattr(target, "__total__") and target.__total__:
        return all(k in k_obj for k in ks)
    else:
        return any(k in k_obj for k in ks)
