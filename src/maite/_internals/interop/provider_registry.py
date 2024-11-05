# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    get_args as get_type_args,
    overload,
)

from maite._internals.protocols.providers import (
    DatasetProvider,
    MetricProvider,
    ModelProvider,
)
from maite.errors import InternalError, InvalidArgument

ArtifactName = Literal["model", "dataset", "metric"]
AnyProvider = Union[ModelProvider, DatasetProvider, MetricProvider]

_PROVIDER_PROTOCOLS: tuple[type[AnyProvider], ...] = tuple(get_type_args(AnyProvider))

# And from that build the enum
# The enum gives us a compact way to map a desired artifact on the proper protocol interface
# name -> tag: ProviderProtocolTag[name]
# tag -> Protocol: PROVIDER_PROTOCOLS[tag.value]
_ProviderProtocolTag = Enum(
    "_ProviderProtocolTag", ((x, i) for i, x in enumerate(get_type_args(ArtifactName)))
)


#
# Provider Registry Internals
#


class _RegisteredProviderInfo(NamedTuple):
    """The value used to store the registered implementation by the registry

    Attributes
    ----------
    impl: type[AnyProvider]
      The type which has been registered as a provider
    implements:
      A tuple of ProviderProtocolTag's indicating which protocols ``impl`` supports

    Notes
    -----
    Why cache the implements parameter?
        The runtime checks on protocols can be expensive, so we do them once at
        registration time and cache the results to check against at the time when an
        implementation is looked up for use.
    """

    impl: type[AnyProvider]
    implements: tuple[_ProviderProtocolTag, ...]


class _ProviderRegistry:
    """The registry for provider implementations
    This class holds the mapping of key names -> provider implementations for use by
    the factories

    Attributes
    ----------
    registered_provider : dict[str, RegisteredProviderInfo]
        Maps registered names -> provider implementation types

    Methods
    -------
    register_impl(impl_type: type[AnyProvider], key: str) -> None
        Register a provider under the given key
    get_impl(key: str, enforce_protocol: ArtifactName | None = None) -> type[AnyProvider]
        Lookup the Provider type associated with a key
    list_registered(enforce_protocol: ArtifactName | None = None) -> list[str]
        List provider keys registered (optionally limit to specific protocol)

    """

    registered_providers: ClassVar[dict[str, _RegisteredProviderInfo]] = {}

    def __init__(self, *args, **kwargs):
        # Error on __init__ no need to instantiate the class Technically, this would be
        # "safe" in that the instance would hold a reference to the dict defined above,
        # but there is no need to allow/encourage doing that
        raise InternalError(
            "The ProviderRegistry functionality should be accessed "
            "through the class itself, do not instantiate this type"
        )

    @classmethod
    def register_impl(cls, impl_type: type[AnyProvider], key: str):
        """Register a Provider implementation

        Parameters
        ----------
        impl_type : type[AnyProvider]
           Some type implementing at least one of the provider protocols
        key : str
           The string name that can be used to find that provider implementation

        Examples
        --------
        >>> import torch
        >>> from collections.abc import Iterable
        >>> from maite.protocols.image_classification import Dataset
        >>> class SimpleProvider:
        ...     def help(self, name) -> str:
        ...         ...
        ...     def list_datasets(self) -> Iterable[str]:
        ...         ...
        ...     def load_dataset(self) -> Dataset:
        ...         ...
        >>> _ProviderRegistry.register_impl(SimpleProvider, key="simple_provider")
        """
        if key in cls.registered_providers:
            # Warn on key overwrite, we allow this because there are legitimate uses for
            # the pattern, extending a native provider to add some special checks
            # specific to that user's org/standards/security protocols for examples.  We
            # still warn so that someone knows they have done this. They can filter the
            # warning around the registration site if they wish to have a warning-free
            # experience
            warnings.warn(
                f"Attempting to register provider {impl_type.__name__} under "
                f"name {key} which will overwrite the existing provider "
                f"{cls.registered_providers[key].impl.__name__} registered under "
                "that name"
            )
        implements_protocols = tuple(
            _ProviderProtocolTag(i)
            for i, p in enumerate(_PROVIDER_PROTOCOLS)
            if isinstance(impl_type, p)
        )
        if not implements_protocols:
            # Error when the implementation does not satisfy any protocols. They can use
            # a provider-like type without registration if they have a very odd
            # situation that requires it, but passing the provider directly should be
            # possible anywhere a factory-key is also acceptable, checks at those
            # interact should be static.  Explicitly disallow registration of incomplete
            # provider implementations to prevent them using one which is ill-formed
            # unless they know what they are doing
            raise InvalidArgument(
                f"Attempting to register a provider {impl_type.__name__} "
                f"which does not satisfy any of the protocols "
                f"{_PROVIDER_PROTOCOLS}."
            )
        cls.registered_providers[key] = _RegisteredProviderInfo(
            impl_type, implements_protocols
        )

    @classmethod
    def list_registered(cls, enforce_protocol: ArtifactName | None = None) -> list[str]:
        """List the implementations registered which satisfy requirements to be a provider for a particular artifact"""
        if enforce_protocol:
            enforce_tag = _ProviderProtocolTag[enforce_protocol]
            return [
                k
                for k in cls.registered_providers.keys()
                if enforce_tag in cls.registered_providers[k].implements
            ]

        return list(cls.registered_providers.keys())

    @classmethod
    def get_impl(
        cls,
        key: str,
        enforce_protocol: ArtifactName | None = None,
    ) -> type[AnyProvider]:
        """Get the type associated with the provider key

        Parameters
        ----------
        key : str
           The name used to register the provider
        enforce_protocol : ArtifactName, default = None
            The default behavior, when the value is None, is to not enforce a specific
            artifact. Any registered type will implement at least one complete provider
            protocol.

        Returns
        -------
        type[AnyProvider]
           A type (class) that implements at least one of the provider protocols. If
           restricted with ``enforce_protocol`` registered types which do not implement
           that specific artifact protocol will be considered invalid.

        Raises
        ------
        InvalidArgument
           If the key is not registered, or the associated type does not implement the
           protocol when using ``enforce_protocol``

        """
        impl_info = cls.registered_providers.get(key, None)
        if impl_info is None:
            registered_msg = "\n\t".join(cls.list_registered(enforce_protocol))
            raise InvalidArgument(
                f"Unable to find provider registered for key:  {key}, "
                f"the following are registered: \n\t{registered_msg}\n"
            )

        if enforce_protocol:
            enforce_tag = _ProviderProtocolTag[enforce_protocol]
            if enforce_tag not in impl_info.implements:
                implements_msg = "\n\t".join(i.name for i in impl_info.implements)
                raise InvalidArgument(
                    f"The key '{key}' does not implement the interface "
                    f"for {enforce_tag.name} providers. The associated "
                    f"type {impl_info.impl.__name__} implements the "
                    f"following providers:\n {implements_msg}"
                )

        return impl_info.impl


#
# Registration Methods
#

PT = TypeVar("PT", bound=AnyProvider)


@overload
def register_provider(provider_type: type[PT], *, key: str | None = None) -> type[PT]:
    ...


@overload
def register_provider(
    provider_type: None,
) -> Callable[[type[PT]], type[PT]]:
    ...


@overload
def register_provider(*, key: str | None = None) -> Callable[[type[PT]], type[PT]]:
    ...


@overload
def register_provider(provider_type: type[PT]) -> type[PT]:
    ...


def register_provider(
    provider_type: type[PT] | None = None, *, key: str | None = None
) -> Callable[[type[PT]], type[PT]] | type[PT]:
    """
    Register a provider.

    Parameters
    ----------
    provider_type : type[AnyProvider]
       A class which implements one of the provider protocols. The registry will enforce this requirement dynamically.
    key : str, default = None
       The key this implementation will be associated with in the registry.

    Returns
    -------
    AnyProvider
       The registered implementation, unchanged.

    Raises
    ------
    InvalidAugmentError
       If the ``provider_type`` does not implement at least one of the provider protocols
       If a key is not provided, and the class does not define the PROVIDER_REGISTRY_KEY
       attribute.

    Examples
    --------
    This utility can be used in several ways, lets begin with a simple, valid Provider type

    >>> from maite.protocols.object_detection import Dataset
    >>> class SimpleProvider:
    ...     def __init__(self, data: Dataset, name: str):
    ...         self.dataset = data
    ...         self.name = name
    ...     def help(self, name):
    ...         return "This provider only has one dataset, there is no information about it"
    ...     def list_datasets(self):
    ...         return [self.name]
    ...     def load_dataset(self):
    ...         return self.dataset

    The type can be registered with an explicit key, by calling the function

    >>> register_provider(SimpleProvider, key="simple") # type: ignore
    <class 'src.maite._internals.interop.provider_registry.SimpleProvider'>

    Classes which define the ``PROVIDER_REGISTRY_KEY`` attribute can be registered
    without an explicit ``key`` keyword argument

    >>> class SimpleProviderWithKeyDef(SimpleProvider):
    ...     PROVIDER_REGISTRY_KEY = "another_simple"
    >>> register_provider(SimpleProviderWithKeyDef) # type: ignore
    <class 'src.maite._internals.interop.provider_registry.SimpleProviderWithKeyDef'>

    This function can also act as a decorator on the class definition

    >>> @register_provider # type: ignore
    ... class SimpleProviderWithKeyDefDecorated(SimpleProvider):
    ...     PROVIDER_REGISTRY_KEY = "decorated_simple"

    Or without implying the key from the class attribute

    >>> @register_provider(key="key_decorated") # type: ignore
    ... class SimpleProviderDecorated(SimpleProvider):
    ...     ...

    All of these implement the DatasetProvider protocol so they will be registered with
    the factory and associated with "dataset" protocol.

    >>> list_providers(enforce_protocol='dataset')  # doctest: +SKIP
    ['simple', 'another_simple', 'decorated_simple', 'key_decorated'] NOTE: registration of providers with a singleton (i.e. the registry) is creating a coupling between tests, need to figure out how to handle it

    However, they will not be associated with other artifact types as they don't
    implement the necessary interface.

    >>> list_providers(enforce_protocol='model')
    []
    """

    def _registration_helper(
        provider_type: type[PT], *, key: str | None = None
    ) -> type[PT]:
        if key is None:
            key = getattr(provider_type, "PROVIDER_REGISTRY_KEY", None)

        if key is None:
            raise InvalidArgument(
                "Registering a provider implementation without a `key` keyword argument requires the type define the class attribute PROVIDER_REGISTRY_KEY"
            )

        _ProviderRegistry.register_impl(provider_type, key)
        return provider_type

    def wrap(provider_t):
        return _registration_helper(provider_t, key=key)

    if provider_type is None:
        # called like @register_provider()
        return wrap

    # called like @register_provider
    return wrap(provider_type)


#
# Query the registry
#


def list_providers(enforce_protocol: ArtifactName | None = None) -> list[str]:
    """
    List registered provider keys.

    Parameters
    ----------
    enforce_protocol : ArtifactName, default None
       When provided results will be filtered to registered providers which can provide
       the specified artifact type. The default behavior is to list all providers.

    Returns
    -------
    list[str]
       A list of keys which be used to look up a provider which is able to provide the
       specified artifact type.

    Examples
    --------
    >>> sorted(list_providers())  # doctest: +SKIP
    # contains native providers only, NOTE: registration of providers with a singleton (i.e. the registry) is creating a coupling between tests, need to figure out how to handle it
    ... [...]

    >>> from maite.protocols.object_detection import Dataset
    >>> class MyProvider:
    ...     def __init__(self, data: Dataset, name: str):
    ...         self.dataset = data
    ...         self.name = name
    ...     def help(self, name):
    ...         return "This provider only has one dataset, there is no information about it"
    ...     def list_datasets(self):
    ...         return [self.name]
    ...     def load_dataset(self):
    ...         return self.dataset

    >>> register_provider(MyProvider, key='my_provider') # type: ignore
    <class 'src.maite._internals.interop.provider_registry.MyProvider'>
    >>> 'my_provider' in list_providers()
    True
    """
    return _ProviderRegistry.list_registered(enforce_protocol)


def get_provider_type(
    key: str, enforce_protocol: ArtifactName | None = None
) -> type[AnyProvider]:
    """
    Get the type from the registry based on its key.

    Parameters
    ----------
    key : str
       The key associated with the desired provider type in the registry.
    enforce_protocol : ArtifactName, default None
       When provided, an error is raised if the provider does is not able to provide the specified artifact type.

    Returns
    -------
    type[AnyProvider]
       The provider type registered with the given key.

    Raises
    ------
    InvalidArgumentError
       When there is no suitable provider registered with the given key.

    Examples
    --------
    >>> class MyProvider:
    ...     def list_datasets(self):
    ...         ...
    ...     def load_dataset(self):
    ...         ...
    ...     def help(self):
    ...         ...
    >>> register_provider(MyProvider, key='my_provider') # type: ignore
    <class 'src.maite._internals.interop.provider_registry.MyProvider'>
    >>> get_provider_type('my_provider') is MyProvider
    True
    """
    return _ProviderRegistry.get_impl(key, enforce_protocol)


def create_provider(
    key: str,
    enforce_protocol: ArtifactName | None = None,
    provider_args: tuple[Any, ...] | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> AnyProvider:
    """
    Create and return an instance of the specified provider type.

    Parameters
    ----------
    key : str
       The key associated with the desired provider type in the registry.
    enforce_protocol : ArtifactName, default None
       When provided, an error is raised if the provider does is not able to provide the specified artifact type.
    provider_args : tuple[Any, ...], default None
       Args to forward to the provider type's __init__.
    provider_kwargs : dict[str, Any], default None
       Kwargs to forward to the provider type's __init__.

    Returns
    -------
    AnyProvider
       An instance of the provider registered with the given key.

    Raises
    ------
    InvalidArgument
        When there is no suitable provider registered with the given key.

    Notes
    -----
    This method will invoke the type's __init__ which may raise other exceptions than
    those listed here.

    Examples
    --------
    >>> @register_provider(key='my_provider') # type: ignore
    ... class MyProvider:
    ...    def __init__(self, arg1, arg2):
    ...        self.arg1 = arg1
    ...        self.arg2 = arg2
    ...    def list_datasets(self):
    ...         ...
    ...    def load_dataset(self):
    ...         ...
    ...    def help(self):
    ...         ...
    >>> p = create_provider(key='my_provider', provider_args=('a', 'b'))
    >>> p.arg1 # type: ignore
    'a'
    >>> p.arg2 # type: ignore
    'b'
    """
    _args = provider_args or ()
    _kwargs = provider_kwargs or {}
    return get_provider_type(key, enforce_protocol)(*_args, **_kwargs)
