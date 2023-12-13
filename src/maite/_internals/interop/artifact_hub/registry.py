# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import warnings
from typing import Type

from maite.errors import InternalError, InvalidArgument
from maite.protocols import ArtifactHubEndpoint


class HubEndpointRegistry:
    """The registry for ArtifactHubProvider endpoint types.

    This class holds the mapping of key names -> endpoint implementations

    Attributes
    ----------
    registered_endpoints : ClassVar[Dict[str, ]]
        Maps registered names -> provider implementation types

    Methods
    -------
    register_impl(impl_type: Type[ArtifactHubEndpoint], spec_tag: str)
        Register an Endpoint type with a spec prefix tag
    get_impl(spec_tag: str) -> Type[ArtifactHubEndpoint]
        Lookup the Endpoint type associated with the given spec tag
    list_registered_specs() -> List[str]
        List Endpoint spec tags registered

    """

    registered_endpoints = {}

    def __init__(self, *args, **kwargs):
        # Error on __init__ no need to instantiate the class. Technically, this would be
        # "safe" in that the instance would hold a reference to the dict defined above,
        # but there is no need to allow/encourage doing that
        raise InternalError(
            "The ProviderRegistry functionality should be accessed "
            "through the class itself, do not instantiate this type"
        )

    @classmethod
    def register_impl(cls, impl_type: Type[ArtifactHubEndpoint], spec_tag: str):
        """Register an Endpoint implementation

        Parameters
        ----------
        impl_type : Type[ArtifactHubEndpoint]
          Some type implementing the required protocol to act as a hub endpoint
        spec_tag : str
          The spec prefix that will indicate this Endpoint type is to be used

        """
        if spec_tag in cls.registered_endpoints:
            warnings.warn(
                f"Attempting to register endpoint {impl_type.__name__} under "
                f"spec tag {spec_tag} which will overwrite the existing endpoint "
                f"{cls.registered_endpoints[spec_tag].__name__} registered under "
                "that name"
            )
        if not isinstance(impl_type, ArtifactHubEndpoint):
            raise InvalidArgument(
                "Attempting to register a hub endpoint type which does not satisfy the protocol for a hub endpoint."
            )
        cls.registered_endpoints[spec_tag] = impl_type

    @classmethod
    def list_registered_specs(cls):
        return list(cls.registered_endpoints.keys())

    @classmethod
    def get_endpoint_impl(cls, spec):
        """Get the type associated with the given spec

        Parameters
        ----------
        spec : str
          The spec tag used to register the type

        """
        spec_tag = spec
        impl_type = cls.registered_endpoints.get(spec_tag, None)
        if impl_type is None:
            registered_msg = "\n\t".join(cls.registered_endpoints.keys())
            raise InvalidArgument(
                f"Unable to find interface for spec tag {spec_tag}, the following are registered: \n{registered_msg}\n"
                "Register a custom endpoint interface by inheriting from 'HubEndpoint'."
            )
        return impl_type
