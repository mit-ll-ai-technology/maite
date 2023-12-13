# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import ModuleType
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Pattern,
    Type,
    TypeVar,
    cast,
    overload,
)

from maite._internals.interop.provider import ArtifactName, register_provider
from maite._internals.protocols.typing import (
    AnyEntrypoint,
    DatasetEntrypoint,
    MetricEntrypoint,
    ModelEntrypoint,
)
from maite.errors import InvalidArgument
from maite.protocols import ArtifactHubEndpoint, Dataset, Metric, Model, TaskName

from .deduction import make_entrypoint_deduction_filter
from .module_utils import import_hubconf
from .registry import HubEndpointRegistry


def split_spec_path(spec_path):
    spec, path = spec_path.split("::", 1)
    return spec, path


#
# ArtifactHubProvider type
#

HubEP_T = TypeVar("HubEP_T", bound=ArtifactHubEndpoint)
T_co = TypeVar("T_co", covariant=True)


@register_provider(key="artifact_hub")
class ArtifactHubProvider:
    def __init__(self, spec_path: str, **endpoint_options: Any) -> None:
        spec, path = split_spec_path(spec_path)
        endpoint_type: Type[
            ArtifactHubEndpoint
        ] = HubEndpointRegistry.get_endpoint_impl(spec)
        self.endpoint: ArtifactHubEndpoint = endpoint_type(path, **endpoint_options)

    def update_endpoint_options(self, **endpoint_options: Any) -> None:
        self.endpoint = self.endpoint.update_options(**endpoint_options)

    def __repr__(self) -> str:
        return f"ArtifactHubProvider(endpoint={self.endpoint})"

    def _get_module(self, **endpoint_options: Any) -> ModuleType:
        local_dir = self.endpoint.get_cache_or_reload(**endpoint_options)
        return import_hubconf(local_dir, "hubconf.py")

    def list(
        self,
        *,
        name: str | None = None,
        artifact_type: ArtifactName | None = None,
        task: TaskName | None = None,
        require_deduction: bool = True,
        filter_regex: str | Pattern[str] | None = None,
        filter_str: str | List[str] | None = None,
    ) -> List[str]:
        hub_module = self._get_module()
        filter = make_entrypoint_deduction_filter(
            name=name,
            filter_type=artifact_type,
            filter_task=task,
            require_deduction=require_deduction,
            filter_regex=filter_regex,
            filter_str=filter_str,
        )
        return [
            ep_name
            for ep_name in dir(hub_module)
            if filter(getattr(hub_module, ep_name))
        ]

    def get_entrypoint(self, name: str) -> AnyEntrypoint:
        hub_module = self._get_module()
        ep = getattr(hub_module, name, None)
        if not callable(ep):
            raise InvalidArgument(f"{self} does not have an entrypoint {name}")
        return ep

    # Common component for provider protocols

    def help(self, name: str) -> str:
        return self.get_entrypoint(name).__doc__ or ""

    # Dataset Provider protocol

    def list_datasets(
        self,
        *,
        dataset_name: str | None = None,
        task: TaskName | None = None,
        filter_regex: str | Pattern[str] | None = None,
        filter_str: str | List[str] | None = None,
    ) -> Iterable[Any]:
        return self.list(
            artifact_type="dataset",
            name=dataset_name,
            task=task,
            require_deduction=True,
            filter_regex=filter_regex,
            filter_str=filter_str,
        )

    def load_dataset(
        self,
        *,
        dataset_name: str,
        task: TaskName | None = None,
        split: str | None = None,
        **entrypoint_options: Any,
    ) -> Dataset[Any]:
        ep = self.get_entrypoint(dataset_name)
        filter = make_entrypoint_deduction_filter(
            filter_type="dataset", filter_task=task, require_deduction=True
        )
        if not filter(ep):
            raise InvalidArgument(
                f"Entrypoint {dataset_name} does exists, but does not return a Dataset suitable for {task}."
            )
        # cast is required, type checker can't detect filter limiting the type of ep
        ep = cast(DatasetEntrypoint, ep)
        return ep(split=split, **entrypoint_options)

    # Metric Provider protocol

    def list_metrics(
        self,
        *,
        metric_name: str | None = None,
        task: TaskName | None = None,
        filter_regex: str | Pattern[str] | None = None,
        filter_str: str | List[str] | None = None,
    ) -> Iterable[str]:
        return self.list(
            artifact_type="metric",
            name=metric_name,
            task=task,
            require_deduction=True,
            filter_regex=filter_regex,
            filter_str=filter_str,
        )

    def load_metric(
        self,
        *,
        metric_name: str,
        task: TaskName | None = None,
        **entrypoint_options: Any,
    ) -> Metric[Any, Any]:
        ep = self.get_entrypoint(metric_name)
        filter = make_entrypoint_deduction_filter(
            filter_type="metric", filter_task=task, require_deduction=True
        )
        if not filter(ep):
            raise InvalidArgument(
                f"Entrypoint {metric_name} does exists, but does not return a Metric suitable for {task}."
            )
        # cast is required, type checker can't detect filter limiting the type of ep
        ep = cast(MetricEntrypoint, ep)
        return ep(**entrypoint_options)

    # Model Provider protocol

    def list_models(
        self,
        *,
        filter_str: str | List[str] | None = None,
        model_name: str | None = None,
        task: TaskName | None = None,
        filter_regex: str | Pattern[str] | None = None,
    ) -> Iterable[str]:
        return self.list(
            artifact_type="model",
            task=task,
            name=model_name,
            require_deduction=True,
            filter_regex=filter_regex,
            filter_str=filter_str,
        )

    def load_model(
        self,
        *,
        model_name: str,
        task: TaskName | None = None,
        **entrypoint_options: Any,
    ) -> Model[Any, Any]:
        ep = self.get_entrypoint(model_name)
        filter = make_entrypoint_deduction_filter(
            filter_type="model", filter_task=task, require_deduction=True
        )
        if not filter(ep):
            raise InvalidArgument(
                f"Entrypoint {model_name} does exists, but does not return a Model suitable for {task}."
            )
        # cast is required, type checker can't detect filter limiting the type of ep
        ep = cast(ModelEntrypoint, ep)
        return ep(**entrypoint_options)

    #
    # Endpoint Registration API
    #

    @overload
    @staticmethod
    def register_endpoint(
        endpoint_type: Type[HubEP_T],
    ) -> Type[HubEP_T]:
        ...

    @overload
    @staticmethod
    def register_endpoint(
        endpoint_type: None,
    ) -> Callable[[Type[HubEP_T]], Type[HubEP_T]]:
        ...

    @overload
    @staticmethod
    def register_endpoint(
        *, spec_tag: str | None = None
    ) -> Callable[[Type[HubEP_T]], Type[HubEP_T]]:
        ...

    @overload
    @staticmethod
    def register_endpoint(endpoint_type: Type[HubEP_T], spec_tag: str) -> Type[HubEP_T]:
        ...

    @staticmethod
    def register_endpoint(
        endpoint_type: Type[HubEP_T] | None = None,
        spec_tag: str | None = None,
    ) -> Callable[[Type[HubEP_T]], Type[HubEP_T]] | Type[HubEP_T]:
        """Register a provider implementation associated with a spec_tag

        Parameters
        ----------
        entrypoint_type : Type[ArtifactHubEntrypoint]
          A class implementing the ``ArtifactHubEntrypoint`` protocol
        spec_tag : str, default = None
          The spec tag that will select this endpoint type when found in the target URL

        Returns
        -------
        entrypoint_type : Type[ArtifactHubEntrypoint]
          The registered type unchanged

        Raises
        ------
        InvalidArgumentError
          When the provided type does not implement the protocol

        Examples
        --------
        This registration utility can be used in several ways, lets begin with a simple endpoint type

        >>> class SimpleEndpoint:
        ...     def get_cache_or_reload(self, *args, **kwargs):
        ...         return "/path/to/some/dir"

        We can register the above type with a spec tag using the function

        >>> register_endpoint(SimpleEndpoint, spec_tag="simple")

        Classes which define the ``ENDPOINT_REGISTRY_TAG`` attribute can be registered with an explicit ``spec_tag``

        >>> class SimpleEndpointWithKeyDef(SimpleEndpoint):
        ...     ENDPOINT_REGISTRY_TAG = "simple_with_classvar"
        >>> register_endpoint(SimpleEndpointWithKeyDef)

        We can also use the function as a class decorator

        >>> @register_endpoint
        >>> class SimpleEndpointWithKeyDefDecorated(SimpleEndpoint):
        ...     ENDPOINT_REGISTRY_TAG = "simple_with_classvar_decorated"

        Note that the ``ENDPOINT_REGISTRY_TAG`` must be defined by the class when used like
        the above example. If you want to register an endpoint using the decorator without
        putting the tag in the class definition, you may pass the ``spec_tag`` keyword
        argument to the decorator

        >>> @register_endpoint(spec_tag="decorated_with_kwarg")
        >>> class SimpleEndpointDecorated(SimpleEndpoint):
        ...     ...

        """

        def _registration_helper(
            endpoint_t: Type[HubEP_T], tag: Optional[str] = None
        ) -> Type[HubEP_T]:
            if tag is None:
                tag = getattr(endpoint_t, "ENDPOINT_REGISTRY_TAG", None)
            if tag is None:
                raise InvalidArgument(
                    "Registering an endpoint implementation without a `spec_tag` keyword argument requires the type define the class attribute `ENDPOINT_REGISTRY_TAG`."
                )
            HubEndpointRegistry.register_impl(endpoint_t, spec_tag=tag)
            return endpoint_t

        def wrap(endpoint_t):
            return _registration_helper(endpoint_t, tag=spec_tag)

        if endpoint_type is None:
            # called like @ArtifactHubProvider.register_endpoint()
            return wrap

        # called like @ArtifactHubProvider.register_endpoint
        return wrap(endpoint_type)

    @staticmethod
    def list_registered_endpoints() -> List[str]:
        return HubEndpointRegistry.list_registered_specs()
