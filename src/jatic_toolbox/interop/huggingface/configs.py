from jatic_toolbox._internals.interop.huggingface.configs import (
    create_huggingface_dataset_config,
    create_huggingface_model_config,
)

__all__ = ["create_huggingface_dataset_config", "create_huggingface_model_config"]


# import warnings
# from typing import Any

# from datasets.load import load_dataset
# from huggingface_hub.hf_api import list_models
# from hydra_zen import ZenStore, builds

# from jatic_toolbox._internals.interop.hydra_zen import jatic_store
# from jatic_toolbox.interop import huggingface

# jatic_datasets = jatic_store(group="dataset")
# jatic_models = jatic_store(group="model")


# def create_huggingface_dataset_config(path: str, **dataset_kwargs: Any) -> ZenStore:
#     """
#     Create HuggingFace Datasets.

#     Parameters
#     ----------
#     path : str
#         HuggingFace dataset path used in `datasets.load_dataset`.

#     **dataset_kwargs : Any
#         Keyword arguments for HuggingFace's `datasets.load_dataset`.

#     Returns
#     -------
#     ZenStore
#         JATIC datsets config store with added HuggingFace dataset configurations.

#     Examples
#     --------
#     >> jatic_datasets = ccreate_huggingface_dataset_config("Bingsu/Cat_and_Dog")
#     {'dataset': ['Bingsu__Cat_and_Dog', 'biglam__nls_chapbook_illustrations']}
#     """
#     name = path.replace("/", "__")
#     if name not in [m[1] for m in jatic_store["dataset"]]:
#         jatic_datasets(
#             builds(load_dataset, path, populate_full_signature=True, **dataset_kwargs),
#             name=name,
#         )
#     return jatic_datasets


# def create_huggingface_model_config(**list_models_kwargs: Any) -> ZenStore:
#     """
#     Create HuggingFace Model Confgurations.

#     This function uses `huggingface_hub.list_models` to search models
#     defined by the parameters.

#     Parameters
#     ----------
#     **list_models_kwargs : Any
#         Keyword arguments for HuggingFace's `huggingface_hub.list_models`.

#     Returns
#     -------
#     ZenStore
#         JATIC models configuration store with HuggingFace model configurations.

#     Examples
#     --------
#     >> jatic_configs = create_huggingface_model_config(filter="detr")

#     >> from huggingface_hub import (
#     ...  HfApi,
#     ...  ModelFilter,
#     ...  ModelSearchArguments)
#     >> api = HfApi()
#     >> model_args = ModelSearchArguments()
#     >> filt = ModelFilter(
#     ... task=model_args.pipeline_tag.ObjectDetection,
#     ... model_name="detr",
#     ... library=model_args.library.PyTorch)
#     >> jatic_configs = create_huggingface_model_config(filter=filt)
#     >> jatic_configs
#     {'model': [''facebook__detr-resnet-101-dc5', ...]}
#     """
#     models = list_models(**list_models_kwargs)
#     if len(models) == 0:
#         warnings.warn("`huggingface_hub.list_models` returned an empty list of models")
#         return jatic_models

#     _found = 0
#     for m in models:
#         if (
#             m.pipeline_tag is not None
#             and "object-detection" in m.pipeline_tag
#             and m.modelId is not None
#         ):
#             name = m.modelId.replace("/", "__")

#             if name not in [m[1] for m in jatic_store["model"]]:
#                 _found = 1
#                 jatic_models(
#                     builds(
#                         huggingface.object_detection.HuggingFaceObjectDetector,
#                         model=m.modelId,
#                     ),
#                     name=name,
#                 )
#             else:  # pragma: no cover
#                 _found = 2

#     if _found == 0:  # pragma: no cover
#         warnings.warn(
#             "No model config created, please check your search parameters. Note: this function only supports `object-detection` tasks."
#         )

#     return jatic_models
