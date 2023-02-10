from hydra_zen import ZenStore

from .store import jatic_store

try:
    from .smqtk_configs import create_smqtk_model_config
except ImportError:

    def create_smqtk_model_config() -> ZenStore:
        # doc-ignore: GL08
        raise ImportError(
            "Cannot create SMQTK configuration because SMQTK is not installed."
        )


try:
    from .hf_configs import (
        create_huggingface_dataset_config,
        create_huggingface_model_config,
    )
except ImportError:

    def create_huggingface_dataset_config(**kwargs) -> ZenStore:
        # doc-ignore: GL08
        raise ImportError(
            "Cannot create HuggingFace configuration because HuggingFace `datasets` is not installed."
        )

    def create_huggingface_model_config(**kwargs) -> ZenStore:
        # doc-ignore: GL08
        raise ImportError(
            "Cannot create HuggingFace configuration because HuggingFace `transformers` is not installed."
        )


__all__ = [
    "jatic_store",
    "create_huggingface_dataset_config",
    "create_huggingface_model_config",
    "create_smqtk_model_config",
]
