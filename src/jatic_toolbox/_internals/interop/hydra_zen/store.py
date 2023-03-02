from hydra_zen import ZenStore

__all__ = ["jatic_store"]

jatic_store: ZenStore = ZenStore(name="jatic-interop", overwrite_ok=False)
