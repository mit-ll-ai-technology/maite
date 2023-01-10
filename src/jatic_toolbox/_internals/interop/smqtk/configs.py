from hydra_zen import ZenStore, builds

from jatic_toolbox._internals.interop.hydra_zen import jatic_store

from . import object_detection

jatic_datasets = jatic_store(group="dataset")
jatic_models = jatic_store(group="model")


def create_smqtk_model_config() -> ZenStore:
    """
    Create SMQTK CenterNet Confgurations.

    Returns
    -------
    ZenStore
        JATIC models configuration store with SMQTK CenterNet configurations.

    Examples
    --------
    >> jatic_configs = create_smqtk_model_config()
    >> jatic_configs["model"]
    {('model', 'smqtk__centernet-resnet50'): types.Builds_CenterNetVisdrone,
     ('model', 'smqtk__centernet-resnet18'): types.Builds_CenterNetVisdrone,
     ('model', 'smqtk__centernet-res2net50'): types.Builds_CenterNetVisdrone}
    """

    for k in object_detection._MODELS.keys():
        name = f"smqtk__centernet-{k}"

        if name not in [m[1] for m in jatic_store["model"]]:
            jatic_models(
                builds(
                    object_detection.CenterNetVisdrone,
                    model=k,
                    populate_full_signature=True,
                ),
                name=name,
            )

    return jatic_models
