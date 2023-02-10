import pytest
from hydra_zen import ZenStore

from jatic_toolbox.interop.hydra_zen import create_smqtk_model_config


@pytest.mark.parametrize(
    "creator",
    [
        create_smqtk_model_config,
    ],
)
def test_init_hydra_zen_creators(creator):
    output = creator()
    assert isinstance(output, ZenStore)
