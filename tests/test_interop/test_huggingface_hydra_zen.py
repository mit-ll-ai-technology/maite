import pytest
from hydra_zen import ZenStore

from jatic_toolbox.interop.hydra_zen import (
    create_huggingface_dataset_config,
    create_huggingface_model_config,
)


@pytest.mark.parametrize(
    "creator",
    [
        lambda: create_huggingface_dataset_config("Bingsu/Cat_and_Dog"),
        lambda: create_huggingface_model_config(filter="detr"),
    ],
)
def test_init_hydra_zen_creators(creator):
    output = creator()
    assert isinstance(output, ZenStore)


def test_init_hydra_zen_hf_model_warns():
    with pytest.warns(UserWarning):
        create_huggingface_model_config(filter="foo")
