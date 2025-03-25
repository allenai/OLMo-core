import importlib.util

import pytest

from olmo_core.float8.ao import AOCastConfig, AOFloat8LinearConfig, AOScalingType


def has_torchao() -> bool:
    return importlib.util.find_spec("torchao") is not None


@pytest.mark.skipif(not has_torchao(), reason="Requires torchao")
def test_ao_float8_linear_config():
    from torchao.float8.config import Float8LinearConfig, ScalingType

    assert isinstance(AOFloat8LinearConfig().to_ao_type(), Float8LinearConfig)
    assert AOFloat8LinearConfig(emulate=True).to_ao_type().emulate
    assert (
        AOFloat8LinearConfig(cast_config_input=AOCastConfig(scaling_type=AOScalingType.disabled))
        .to_ao_type()
        .cast_config_input.scaling_type
        == ScalingType.DISABLED
    )
