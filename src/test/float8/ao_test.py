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

    float8_config = AOFloat8LinearConfig.recommended()
    assert float8_config.enable_fsdp_float8_all_gather
    assert float8_config.force_recompute_fp8_weight_in_bwd
    assert float8_config.round_scales_to_power_of_2


@pytest.mark.skipif(not has_torchao(), reason="Requires torchao")
def test_ao_mx_linear_config():
    from torchao.prototype.mx_formats.config import MXLinearConfig

    from olmo_core.config import DType
    from olmo_core.float8.ao import (
        AOKernelPreference,
        AOMXFP8Dim1CastKernelChoice,
        AOMXLinearConfig,
        AOScaleCalculationMode,
    )

    assert isinstance(AOMXLinearConfig().to_ao_type(), MXLinearConfig)
    assert AOMXLinearConfig(block_size=32).to_ao_type().block_size == 32
    assert AOMXLinearConfig(elem_dtype=DType.float8_e4m3fn).to_ao_type().elem_dtype.itemsize == 1
    assert (
        AOMXLinearConfig(kernel_preference=AOKernelPreference.auto).to_ao_type().kernel_preference
        == "auto"
    )
    assert (
        AOMXLinearConfig(mxfp8_cast_kernel_choice=AOMXFP8Dim1CastKernelChoice.cuda)
        .to_ao_type()
        .mxfp8_cast_kernel_choice.value
        == "cuda"
    )
    assert (
        AOMXLinearConfig(scale_calculation_mode=AOScaleCalculationMode.rceil)
        .to_ao_type()
        .scale_calculation_mode.value
        == "rceil"
    )

    mxfp8_rceil_config = AOMXLinearConfig.mxfp8_cublas_rceil()
    assert mxfp8_rceil_config.kernel_preference == AOKernelPreference.auto
    assert mxfp8_rceil_config.mxfp8_cast_kernel_choice == AOMXFP8Dim1CastKernelChoice.cuda
    assert mxfp8_rceil_config.scale_calculation_mode == AOScaleCalculationMode.rceil
