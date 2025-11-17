import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import (
    FusedRMSNorm,
    L2Norm,
    LayerNormConfig,
    LayerNormType,
    RMSNorm,
)
from olmo_core.testing import requires_flash_attn_2, requires_gpu


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("bias", [pytest.param(True, id="bias"), pytest.param(False, id="no-bias")])
@pytest.mark.parametrize(
    "dtype", [pytest.param(torch.float32, id="fp32"), pytest.param(torch.bfloat16, id="bf16")]
)
def test_fused_rms_norm(bias, dtype):
    dim = 64
    norm = RMSNorm(size=dim, bias=bias, init_device="cuda")
    norm_fused = FusedRMSNorm(size=dim, bias=bias, init_device="cuda")

    x = torch.randn(4, dim, device="cuda", dtype=dtype)
    y1 = norm(x)
    y2 = norm_fused(x)
    torch.testing.assert_close(y1, y2)


def test_layer_norm_builder_config():
    norm = LayerNormConfig(name=LayerNormType.l2_norm).build(size=1024)
    assert isinstance(norm, L2Norm)

    with pytest.raises(OLMoConfigurationError):
        LayerNormConfig(name=LayerNormType.l2_norm, elementwise_affine=True).build(size=1024)
