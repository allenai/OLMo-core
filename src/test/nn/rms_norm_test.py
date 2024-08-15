import torch

from olmo_core.nn.rms_norm import FusedRMSNorm, RMSNorm

from ..utils import requires_flash_attn, requires_gpu


@requires_gpu
@requires_flash_attn
def test_fused_rms_norm():
    dim = 64
    norm = RMSNorm(size=dim, init_device="cuda")
    norm_fused = FusedRMSNorm(size=dim, init_device="cuda")

    x = torch.randn(4, dim, device="cuda")
    y1 = norm(x)
    y2 = norm_fused(x)
    torch.testing.assert_close(y1, y2)
