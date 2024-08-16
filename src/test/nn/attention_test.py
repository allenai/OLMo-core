from typing import Any, Dict

import pytest
import torch

from olmo_core.nn.attention import Attention, FusedAttention

from ..utils import requires_flash_attn, requires_gpu


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16")])
def test_fused_attention_against_non_fused(dtype):
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32
    batch_size = 2
    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device="cuda",
    )

    attention = Attention(**kwargs)
    fused_att = FusedAttention(**kwargs)

    x1 = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")
    x2 = x1.clone()

    with torch.autocast("cuda", dtype=dtype, enabled=True):
        y1 = attention(x1)
        y2 = fused_att(x2)

    torch.testing.assert_close(y1, y2)
