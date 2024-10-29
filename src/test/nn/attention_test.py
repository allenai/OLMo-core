from typing import Any, Dict, Optional

import pytest
import torch

from olmo_core.nn.attention import Attention, FusedAttention
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType

from ..utils import DEVICES, FLASH_MARKS, GPU_MARKS, requires_flash_attn, requires_gpu


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16", marks=GPU_MARKS),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "n_kv_heads",
    [pytest.param(None, id="MHA"), pytest.param(1, id="MQA"), pytest.param(4, id="GQA")],
)
@pytest.mark.parametrize(
    "use_flash",
    [pytest.param(True, id="flash", marks=FLASH_MARKS), pytest.param(False, id="torch-SDPA")],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"clip_qkv": 8.0}, id="QKV-clip"),
        pytest.param({"rope": RoPEConfig()}, id="rope"),
        pytest.param({"rope": RoPEConfig(name=RoPEType.complex)}, id="complex-rope"),
        pytest.param({"qk_norm": LayerNormConfig()}, id="qk-norm"),
    ],
)
def test_attention(
    dtype: torch.dtype,
    device: torch.device,
    n_kv_heads: Optional[int],
    use_flash: bool,
    kwargs: Dict[str, Any],
):
    if use_flash and dtype == torch.float32:
        pytest.skip("flash requires a low precision dtype")

    if dtype == torch.bfloat16 and device.type == "cpu":
        pytest.skip("bf16 requires GPU")

    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32

    attention = Attention(
        d_model=d_model,
        n_heads=8,
        n_kv_heads=n_kv_heads,
        use_flash=use_flash,
        init_device=device.type,
        **kwargs,
    )

    x1 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x2 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        y1 = attention(x1)
        y2 = attention(x2)
        y = attention(x)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16")])
@pytest.mark.parametrize(
    "use_flash", [pytest.param(True, id="flash"), pytest.param(False, id="torch-SDPA")]
)
def test_fused_attention_against_non_fused(dtype: torch.dtype, use_flash: bool):
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32
    batch_size = 2
    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device="cuda",
    )

    attention = Attention(use_flash=use_flash, **kwargs)
    fused_att = FusedAttention(**kwargs)

    # Make sure weights match.
    with torch.no_grad():
        fused_att.w_out.load_state_dict(attention.w_out.state_dict())
        fused_att.w_qkv.weight.copy_(
            torch.cat([attention.w_q.weight, attention.w_k.weight, attention.w_v.weight])
        )
        fused_att.w_qkv.bias.copy_(
            torch.cat([attention.w_q.bias, attention.w_k.bias, attention.w_v.bias])
        )

    x1 = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")
    x2 = x1.clone()

    with torch.autocast("cuda", dtype=dtype, enabled=True):
        y1 = attention(x1)
        y2 = fused_att(x2)

    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_flash_attn
def test_fused_attention_with_rope():
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32

    fused_att = FusedAttention(
        d_model=d_model, n_heads=8, rope=RoPEConfig(name=RoPEType.fused), init_device="cuda"
    )

    x1 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x2 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = fused_att(x1)
        y2 = fused_att(x2)
        y = fused_att(x)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)


@requires_gpu
@requires_flash_attn
def test_attention_with_intra_document_masking():
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32

    attention = Attention(d_model=d_model, n_heads=8, init_device="cuda", use_flash=True)
    fused_att = FusedAttention(d_model=d_model, n_heads=8, init_device="cuda")

    # Make sure weights match.
    with torch.no_grad():
        fused_att.w_out.load_state_dict(attention.w_out.state_dict())
        fused_att.w_qkv.weight.copy_(
            torch.cat([attention.w_q.weight, attention.w_k.weight, attention.w_v.weight])
        )
        fused_att.w_qkv.bias.copy_(
            torch.cat([attention.w_q.bias, attention.w_k.bias, attention.w_v.bias])
        )

    x = torch.randn(2, seq_len, d_model, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = attention(x.clone())
        y2 = attention(
            x.clone(),
            max_doc_len=seq_len,
            cu_doc_lens=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device="cuda"),
        )

        y1_fused = fused_att(x.clone())
        y2_fused = fused_att(
            x.clone(),
            max_doc_len=seq_len,
            cu_doc_lens=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device="cuda"),
        )

    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(y1_fused, y2_fused)
    torch.testing.assert_close(y1, y1_fused)
    torch.testing.assert_close(y2, y2_fused)
