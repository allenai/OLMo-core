from typing import Any, Dict, Optional

import pytest
import torch

from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    FusedAttention,
    RingAttentionZigZagLoadBalancer,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import (
    DEVICES,
    FLASH_MARKS,
    GPU_MARKS,
    requires_flash_attn,
    requires_gpu,
)


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
    [pytest.param(None, id="MHA"), pytest.param(1, id="MQA"), pytest.param(2, id="GQA")],
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
        n_heads=4,
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


@pytest.mark.parametrize(
    "attn_config",
    [
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=True),
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=False),
        AttentionConfig(
            name=AttentionType.default, n_heads=8, bias=False, qk_norm=LayerNormConfig()
        ),
    ],
)
def test_attention_builder_config(attn_config: AttentionConfig):
    d_model = 64

    attn = attn_config.build(d_model, layer_idx=0, n_layers=1)

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in attn.parameters())
    assert attn_config.num_params(d_model) == n_params


def _get_lb(rank: int, world_size: int) -> RingAttentionZigZagLoadBalancer:
    return RingAttentionZigZagLoadBalancer(cp_rank=rank, cp_world_size=world_size)


def test_zig_zag_load_balancer_padding():
    x, padding_added = _get_lb(0, 4).pad(torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0), 1, -1)
    assert x.tolist() == [[0, 1, 2, 3, 4, 5, -1, -1]]
    assert padding_added == 2


def test_zig_zag_load_balancer_shard():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
    assert _get_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [
        [
            0,
            7,
        ]
    ]
    assert _get_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [
        [
            3,
            4,
        ]
    ]


def test_zig_zag_load_balancer_shard_with_padding():
    x = torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0)
    assert _get_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[0].tolist() == [
        [
            0,
            -1,
        ]
    ]
    assert _get_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[0].tolist() == [
        [
            3,
            4,
        ]
    ]


def test_zig_zag_load_balancer_shard_by_document():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 8, 12])

    assert _get_lb(0, 2).batch_shard_by_document(inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens)[
        0
    ][0].tolist() == [
        [
            0,
            1,
            6,
            7,
            8,
            11,
        ]
    ]

    assert _get_lb(1, 2).batch_shard_by_document(inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens)[
        0
    ][0].tolist() == [
        [
            2,
            3,
            4,
            5,
            9,
            10,
        ]
    ]


def test_zig_zag_load_balancer_shard_by_document_with_padding():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 7, 10])

    res, opts = _get_lb(0, 2).batch_shard_by_document(
        inputs=[x],
        seq_dims=[1],
        cu_doc_lens=cu_doc_lens,
        pad_values=[-1],
    )
    new_doc_lens = opts["cu_doc_lens"]
    assert new_doc_lens.tolist() == [0, 4, 6]
    assert res[0].tolist() == [
        [
            0,
            1,
            6,
            -1,
            7,
            -1,
        ]
    ]
