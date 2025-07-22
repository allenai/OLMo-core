from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    FusedAttention,
    RingAttentionZigZagLoadBalancer,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.flash_attn_api import (
    dispatch_flash_attn,
    dispatch_flash_attn_with_kvcache,
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
        pytest.param({"qk_norm": LayerNormConfig(), "use_head_qk_norm": True}, id="head-qk-norm"),
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


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16")])
@pytest.mark.parametrize(
    "n_kv_heads",
    [pytest.param(None, id="MHA"), pytest.param(1, id="MQA"), pytest.param(2, id="GQA")],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no-opts"),
        pytest.param({"clip_qkv": 8.0}, id="QKV-clip"),
        pytest.param({"rope": RoPEConfig()}, id="rope"),
        pytest.param({"qk_norm": LayerNormConfig()}, id="qk-norm"),
        pytest.param({"qk_norm": LayerNormConfig(), "use_head_qk_norm": True}, id="head-qk-norm"),
    ],
)
def test_attention_kv_caching(
    dtype: torch.dtype,
    n_kv_heads: Optional[int],
    kwargs: Dict[str, Any],
):
    torch.random.manual_seed(0)

    d_model = 128
    n_heads = 8
    max_seq_len = 128
    batch_size = 1
    prefill_len = 63
    decode_len = 1
    total_len = prefill_len + decode_len
    assert total_len <= max_seq_len

    # Initialize attention module
    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        use_flash=True,
        init_device="cuda",
        **kwargs,
    )

    # Input tensor
    x = torch.randn(batch_size, total_len, d_model, dtype=dtype, device="cuda")

    # 1. Combined forward pass (for comparison)
    with patch(
        "olmo_core.nn.attention.dispatch_flash_attn", wraps=dispatch_flash_attn
    ) as mock_dispatch:
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            y_combined = attention(x)
        mock_dispatch.assert_called_once()
        (q_combined, k_combined, v_combined), _ = mock_dispatch.call_args

    # 2. Prefill + decode
    x_prefill = x[:, :prefill_len, :]
    x_decode = x[:, prefill_len:total_len, :]

    # Initialize KV cache
    _n_kv_heads = n_kv_heads or n_heads
    head_dim = d_model // n_heads
    k_cache = torch.zeros(
        batch_size, max_seq_len, _n_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v_cache = torch.zeros(
        batch_size, max_seq_len, _n_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    assert torch.all(k_cache == 0)
    assert torch.all(v_cache == 0)

    # Prefill step
    cache_seqlens_prefill = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    with patch(
        "olmo_core.nn.attention.dispatch_flash_attn", wraps=dispatch_flash_attn
    ) as mock_dispatch:
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            y_prefill = attention(
                x_prefill,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens_prefill,
                prefill_kv_cache=True,
            )
        mock_dispatch.assert_called_once()
        (q_prefill, k_prefill, v_prefill), _ = mock_dispatch.call_args

    # Check that the inputs and outputs to the flash attention kernel are the same
    torch.testing.assert_close(q_combined[:, :prefill_len, :, :], q_prefill)
    torch.testing.assert_close(k_combined[:, :prefill_len, :, :], k_prefill)
    torch.testing.assert_close(v_combined[:, :prefill_len, :, :], v_prefill)

    # Check cache state after prefill
    k_prefill_region = k_cache[:, :prefill_len, :, :]
    v_prefill_region = v_cache[:, :prefill_len, :, :]
    k_zero_fraction = (k_prefill_region == 0).float().mean().item()
    v_zero_fraction = (v_prefill_region == 0).float().mean().item()
    assert k_zero_fraction < 0.01, f"Prefill k cache has {k_zero_fraction:.2%} zeros"
    assert v_zero_fraction < 0.01, f"Prefill v cache has {v_zero_fraction:.2%} zeros"
    assert torch.all(k_cache[:, prefill_len:, :, :] == 0)
    assert torch.all(v_cache[:, prefill_len:, :, :] == 0)

    # Decode step
    cache_seqlens_decode = torch.full((batch_size,), prefill_len, dtype=torch.int32, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_decode = attention(
            x_decode,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens_decode,
            prefill_kv_cache=False,
        )

    # Check cache state after decode
    k_total_region = k_cache[:, :total_len, :, :]
    v_total_region = v_cache[:, :total_len, :, :]
    k_zero_fraction = (k_total_region == 0).float().mean().item()
    v_zero_fraction = (v_total_region == 0).float().mean().item()
    assert k_zero_fraction < 0.01, f"Decode k cache has {k_zero_fraction:.2%} zeros"
    assert v_zero_fraction < 0.01, f"Decode v cache has {v_zero_fraction:.2%} zeros"
    assert torch.all(k_cache[:, total_len:, :, :] == 0)
    assert torch.all(v_cache[:, total_len:, :, :] == 0)

    # 3. Compare results
    torch.testing.assert_close(y_combined[:, :prefill_len, :], y_prefill)

    # Decode comparison needs looser tolerances due to different computation paths (and matmul shapes etc).
    torch.testing.assert_close(
        y_combined[:, prefill_len:total_len, :],
        y_decode,
        atol=1e-3,
        rtol=0.015,
    )


@requires_gpu
@requires_flash_attn
def test_attention_prefill_forward_pass():
    torch.random.manual_seed(0)

    d_model = 64
    n_heads = 4
    max_seq_len = 64
    batch_size = 2
    seq_len = 32
    dtype = torch.bfloat16

    attention = Attention(d_model=d_model, n_heads=n_heads, use_flash=True, init_device="cuda")

    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")
    cache_shape = (batch_size, max_seq_len, n_heads, d_model // n_heads)
    k_cache = torch.zeros(cache_shape, dtype=dtype, device="cuda")
    v_cache = torch.zeros(cache_shape, dtype=dtype, device="cuda")
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_standard = attention(x)
        y_prefill = attention(
            x, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, prefill_kv_cache=True
        )

    torch.testing.assert_close(y_standard, y_prefill)


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


@pytest.mark.parametrize(
    "force_first, force_last, layer_idx, expected_window_size, expected_should_use_swa",
    [
        # Test with forcing full attention on neither first nor last layer.
        (False, False, 0, 1024, True),  # Pattern start
        (False, False, 1, 2048, True),  # Pattern middle
        (False, False, 2, -1, False),  # Pattern end
        (False, False, 11, -1, False),  # Last layer, pattern end
        (True, False, 1, 1024, True),  # Effective layer=0
        (True, False, 11, 2048, True),  # Effective layer=10
        # Test with forcing full attention only on the last layer.
        (False, True, 0, 1024, True),  # First layer, not forced
        (False, True, 11, -1, False),  # Forced last
        # Test with forcing full attention on both first and last layers.
        (True, True, 0, -1, False),  # Forced first
        (True, True, 1, 1024, True),  # Effective layer=0
        (True, True, 11, -1, False),  # Forced last
    ],
)
def test_sliding_window_attention_config_window_size(
    force_first: bool,
    force_last: bool,
    layer_idx: int,
    expected_window_size: int,
    expected_should_use_swa: bool,
):
    n_layers = 12
    pattern = [1024, 2048, -1]

    config = SlidingWindowAttentionConfig(
        pattern=pattern,
        force_full_attention_on_first_layer=force_first,
        force_full_attention_on_last_layer=force_last,
    )

    assert config._get_window_size(layer_idx, n_layers) == expected_window_size
    assert config.should_use_swa(layer_idx, n_layers) == expected_should_use_swa


def test_sliding_window_attention_config_get_window_size_error():
    n_layers = 12
    pattern = [1024, 2048, -1]
    config = SlidingWindowAttentionConfig(
        pattern=pattern,
        force_full_attention_on_first_layer=True,
        force_full_attention_on_last_layer=True,
    )

    assert config.get_window_size(1, n_layers) == 1024  # This layer uses SWA
    with pytest.raises(ValueError):
        config.get_window_size(0, n_layers)  # This layer uses full attention


def test_sliding_window_attention_config_invalid_pattern_error():
    with pytest.raises(OLMoConfigurationError):
        bad_config = SlidingWindowAttentionConfig(
            pattern=[0], force_full_attention_on_first_layer=False
        )
        bad_config._get_window_size(0, n_layers=12)
