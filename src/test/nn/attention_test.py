from typing import Any, Dict, Optional

import pytest
import torch

from olmo_core.data.utils import attention_mask_to_cache_leftpad
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    FusedAttention,
    RingAttentionZigZagLoadBalancer,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import DEVICES, FLASH_MARKS, GPU_MARKS, requires_flash_attn, requires_gpu
from olmo_core.utils import seed_all


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

    seed_all(0)

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
    seed_all(0)

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
    seed_all(0)

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
    seed_all(0)

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
@pytest.mark.parametrize("batch_size", [1, 2])
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
def test_attention_kv_caching(batch_size: int, n_kv_heads: Optional[int], kwargs: Dict[str, Any]):
    seed_all(0)

    d_model = 512
    n_heads = 8
    max_seq_len = 512
    prefill_len = 508
    decode_steps = 1
    total_len = prefill_len + decode_steps
    assert total_len <= max_seq_len

    # Initialize attention module
    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        use_flash=True,
        init_device="cuda",
        dtype=torch.float32,
        **kwargs,
    )

    # Input tensor
    x = torch.randn(batch_size, total_len, d_model, dtype=torch.bfloat16, device="cuda")

    # 1. Combined forward pass (for comparison)
    seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_combined = attention(x)

    # 2. Prefill + multiple decode steps with KV cache
    x_prefill = x[:, :prefill_len, :]

    # Create attention mask with left padding (simulate variable length sequences)
    # For simplicity, we'll use no padding here
    attention_mask = torch.ones(batch_size, prefill_len, dtype=torch.bool, device="cuda")
    cache_leftpad, seq_lens = attention_mask_to_cache_leftpad(attention_mask)

    # Allocate KV cache
    attention.reset_kv_cache(
        use_cache=True, batch_size=batch_size, max_seq_len=max_seq_len, dtype=torch.bfloat16
    )

    # First pass with allocated KV cache - this will populate the cache
    seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_prefill = attention(
            x_prefill, use_cache=True, cache_leftpad=cache_leftpad, seq_lens=seq_lens
        )  # (B, P, D)

    assert y_prefill.shape == (batch_size, prefill_len, d_model), "Prefill output shape mismatch"

    # Multiple decode steps
    y_decode_steps = []
    for step in range(decode_steps):
        x_decode = x[:, prefill_len + step : prefill_len + step + 1, :]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # For decoding single tokens, seq_lens is 1
            decode_seq_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")
            y_decode = attention(
                x_decode, use_cache=True, cache_leftpad=cache_leftpad, seq_lens=decode_seq_lens
            )  # (B, 1, D)
        y_decode_steps.append(y_decode)

    # Concatenate decode outputs
    y_decode_combined = torch.cat(y_decode_steps, dim=1)  # (B, D_steps, D)
    assert y_decode_combined.shape == (batch_size, decode_steps, d_model), (
        "Decode output shape mismatch"
    )

    # Reference without KV cache
    attention.free_kv_cache()
    seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_reference_prefill = attention(x_prefill)

    # 3. Compare results
    torch.testing.assert_close(
        y_reference_prefill,
        y_prefill,
        rtol=1e-5 if batch_size > 1 else None,
        atol=5e-3 if batch_size > 1 else None,
        msg=lambda s: f"Prefill reference outputs don't match: {s}",
    )
    torch.testing.assert_close(
        y_combined[:, :prefill_len, :],
        y_prefill,
        rtol=1e-5 if batch_size > 1 else None,
        atol=5e-3 if batch_size > 1 else None,
        msg=lambda s: f"Prefill outputs don't match: {s}",
    )

    # Decode comparison needs looser tolerances due to different computation paths (and matmul shapes etc).
    torch.testing.assert_close(
        y_combined[:, prefill_len:, :],
        y_decode_combined,
        rtol=1e-5 if batch_size > 1 else None,
        atol=5e-3 if batch_size > 1 else None,
        msg=lambda s: f"Outputs that leverage the KV-cache don't match: {s}",
    )


@requires_gpu
@requires_flash_attn
def test_attention_kv_cache_update():
    seed_all(0)

    d_model = 64
    n_heads = 8
    n_kv_heads = 2
    batch_size = 2
    max_seq_len = 64
    prefill_len = 30
    decode_steps = 5
    dtype = torch.bfloat16

    # Initialize attention module
    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        use_flash=True,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Initialize cache
    attention.reset_kv_cache(
        use_cache=True, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype
    )
    assert attention.k_cache is not None
    assert attention.v_cache is not None
    assert attention.cache_seqlens is not None

    # Manually set cache contents as if we just did a prefill.
    prefill_input = torch.randn(batch_size, prefill_len, d_model, dtype=dtype, device="cuda")
    # Create attention mask (no padding for simplicity)
    attention_mask = torch.ones(batch_size, prefill_len, dtype=torch.bool, device="cuda")
    cache_leftpad, seq_lens = attention_mask_to_cache_leftpad(attention_mask)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        attention(prefill_input, use_cache=True, cache_leftpad=cache_leftpad, seq_lens=seq_lens)

    k_at_prev_write_pos: Optional[torch.Tensor] = None
    v_at_prev_write_pos: Optional[torch.Tensor] = None

    # Loop over decode steps.
    for step in range(decode_steps):
        # Store cache state before the decode step.
        k_cache_before = attention.k_cache.clone()
        v_cache_before = attention.v_cache.clone()
        cache_seqlens_before = attention.cache_seqlens.clone()

        # Single decode step.
        decode_input = torch.randn(batch_size, 1, d_model, dtype=dtype, device="cuda")
        decode_seq_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            attention(
                decode_input, use_cache=True, cache_leftpad=cache_leftpad, seq_lens=decode_seq_lens
            )

        # Check that cache has been updated.
        assert not torch.equal(k_cache_before, attention.k_cache)
        assert not torch.equal(v_cache_before, attention.v_cache)
        assert torch.all(attention.cache_seqlens == cache_seqlens_before + 1)

        # Check that the update happened at the right position.
        k_at_current_write_pos_list = []
        v_at_current_write_pos_list = []
        for i in range(batch_size):
            current_write_pos = cache_seqlens_before[i]
            # Check that the cache *before* the new token is unchanged.
            torch.testing.assert_close(
                k_cache_before[i, :current_write_pos, :, :],
                attention.k_cache[i, :current_write_pos, :, :],
            )
            torch.testing.assert_close(
                v_cache_before[i, :current_write_pos, :, :],
                attention.v_cache[i, :current_write_pos, :, :],
            )
            # Check that the cache *after* the new token is unchanged.
            torch.testing.assert_close(
                k_cache_before[i, current_write_pos + 1 :, :, :],
                attention.k_cache[i, current_write_pos + 1 :, :, :],
            )
            torch.testing.assert_close(
                v_cache_before[i, current_write_pos + 1 :, :, :],
                attention.v_cache[i, current_write_pos + 1 :, :, :],
            )
            # Check that the cache at the new token position is not all zeros.
            assert not torch.all(attention.k_cache[i, current_write_pos, :, :] == 0)
            assert not torch.all(attention.v_cache[i, current_write_pos, :, :] == 0)

            # New check: ensure previous write is untouched.
            if step > 0:
                assert k_at_prev_write_pos is not None and v_at_prev_write_pos is not None
                prev_write_pos = current_write_pos - 1
                torch.testing.assert_close(
                    k_at_prev_write_pos[i],
                    attention.k_cache[i, prev_write_pos, :, :],
                    msg=f"step {step}, batch {i}",
                )
                torch.testing.assert_close(
                    v_at_prev_write_pos[i],
                    attention.v_cache[i, prev_write_pos, :, :],
                    msg=f"step {step}, batch {i}",
                )

            k_at_current_write_pos_list.append(attention.k_cache[i, current_write_pos, :, :])
            v_at_current_write_pos_list.append(attention.v_cache[i, current_write_pos, :, :])

        # Store the written slice for the next iteration's check.
        k_at_prev_write_pos = torch.stack(k_at_current_write_pos_list)
        v_at_prev_write_pos = torch.stack(v_at_current_write_pos_list)


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_attention_prefill_forward_pass(batch_size: int):
    seed_all(0)

    d_model = 64
    n_heads = 4
    max_seq_len = 128
    seq_len = 124
    dtype = torch.bfloat16

    attention = Attention(d_model=d_model, n_heads=n_heads, use_flash=True, init_device="cuda")

    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")

    # Standard forward pass without KV cache
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_standard = attention(x)

    # Forward pass with KV cache allocated
    attention.reset_kv_cache(
        use_cache=True, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype
    )

    # Create attention mask (no padding)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device="cuda")
    cache_leftpad, seq_lens = attention_mask_to_cache_leftpad(attention_mask)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_with_cache = attention(x, use_cache=True, cache_leftpad=cache_leftpad, seq_lens=seq_lens)

    torch.testing.assert_close(y_standard, y_with_cache)


@requires_gpu
@requires_flash_attn
def test_attention_kv_caching_with_leftpad():
    """Test KV caching with left-padded attention masks."""
    seed_all(0)

    batch_size = 2
    d_model = 128
    n_heads = 8
    max_seq_len = 100
    dtype = torch.bfloat16

    # Initialize attention module
    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        use_flash=True,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Create inputs with different sequence lengths (simulated with left padding)
    # Sequence 1: 3 padding tokens + 7 real tokens
    # Sequence 2: 5 padding tokens + 5 real tokens
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")

    # Create attention mask with left padding
    attention_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 3 padding tokens
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 5 padding tokens
        ],
        dtype=torch.bool,
        device="cuda",
    )

    # Convert to cache_leftpad
    cache_leftpad, seq_lens = attention_mask_to_cache_leftpad(attention_mask)
    assert cache_leftpad.tolist() == [3, 5]
    assert seq_lens.tolist() == [7, 5]

    # Test with KV cache
    attention.reset_kv_cache(
        use_cache=True, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_with_cache = attention(x, use_cache=True, cache_leftpad=cache_leftpad, seq_lens=seq_lens)

    assert y_with_cache.shape == (batch_size, seq_len, d_model)

    # Test incremental decoding
    new_token = torch.randn(batch_size, 1, d_model, dtype=dtype, device="cuda")
    decode_seq_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_decode = attention(
            new_token, use_cache=True, cache_leftpad=cache_leftpad, seq_lens=decode_seq_lens
        )

    assert y_decode.shape == (batch_size, 1, d_model)

    # Clean up
    attention.free_kv_cache()


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
