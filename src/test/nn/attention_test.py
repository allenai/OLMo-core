from typing import Any, Dict, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.tensor import Shard, init_device_mesh

from olmo_core.data.utils import attention_mask_to_cache_leftpad
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    FusedAttention,
    NormalizedAttention,
    RingAttentionLoadBalancerType,
    RingAttentionZigZagLoadBalancer,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import (
    BACKENDS,
    DEVICES,
    FLASH_2_MARKS,
    FLASH_3_MARKS,
    GPU_MARKS,
    TE_MARKS,
    requires_flash_attn_2,
    requires_gpu,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.testing.utils import requires_compute_capability
from olmo_core.utils import get_default_device, seed_all

BF16_RTOL = 1e-5
BF16_ATOL = 5e-3


@pytest.mark.parametrize(
    "window_size",
    [
        pytest.param((-1, -1), id="full"),
        pytest.param((8, 8), id="SWA"),
    ],
)
@pytest.mark.parametrize("n_kv_heads", [None, 4])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize(
    "backend_name",
    [AttentionBackendName.flash_2, AttentionBackendName.flash_3, AttentionBackendName.te],
)
@requires_gpu
def test_attention_backend(
    backend_name: AttentionBackendName,
    head_dim: int,
    n_heads: int,
    n_kv_heads: Optional[int],
    window_size: Tuple[int, int],
    dtype: torch.dtype = torch.bfloat16,
):
    try:
        backend_name.assert_supported()
        backend = backend_name.build(
            head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, window_size=window_size
        )
        default = AttentionBackendName.torch.build(
            head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, window_size=window_size
        )
    except RuntimeError as e:
        pytest.skip(str(e))

    seed_all(0)
    B, T = 2, 16

    q = torch.randn(B, T, n_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(B, T, n_kv_heads or n_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(B, T, n_kv_heads or n_heads, head_dim, device="cuda", dtype=dtype)

    att = backend((q, k, v)).view(B, T, -1)
    att_reference = default((q, k, v)).view(B, T, -1)
    torch.testing.assert_close(att, att_reference)


@pytest.mark.parametrize("attention_cls", [Attention, NormalizedAttention])
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
    "backend",
    [
        pytest.param("flash_2", id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param("flash_3", id="flash-attn-3", marks=FLASH_3_MARKS),
        pytest.param("torch", id="torch-SDPA"),
        pytest.param("te", id="te-attn", marks=TE_MARKS),
    ],
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
    attention_cls,
    dtype: torch.dtype,
    device: torch.device,
    n_kv_heads: Optional[int],
    backend: str,
    kwargs: Dict[str, Any],
):
    if backend in ("flash_2", "flash_3") and dtype == torch.float32:
        pytest.skip("flash-attn requires a low precision dtype")
    if dtype == torch.bfloat16 and device.type == "cpu":
        pytest.skip("bf16 requires GPU")
    if attention_cls is NormalizedAttention:
        if "clip_qkv" in kwargs:
            pytest.skip("clip_qkv is not supported for NormalizedAttention")
        if "use_head_qk_norm" in kwargs:
            pytest.skip("use_head_qk_norm is not supported for NormalizedAttention")
        if backend in ("flash_2", "flash_3", "te"):
            pytest.xfail(
                f"NormalizedAttention is broken with '{backend}' backend because it creates activation tensors in fp32"
            )

    seed_all(0)

    d_model = 128
    seq_len = 32

    attention = attention_cls(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        backend=backend,
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(AttentionBackendName.flash_2, id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param(AttentionBackendName.flash_3, id="flash-attn-2", marks=FLASH_3_MARKS),
        pytest.param(AttentionBackendName.torch, id="torch-SDPA"),
        pytest.param(AttentionBackendName.te, id="te-attn", marks=TE_MARKS),
    ],
)
@pytest.mark.parametrize(
    "window_size",
    [pytest.param(None, id="full"), pytest.param(16, id="sliding")],
)
@pytest.mark.parametrize(
    "intra_doc_masking",
    [pytest.param(False, id="no-doc-masking"), pytest.param(True, id="doc-masking")],
)
def test_sdpa(
    device: torch.device,
    dtype: torch.dtype,
    backend_name: AttentionBackendName,
    window_size: Optional[int],
    intra_doc_masking: bool,
):
    if (
        backend_name in (AttentionBackendName.flash_2, AttentionBackendName.flash_3)
        and dtype == torch.float32
    ):
        pytest.skip("flash-attn requires a low precision dtype")
    if (
        backend_name
        in (AttentionBackendName.flash_2, AttentionBackendName.flash_3, AttentionBackendName.te)
        and device.type == "cpu"
    ):
        pytest.skip(f"{backend_name} backend requires GPU")
    if backend_name == AttentionBackendName.torch and intra_doc_masking:
        pytest.skip("intra-document masking is not supported by torch backend")

    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32
    batch_size = 2
    n_heads = 8
    if intra_doc_masking:
        doc_lens = torch.tensor([[0, 4, 16, 12], [8, 8, 8, 8]], dtype=torch.int32, device=device)
        max_doc_len = int(torch.max(doc_lens))
        cu_doc_lens = torch.cumsum(doc_lens.flatten(), dim=0, dtype=torch.int32)
        assert int(cu_doc_lens[-1]) == batch_size * seq_len
    else:
        doc_lens = None
        max_doc_len = None
        cu_doc_lens = None

    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device=device.type,
        window_size=window_size,
    )

    attention = Attention(backend=backend_name, **kwargs)

    q = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)

    with torch.no_grad():
        mask_len = batch_size * seq_len if intra_doc_masking else seq_len
        attn_mask = torch.ones(mask_len, mask_len, dtype=torch.bool, device=device).tril(diagonal=0)
        is_causal = False

        if window_size is not None:
            attn_mask = torch.logical_and(
                attn_mask,
                torch.ones(mask_len, mask_len, dtype=torch.bool, device=device).triu(
                    diagonal=1 - window_size
                ),
            )
        if intra_doc_masking:
            assert doc_lens is not None
            attn_mask = torch.logical_and(
                attn_mask,
                torch.block_diag(
                    *[
                        torch.ones(int(doc_len), int(doc_len), dtype=torch.bool, device=device)
                        for doc_len in doc_lens.flatten()
                    ]
                ),
            )

        if window_size is None and not intra_doc_masking:
            attn_mask = None
            is_causal = True

        # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
        y1 = (
            F.scaled_dot_product_attention(
                q.view(q.shape[0] * q.shape[1] // mask_len, mask_len, *q.shape[2:]).transpose(1, 2),
                k.view(k.shape[0] * k.shape[1] // mask_len, mask_len, *k.shape[2:]).transpose(1, 2),
                v.view(v.shape[0] * v.shape[1] // mask_len, mask_len, *v.shape[2:]).transpose(1, 2),
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            .transpose(1, 2)
            .contiguous()
        )
        try:
            y2 = attention.sdpa(
                q,
                k,
                v,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            ).view_as(y1)
        except RuntimeError:
            if backend_name == AttentionBackendName.te and intra_doc_masking:
                pytest.xfail("intra-document masking is currently broken in te backend")
            raise

    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16")])
@pytest.mark.parametrize(
    "use_flash", [pytest.param(True, id="flash_2"), pytest.param(False, id="torch-SDPA")]
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
@requires_flash_attn_2
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
@requires_flash_attn_2
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
@requires_flash_attn_2
@requires_compute_capability(min_cc=9)  # flash-attn bf16 precision is worse on A100s (cc=8)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "n_kv_heads",
    [pytest.param(None, id="MHA"), pytest.param(2, id="GQA")],
)
@pytest.mark.parametrize(
    "use_rope",
    [pytest.param(True, id="rope"), pytest.param(False, id="no-rope")],
)
def test_attention_kv_caching(batch_size: int, n_kv_heads: Optional[int], use_rope: bool):
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
        rope=RoPEConfig() if use_rope else None,
        use_flash=True,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Input tensor
    x = torch.randn(batch_size, total_len, d_model, dtype=torch.bfloat16, device="cuda")

    # 1. Combined forward pass (for comparison)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_combined = attention(x)

    # 2. Prefill + multiple decode steps with KV cache
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    x_prefill = x[:, :prefill_len, :]
    attention_mask = torch.ones(batch_size, prefill_len, dtype=torch.bool, device="cuda")
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)

    # First pass with allocated KV cache - this will populate the cache
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_prefill = attention(x_prefill, cache_leftpad=cache_leftpad)

    # Multiple decode steps
    y_decode_steps = []
    for step in range(decode_steps):
        x_decode = x[:, prefill_len + step : prefill_len + step + 1, :]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y_decode = attention(x_decode, cache_leftpad=None)
        y_decode_steps.append(y_decode)
    y_decode_combined = torch.cat(y_decode_steps, dim=1)

    # 3. Compare results
    # Check that prefill output matches the corresponding part of the combined output
    torch.testing.assert_close(
        y_combined[:, :prefill_len, :],
        y_prefill,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
        msg="Prefill outputs don't match",
    )

    # Check that decode outputs match the corresponding part of the combined output
    torch.testing.assert_close(
        y_combined[:, prefill_len:, :],
        y_decode_combined,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
        msg="Decode outputs with KV-cache don't match",
    )


@requires_gpu
@requires_flash_attn_2
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
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    assert attention.kv_cache_manager is not None

    # Manually set cache contents as if we just did a prefill.
    prefill_input = torch.randn(batch_size, prefill_len, d_model, dtype=dtype, device="cuda")
    attention_mask = torch.ones(batch_size, prefill_len, dtype=torch.bool, device="cuda")
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        attention(prefill_input, cache_leftpad=cache_leftpad)

    k_at_prev_write_pos: Optional[torch.Tensor] = None
    v_at_prev_write_pos: Optional[torch.Tensor] = None

    for step in range(decode_steps):
        # Store cache state before the decode step.
        k_cache_before = attention.kv_cache_manager.k_cache.clone()
        v_cache_before = attention.kv_cache_manager.v_cache.clone()
        cache_seqlens_before = attention.kv_cache_manager.cache_seqlens.clone()

        decode_input = torch.randn(batch_size, 1, d_model, dtype=dtype, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            attention(decode_input, cache_leftpad=None)

        # Check that cache has been updated.
        assert not torch.equal(k_cache_before, attention.kv_cache_manager.k_cache)
        assert not torch.equal(v_cache_before, attention.kv_cache_manager.v_cache)
        assert attention.kv_cache_manager.cache_seqlens == cache_seqlens_before + 1

        # Check that the update happened at the right position.
        current_write_pos = cache_seqlens_before.item()
        k_cache_after = attention.kv_cache_manager.k_cache
        v_cache_after = attention.kv_cache_manager.v_cache

        # Check that the cache *before* the new token is unchanged.
        torch.testing.assert_close(
            k_cache_before[:, :current_write_pos, :, :],
            k_cache_after[:, :current_write_pos, :, :],
        )
        torch.testing.assert_close(
            v_cache_before[:, :current_write_pos, :, :],
            v_cache_after[:, :current_write_pos, :, :],
        )

        # Check that the cache *after* the new token is unchanged.
        torch.testing.assert_close(
            k_cache_before[:, current_write_pos + 1 :, :, :],
            k_cache_after[:, current_write_pos + 1 :, :, :],
        )
        torch.testing.assert_close(
            v_cache_before[:, current_write_pos + 1 :, :, :],
            v_cache_after[:, current_write_pos + 1 :, :, :],
        )

        # Check that the cache at the new token position is not all zeros.
        assert not torch.all(k_cache_after[:, current_write_pos, :, :] == 0)
        assert not torch.all(v_cache_after[:, current_write_pos, :, :] == 0)

        # New check: ensure previous write is untouched.
        if step > 0:
            assert k_at_prev_write_pos is not None and v_at_prev_write_pos is not None
            prev_write_pos = current_write_pos - 1
            torch.testing.assert_close(
                k_at_prev_write_pos,
                k_cache_after[:, prev_write_pos, :, :],
                msg=f"step {step}",
            )
            torch.testing.assert_close(
                v_at_prev_write_pos,
                v_cache_after[:, prev_write_pos, :, :],
                msg=f"step {step}",
            )

        # Store the written slice for the next iteration's check.
        k_at_prev_write_pos = k_cache_after[:, current_write_pos, :, :].clone()
        v_at_prev_write_pos = v_cache_after[:, current_write_pos, :, :].clone()


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("batch_size", [1, 8])
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
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device="cuda")
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_with_cache = attention(x, cache_leftpad=cache_leftpad)

    torch.testing.assert_close(y_standard, y_with_cache)


@requires_gpu
@requires_flash_attn_2
def test_attention_kv_cache_write_position():
    """Test KV caching with left-padded attention masks."""
    seed_all(0)

    batch_size = 2
    d_model = 128
    n_heads = 8
    max_seq_len = 100
    dtype = torch.bfloat16

    attention = Attention(
        d_model=d_model, n_heads=n_heads, use_flash=True, init_device="cuda", dtype=torch.float32
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
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)
    assert cache_leftpad.tolist() == [3, 5]

    # 1. Test prefill with KV cache
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    assert attention.kv_cache_manager is not None

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_prefill = attention(x, cache_leftpad=cache_leftpad)

    assert y_prefill.shape == (batch_size, seq_len, d_model)
    torch.testing.assert_close(attention.kv_cache_manager.cache_leftpad, cache_leftpad)

    # Check zero/non-zero structure in the cache after prefill
    k_cache = attention.kv_cache_manager.k_cache
    v_cache = attention.kv_cache_manager.v_cache
    for i in range(batch_size):
        lp = int(cache_leftpad[i].item())
        content_len = seq_len - lp
        # Padded region should be zero
        if lp > 0:
            assert torch.all(k_cache[i, :lp] == 0)
            assert torch.all(v_cache[i, :lp] == 0)
        # Content region should be non-zero
        assert not torch.all(k_cache[i, lp : lp + content_len] == 0)
        assert not torch.all(v_cache[i, lp : lp + content_len] == 0)
        # Region after content should be zero
        assert torch.all(k_cache[i, lp + content_len :] == 0)
        assert torch.all(v_cache[i, lp + content_len :] == 0)

    # 2. Test incremental decoding
    new_token = torch.randn(batch_size, 1, d_model, dtype=dtype, device="cuda")
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()
    seqlens_before = attention.kv_cache_manager.cache_seqlens.clone()

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_decode = attention(new_token, cache_leftpad=None)

    assert y_decode.shape == (batch_size, 1, d_model)
    assert torch.all(attention.kv_cache_manager.cache_seqlens == (seqlens_before + 1))

    # Verify that only the new write position per batch changed
    for i in range(batch_size):
        write_pos = int(seqlens_before)
        k_cache_new = attention.kv_cache_manager.k_cache[i]
        v_cache_new = attention.kv_cache_manager.v_cache[i]

        # Regions before the write position should be unchanged
        if write_pos > 0:
            torch.testing.assert_close(k_cache_before[i, :write_pos], k_cache_new[:write_pos])
            torch.testing.assert_close(v_cache_before[i, :write_pos], v_cache_new[:write_pos])

        # The write position must now be non-zero
        assert not torch.all(k_cache_new[write_pos] == 0)
        assert not torch.all(v_cache_new[write_pos] == 0)

        # Region after the write position should remain zeros (unchanged)
        torch.testing.assert_close(k_cache_before[i, write_pos + 1 :], k_cache_new[write_pos + 1 :])
        torch.testing.assert_close(v_cache_before[i, write_pos + 1 :], v_cache_new[write_pos + 1 :])


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("use_rope", [True, False], ids=["rope", "no-rope"])
def test_attention_leftpad_shift_equivalence(use_rope):
    """The same content, presented with different left-padding, should produce identical outputs on the valid region."""
    seed_all(0)

    d_model = 128
    n_heads = 8
    dtype = torch.bfloat16
    kv_cache_max_len = 100

    # Shared content of length L
    len_content = 7
    x_shared = torch.randn(1, len_content, d_model, dtype=dtype, device="cuda")
    x_next_shared = torch.randn(1, 1, d_model, dtype=dtype, device="cuda")

    # Two different left-padding amounts for the same content
    pad_a = 3
    pad_b = 8

    # Build masks to derive correct cache_leftpad and seq_lens
    max_len_a = pad_a + len_content
    mask_a = torch.tensor([[0] * pad_a + [1] * len_content], dtype=torch.bool, device="cuda")
    cache_leftpad_a = attention_mask_to_cache_leftpad(mask_a)

    max_len_b = pad_b + len_content
    mask_b = torch.tensor([[0] * pad_b + [1] * len_content], dtype=torch.bool, device="cuda")
    cache_leftpad_b = attention_mask_to_cache_leftpad(mask_b)

    # Build left-padded inputs so padding tokens are present and must be ignored by the kernel
    x_a = torch.zeros(1, max_len_a, d_model, dtype=dtype, device="cuda")
    x_b = torch.zeros(1, max_len_b, d_model, dtype=dtype, device="cuda")
    x_a[:, -len_content:, :] = x_shared
    x_b[:, -len_content:, :] = x_shared

    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        rope=RoPEConfig() if use_rope else None,
        use_flash=True,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Run with leftpad A
    attention.init_kv_cache_manager(1, kv_cache_max_len)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        # Prefill
        y_a = attention(x_a, cache_leftpad=cache_leftpad_a)

        # Decode one more token using the KV cache
        y_a_next = attention(x_next_shared)

    # Run with leftpad B
    attention.init_kv_cache_manager(1, kv_cache_max_len)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        # Prefill
        y_b = attention(x_b, cache_leftpad=cache_leftpad_b)

        # Decode one more token using the KV cache (same next token content)
        y_b_next = attention(x_next_shared)

    # Without RoPE, leftpad shift should not change outputs on the valid region.
    torch.testing.assert_close(
        y_a[:, -len_content:, :], y_b[:, -len_content:, :], rtol=BF16_RTOL, atol=BF16_ATOL
    )

    # Also validate the decode step equivalence (single-token outputs should match)
    torch.testing.assert_close(y_a_next, y_b_next, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize(
    "attn_config",
    [
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=True),
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=False),
        AttentionConfig(
            name=AttentionType.default, n_heads=8, bias=False, qk_norm=LayerNormConfig()
        ),
        # GQA with QK norm - regression test for k_norm size calculation
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                n_kv_heads=2,
                bias=False,
                qk_norm=LayerNormConfig(),
            ),
            id="GQA-qk-norm",
        ),
        # MQA with QK norm - regression test for k_norm size calculation
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                n_kv_heads=1,
                bias=False,
                qk_norm=LayerNormConfig(),
            ),
            id="MQA-qk-norm",
        ),
        # OLMo 3 32B-like config (scaled down) - regression test for k_norm size calculation
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=40,
                n_kv_heads=8,
                bias=False,
                qk_norm=LayerNormConfig(),
            ),
            id="OLMo3-32B-like-qk-norm",
        ),
    ],
)
def test_attention_builder_config(attn_config: AttentionConfig):
    # Use d_model that's divisible by max n_heads in our test configs (40)
    d_model = 160

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


def _run_tensor_parallel_attention(
    checkpoint_dir: str, inputs_path: str, outputs_path: str, attn_kwargs: Dict[str, Any]
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    attn = Attention(init_device=device.type, **attn_kwargs)

    # Shard sequence dim in/out like the transformer block does.
    attn.apply_tp(mesh["tp"], input_layout=Shard(1), output_layout=Shard(1), use_local_output=False)
    load_model_and_optim_state(checkpoint_dir, attn)

    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk = x.size(1) // world_size
    x_local = x[:, rank * chunk : (rank + 1) * chunk, :]
    y_local = attn(x_local)

    # Backward to exercise graph in TP mode.
    y_local.sum().backward()

    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk : (rank + 1) * chunk, :]
    torch.testing.assert_close(y_ref_local, y_local.to_local())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "attn_kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"rope": RoPEConfig()}, id="rope"),
        pytest.param({"qk_norm": LayerNormConfig()}, id="qk-layernorm"),
        pytest.param({"qk_norm": LayerNormConfig(), "rope": RoPEConfig()}, id="qk-layernorm-rope"),
        pytest.param(
            {"qk_norm": LayerNormConfig(), "use_head_qk_norm": True},
            id="headwise-qk-layernorm",
        ),
        pytest.param(
            {"qk_norm": LayerNormConfig(), "use_head_qk_norm": True, "rope": RoPEConfig()},
            id="headwise-qk-layernorm-rope",
        ),
    ],
)
def test_tensor_parallel_attention(backend: str, attn_kwargs: Dict[str, Any], tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")

    seed_all(0)
    attn_kwargs.update({"d_model": 128, "n_heads": 8, "use_flash": False})
    attn = Attention(init_device=device.type, **attn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, attn_kwargs["d_model"], device=device)
    y = attn(x)

    outputs_path = tmp_path / "attn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "attn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, attn)

    run_distributed_test(
        _run_tensor_parallel_attention,
        backend=backend,
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, attn_kwargs),
    )


def _run_context_parallel_attention(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    attn_kwargs: Dict[str, Any],
    load_balancer_type,
    head_stride: int,
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    attn = Attention(init_device=device.type, **attn_kwargs)
    attn.apply_cp(mesh["cp"], load_balancer_type, head_stride=head_stride)
    load_model_and_optim_state(checkpoint_dir, attn)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        y_local = attn(x_local)

    # Backward to exercise graph in CP mode.
    y_local.sum().backward()

    # Load the reference output and split it across ranks on the sequence dimension.
    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    # Compare the local output with the reference output.
    torch.testing.assert_close(y_ref_local, y_local)


@requires_multi_gpu
@requires_flash_attn_2
@pytest.mark.parametrize(
    "load_balancer_type",
    [pytest.param(RingAttentionLoadBalancerType.zig_zag, id="zig_zag")],
)
@pytest.mark.parametrize("head_stride", [pytest.param(1), pytest.param(8)])
@pytest.mark.skip("known precision issues with ring-flash-attn")
def test_context_parallel_attention(load_balancer_type, head_stride: int, tmp_path):
    seed_all(0)
    device = torch.device("cuda")

    # CP requires flash-attn and low precision dtypes.
    attn_kwargs: Dict[str, Any] = {"d_model": 128, "n_heads": 8, "use_flash": True}
    attn = Attention(init_device=device.type, **attn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, attn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = attn(x)

    outputs_path = tmp_path / "attn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "attn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, attn)

    run_distributed_test(
        _run_context_parallel_attention,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            inputs_path,
            outputs_path,
            attn_kwargs,
            load_balancer_type,
            head_stride,
        ),
    )
