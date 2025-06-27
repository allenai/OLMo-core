import contextlib
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    FusedAttention,
    RingAttentionZigZagLoadBalancer,
    SlidingWindowAttentionConfig,
    get_flex_attn_causal_block_mask,
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


@contextlib.contextmanager
def _allow_fp16_bf16_reduction_math_sdp(enabled: bool):
    math_sdpa_low_precision_allowed = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()

    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(enabled)
        yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(math_sdpa_low_precision_allowed)


# Implementation adapted from https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    full_precision: bool = True,
    use_math_backend: bool = False,
) -> torch.Tensor:
    # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
    # shape: (batch_size, n_heads, seq_len, head_dim),
    #        (batch_size, n_kv_heads, seq_len, head_dim),
    #        (batch_size, n_kv_heads, seq_len, head_dim)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    if not full_precision and not use_math_backend:
        raise ValueError("Math backend must be used when full precision is not desired")

    with contextlib.ExitStack() as stack:
        if use_math_backend:
            stack.enter_context(_allow_fp16_bf16_reduction_math_sdp(not full_precision))
            stack.enter_context(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH))

        att = (
            F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
            .transpose(1, 2)
            .contiguous()
        )

    return att


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
    "use_flash, use_flex_attn",
    [
        pytest.param(True, False, id="flash", marks=FLASH_MARKS),
        pytest.param(False, True, id="torch-flex-attn"),
        pytest.param(False, False, id="torch-SDPA"),
    ],
)
@pytest.mark.parametrize(
    "window_size",
    [pytest.param(None, id="full"), pytest.param(16, id="sliding")],
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
    use_flex_attn: bool,
    window_size: Optional[int],
    kwargs: Dict[str, Any],
):
    if use_flash and dtype == torch.float32:
        pytest.skip("flash requires a low precision dtype")

    if window_size is not None and not use_flash and not use_flex_attn:
        pytest.skip("sliding window attention requires flash or flex attention")

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
        use_flex_attn=use_flex_attn,
        init_device=device.type,
        window_size=window_size,
        **kwargs,
    )

    block_mask = None
    if use_flex_attn:
        block_mask = get_flex_attn_causal_block_mask(seq_len, device, attention.window_size)

    x1 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x2 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        y1 = attention(x1, block_mask=block_mask)
        y2 = attention(x2, block_mask=block_mask)
        y = attention(x, block_mask=block_mask)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        # pytest.param(torch.bfloat16, id="bf16"),  No bf16 because of numerical instability
        pytest.param(torch.float32, id="fp32"),
    ],
)
def test_flex_attention_against_sdpa(device: torch.device, dtype: torch.dtype):
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32
    batch_size = 2
    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device=device.type,
    )

    attention = Attention(**kwargs)
    flex_att = Attention(use_flex_attn=True, **kwargs)

    block_mask = get_flex_attn_causal_block_mask(
        seq_len,
        device,
        flex_att.window_size,
        block_size=8,
    )

    # Make sure weights match.
    with torch.no_grad():
        flex_att.w_out.load_state_dict(attention.w_out.state_dict())
        flex_att.w_q.load_state_dict(attention.w_q.state_dict())
        flex_att.w_k.load_state_dict(attention.w_k.state_dict())
        flex_att.w_v.load_state_dict(attention.w_v.state_dict())

    x1 = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)
    x2 = x1.clone()

    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        y1 = attention(x1)
        y2 = flex_att(x2, block_mask=block_mask)

    torch.testing.assert_close(y1, y2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "use_flash, use_flex_attn",
    [
        pytest.param(True, False, id="flash", marks=FLASH_MARKS),
        pytest.param(False, True, id="torch-flex-attn"),
        pytest.param(False, False, id="torch-SDPA"),
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
    use_flash: bool,
    use_flex_attn: bool,
    window_size: Optional[int],
    intra_doc_masking: bool,
):
    if use_flash and dtype == torch.float32:
        pytest.skip("flash requires a low precision dtype")

    if use_flash and device.type == "cpu":
        pytest.skip("flash requires gpu")

    if not use_flash and not use_flex_attn and window_size is not None:
        pytest.skip("sliding window attention is not supported by torch SDPA")

    if not use_flash and not use_flex_attn and intra_doc_masking:
        pytest.skip("intra-document masking is not supported by torch SDPA")

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
        # cu_doc_lens = torch.arange(0, batch_size * seq_len, max_doc_len, dtype=torch.int32, device="cuda")
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

    attention = Attention(use_flash=use_flash, use_flex_attn=use_flex_attn, **kwargs)
    block_mask = None
    if use_flex_attn:
        block_mask = get_flex_attn_causal_block_mask(
            seq_len,
            device,
            attention.window_size,
            doc_lens=doc_lens,
            block_size=8,
        )

    q = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)

    # SDPA backends yield slightly different results in lower precisions. Documentation says the Math
    # backend perform operations in full precision, so we use that backend and make our eager baseline
    # use full precision.
    with torch.no_grad():
        mask_len = batch_size * seq_len if intra_doc_masking else seq_len
        attn_mask = torch.ones(mask_len, mask_len, dtype=torch.bool, device=device).tril(diagonal=0)

        if window_size is not None:
            attn_mask = torch.logical_and(
                attn_mask,
                torch.ones(mask_len, mask_len, dtype=torch.bool, device=device).triu(
                    diagonal=-window_size
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

        y1 = scaled_dot_product_attention(
            q.view(q.shape[0] * q.shape[1] // mask_len, mask_len, *q.shape[2:]),
            k.view(k.shape[0] * k.shape[1] // mask_len, mask_len, *k.shape[2:]),
            v.view(v.shape[0] * v.shape[1] // mask_len, mask_len, *v.shape[2:]),
            is_causal=False,
            attn_mask=attn_mask,
            use_math_backend=use_flex_attn,
            full_precision=not use_flex_attn,
        )
        y2 = attention.sdpa(
            q, k, v, max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens, block_mask=block_mask
        ).view_as(y1)

    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_flash_attn
def test_flash_sdpa():
    torch.random.manual_seed(0)

    dtype = torch.bfloat16
    device = torch.device("cuda")

    d_model = 128
    seq_len = 32
    batch_size = 2
    n_heads = 8
    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device=device.type,
    )

    attention = Attention(**kwargs)
    flash_att = Attention(use_flash=True, **kwargs)

    q = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)

    with torch.no_grad(), torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        y1 = attention.sdpa(q, k, v)
        y2 = flash_att.sdpa(q, k, v)

    torch.testing.assert_close(y1, y2)


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
def test_fused_attention_with_intra_document_masking():
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
        max_doc_len = seq_len
        cu_doc_lens = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device="cuda")

        y1 = attention(x.clone())
        y2 = attention(x.clone(), max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)

        y1_fused = fused_att(x.clone())
        y2_fused = fused_att(x.clone(), max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)

    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(y1_fused, y2_fused)
    torch.testing.assert_close(y1, y1_fused)
    torch.testing.assert_close(y2, y2_fused)


@requires_gpu
@requires_flash_attn
def test_fused_attention_with_intra_document_masking_small_docs():
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
        max_doc_len = 16
        cu_doc_lens = torch.tensor([0, 4, 20, 32, 40, 48, 56, 64], dtype=torch.int32, device="cuda")

        y1 = attention(x.clone())
        y2 = attention(x.clone(), max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)

        y1_fused = fused_att(x.clone())
        y2_fused = fused_att(x.clone(), max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)

    torch.testing.assert_close(y1, y1_fused)
    torch.testing.assert_close(y2, y2_fused)
    assert not torch.allclose(y1, y2), "Intra document masking should yield different results"
    assert not torch.allclose(
        y1_fused, y2_fused
    ), "Intra document masking should yield different results"


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
