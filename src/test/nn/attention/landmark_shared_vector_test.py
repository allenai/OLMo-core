"""Tests for :class:`SharedVectorLandmarkAttention`.

The fused-kernel (``use_kernel=True``) ``head_dim`` output is exercised by the existing fast-landmark
kernel tests; these CPU tests cover the eager reference path and the (kernel-shared) ``vec_dim`` tail
computation:

* the eager ``head_dim`` output equals ordinary (non-compressive) landmark attention;
* the tail from :meth:`SharedVectorLandmarkAttention._shared_vector_tail` (used by *both* paths)
  equals a dense brute-force tail derived from the full landmark grouped-softmax probabilities.
"""

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    LandmarkAttention,
    SharedVectorLandmarkAttention,
)
from olmo_core.nn.attention.landmark import build_landmark_masks, landmark_grouped_softmax


def _build(
    *,
    d_model: int = 64,
    n_heads: int = 4,
    n_kv_heads: int = 4,
    head_dim: int = 16,
    mem_freq: int = 15,
    vec_dim: int = 32,
    dtype: torch.dtype = torch.float32,
) -> SharedVectorLandmarkAttention:
    m = SharedVectorLandmarkAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        mem_freq=mem_freq,
        vec_dim=vec_dim,
        use_kernel=False,
        bias=False,
        dtype=dtype,
    )
    # Randomize the (zero-initialized) base and w_out_vec so tests exercise a non-trivial tail path.
    with torch.no_grad():
        m.base.normal_(std=0.1)
        m.w_out_vec.weight.normal_(std=0.1)
    return m


def _qkv(m: SharedVectorLandmarkAttention, B: int, T: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(0)
    shape = (B, m.n_heads, T, m.head_dim)
    q = torch.randn(shape, generator=g, dtype=dtype)
    k = torch.randn(shape, generator=g, dtype=dtype)
    v = torch.randn(shape, generator=g, dtype=dtype)
    return q, k, v


def _dense_tail_reference(
    m: SharedVectorLandmarkAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Brute-force tail from the dense landmark grouped-softmax probabilities ``P``.

    ``tail_i = sum_j P_ij * (base if j in query i's block else e_{block(j)})``.
    """
    B, H, T, _ = q.shape
    Lb = m.block_size
    attn_mask, is_mem, last_section_mask = build_landmark_masks(T, Lb, q.device, q.dtype)
    attn = torch.matmul(q, k.transpose(-1, -2)) * m.softmax_scale + attn_mask
    attn = torch.maximum(attn, torch.tensor(torch.finfo(attn.dtype).min, dtype=attn.dtype))
    P = landmark_grouped_softmax(
        attn,
        dim=-1,
        is_mem=is_mem.expand(B, H, T, T),
        last_section_mask=last_section_mask.expand(B, 1, T, T),
    )

    mem_pos = torch.arange(Lb - 1, T, Lb)
    v_lm = v[:, :, mem_pos, :]  # (B, H, nb, D)
    e = torch.einsum("bhnd,hde->bhne", v_lm, m.weight_landmark)  # (B, H, nb, vec)
    block_of = torch.arange(T) // Lb
    E_full = e[:, :, block_of, :]  # (B, H, T, vec)
    same_block = (block_of.view(T, 1) == block_of.view(1, T)).to(P.dtype)  # (T, T)

    cross = torch.matmul(P * (1.0 - same_block), E_full)  # (B, H, T, vec)
    same_mass = (P * same_block).sum(dim=-1)  # (B, H, T)
    return cross + same_mass.unsqueeze(-1) * m.base.view(1, H, 1, m.vec_dim)


def test_eager_main_matches_non_compressive_landmark():
    m = _build()
    T = m.block_size * 3
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    main = m._main_dense(q, k, v)

    ref = LandmarkAttention(
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_kv_heads=m.n_kv_heads,
        head_dim=m.head_dim,
        mem_freq=m.mem_freq,
        bias=False,
    )
    ref_out = ref._eager_forward(q, k, v)
    torch.testing.assert_close(main, ref_out)


def test_tail_matches_dense_bruteforce():
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)  # non-trivial landmark map
    T = m.block_size * 4
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    tail = m._shared_vector_tail(q, k, v)
    ref = _dense_tail_reference(m, q, k, v)
    torch.testing.assert_close(tail, ref, atol=1e-4, rtol=1e-4)


def test_first_block_tail_is_base():
    """Queries in block 0 have no past block, so their entire tail is the learned base vector."""
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
    T = m.block_size * 3
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    tail = m._shared_vector_tail(q, k, v)  # (B, H, T, vec)
    Lb = m.block_size
    first_block = tail[:, :, :Lb, :]  # queries in block 0
    expected = m.base.view(1, m.n_heads, 1, m.vec_dim).expand_as(first_block)
    torch.testing.assert_close(first_block, expected, atol=1e-5, rtol=1e-4)


def test_shapes_and_split_projection():
    m = _build(vec_dim=8)
    B, T = 2, m.block_size * 3
    q, k, v = _qkv(m, B=B, T=T, dtype=torch.float32)

    # head_dim output branch is unchanged; the vec branch has its own projection.
    assert m._main(q, k, v).shape == (B, m.n_heads, T, m.head_dim)
    assert m._shared_vector_tail(q, k, v).shape == (B, m.n_heads, T, m.vec_dim)
    assert m.w_out.in_features == m.n_heads * m.head_dim  # base shape -> loads from base checkpoint
    assert m.w_out_vec.in_features == m.n_heads * m.vec_dim

    x = torch.randn(B, T, m.d_model)
    out = m(x)
    assert out.shape == (B, T, m.d_model)


def test_zero_init_tail_reproduces_plain_landmark():
    """With the default (zero) w_out_vec/base init, the vec tail contributes nothing, so the module
    equals a plain fast-landmark model sharing the same q/k/v/out weights."""
    m = SharedVectorLandmarkAttention(
        d_model=64, n_heads=4, n_kv_heads=4, head_dim=16, mem_freq=15, use_kernel=False, bias=False
    )  # default init: w_out_vec = 0, base = 0
    x = torch.randn(2, m.block_size * 3, m.d_model)
    out = m(x)

    ref = LandmarkAttention(
        d_model=64, n_heads=4, n_kv_heads=4, head_dim=16, mem_freq=15, bias=False
    )
    # Share the loaded/base weights (q/k/v/out) between the two modules.
    ref_keys = set(ref.state_dict().keys())
    ref.load_state_dict({k: v for k, v in m.state_dict().items() if k in ref_keys}, strict=False)
    torch.testing.assert_close(out, ref(x), atol=1e-5, rtol=1e-4)


def test_backward_populates_new_param_grads():
    m = _build()
    B, T = 1, m.block_size * 3
    x = torch.randn(B, T, m.d_model)
    m(x).sum().backward()
    assert m.weight_landmark.grad is not None and torch.isfinite(m.weight_landmark.grad).all()
    assert m.base.grad is not None and torch.isfinite(m.base.grad).all()


def test_gate_rejected():
    from olmo_core.nn.attention import GateConfig, GateGranularity

    with pytest.raises(OLMoConfigurationError):
        SharedVectorLandmarkAttention(
            d_model=64,
            n_heads=4,
            head_dim=16,
            mem_freq=15,
            use_kernel=False,
            gate=GateConfig(granularity=GateGranularity.elementwise),
        )


def test_config_build_and_validation():
    # vec_dim without shared_vector_landmark -> error
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(
            name=AttentionType.fast_landmark, n_heads=4, head_dim=16, mem_freq=15, vec_dim=8
        ).build(d_model=64, layer_idx=0, n_layers=1)

    # valid build produces the class with the configured vec_dim and widened w_out
    m = AttentionConfig(
        name=AttentionType.shared_vector_landmark,
        n_heads=4,
        n_kv_heads=4,
        head_dim=16,
        mem_freq=15,
        vec_dim=8,
        bias=False,
    ).build(d_model=64, layer_idx=0, n_layers=1)
    assert isinstance(m, SharedVectorLandmarkAttention)
    assert m.vec_dim == 8
    assert m.w_out.in_features == 4 * 16  # unchanged base shape
    assert m.w_out_vec.in_features == 4 * 8
