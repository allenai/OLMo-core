"""Tests for :class:`DocumentCompressiveLandmarkAttention` -- the *compressive* analogue of
:class:`DocumentLandmarkAttention` (compressive landmark grouped softmax + chunked-document masking)
-- and for the generalization of the layer-dependent ``"hierarchical_dilated"`` cross-document
visibility policy to **all** document-chunked families (dense / landmark / compressive).
"""

import math

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    DocumentCompressiveLandmarkAttention,
    DocumentLandmarkAttention,
)
from olmo_core.nn.attention.chunked_mask import (
    AttentionPattern,
    build_chunked_allowed_mask,
)
from olmo_core.nn.attention.landmark import (
    compressive_landmark_grouped_softmax,
    landmark_grouped_softmax,
)
from olmo_core.nn.attention.landmark_compressive import (
    fused_compressive_landmark_attention,
)
from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import requires_gpu


def _doc_compressive_attention(*, mem_freq: int = 3, cross_doc_mode: str = "chunked", **kw):
    config = AttentionConfig(
        name=AttentionType.document_compressive_landmark,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        mem_freq=mem_freq,
        nonselected_landmark_mass=0.1,
        cross_doc_mode=cross_doc_mode,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        **kw,
    )
    attn = config.build(64, layer_idx=0, n_layers=1)
    assert isinstance(attn, DocumentCompressiveLandmarkAttention)
    return attn


def _eager_compressive_reference(q, k, v, block_size):
    """Independent dense compressive-landmark oracle (full-context, causal); same as the one in
    ``landmark_compressive_kernel_test`` that the fused kernel is validated against."""
    B, H, T, d = q.shape
    device = q.device
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(-1, -2)).float() * scale
    neg_inf = torch.finfo(scores.dtype).min
    pos = torch.arange(T, device=device)
    sec = pos // block_size
    is_mem = (pos % block_size) == (block_size - 1)
    causal = pos[None, :] <= pos[:, None]
    same_block = sec[None, :] == sec[:, None]
    past_block = sec[None, :] < sec[:, None]
    kmem = is_mem[None, :]
    local_content = same_block & (~kmem) & causal
    past_landmark = past_block & kmem
    gate_set = (local_content | past_landmark).view(1, 1, T, T)
    gate_w = torch.softmax(scores.masked_fill(~gate_set, neg_inf), dim=-1)
    within = torch.softmax(scores.reshape(B, H, T, T // block_size, block_size), dim=-1)
    within = within.reshape(B, H, T, T)
    block_gate = gate_w[..., is_mem]
    block_gate_full = block_gate.repeat_interleave(block_size, dim=-1)
    past_mask = past_block.view(1, 1, T, T)
    local_mask = local_content.view(1, 1, T, T)
    final = torch.where(past_mask, block_gate_full * within, torch.zeros_like(within))
    final = torch.where(local_mask, gate_w, final)
    return final.to(v.dtype) @ v


# ---------------------------------------------------------------------------
# compressive_landmark_grouped_softmax (the eager primitive)
# ---------------------------------------------------------------------------


def test_compressive_grouped_softmax_matches_oracle():
    """The eager compressive grouped softmax must reproduce the independent dense compressive oracle
    (no chunk mask, single document) -- the same oracle the fused kernel is validated against."""
    torch.manual_seed(0)
    B, H, T, d = 2, 3, 16, 8
    block_size = 4
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)
    scale = 1.0 / math.sqrt(d)

    attn = _doc_compressive_attention(mem_freq=block_size - 1)
    attn_mask, is_mem, lsm = attn._landmark_masks(T, q.device, torch.float32, batch_size=B)
    logits = (q @ k.transpose(-1, -2)) * scale + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    probs = compressive_landmark_grouped_softmax(
        logits, dim=-1, is_mem=is_mem.expand(B, H, T, T), last_section_mask=lsm.expand(B, 1, T, T)
    )
    out_mine = probs @ v
    out_ref = _eager_compressive_reference(q, k, v, block_size)

    assert torch.allclose(probs.sum(-1), torch.ones(B, H, T), atol=1e-5)
    assert torch.allclose(out_mine, out_ref, atol=1e-5)


def test_compressive_grouped_softmax_differs_from_normal_landmark():
    """The defining difference: the landmark token now contributes value (within-block softmax over
    the full block), so the probabilities must differ from the non-compressive grouped softmax."""
    torch.manual_seed(0)
    B, H, T, d = 1, 2, 16, 8
    block_size = 4
    q, k = torch.randn(B, H, T, d), torch.randn(B, H, T, d)
    attn = _doc_compressive_attention(mem_freq=block_size - 1)
    attn_mask, is_mem, lsm = attn._landmark_masks(T, q.device, torch.float32, batch_size=B)
    logits = (q @ k.transpose(-1, -2)) * (d**-0.5) + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    kw = dict(dim=-1, is_mem=is_mem.expand(B, H, T, T), last_section_mask=lsm.expand(B, 1, T, T))
    p_comp = compressive_landmark_grouped_softmax(logits, **kw)
    p_norm = landmark_grouped_softmax(logits, **kw)
    assert not torch.allclose(p_comp, p_norm, atol=1e-3)


# ---------------------------------------------------------------------------
# DocumentCompressiveLandmarkAttention -- config / masking / training
# ---------------------------------------------------------------------------


def test_document_compressive_config_builds():
    attn = _doc_compressive_attention(mem_freq=3)
    assert attn.block_size == 4
    assert attn.cross_doc_mode == "chunked"
    assert attn.nonselected_landmark_mass == 0.1


def test_document_compressive_kernel_rejected():
    # The fused compressive kernel has no chunk-mask path; the variant is eager-only.
    with pytest.raises(OLMoConfigurationError):
        DocumentCompressiveLandmarkAttention(
            mem_freq=3, n_heads=8, head_dim=8, d_model=64, use_kernel=True
        )


def test_document_compressive_unknown_mode_rejected():
    with pytest.raises(OLMoConfigurationError):
        DocumentCompressiveLandmarkAttention(
            mem_freq=3, n_heads=8, head_dim=8, d_model=64, cross_doc_mode="nope"
        )


def test_document_compressive_bad_alpha_rejected():
    with pytest.raises(OLMoConfigurationError):
        DocumentCompressiveLandmarkAttention(
            mem_freq=3, n_heads=8, head_dim=8, d_model=64, nonselected_landmark_mass=1.0
        )


def test_document_compressive_isolation_and_free_bridge():
    """ISOLATION: a context query in chunk 1 cannot see chunk 0 (and chunk 0 sees only its own
    chunk); the FREE query/answer bridges back to earlier chunks. Mirrors the landmark doc-chunked
    semantics, computed through the *compressive* grouped softmax."""
    attn = _doc_compressive_attention(mem_freq=3)
    attn.eval()
    T = 12
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    attn._chunk_ids = chunk_ids
    attn_mask, is_mem, lsm = attn._landmark_masks(
        T, torch.device("cpu"), torch.float32, batch_size=1
    )
    attn._chunk_ids = None
    torch.manual_seed(1)
    logits = torch.randn(1, 1, T, T) + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    probs = compressive_landmark_grouped_softmax(
        logits, dim=-1, is_mem=is_mem.expand(1, 1, T, T), last_section_mask=lsm
    )[0, 0]

    assert torch.allclose(probs.sum(-1), torch.ones(T), atol=1e-5)
    assert torch.allclose(probs[5, 0:4], torch.zeros(4), atol=1e-6)  # chunk1 isolated from chunk0
    assert torch.allclose(probs[2, 3:], torch.zeros(T - 3), atol=1e-6)  # chunk0 query stays local
    assert probs[10, 0:8].sum() > 1e-4  # FREE query bridges back


def test_document_compressive_bridge_uses_landmark_value():
    """The compressive bridge: a FREE query gives an earlier chunk's *landmark* token nonzero weight
    (the compressed block summary), whereas the non-compressive grouped softmax gives it exactly 0.
    """
    attn = _doc_compressive_attention(mem_freq=3)
    T = 12
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    attn._chunk_ids = chunk_ids
    attn_mask, is_mem, lsm = attn._landmark_masks(
        T, torch.device("cpu"), torch.float32, batch_size=1
    )
    attn._chunk_ids = None
    torch.manual_seed(2)
    logits = torch.randn(1, 1, T, T) + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    kw = dict(dim=-1, is_mem=is_mem.expand(1, 1, T, T), last_section_mask=lsm)
    p_comp = compressive_landmark_grouped_softmax(logits, **kw)[0, 0]
    p_norm = landmark_grouped_softmax(logits, **kw)[0, 0]
    # Landmark (block-end) columns 3 and 7 are past blocks for the FREE query 10.
    assert p_comp[10, 3] > 1e-5 and p_comp[10, 7] > 1e-5
    assert p_norm[10, 3] < 1e-7 and p_norm[10, 7] < 1e-7


def test_document_compressive_forward_backward_finite():
    attn = _doc_compressive_attention(mem_freq=3)
    x = torch.randn(1, 12, 64, requires_grad=True)
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    out = attn(x, chunk_ids=chunk_ids)
    assert out.shape == (1, 12, 64) and torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_document_compressive_no_chunk_ids_matches_oracle():
    """Without chunk_ids the variant must reduce to plain compressive landmark attention (the
    full-context oracle), i.e. the chunked mask is a no-op."""
    attn = _doc_compressive_attention(mem_freq=3)
    attn.eval()
    B, T, d_model = 2, 12, 64
    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out = attn(x)  # no chunk_ids
    # Reconstruct q,k,v the module used and compare to the oracle.
    q, k, v = attn._prepare_qkv(x)
    from olmo_core.nn.attention.landmark import repeat_kv

    n_rep = q.shape[2] // k.shape[2]
    qh = q.transpose(1, 2)
    kh = repeat_kv(k.transpose(1, 2), n_rep)
    vh = repeat_kv(v.transpose(1, 2), n_rep)
    ref = _eager_compressive_reference(qh, kh, vh, attn.block_size)
    ref = ref.transpose(1, 2).contiguous().view(B, T, -1)
    ref = attn.w_out(attn._apply_gate(ref, x))
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# hierarchical_dilated generalized to landmark + compressive families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, cls, extra",
    [
        (AttentionType.document_landmark, DocumentLandmarkAttention, {}),
        (
            AttentionType.document_compressive_landmark,
            DocumentCompressiveLandmarkAttention,
            {"nonselected_landmark_mass": 0.1},
        ),
    ],
)
def test_hierarchical_dilated_landmark_families(name, cls, extra):
    """``"hierarchical_dilated"`` is a cross-document *visibility* policy orthogonal to the attention
    mechanism, so it must work for the landmark and compressive families too: build + forward/backward
    finite, ``layer_idx`` threaded, and the same layer-dependent visibility as the dense mask."""
    T = 16
    chunk_ids = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4 + [-1] * 4])  # 3 ctx chunks + FREE answer

    def build(layer_idx):
        cfg = AttentionConfig(
            name=name,
            n_heads=8,
            n_kv_heads=2,
            head_dim=8,
            bias=False,
            mem_freq=3,
            cross_doc_mode="hierarchical_dilated",
            dilation_n=2,
            dilation_m=2,
            qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
            use_head_qk_norm=True,
            **extra,
        )
        return cfg.build(64, layer_idx=layer_idx, n_layers=4)

    a0, a2 = build(0), build(2)
    assert isinstance(a0, cls) and a0.layer_idx == 0 and a2.layer_idx == 2
    assert a0.cross_doc_mode == "hierarchical_dilated"

    for a in (a0, a2):
        x = torch.randn(1, T, 64, requires_grad=True)
        out = a(x, chunk_ids=chunk_ids)
        assert out.shape == (1, T, 64) and torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    # Layer-dependent visibility must match the dense hierarchical allowed-mask exactly (same plumbing).
    pat = AttentionPattern(name="hierarchical_dilated", dilation_n=2, dilation_m=2)
    allow0 = build_chunked_allowed_mask(pat, chunk_ids, layer_idx=0)[0]
    allow2 = build_chunked_allowed_mask(pat, chunk_ids, layer_idx=2)[0]
    # layer 0: stride 1, n=2 -> chunk-2 query sees chunk 1 but not chunk 0.
    assert allow0[11, 4:8].any() and not allow0[11, 0:4].any()
    # deeper layer widens the stride -> visibility changes.
    assert not torch.equal(allow0, allow2)


def test_hierarchical_dilated_dilation_requires_doc_chunked_family():
    # dilation knobs are rejected on non-document-chunked attention.
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(
            name=AttentionType.fast_landmark,
            n_heads=8,
            head_dim=8,
            mem_freq=3,
            dilation_n=2,
        ).build(64, layer_idx=0, n_layers=1)


# ---------------------------------------------------------------------------
# GPU: eager compressive grouped softmax vs the fused compressive kernel (parity)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("head_dim, mem_freq", [(64, 15), (256, 63)])
def test_compressive_eager_softmax_matches_fused_kernel(head_dim: int, mem_freq: int):
    """The eager compressive grouped softmax (used by DocumentCompressiveLandmarkAttention) must match
    the fused compressive kernel (no chunk mask, single document) -- forward parity, fp32 ~1e-4."""
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads, T = 2, 4, block_size * 4
    q = torch.randn(B, n_heads, T, head_dim, device="cuda")
    k = torch.randn(B, n_heads, T, head_dim, device="cuda")
    v = torch.randn(B, n_heads, T, head_dim, device="cuda")
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    scale = head_dim**-0.5

    out_fused = fused_compressive_landmark_attention(
        q, k, v, is_mem, sm_scale=scale, block_size=block_size
    )

    # Eager path: build the landmark masks + run the eager compressive grouped softmax.
    finfo_min = torch.finfo(torch.float32).min
    pos = torch.arange(T, device="cuda")
    causal = torch.where(pos[None, :] <= pos[:, None], 0.0, finfo_min).view(1, 1, T, T)
    ism = is_mem.view(1, 1, 1, T)
    mem_ids = torch.where(causal < -1, -1, torch.cumsum(ism, -1) - ism.int())
    lsm = torch.amax(mem_ids, -1, keepdim=True) == mem_ids
    attn_mask = causal.clone()
    attn_mask.masked_fill_(lsm & ism, finfo_min)
    lsm = lsm & (attn_mask > -1)
    ism = ism & (attn_mask > -1)
    logits = (q @ k.transpose(-1, -2)) * scale + attn_mask
    logits = torch.maximum(logits, torch.tensor(finfo_min, device="cuda"))
    probs = compressive_landmark_grouped_softmax(
        logits,
        dim=-1,
        is_mem=ism.expand(B, n_heads, T, T),
        last_section_mask=lsm.expand(B, 1, T, T),
    )
    out_eager = probs @ v
    torch.testing.assert_close(out_fused, out_eager, rtol=1e-4, atol=1e-4)
