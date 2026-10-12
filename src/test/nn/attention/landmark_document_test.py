"""Tests for DocumentLandmarkAttention (OLMo-core grouped-softmax landmark + corpus-reasoning
chunked-document masking) and the ported chunked_mask pattern family."""

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    DocumentLandmarkAttention,
)
from olmo_core.nn.attention.chunked_mask import (
    PAD_CHUNK_ID,
    AttentionPattern,
    build_chunk_ids_from_tokens,
    build_chunked_allowed_mask,
)
from olmo_core.nn.attention.landmark import landmark_grouped_softmax
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType


def _doc_landmark_attention(*, mem_freq: int = 3, cross_doc_mode: str = "chunked", **kw):
    config = AttentionConfig(
        name=AttentionType.document_landmark,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        mem_freq=mem_freq,
        cross_doc_mode=cross_doc_mode,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        **kw,
    )
    attn = config.build(64, layer_idx=0, n_layers=1)
    assert isinstance(attn, DocumentLandmarkAttention)
    return attn


# ---------------------------------------------------------------------------
# build_chunked_allowed_mask (ported corpus-reasoning logic)
# ---------------------------------------------------------------------------


def test_chunked_pattern_isolates_context_bridges_free():
    # chunk0 = 0..3, chunk1 = 4..7, FREE (query/answer) = 8..11.
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    allowed = build_chunked_allowed_mask(AttentionPattern(name="chunked"), chunk_ids)[0]
    # context chunk1 query (5) cannot see chunk0 (0..3)
    assert not allowed[5, 0:4].any()
    # context chunk1 query (5) can see its own chunk causally (4..5)
    assert allowed[5, 4:6].all() and not allowed[5, 6:].any()
    # FREE query (10) sees everything causal up to itself
    assert allowed[10, : 11].all()
    # causal: nobody sees the future
    assert not allowed[3, 4:].any()


def test_chunked_pattern_pad_never_attended():
    chunk_ids = torch.tensor([[0, 0, 0, 0, -1, -1, PAD_CHUNK_ID, PAD_CHUNK_ID]])
    allowed = build_chunked_allowed_mask(AttentionPattern(name="chunked"), chunk_ids)[0]
    # Pad keys are attended by nobody, and pad queries attend nobody -- except the diagonal NaN guard
    # (a fully-masked row would NaN the softmax; those positions are dropped by the loss mask anyway).
    off_diag = ~torch.eye(8, dtype=torch.bool)
    assert not (allowed[:, 6:] & off_diag[:, 6:]).any()  # pad keys: only the self-diagonal
    assert not (allowed[6:, :] & off_diag[6:, :]).any()  # pad queries: only themselves
    assert allowed[6, 6] and allowed[7, 7]  # diagonal guard present


def test_standard_pattern_is_plain_causal():
    chunk_ids = torch.tensor([[0, 0, 1, 1, -1, -1]])
    allowed = build_chunked_allowed_mask(AttentionPattern(name="standard"), chunk_ids)[0]
    S = 6
    expected = torch.tril(torch.ones(S, S, dtype=torch.bool))
    assert torch.equal(allowed, expected)


def test_doc_window_pattern():
    # 3 chunks of 2 tokens; window k=1 -> chunk i sees chunks i-1, i.
    chunk_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    allowed = build_chunked_allowed_mask(AttentionPattern(name="doc_window", doc_window_k=1), chunk_ids)[0]
    # chunk2 query (5) sees chunk1 (2..3) and chunk2 (4..5) but not chunk0 (0..1)
    assert allowed[5, 2:6].all() and not allowed[5, 0:2].any()


def test_last_token_anchor_pattern():
    chunk_ids = torch.tensor([[0, 0, 1, 1, -1, -1]])
    # anchors at each chunk's last token (1 and 3)
    is_anchor = torch.tensor([[False, True, False, True, False, False]])
    allowed = build_chunked_allowed_mask(
        AttentionPattern(name="last_token_anchor"), chunk_ids, is_anchor=is_anchor
    )[0]
    # chunk1 query (3) can see chunk0's anchor (token 1) but not chunk0's non-anchor (token 0)
    assert allowed[3, 1] and not allowed[3, 0]


# ---------------------------------------------------------------------------
# build_chunk_ids_from_tokens (runtime role reconstruction)
# ---------------------------------------------------------------------------


def test_build_chunk_ids_from_tokens():
    DS, DE, EOS = 100, 101, 102
    # row: [DS d d DE] free free [DS d DE] eos pad pad
    ids = torch.tensor([[100, 5, 6, 101, 7, 8, 100, 9, 101, 102, 0, 0]])
    roles = build_chunk_ids_from_tokens(ids, DS, DE, EOS, mode="chunked")
    expected = torch.tensor([[0, 0, 0, 0, -1, -1, 1, 1, 1, -1, -2, -2]], dtype=torch.int32)
    assert torch.equal(roles, expected)


def test_build_chunk_ids_sink_mode():
    DS, DE, EOS = 100, 101, 102
    ids = torch.tensor([[7, 8, 100, 9, 101, 102, 0]])  # prefix 7,8 -> SINK in modified_swa
    roles = build_chunk_ids_from_tokens(ids, DS, DE, EOS, mode="modified_swa")
    assert roles.tolist() == [[-3, -3, 0, 0, 0, -1, -2]]


# ---------------------------------------------------------------------------
# DocumentLandmarkAttention
# ---------------------------------------------------------------------------


def test_document_landmark_config_builds():
    attn = _doc_landmark_attention(mem_freq=3)
    assert attn.block_size == 4
    assert attn.cross_doc_mode == "chunked"


def test_document_landmark_kernel_opt_in_accepted():
    # use_kernel is now supported (routes to the fast fused kernel with the per-token chunk mask at
    # train time; falls back to eager on CPU / without chunk_ids / during top-k eval).
    attn = DocumentLandmarkAttention(mem_freq=3, n_heads=8, head_dim=8, d_model=64, use_kernel=True)
    assert attn._use_chunk_kernel is True


def test_document_landmark_unknown_mode_rejected():
    with pytest.raises(OLMoConfigurationError):
        DocumentLandmarkAttention(mem_freq=3, n_heads=8, head_dim=8, d_model=64, cross_doc_mode="nope")


def test_document_landmark_masks_drive_grouped_softmax():
    # chunk0 = 0..3, chunk1 = 4..7, FREE query/answer = 8..11 (block_size=4).
    attn = _doc_landmark_attention(mem_freq=3)
    attn.eval()
    T = 12
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    attn._chunk_ids = chunk_ids
    attn_mask, is_mem, lsm = attn._landmark_masks(T, torch.device("cpu"), torch.float32, batch_size=1)
    attn._chunk_ids = None
    logits = torch.randn(1, 1, T, T) + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    probs = landmark_grouped_softmax(logits, dim=-1, is_mem=is_mem.expand(1, 1, T, T), last_section_mask=lsm)[0, 0]

    assert torch.allclose(probs.sum(-1), torch.ones(T), atol=1e-5)
    # context chunk1 query isolated from chunk0
    assert torch.allclose(probs[5, 0:4], torch.zeros(4), atol=1e-6)
    # context chunk0 query sees only within its chunk
    assert torch.allclose(probs[2, 3:], torch.zeros(T - 3), atol=1e-6)
    # FREE query/answer bridges back to earlier chunks
    assert probs[10, 0:8].sum() > 1e-4


def test_document_landmark_no_chunk_ids_matches_normal_landmark():
    # Without chunk_ids the variant must reduce to normal landmark (same q,k,v -> same eager output).
    from olmo_core.nn.attention import LandmarkAttention

    doc = _doc_landmark_attention(mem_freq=3)
    base_cfg = AttentionConfig(
        name=AttentionType.landmark,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        mem_freq=3,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
    )
    base = base_cfg.build(64, layer_idx=0, n_layers=1)
    assert isinstance(base, LandmarkAttention)
    base.load_state_dict(doc.state_dict())
    doc.eval()
    base.eval()
    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        assert torch.allclose(doc(x), base(x), atol=1e-5)


def test_document_landmark_forward_with_chunk_ids_runs():
    attn = _doc_landmark_attention(mem_freq=3)
    attn.eval()
    B, T, d_model = 1, 12, 64
    x = torch.randn(B, T, d_model)
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    with torch.no_grad():
        out = attn(x, chunk_ids=chunk_ids)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_document_landmark_eager_training_backward():
    attn = _doc_landmark_attention(mem_freq=3)
    x = torch.randn(1, 12, 64, requires_grad=True)
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    attn(x, chunk_ids=chunk_ids).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("top_k", [None, 1, 2])
def test_document_landmark_cached_decode_matches_eager(top_k):
    """The cached generation path (chunked-masked prefill + plain landmark decode) must produce the
    same attention outputs as re-feeding the whole growing sequence through the eager forward -- the
    property that makes the KV-cached eval (O(T^2 + gen*T)) a drop-in for the eager loop (O(gen*T^2)).
    Validated with and without hard top-k landmark retrieval."""
    torch.manual_seed(0)
    attn = _doc_landmark_attention(mem_freq=3)  # block_size = 4
    attn.eval()
    Lb = attn.block_size
    d_model = 64
    P, T = 12, 20  # prompt 0..11 (two context chunks + a FREE block); 12..19 are generated FREE tokens
    x = torch.randn(1, T, d_model)
    full_chunks = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1] + [-1] * (T - P)])

    if top_k is not None:
        attn.set_eval_top_k(top_k)

    # Eager reference: forward over the growing sequence (right-padded to a block multiple, as the eager
    # eval loop does), reading the next-token row at the last real position.
    ref = []
    with torch.no_grad():
        for p in range(P - 1, T):
            L = p + 1
            pad_to = ((L + Lb - 1) // Lb) * Lb
            xp = torch.cat([x[:, :L], torch.zeros(1, pad_to - L, d_model)], dim=1)
            cp = torch.cat([full_chunks[:, :L], torch.full((1, pad_to - L), -2)], dim=1)
            ref.append(attn(xp, chunk_ids=cp)[:, L - 1])

    # Cached path: single-shot prefill of the prompt, then incremental single-token decode.
    attn.init_kv_cache_manager(batch_size=1, max_seq_len=T)
    got = []
    with torch.no_grad():
        got.append(attn(x[:, :P], chunk_ids=full_chunks[:, :P])[:, -1])  # prefill -> row P-1
        for p in range(P, T):
            got.append(attn(x[:, p : p + 1], chunk_ids=full_chunks[:, p : p + 1])[:, -1])
    attn.kv_cache_manager = None

    assert torch.allclose(torch.stack(ref), torch.stack(got), atol=1e-5)
