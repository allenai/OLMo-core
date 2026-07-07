"""Tests for DocumentChunkedAttention (dense full attention restricted by the chunked-document
mask -- the non-landmark analogue of DocumentLandmarkAttention)."""

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    DocumentChunkedAttention,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType


def _attention(name: AttentionType, *, cross_doc_mode=None, **kw):
    config = AttentionConfig(
        name=name,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        cross_doc_mode=cross_doc_mode,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        **kw,
    )
    return config.build(64, layer_idx=0, n_layers=1)


def _doc_chunked_attention(*, cross_doc_mode: str = "chunked", **kw):
    attn = _attention(AttentionType.document_chunked, cross_doc_mode=cross_doc_mode, **kw)
    assert isinstance(attn, DocumentChunkedAttention)
    return attn


# A 3-chunk-ish layout: chunk0 = 0..3, chunk1 = 4..7, FREE (query/answer) = 8..11.
CHUNK_IDS = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])


def test_document_chunked_config_builds():
    attn = _doc_chunked_attention()
    assert isinstance(attn, DocumentChunkedAttention)
    assert attn.cross_doc_mode == "chunked"


def test_document_chunked_unknown_mode_rejected():
    with pytest.raises(OLMoConfigurationError):
        DocumentChunkedAttention(n_heads=8, head_dim=8, d_model=64, cross_doc_mode="nope")


def test_document_chunked_mem_freq_rejected():
    # document_chunked is not a landmark variant; mem_freq must be rejected by the config.
    with pytest.raises(OLMoConfigurationError):
        _attention(AttentionType.document_chunked, mem_freq=3)


def test_document_chunked_no_chunk_ids_matches_default_causal():
    # Without chunk_ids the variant must reduce to ordinary causal attention.
    doc = _doc_chunked_attention()
    base = _attention(AttentionType.default)
    assert isinstance(base, Attention) and not isinstance(base, DocumentChunkedAttention)
    base.load_state_dict(doc.state_dict())
    doc.eval()
    base.eval()
    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        assert torch.allclose(doc(x), base(x), atol=1e-5)


def test_document_chunked_isolates_context_bridges_free():
    # Perturbing a chunk-0 token must NOT change a chunk-1 query's output (chunks isolated), but MUST
    # change a FREE query's output (FREE bridges across chunks).
    attn = _doc_chunked_attention()
    attn.eval()
    x = torch.randn(1, 12, 64)
    x2 = x.clone()
    x2[0, 1] += 1.0  # perturb a chunk-0 position
    with torch.no_grad():
        out = attn(x, chunk_ids=CHUNK_IDS)
        out2 = attn(x2, chunk_ids=CHUNK_IDS)
    # chunk-1 query (position 5) is isolated from chunk-0 -> unchanged.
    assert torch.allclose(out[0, 5], out2[0, 5], atol=1e-6)
    # FREE query (position 10) attends every earlier chunk -> changed.
    assert not torch.allclose(out[0, 10], out2[0, 10], atol=1e-4)


def test_document_chunked_forward_with_chunk_ids_runs():
    attn = _doc_chunked_attention()
    attn.eval()
    B, T, d_model = 1, 12, 64
    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out = attn(x, chunk_ids=CHUNK_IDS)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_document_chunked_eager_training_backward():
    attn = _doc_chunked_attention()
    x = torch.randn(1, 12, 64, requires_grad=True)
    attn(x, chunk_ids=CHUNK_IDS).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# FlexAttention regression: the flex mask_mod must be bit-identical to the dense allowed-mask, and
# (GPU) the flex forward must match the materialized-mask forward.
# ---------------------------------------------------------------------------

from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    AttentionPattern,
    build_chunked_allowed_mask,
    build_chunked_mask_mod,
)


@pytest.mark.parametrize(
    "pattern",
    [
        AttentionPattern(name="chunked"),
        AttentionPattern(name="standard"),
        AttentionPattern(name="doc_window", doc_window_k=1),
    ],
)
def test_flex_mask_mod_matches_dense_allowed_mask(pattern):
    # The flex mask_mod (point predicate) must materialize to EXACTLY the dense allowed-mask, so the
    # block-sparse FlexAttention kernel computes the same masked softmax as the fallback path.
    chunk_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, -1, -1, -2, -2, -1]])
    T = chunk_ids.shape[1]
    mask_mod = build_chunked_mask_mod(pattern, chunk_ids)
    q_idx = torch.arange(T).view(T, 1)
    kv_idx = torch.arange(T).view(1, T)
    got = mask_mod(0, 0, q_idx, kv_idx)  # (T, T) bool, broadcast over the index grid
    expected = build_chunked_allowed_mask(pattern, chunk_ids)[0]
    assert torch.equal(got, expected)


@pytest.mark.gpu
def test_flex_matches_materialized_forward_on_gpu():
    # On CUDA the default path uses FlexAttention; forcing the dense materialized mask must give the
    # same output (numerical parity), and both must differ from full causal attention.
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    import olmo_core.nn.attention.document_chunked as dcm

    if not dcm._HAS_FLEX:
        pytest.skip("FlexAttention unavailable")
    torch.manual_seed(0)
    attn = _doc_chunked_attention().to("cuda").to(torch.bfloat16)
    attn.eval()
    dcm._FLEX_MIN_SEQ_LEN = (
        0  # force the flex path at this tiny T (it is normally gated to long ctx)
    )
    B, T, d = 2, 16, 64
    x = torch.randn(B, T, d, device="cuda", dtype=torch.bfloat16)
    chunk_ids = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]], device="cuda"
    ).expand(B, T)
    with torch.no_grad():
        attn._force_eager_mask = False
        flex_out = attn(x, chunk_ids=chunk_ids)
        attn._force_eager_mask = True
        dense_out = attn(x, chunk_ids=chunk_ids)
        causal_out = attn(x)  # no chunk_ids -> plain causal
    assert torch.allclose(flex_out, dense_out, atol=2e-2, rtol=0)
    assert not torch.allclose(flex_out, causal_out, atol=1e-2)


def test_random_doc_pattern_dense_flex_parity_and_semantics():
    """random_doc: each context doc attends itself + a seeded-random subset of EARLIER docs; free
    tokens stay global; dense mask == flex mask_mod; doc_keep_prob=0 collapses to isolated 'chunked'.
    """
    import torch

    from olmo_core.nn.attention.chunked_mask import (
        FREE_CHUNK_ID,
        AttentionPattern,
        build_chunked_allowed_mask,
        build_chunked_mask_mod,
    )

    S_pre, ndocs, dlen, S_post = 3, 20, 5, 2
    ids = [FREE_CHUNK_ID] * S_pre
    for d in range(ndocs):
        ids += [d] * dlen
    ids += [FREE_CHUNK_ID] * S_post
    cids = torch.tensor(ids).unsqueeze(0)
    S = cids.shape[1]

    pat = AttentionPattern(name="random_doc", doc_keep_prob=0.1, random_seed=42)
    M = build_chunked_allowed_mask(pat, cids)[0]

    # dense == flex mask_mod, elementwise
    mm = build_chunked_mask_mod(pat, cids)
    for q in range(S):
        for k in range(S):
            v = bool(mm(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(k)))
            assert v == bool(M[q, k]), (q, k)

    # no future-doc attention; own doc always attended; kept-earlier fraction is in [0, 1)
    for i in range(ndocs):
        ti = S_pre + i * dlen
        assert bool(M[ti, ti])  # self
        for j in range(i + 1, ndocs):  # future docs never attended
            assert not bool(M[ti, S_pre + j * dlen])

    # seed changes the graph; deterministic for a fixed seed
    M2 = build_chunked_allowed_mask(
        AttentionPattern(name="random_doc", doc_keep_prob=0.1, random_seed=7), cids
    )[0]
    assert not bool((M == M2).all())
    M2b = build_chunked_allowed_mask(
        AttentionPattern(name="random_doc", doc_keep_prob=0.1, random_seed=7), cids
    )[0]
    assert bool((M2 == M2b).all())

    # doc_keep_prob=0 == isolated 'chunked'
    M0 = build_chunked_allowed_mask(AttentionPattern(name="random_doc", doc_keep_prob=0.0), cids)[0]
    Mc = build_chunked_allowed_mask(AttentionPattern(name="chunked"), cids)[0]
    assert bool((M0 == Mc).all())
