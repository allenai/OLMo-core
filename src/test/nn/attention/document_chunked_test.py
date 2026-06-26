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
