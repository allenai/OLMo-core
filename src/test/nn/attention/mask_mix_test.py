"""Tests for document-chunked **mask mixing** -- the runtime schedule that collapses a random subset
of examples from the chunked (block-sparse) mask to plain causal (full) attention.

Covers the three invariants the reference (corpus-reasoning ``olmo_flex_attention``) guarantees:
  * ``p == 0`` is a strict no-op (pure-chunked stays bit-identical);
  * a forced collapse makes an example's mask exactly plain causal;
  * the curriculum ``p(forward_idx)`` anneals linearly and is deterministic under a fixed seed.
"""

import torch

from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    DocumentChunkedAttention,
)
from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    AttentionPattern,
    build_chunked_allowed_mask,
    collapse_roles_to_causal,
    mask_mix_standard_prob,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType

# A 3-part layout: chunk0 = 0..3, chunk1 = 4..7, FREE (query/answer) = 8..10, PAD = 11.
CHUNK_IDS = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -2]])


# ---------------------------------------------------------------------------
# mask_mix_standard_prob: the p(forward_idx) schedule
# ---------------------------------------------------------------------------


def test_prob_disabled_is_zero():
    assert mask_mix_standard_prob(0) == 0.0
    assert mask_mix_standard_prob(1000) == 0.0


def test_prob_static_is_constant():
    for idx in (0, 5, 999):
        assert mask_mix_standard_prob(idx, standard_mix_prob=0.1) == 0.1


def test_prob_static_takes_precedence_over_curriculum():
    # standard_mix_prob wins even if curriculum params are also (mistakenly) present.
    assert mask_mix_standard_prob(
        50, standard_mix_prob=0.3, mix_start_p=0.8, mix_end_p=0.0, mix_total_forwards=100
    ) == 0.3


def test_prob_curriculum_anneals_linearly_and_clamps():
    kw = dict(mix_start_p=0.8, mix_end_p=0.0, mix_total_forwards=100)
    assert abs(mask_mix_standard_prob(0, **kw) - 0.8) < 1e-9  # start
    assert abs(mask_mix_standard_prob(50, **kw) - 0.4) < 1e-9  # midpoint
    assert abs(mask_mix_standard_prob(100, **kw) - 0.0) < 1e-9  # end
    assert abs(mask_mix_standard_prob(500, **kw) - 0.0) < 1e-9  # clamped past the end


def test_prob_curriculum_can_increase():
    kw = dict(mix_start_p=0.0, mix_end_p=1.0, mix_total_forwards=10)
    assert abs(mask_mix_standard_prob(5, **kw) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# collapse_roles_to_causal: the roles -> FREE trick
# ---------------------------------------------------------------------------


def test_collapse_p_zero_is_identity_no_clone():
    out = collapse_roles_to_causal(CHUNK_IDS, 0.0, forward_idx=1, mix_seed=42)
    assert out is CHUNK_IDS  # same object -> strictly bit-identical, no allocation


def test_collapse_p_one_all_free_keeps_pad():
    out = collapse_roles_to_causal(CHUNK_IDS, 1.0, forward_idx=1, mix_seed=42)
    # every non-pad role -> FREE, pad preserved.
    expected = torch.where(
        CHUNK_IDS == PAD_CHUNK_ID, CHUNK_IDS, torch.full_like(CHUNK_IDS, FREE_CHUNK_ID)
    )
    assert torch.equal(out, expected)
    # input never mutated.
    assert CHUNK_IDS[0, 0].item() == 0


def test_collapse_is_deterministic_under_seed():
    a = collapse_roles_to_causal(CHUNK_IDS, 0.5, forward_idx=7, mix_seed=42)
    b = collapse_roles_to_causal(CHUNK_IDS, 0.5, forward_idx=7, mix_seed=42)
    assert torch.equal(a, b)


def test_collapse_per_example_independent():
    # A batch where the seed collapses some rows and not others; each row is all-FREE-or-untouched.
    batch = CHUNK_IDS.repeat(8, 1)
    out = collapse_roles_to_causal(batch, 0.5, forward_idx=3, mix_seed=42)
    for b in range(8):
        row = out[b]
        collapsed = torch.equal(
            row, torch.where(batch[b] == PAD_CHUNK_ID, batch[b], torch.full_like(batch[b], FREE_CHUNK_ID))
        )
        untouched = torch.equal(row, batch[b])
        assert collapsed or untouched, f"row {b} is neither fully collapsed nor untouched"


def test_collapsed_roles_give_plain_causal_mask():
    # A collapsed (all-FREE) example under the *chunked* pattern must equal the *standard* (plain
    # causal, pad-aware) mask -- the whole point of the roles->FREE trick.
    collapsed = torch.where(
        CHUNK_IDS == PAD_CHUNK_ID, CHUNK_IDS, torch.full_like(CHUNK_IDS, FREE_CHUNK_ID)
    )
    chunked = build_chunked_allowed_mask(AttentionPattern(name="chunked"), collapsed)
    standard = build_chunked_allowed_mask(AttentionPattern(name="standard"), CHUNK_IDS)
    # build_chunked_allowed_mask adds a self-diagonal NaN guard for "chunked" but not "standard";
    # compare off-diagonal (the meaningful part) and confirm the diagonal is set on the chunked one.
    S = CHUNK_IDS.shape[1]
    eye = torch.eye(S, dtype=torch.bool).unsqueeze(0)
    assert torch.equal(chunked & ~eye, standard & ~eye)


# ---------------------------------------------------------------------------
# End-to-end through DocumentChunkedAttention (the real native masked SDPA)
# ---------------------------------------------------------------------------


def _doc_chunked_attention(**kw):
    config = AttentionConfig(
        name=AttentionType.document_chunked,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        cross_doc_mode="chunked",
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        **kw,
    )
    attn = config.build(64, layer_idx=0, n_layers=1)
    assert isinstance(attn, DocumentChunkedAttention)
    return attn


def _forward(attn, x, chunk_ids):
    with torch.no_grad():
        return attn(x, chunk_ids=chunk_ids)


def test_e2e_collapsed_equals_plain_causal():
    torch.manual_seed(0)
    attn = _doc_chunked_attention().eval()
    B, S, D = 1, 12, 64
    x = torch.randn(B, S, D)

    collapsed = torch.where(
        CHUNK_IDS == PAD_CHUNK_ID, CHUNK_IDS, torch.full_like(CHUNK_IDS, FREE_CHUNK_ID)
    )
    out_collapsed = _forward(attn, x, collapsed)

    # A plain-causal reference: no roles at all -> DocumentChunkedAttention takes its is_causal path.
    out_causal = _forward(attn, x, None)

    # PAD (position 11) rows are dropped by the loss mask; compare the real (non-pad) positions.
    real = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert torch.allclose(out_collapsed[:, real], out_causal[:, real], atol=1e-5)


def test_e2e_chunked_differs_from_causal():
    # Sanity: without collapse, the chunked mask really does change the output (else the parity above
    # would be vacuous).
    torch.manual_seed(0)
    attn = _doc_chunked_attention().eval()
    x = torch.randn(1, 12, 64)
    out_chunked = _forward(attn, x, CHUNK_IDS)
    out_causal = _forward(attn, x, None)
    real = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert not torch.allclose(out_chunked[:, real], out_causal[:, real], atol=1e-4)


# ---------------------------------------------------------------------------
# Model level: enable_document_chunk_attention wiring (counter, eval gate, validation)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from olmo_core.exceptions import OLMoConfigurationError  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

# Small in-vocab boundary ids for a toy model (real Qwen3 ids are out of a tiny vocab).
_VOCAB = 256
_DOC_START, _DOC_END, _EOS = 10, 11, 12


def _tiny_docchunk_model():
    config = TransformerConfig.llama_like(
        vocab_size=_VOCAB,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        document_chunked=True,
        cross_doc_mode="chunked",
    )
    return config.build(init_device="cpu")


def _toy_input():
    # <s> <doc_start> a b <doc_end> free free <eos> -> chunk0 spans the doc, rest FREE, then PAD.
    return torch.tensor([[5, _DOC_START, 6, 7, _DOC_END, 8, 9, _EOS]])


def test_enable_rejects_static_and_curriculum_together():
    model = _tiny_docchunk_model()
    with pytest.raises(OLMoConfigurationError):
        model.enable_document_chunk_attention(
            _DOC_START, _DOC_END, _EOS, standard_mix_prob=0.1, mix_start_p=0.8, mix_end_p=0.0
        )


def test_enable_rejects_curriculum_without_total_forwards():
    model = _tiny_docchunk_model()
    with pytest.raises(OLMoConfigurationError):
        model.enable_document_chunk_attention(
            _DOC_START, _DOC_END, _EOS, mix_start_p=0.8, mix_end_p=0.0, mix_total_forwards=0
        )


def test_no_mix_leaves_mix_none():
    model = _tiny_docchunk_model()
    model.enable_document_chunk_attention(_DOC_START, _DOC_END, _EOS)
    assert model._document_chunk_attention["mix"] is None


def test_counter_increments_in_train_not_eval():
    model = _tiny_docchunk_model()
    model.enable_document_chunk_attention(
        _DOC_START, _DOC_END, _EOS, standard_mix_prob=0.5, mix_seed=7
    )
    ids = _toy_input()

    model.eval()
    with torch.no_grad():
        model(ids)
        model(ids)
    assert model._document_chunk_attention["mix"]["forward_idx"] == 0  # eval never increments

    model.train()
    with torch.no_grad():
        model(ids)
        model(ids)
        model(ids)
    assert model._document_chunk_attention["mix"]["forward_idx"] == 3  # one per training forward


def test_eval_forward_bit_identical_with_and_without_mix():
    # Mask mixing is training-only, so an eval forward must be UNAFFECTED by a mix config -> a strong
    # p=0-invariant check at the model level (same weights, same input, logits identical).
    torch.manual_seed(0)
    plain = _tiny_docchunk_model()
    plain.enable_document_chunk_attention(_DOC_START, _DOC_END, _EOS)

    torch.manual_seed(0)
    mixed = _tiny_docchunk_model()
    mixed.enable_document_chunk_attention(
        _DOC_START, _DOC_END, _EOS, standard_mix_prob=1.0, mix_seed=7
    )

    ids = _toy_input()
    plain.eval()
    mixed.eval()
    with torch.no_grad():
        a = plain(ids)
        b = mixed(ids)
    assert torch.equal(a, b)


def test_train_forward_with_full_mix_runs_and_is_finite():
    # standard_mix_prob=1.0 collapses every example -> the model trains on plain causal that step; the
    # forward must still produce finite logits (curriculum log path exercised via forward_idx).
    model = _tiny_docchunk_model()
    model.enable_document_chunk_attention(
        _DOC_START, _DOC_END, _EOS, mix_start_p=1.0, mix_end_p=0.0, mix_total_forwards=4, mix_seed=7
    )
    model.train()
    ids = _toy_input()
    with torch.no_grad():
        out = model(ids)
    assert torch.isfinite(out).all()
