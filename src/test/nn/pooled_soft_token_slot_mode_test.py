"""Tests for the soft-token SLOT MODES (``mean`` / ``cmean`` / ``cent_cmean``).

The plain mean input embedding of a pooled document is ~94% shared common-word mass, which is what
makes it unreadable (``records/outlier-richer-slot-probe.md``); ``cmean`` drops a stop set of ids
from the mean and ``cent_cmean`` additionally centres and rescales it. ``mean`` must stay
bit-identical to the pre-2026-09-15 path.
"""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder
from olmo_core.nn.pooled_soft_token import (
    SLOT_MODES,
    apply_slot_mode,
    build_slot_stop_ids,
)
from olmo_core.nn.transformer import TransformerConfig

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
STOP = 100  # the "common word" id every document is padded with
IGN = -100


def _row(docs):
    """``[free] + <doc_start> tokens <doc_end> ... + [q, EOS]`` for a list of token lists."""
    ids = [11, 12]
    for toks in docs:
        ids += [DOC_START, *toks, DOC_END]
    ids += [21, 22, EOS]
    return ids


def _model(slot_mode="mean", stop_ids=None, keep=None):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DOC_START,
        DOC_END,
        EOS,
        placeholder_id=PLACEHOLDER,
        keep_prob=0.0,
        slot_mode=slot_mode,
        slot_stop_ids=stop_ids,
    )
    if keep is not None:
        model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
    model.train()
    return model


def _slot_feats(model, ids):
    x = torch.tensor([ids])
    lab = torch.full_like(x, IGN)
    lab[0, -3:-1] = x[0, -2:]
    _, inject, _ = model._compact_pooled_soft_tokens(x, lab, IGN)
    rows, cols, feats = inject
    return feats


DOCS = [[201, 202, STOP, STOP], [301, 302, 303, STOP], [401, STOP, STOP, 402]]


def test_default_mode_is_the_plain_mean_bit_identical():
    """``slot_mode="mean"`` (the default) must equal the historical ``index_add`` plain mean."""
    ids = _row(DOCS)
    keep = torch.zeros(1, len(DOCS), dtype=torch.bool)
    model = _model(keep=keep)
    feats = _slot_feats(model, ids)
    emb = model.embeddings(torch.tensor([ids]))[0]
    cid = build_chunk_ids_from_tokens(
        torch.tensor([ids]), doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS
    )[0]
    want = torch.stack([emb[cid == d].mean(0) for d in range(len(DOCS))])
    assert feats.shape == want.shape
    assert torch.equal(feats, want)


def test_cmean_drops_exactly_the_stop_ids():
    ids = _row(DOCS)
    keep = torch.zeros(1, len(DOCS), dtype=torch.bool)
    stop = [STOP, DOC_START, DOC_END]
    model = _model(slot_mode="cmean", stop_ids=stop, keep=keep)
    feats = _slot_feats(model, ids)
    emb = model.embeddings(torch.tensor([ids]))[0]
    cid = build_chunk_ids_from_tokens(
        torch.tensor([ids]), doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS
    )[0]
    row = torch.tensor(ids)
    for d in range(len(DOCS)):
        sel = (cid == d) & ~torch.isin(row, torch.tensor(stop))
        assert torch.allclose(feats[d], emb[sel].mean(0), atol=1e-5)
    # and it is NOT the plain mean (the stop tokens did carry mass)
    plain = torch.stack([emb[cid == d].mean(0) for d in range(len(DOCS))])
    assert not torch.allclose(feats, plain, atol=1e-4)


def test_cent_cmean_is_row_centred_and_rescaled():
    ids = _row(DOCS)
    keep = torch.zeros(1, len(DOCS), dtype=torch.bool)
    stop = [STOP, DOC_START, DOC_END]
    model = _model(slot_mode="cent_cmean", stop_ids=stop, keep=keep)
    feats = _slot_feats(model, ids)
    emb = model.embeddings(torch.tensor([ids]))[0]
    cid = build_chunk_ids_from_tokens(
        torch.tensor([ids]), doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS
    )[0]
    row = torch.tensor(ids)
    content = (cid >= 0) & ~torch.isin(row, torch.tensor(stop))
    centre = emb[content].mean(0)
    target = emb[content].norm(dim=-1).mean()
    # every slot sits on the sphere of the row's mean real-token norm
    assert torch.allclose(feats.norm(dim=-1), target.expand(len(DOCS)), atol=1e-4)
    # and the pre-rescale vectors are the content means minus the row centroid, so their
    # (unweighted-by-doc-length) mean over documents is much closer to zero than before centring
    dirs = torch.stack(
        [
            (emb[(cid == d) & content.clone()].mean(0) - centre)
            / (emb[(cid == d) & content.clone()].mean(0) - centre).norm()
            for d in range(len(DOCS))
        ]
    )
    assert torch.allclose(feats / feats.norm(dim=-1, keepdim=True), dirs, atol=1e-4)


def test_cent_cmean_row_mean_of_content_means_is_zero():
    """The centre really is the row's content mean: subtracting it zeroes the token-weighted mean
    over the row's documents."""
    torch.manual_seed(0)
    emb = torch.randn(1, 24, 8)
    input_ids = torch.full((1, 24), 300)
    input_ids[0, ::4] = STOP  # a "common word" every 4 tokens
    chunk_ids = torch.tensor([[0] * 8 + [1] * 8 + [2] * 8])
    stop_mask = torch.zeros(1000, dtype=torch.bool)
    stop_mask[STOP] = True
    plain = torch.zeros(1, 3, 8)
    cm, n_fb = apply_slot_mode(
        emb, input_ids, chunk_ids, 3, plain, mode="cmean", stop_mask=stop_mask
    )
    assert n_fb == 0
    content = input_ids[0] != STOP
    # token-count-weighted mean of the per-doc content means == the row centroid
    counts = torch.stack([(chunk_ids[0] == d)[content].sum() for d in range(3)]).float()
    centre = emb[0][content].mean(0)
    assert torch.allclose((cm[0] * counts[:, None]).sum(0) / counts.sum(), centre, atol=1e-5)
    cc, _ = apply_slot_mode(
        emb, input_ids, chunk_ids, 3, plain, mode="cent_cmean", stop_mask=stop_mask
    )
    # after centring, the same weighted mean of the DIRECTIONS' pre-scale vectors is zero
    pre = cm[0] - centre
    assert torch.allclose((pre * counts[:, None]).sum(0) / counts.sum(), torch.zeros(8), atol=1e-5)
    target = emb[0][content].norm(dim=-1).mean()
    assert torch.allclose(cc[0].norm(dim=-1), target.expand(3), atol=1e-4)


def test_all_stop_document_falls_back_to_the_plain_mean_and_counts():
    """A document whose every token is in the stop set keeps its plain mean, and is counted."""
    emb = torch.randn(1, 12, 8)
    input_ids = torch.full((1, 12), 300)
    input_ids[0, 4:8] = STOP  # doc 1 is entirely common words
    chunk_ids = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4])
    stop_mask = torch.zeros(1000, dtype=torch.bool)
    stop_mask[STOP] = True
    plain = torch.stack([emb[0][chunk_ids[0] == d].mean(0) for d in range(3)]).unsqueeze(0)
    out, n_fb = apply_slot_mode(
        emb, input_ids, chunk_ids, 3, plain, mode="cmean", stop_mask=stop_mask
    )
    assert n_fb == 1
    assert torch.allclose(out[0, 1], plain[0, 1], atol=1e-6)
    assert torch.allclose(out[0, 0], plain[0, 0], atol=1e-6)  # no stop tokens -> unchanged


def test_build_slot_stop_ids_takes_top_k_plus_extras_and_punctuation():
    # id 7 appears most, then 8; 5 is punctuation but rare; 900 is a marker passed as extra.
    tokens = [7] * 10 + [8] * 5 + [9] * 2 + [5]
    stop, shown = build_slot_stop_ids(tokens, top_k=1, extra_ids=(900,))
    assert stop == [7, 900]
    assert shown == ["7"]  # no decoder -> raw ids, most frequent first
    stop2, shown2 = build_slot_stop_ids(
        tokens,
        top_k=1,
        extra_ids=(900,),
        decode=lambda t: {5: ".", 7: " the", 8: "cat", 9: "dog"}.get(t, "x"),
    )
    assert stop2 == [5, 7, 900]
    assert shown2[0] == repr(" the")


def test_unknown_slot_mode_is_refused():
    from olmo_core.exceptions import OLMoConfigurationError

    assert SLOT_MODES == ("mean", "cmean", "cent_cmean")
    with pytest.raises(OLMoConfigurationError):
        _model(slot_mode="nope", stop_ids=[STOP])
    with pytest.raises(OLMoConfigurationError):
        _model(slot_mode="cmean", stop_ids=None)
    with pytest.raises(ValueError):
        apply_slot_mode(
            torch.zeros(1, 4, 8),
            torch.zeros(1, 4, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            1,
            torch.zeros(1, 1, 8),
            mode="mean",
            stop_mask=torch.zeros(1000, dtype=torch.bool),
        )
