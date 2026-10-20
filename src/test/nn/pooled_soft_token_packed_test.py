"""Soft tokens on PACKED rows (segment mode): unpack -> per-example gold keep -> compact -> re-pack.

The packed forward must train exactly what the padded one-example-per-row forward trains: same
kept tokens, same positions, same loss. Also covers the per-example FLOP lengths."""

import pytest
import torch

from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row
from olmo_core.nn.attention.pooled_doc_kv import (
    install_pooled_doc_keep,
    make_fingerprint_keep_docs_fn,
)
from olmo_core.nn.pooled_soft_token import (
    compact_pooled_rows,
    example_lengths,
    repack_compacted_rows,
    unpack_packed_rows,
)
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
IGN = -100


def _example(seed, n_docs, doc_len=12):
    g = torch.Generator().manual_seed(seed)
    ids = [11, 12]
    for _ in range(n_docs):
        ids += [DOC_START, *torch.randint(100, 800, (doc_len,), generator=g).tolist(), DOC_END]
    ids += [21, 22, 23, EOS]
    return ids


def _labels(ids, n_loss=3):
    lab = [IGN] * len(ids)
    e = ids.index(EOS)
    for t in range(e - n_loss, e):
        lab[t] = ids[t + 1]
    return lab


EXAMPLES = [_example(1, 3), _example(2, 5, doc_len=9), _example(3, 2, doc_len=20)]
GOLD = {1: [1], 2: [0, 3], 3: []}  # per example seed; [] = no gold doc (compress everything)


def _packed(pad_tail=7):
    ids = sum(EXAMPLES, []) + [EOS] * pad_tail
    lab = sum((_labels(e) for e in EXAMPLES), []) + [IGN] * pad_tail
    lens = [len(e) for e in EXAMPLES] + [pad_tail]
    return torch.tensor([ids]), torch.tensor([lab]), torch.tensor([lens], dtype=torch.int32)


def _padded():
    L = max(len(e) for e in EXAMPLES) + 3
    ids = torch.tensor([e + [EOS] * (L - len(e)) for e in EXAMPLES])
    lab = torch.tensor([_labels(e) + [IGN] * (L - len(e)) for e in EXAMPLES])
    return ids, lab


def _gold_table():
    return {content_fingerprint_from_row(e, EOS): GOLD[s] for s, e in zip((1, 2, 3), EXAMPLES)}


def test_unpack_matches_padded_rows_and_drops_pad_doc():
    ids, lab, dl = _packed()
    up = unpack_packed_rows(ids, lab, dl, pad_id=EOS, ignore_index=IGN)
    assert up.lengths == [len(e) for e in EXAMPLES]
    for i, e in enumerate(EXAMPLES):
        assert up.input_ids[i, : len(e)].tolist() == e
        assert (up.input_ids[i, len(e) :] == EOS).all()
        assert up.labels[i, : len(e)].tolist() == _labels(e)
    # fingerprints of the unpacked rows are the padded path's
    assert [content_fingerprint_from_row(r.tolist(), EOS) for r in up.input_ids] == list(
        _gold_table()
    )


def test_example_lengths_packed_and_padded():
    ids, _, dl = _packed()
    assert example_lengths(ids, dl, EOS) == [len(e) for e in EXAMPLES]
    pids, _ = _padded()
    assert example_lengths(pids, None, EOS) == [len(e) for e in EXAMPLES]


@pytest.mark.parametrize("pad_multiple", [1, 16])
def test_repack_concatenates_compacted_rows(pad_multiple):
    ids, lab, dl = _packed()
    up = unpack_packed_rows(ids, lab, dl, pad_id=EOS, ignore_index=IGN)
    cids = build_chunk_ids_from_tokens(
        up.input_ids, doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS, mode="chunked"
    )
    keep = torch.zeros(3, int(cids.max()) + 1, dtype=torch.bool)
    keep[0, 1] = True
    cb = compact_pooled_rows(
        up.input_ids, up.labels, cids, keep, placeholder_id=PLACEHOLDER, pad_token_id=EOS,
        ignore_index=IGN, drop_slots=True,
    )
    rid, rlab, rpos, rdl, rmdl = repack_compacted_rows(
        cb, pad_id=EOS, ignore_index=IGN, pad_multiple=pad_multiple
    )
    lens = cb.row_lens.tolist()
    assert rid.shape[1] % pad_multiple == 0 and rid.shape[1] >= sum(lens)
    assert rdl[0, : len(lens)].tolist() == lens and rmdl == [max(rdl[0].tolist())]
    off = 0
    for b, L in enumerate(lens):
        assert rid[0, off : off + L].tolist() == cb.input_ids[b, :L].tolist()
        assert rpos[0, off : off + L].tolist() == cb.position_ids[b, :L].tolist()
        assert rlab[0, off : off + L].tolist() == cb.labels[b, :L].tolist()
        off += L
    assert (rlab[0, off:] == IGN).all()
    # labels survive compaction + re-pack exactly
    assert int((rlab != IGN).sum()) == int((lab != IGN).sum())


def _hybrid(device="cuda"):
    from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.rope import RoPEConfig, RoPEType
    from olmo_core.nn.transformer import (
        TransformerBlockConfig,
        TransformerBlockType,
        TransformerConfig,
    )

    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    ff = FeedForwardConfig(hidden_size=512, bias=False)
    gdn = TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        sequence_mixer=GatedDeltaNetConfig(n_heads=8),
        layer_norm=ln,
        feed_forward=ff,
    )
    attn = TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        attention=AttentionConfig(
            n_heads=8, rope=RoPEConfig(name=RoPEType.default), backend=AttentionBackendName.flash_2
        ),
        layer_norm=ln,
        feed_forward=ff,
    )
    cfg = TransformerConfig(
        d_model=256, vocab_size=1024, n_layers=3, block=gdn, block_overrides={1: attn},
        lm_head=LMHeadConfig(layer_norm=ln, bias=False),
    )
    model = cfg.build(init_device=device)
    model.init_weights()
    model = model.to(torch.bfloat16)
    model.enable_pooled_soft_tokens(
        DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, keep_prob=0.0, drop_slots=True,
        header_extra_tokens=2, keep_token_rule="first_last", keep_token_k=0.2, pad_multiple=16,
    )
    keep_fn = make_fingerprint_keep_docs_fn(
        _gold_table(), doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS, n_random=0
    )
    install_pooled_doc_keep(model, keep_fn)
    model._flop_pad_id = EOS
    model.train()
    return model


@requires_gpu
@requires_fla
def test_packed_segment_mode_matches_padded_path():
    model = _hybrid()
    pids, plab = _padded()
    out_pad = model(pids.cuda(), labels=plab.cuda(), loss_reduction="sum")
    lens_pad = model._flop_example_lens
    model._flop_example_lens = None
    ids, lab, dl = _packed()
    out_pk = model(
        ids.cuda(), labels=lab.cuda(), loss_reduction="sum", doc_lens=dl.cuda(),
        max_doc_lens=[int(dl.max())],
    )
    lens_pk = model._flop_example_lens
    assert torch.allclose(out_pk.loss.float(), out_pad.loss.float(), rtol=2e-2, atol=2e-2), (
        out_pk.loss, out_pad.loss,
    )
    # identical per-example dense and compacted lengths on both paths
    assert lens_pk == lens_pad
    assert lens_pk["dense"] == [len(e) for e in EXAMPLES]
    assert all(a < d for a, d in zip(lens_pk["actual"], lens_pk["dense"]))
    out_pk.loss.backward()


@requires_gpu
@requires_fla
def test_dense_packed_forward_records_example_lengths():
    model = _hybrid()
    model._pooled_soft_tokens = None
    ids, lab, dl = _packed()
    model(ids.cuda(), labels=lab.cuda(), doc_lens=dl.cuda(), max_doc_lens=[int(dl.max())])
    st = model._flop_example_lens
    assert st["dense"] == st["actual"] == [len(e) for e in EXAMPLES]
