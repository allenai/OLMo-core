"""Soft-token keep rules: random phrase spans (``keep_token_rule="spans"``) and the two-tier keep
(``kept_doc_k``: the keep set gets a larger per-document budget instead of staying whole)."""

import torch

from olmo_core.config import DType
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder
from olmo_core.nn.pooled_soft_token import span_scores
from olmo_core.nn.transformer import TransformerConfig

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
IGN = -100


def _row(n_docs=4, doc_len=60):
    ids = [11, 12]
    for d in range(n_docs):
        ids += [DOC_START, *(100 + (7 * d + j) % 700 for j in range(doc_len)), DOC_END]
    ids += [21, 22, EOS]
    return ids


def _labels(ids):
    lab = [IGN] * len(ids)
    e = ids.index(EOS)
    for t in range(e - 2, e):
        lab[t] = ids[t + 1]
    return lab


def _model(**kw):
    cfg = TransformerConfig.olmo2_190M(vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32)
    m = cfg.build(init_device="cpu")
    m.enable_pooled_soft_tokens(DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, keep_prob=0.0,
                                drop_slots=True, header_extra_tokens=2, **kw)
    m.train()
    return m


def _kept_per_doc(m, ids, keep_docs):
    x, lab = torch.tensor([ids]), torch.tensor([_labels(ids)])
    m._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_docs)
    cb, _, _ = m._compact_pooled_soft_tokens(x, lab, IGN)
    cid = build_chunk_ids_from_tokens(x, doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS, mode="chunked")[0]
    pos = cb.position_ids[0, : int(cb.row_lens[0])]
    kept = torch.zeros(len(ids), dtype=torch.bool)
    kept[pos] = True
    out = []
    for d in range(int(cid.max()) + 1):
        body = (cid == d) & (x[0] != DOC_START) & (x[0] != DOC_END)
        out.append((int((kept & body).sum()), int(body.sum()), kept & body))
    return out


def test_span_scores_constant_within_blocks():
    s = span_scores(torch.zeros(2, 50, dtype=torch.long), span_len=12)
    assert s.shape == (2, 50)
    for b in range(2):
        for blk in range(0, 50, 12):
            assert torch.all(s[b, blk : blk + 12] == s[b, blk])
    assert not torch.equal(s[0], s[1])


def test_spans_rule_keeps_contiguous_runs_at_budget():
    torch.manual_seed(0)
    m = _model(keep_token_rule="spans", keep_token_k=0.25, span_len=8)
    res = _kept_per_doc(m, _row(), torch.zeros(1, 4, dtype=torch.bool))
    for k, n, mask in res:
        assert 0.25 * n <= k <= 0.25 * n + 2 + 8  # budget + 2-token prefix, block granularity
        idx = mask.nonzero().flatten().tolist()
        runs = sum(1 for i, j in zip(idx, idx[1:]) if j != i + 1) + 1
        assert runs <= 1 + -(-k // 8) + 1  # a few contiguous spans, not scattered tokens


def test_tier_a_gets_its_own_budget_instead_of_whole():
    torch.manual_seed(0)
    keep = torch.tensor([[True, False, False, False]])
    whole = _kept_per_doc(_model(keep_token_rule="first_last", keep_token_k=0.2), _row(), keep)
    assert whole[0][0] == whole[0][1]  # default: kept doc stays whole
    tier = _kept_per_doc(_model(keep_token_rule="first_last", keep_token_k=0.2, kept_doc_k=0.5), _row(), keep)
    k0, n0, _ = tier[0]
    assert 0.5 * n0 <= k0 < n0  # tier A: about half, not whole
    for k, n, _ in tier[1:]:
        assert k < 0.5 * n  # tier B: the small budget
        assert k == whole[1][0]  # tier B identical to the non-tier rule
