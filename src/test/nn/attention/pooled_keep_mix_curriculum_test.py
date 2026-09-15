"""
The **compression-mixing curriculum** on the pooled keep set.

Each row trains UNCOMPRESSED (every document real) with probability ``p_full``, annealed from
``mix_start_p`` to ``mix_end_p``. Until 2026-09-14 the curriculum lived only inside
``make_fingerprint_keep_docs_fn``, which ``train_ctc_suite.py`` installs only when the run is NOT
``--st-gold-blind`` -- so on a gold-blind arm the ``--st-mix-*`` flags were parsed and silently
ignored (the ds64 ``kvgbmix`` arm was a byte-identical re-run of ``kvgb``; see
records/ds64-handoff.md section 9). These tests pin the gold-blind path down.
"""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.pooled_doc_kv import (
    PooledDocKeepHolder,
    anneal_p_full,
    resolve_keep_docs,
)
from olmo_core.nn.transformer import TransformerConfig

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
IGN = -100
N_DOCS = 4
BATCH = 24


def _row(doc_len: int):
    """``[free, free] + N_DOCS marker-wrapped docs of ``doc_len`` tokens + [q, q, EOS]``."""
    ids = [11, 12]
    for d in range(N_DOCS):
        ids += [DOC_START, *(100 + 7 * d + j for j in range(doc_len)), DOC_END]
    return ids + [21, 22, EOS]


def _batch():
    """A batch whose rows have DIFFERENT document layouts, so their keep draws are independent
    (the fallback draw hashes the row's chunk layout, not its token ids)."""
    rows = [_row(2 + b) for b in range(BATCH)]
    width = max(len(r) for r in rows)
    return torch.tensor([r + [EOS] * (width - len(r)) for r in rows])


def _chunk_ids(ids: torch.Tensor) -> torch.Tensor:
    return build_chunk_ids_from_tokens(
        ids, doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS, mode="chunked"
    )


def _model(keep_prob: float = 0.0, **mix):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, keep_prob=keep_prob, **mix
    )
    model.train()
    return model


def _pooled_rows(model, ids: torch.Tensor):
    """Set of row indices that got at least one soft slot (i.e. were compressed)."""
    cb, _, _ = model._compact_pooled_soft_tokens(ids, None, IGN)
    return set(int(r) for r in cb.soft_rows.tolist())


# --------------------------------------------------------------------------------------
# the annealing helper
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "calls, expected", [(0, 1.0), (5, 0.5), (10, 0.0), (25, 0.0)]  # clamped past the horizon
)
def test_anneal_p_full_is_linear_then_clamped(calls, expected):
    assert anneal_p_full(
        calls, mix_start_p=1.0, mix_end_p=0.0, mix_total_calls=10
    ) == pytest.approx(expected)


def test_anneal_p_full_without_a_horizon_is_constant():
    assert anneal_p_full(99, mix_start_p=0.4, mix_end_p=0.0, mix_total_calls=0) == pytest.approx(
        0.4
    )


# --------------------------------------------------------------------------------------
# resolve_keep_docs: the gold-blind draw
# --------------------------------------------------------------------------------------


def test_resolve_keep_docs_without_mix_is_unchanged():
    """``mix_p_full=0`` (the default) must reproduce the pre-curriculum draw bit for bit."""
    cids = _chunk_ids(_batch())
    base = resolve_keep_docs(cids, N_DOCS, holder=None, keep_prob=0.3, keep_seed=42)
    for call in (0, 7):
        same = resolve_keep_docs(
            cids, N_DOCS, holder=None, keep_prob=0.3, keep_seed=42, mix_p_full=0.0, mix_call=call
        )
        assert torch.equal(base, same)


@pytest.mark.parametrize("mix_call", [0, 5])
def test_resolve_keep_docs_mix_only_adds_whole_real_rows(mix_call):
    cids = _chunk_ids(_batch())
    kw = dict(holder=None, keep_prob=0.3, keep_seed=42)
    base = resolve_keep_docs(cids, N_DOCS, **kw)
    mixed = resolve_keep_docs(cids, N_DOCS, **kw, mix_p_full=0.5, mix_call=mix_call)
    full_rows = mixed.all(dim=1) & ~base.all(dim=1)
    assert full_rows.any(), "p_full=0.5 promoted no row to fully-real"
    assert not full_rows.all(), "p_full=0.5 promoted every row"
    # Rows the curriculum did NOT pick keep exactly the documents they had before.
    untouched = ~full_rows
    assert torch.equal(mixed[untouched], base[untouched])
    # p_full = 1 keeps everything; p_full = 0 keeps nothing extra.
    assert resolve_keep_docs(cids, N_DOCS, **kw, mix_p_full=1.0, mix_call=mix_call).all()


def test_resolve_keep_docs_mix_is_deterministic_and_call_dependent():
    cids = _chunk_ids(_batch())
    kw = dict(holder=None, keep_prob=0.0, keep_seed=42, mix_p_full=0.5)
    a = resolve_keep_docs(cids, N_DOCS, **kw, mix_call=3)
    assert torch.equal(a, resolve_keep_docs(cids, N_DOCS, **kw, mix_call=3))
    assert not torch.equal(a, resolve_keep_docs(cids, N_DOCS, **kw, mix_call=4))


# --------------------------------------------------------------------------------------
# the trainer-visible path: Transformer.enable_pooled_soft_tokens
# --------------------------------------------------------------------------------------


def test_soft_token_compaction_without_mix_flags_is_unchanged():
    """No ``--st-mix-*`` -> the keep set is exactly the seeded ``keep_prob`` draw, and the
    curriculum's call counter never moves."""
    ids = _batch()
    model = _model(keep_prob=0.5)
    cb, _, _ = model._compact_pooled_soft_tokens(ids, None, IGN)
    ref = resolve_keep_docs(
        _chunk_ids(ids),
        N_DOCS,
        holder=None,
        keep_prob=0.5,
        keep_seed=model._pooled_soft_tokens["keep_seed"],
    )
    got = set(zip(cb.soft_rows.tolist(), cb.soft_docs.tolist()))
    want = {(b, d) for b in range(BATCH) for d in range(N_DOCS) if not ref[b, d]}
    assert got == want
    assert model._pooled_soft_tokens["mix_calls"] == 0


@pytest.mark.parametrize("gold_blind", [True, False])
@pytest.mark.parametrize("calls_done, expect_full_rows", [(0, "all"), (5, "some")])
def test_mix_curriculum_applies_on_the_gold_blind_path(gold_blind, calls_done, expect_full_rows):
    """
    With ``keep_prob=0`` every document is pooled unless the curriculum promotes its whole row, so
    the uncompressed rows ARE the mixed rows. Annealed 1.0 -> 0.0 over 10 calls: at call 0 every
    row is full, at call 5 about half are.

    ``gold_blind=False`` (the :func:`install_pooled_doc_keep` hook installed) must NOT apply it
    here -- the hook owns the curriculum on that path, and applying it twice would double the
    uncompressed share.
    """
    ids = _batch()
    model = _model(keep_prob=0.0, mix_start_p=1.0, mix_end_p=0.0, mix_total_calls=10)
    if not gold_blind:
        model._pooled_keep_holder = PooledDocKeepHolder(
            keep_docs=torch.zeros(BATCH, N_DOCS, dtype=torch.bool)
        )
    model._pooled_soft_tokens["mix_calls"] = calls_done
    compressed = _pooled_rows(model, ids)
    n_full = BATCH - len(compressed)
    if not gold_blind:
        assert n_full == 0
        assert model._pooled_soft_tokens["mix_calls"] == calls_done  # counter untouched
        return
    assert model._pooled_soft_tokens["mix_calls"] == calls_done + 1
    if expect_full_rows == "all":
        assert n_full == BATCH
    else:
        assert 0 < n_full < BATCH


def test_mix_curriculum_anneals_away_over_training():
    """The uncompressed share must fall monotonically-ish to zero by the end of the horizon."""
    ids = _batch()
    model = _model(keep_prob=0.0, mix_start_p=1.0, mix_end_p=0.0, mix_total_calls=8)
    shares = []
    for _ in range(9):
        shares.append((BATCH - len(_pooled_rows(model, ids))) / BATCH)
    assert shares[0] == 1.0
    assert shares[-1] == 0.0
    assert shares[1] > shares[-1]
