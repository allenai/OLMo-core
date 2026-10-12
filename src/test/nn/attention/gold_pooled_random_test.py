"""
CPU unit tests for the ``"gold_pooled_random"`` keep policy (pure Python, no GPU).

``gold_pooled_random`` is the inverse of ``gold_plus_random``: every GOLD document is left OUT of
the keep set -- so a pooled-KV / soft-token arm always pools it -- and a fraction of the NON-gold
documents keep their real tokens. The properties that make the arm mean what it claims are all
checked here:

  1. gold is NEVER kept, at any fraction and any seed;
  2. exactly ``n_random`` non-gold documents are kept (not ``len(gold) + n_random``, which is what
     the superficially similar ``"random_nongold"`` control does);
  3. the choice is deterministic given the seeded RNG, and stable across "epochs" (the caller
     re-seeds per fingerprint, so the same row keeps the same documents);
  4. through :func:`make_fingerprint_keep_docs_fn`, a ``--st-keep-frac``-style FRACTION keeps the
     same SHARE of non-gold documents at two very different context widths (n = 14 and n = 56),
     which is the whole point of driving it by fraction rather than count;
  5. defaults are untouched -- the same call with the default ``gold_plus_random`` mode still
     keeps every gold document.
"""

import random

import torch

from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row, select_keep_docs
from olmo_core.nn.attention.pooled_doc_kv import make_fingerprint_keep_docs_fn

EOS, DS, DE = 1000, 1001, 1002


def _row(n_docs: int, body: int = 3):
    """Marker-wrapped ids for ``n_docs`` documents with distinct bodies, then an answer + EOS."""
    ids = [50, 51]
    for d in range(n_docs):
        ids.append(DS)
        ids.extend(100 + d * body + j for j in range(body))
        ids.append(DE)
    ids.extend([90, 91, EOS])
    return ids


def test_gold_is_never_kept_and_count_is_exact():
    present = list(range(14))
    gold = {2, 7, 11}
    for n_random in (0, 1, 3, 5, 11):
        for seed in range(25):
            keep = select_keep_docs(
                present, gold, n_random=n_random, mode="gold_pooled_random",
                rng=random.Random(seed),
            )
            assert not (keep & gold), f"gold leaked into the keep set: {keep & gold}"
            assert keep <= set(present)
            assert len(keep) == min(n_random, len(present) - len(gold))


def test_differs_from_random_nongold_by_the_gold_count():
    """``random_nongold`` keeps ``len(gold) + n_random`` non-gold docs; this one keeps ``n_random``."""
    present, gold = list(range(14)), {2, 7, 11}
    kw = dict(n_random=4, rng=random.Random(0))
    a = select_keep_docs(present, gold, mode="gold_pooled_random", **dict(kw, rng=random.Random(0)))
    b = select_keep_docs(present, gold, mode="random_nongold", **dict(kw, rng=random.Random(0)))
    assert len(a) == 4
    assert len(b) == 4 + len(gold)
    assert not (a & gold) and not (b & gold)


def test_deterministic_given_the_seed():
    present, gold = list(range(20)), {3, 9}
    a = select_keep_docs(present, gold, n_random=6, mode="gold_pooled_random", rng=random.Random(7))
    b = select_keep_docs(present, gold, n_random=6, mode="gold_pooled_random", rng=random.Random(7))
    c = select_keep_docs(present, gold, n_random=6, mode="gold_pooled_random", rng=random.Random(8))
    assert a == b
    assert a != c  # not degenerate


def test_default_mode_unchanged():
    present, gold = list(range(14)), {2, 7, 11}
    keep = select_keep_docs(present, gold, n_random=3, mode="gold_plus_random", rng=random.Random(0))
    assert gold <= keep and len(keep) == len(gold) + 3


def test_fraction_is_length_invariant_end_to_end():
    """n = 14 and n = 56 rows both keep ~1/6 of their NON-gold docs, and never a gold one."""
    for n_docs, n_gold in ((14, 3), (56, 3)):
        ids = _row(n_docs)
        gold = list(range(n_gold))  # docs 0..n_gold-1 are gold
        fp = content_fingerprint_from_row(ids, EOS)
        fn = make_fingerprint_keep_docs_fn(
            {fp: gold},
            doc_start_id=DS, doc_end_id=DE, eos_id=EOS,
            n_random_frac=1.0 / 6.0, mode="gold_pooled_random", seed=11,
        )
        keep = fn(torch.tensor([ids], dtype=torch.long))
        assert keep.shape == (1, n_docs)
        kept = {int(i) for i in keep[0].nonzero().flatten().tolist()}
        assert not (kept & set(gold)), f"gold kept real at n={n_docs}: {kept & set(gold)}"
        expected = max(1, round((n_docs - n_gold) / 6.0))
        assert len(kept) == expected, f"n={n_docs}: kept {len(kept)}, expected {expected}"


def test_fingerprint_miss_keeps_everything_real():
    """An unknown row (e.g. the trainer's synthetic warmup batch) must degrade to full attention."""
    ids = _row(8)
    fn = make_fingerprint_keep_docs_fn(
        {"not-this-row": [0]},
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS,
        n_random_frac=0.5, mode="gold_pooled_random", seed=3,
    )
    keep = fn(torch.tensor([ids], dtype=torch.long))
    assert bool(keep.all())
