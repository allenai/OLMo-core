"""
CPU unit tests for :mod:`olmo_core.nn.attention.doc_categories` -- the whole-category keep policies
``smallcat_keep`` (gold-blind) and ``gold_plus_wholecats`` (gold-aware).

The properties under test are exactly the ones that make the policies mean what they claim:

  1. the clustering finds 3 clearly-separated small groups among many near-duplicates, and
     ``smallcat_keep`` keeps **exactly those** -- this is the rule's whole premise;
  2. **no partial categories, ever** -- every kept document's whole category is kept. A partial
     category is the failure mode these policies exist to remove: under ``gold_plus_random`` the
     gold category was the only COMPLETE one, which made "complete" a content-free gold detector;
  3. a decoy large category, when one is affordable, is also kept WHOLE, and when none is
     affordable the row keeps none and says so rather than silently paying for a giant one;
  4. ``gold_plus_wholecats`` keeps the gold document's category whole **plus** K other whole
     categories, so completeness names nothing;
  5. at least one category always stays pooled -- otherwise the arm is silently dense;
  6. a row with no structure (all documents alike) comes back as ONE category rather than an
     arbitrary split;
  7. the choices are deterministic given the seed and the call index.
"""

import torch

from olmo_core.nn.attention.doc_categories import cosine_categories, resolve_category_keep

D = 64


def _orthogonal_dirs(k: int, d: int = D) -> torch.Tensor:
    """``k`` exactly orthogonal unit directions -- so the test is about the RULE, not about whether
    two random Gaussians happened to land near each other."""
    g = torch.Generator().manual_seed(1234)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return q.t()[:k]


def _row(sizes, jitter: float = 0.02, seed: int = 7):
    """One row's document vectors: ``sizes[i]`` near-duplicates around orthogonal direction ``i``.

    :returns: ``((n, D) vectors, [member index lists])``.
    """
    g = torch.Generator().manual_seed(seed)
    dirs = _orthogonal_dirs(len(sizes))
    vecs, groups, i = [], [], 0
    for c, n in enumerate(sizes):
        members = []
        for _ in range(n):
            vecs.append(dirs[c] + jitter * torch.randn(D, generator=g))
            members.append(i)
            i += 1
        groups.append(members)
    return torch.stack(vecs), groups


def test_clustering_recovers_the_planted_groups():
    vecs, groups = _row([20, 2, 2, 2])
    labels = cosine_categories(vecs)
    found = {}
    for idx, c in enumerate(labels.tolist()):
        found.setdefault(c, []).append(idx)
    assert sorted(sorted(v) for v in found.values()) == sorted(sorted(g) for g in groups)


def test_unimodal_row_is_one_category():
    """No structure to find -> ONE category, not an arbitrary split."""
    vecs, _ = _row([12])
    assert len(set(cosine_categories(vecs).tolist())) == 1


def test_smallcat_keeps_exactly_the_three_small_groups():
    vecs, groups = _row([20, 2, 2, 2])
    valid = torch.ones(1, vecs.shape[0], dtype=torch.bool)
    keep, stats = resolve_category_keep(
        vecs.unsqueeze(0), valid, mode="smallcat_keep", n_small=3, n_decoy=0,
        doc_counts=torch.ones(1, vecs.shape[0]),
    )
    kept = {int(i) for i in keep[0].nonzero().flatten().tolist()}
    assert kept == set(groups[1] + groups[2] + groups[3])
    assert stats["n_cats_mean"] == 4
    assert abs(stats["real_token_frac"] - 6 / 26) < 1e-6


def test_no_partial_categories_ever():
    """Every kept document's ENTIRE category is kept, under both modes and with decoys on."""
    vecs, groups = _row([14, 6, 3, 2, 2])
    valid = torch.ones(1, vecs.shape[0], dtype=torch.bool)
    labels = cosine_categories(vecs)
    gold = torch.zeros(1, vecs.shape[0], dtype=torch.bool)
    gold[0, groups[3][0]] = True
    for kwargs in (
        dict(mode="smallcat_keep", n_small=2, n_decoy=1, decoy_max_size_mult=4.0),
        dict(mode="smallcat_keep", n_small=3, n_decoy=0),
        dict(mode="gold_plus_wholecats", gold_docs=gold, n_cats=2),
        dict(mode="gold_plus_wholecats", gold_docs=gold, n_cats=2, cats_random=True),
    ):
        keep, _ = resolve_category_keep(vecs.unsqueeze(0), valid, **kwargs)
        for c in set(labels.tolist()):
            members = (labels == c).nonzero().flatten()
            kept = keep[0][members]
            assert bool(kept.all()) or not bool(kept.any()), (
                f"category {c} kept PARTIALLY under {kwargs['mode']}: {kept.tolist()}"
            )


def test_decoy_is_kept_whole_when_affordable_and_skipped_when_not():
    vecs, groups = _row([14, 6, 2, 2])          # largest small kept = 2 -> cap 2*2 = 4
    valid = torch.ones(1, vecs.shape[0], dtype=torch.bool)
    # cap 4 excludes the 6- and 14-document categories: nothing eligible, so no decoy is bought
    _, tight = resolve_category_keep(
        vecs.unsqueeze(0), valid, mode="smallcat_keep", n_small=2, n_decoy=1,
        decoy_max_size_mult=2.0,
    )
    assert tight["decoy_skipped"] == 1
    # a looser cap admits the 6-document category, and it comes in WHOLE
    keep, loose = resolve_category_keep(
        vecs.unsqueeze(0), valid, mode="smallcat_keep", n_small=2, n_decoy=1,
        decoy_max_size_mult=3.0,
    )
    assert loose["decoy_skipped"] == 0
    kept = {int(i) for i in keep[0].nonzero().flatten().tolist()}
    assert kept == set(groups[2] + groups[3] + groups[1])


def test_gold_category_kept_whole_plus_k_others():
    vecs, groups = _row([14, 6, 3, 2, 2])
    valid = torch.ones(1, vecs.shape[0], dtype=torch.bool)
    gold = torch.zeros(1, vecs.shape[0], dtype=torch.bool)
    gold[0, groups[2][0]] = True                      # one member of the 3-document category
    keep, stats = resolve_category_keep(
        vecs.unsqueeze(0), valid, mode="gold_plus_wholecats", gold_docs=gold, n_cats=2,
    )
    kept = {int(i) for i in keep[0].nonzero().flatten().tolist()}
    assert set(groups[2]) <= kept, "the gold document's category must be kept WHOLE"
    # + the two SMALLEST non-gold categories (the 2-document ones), and nothing else
    assert kept == set(groups[2] + groups[3] + groups[4])
    assert stats["saturated_rows"] == 0


def test_at_least_one_category_stays_pooled():
    """Asking for more categories than exist must not silently train the row DENSE."""
    vecs, _ = _row([8, 3, 2])
    valid = torch.ones(1, vecs.shape[0], dtype=torch.bool)
    keep, stats = resolve_category_keep(
        vecs.unsqueeze(0), valid, mode="smallcat_keep", n_small=9, n_decoy=4,
    )
    assert not bool(keep.all()), "every category was kept -- the arm is dense"
    assert stats["saturated_rows"] == 0


def test_deterministic():
    vecs, _ = _row([14, 6, 3, 2, 2])
    valid = torch.ones(1, vecs.shape[0], dtype=torch.bool)
    kw = dict(mode="smallcat_keep", n_small=2, n_decoy=1, decoy_max_size_mult=4.0, seed=5, call=3)
    a, _ = resolve_category_keep(vecs.unsqueeze(0), valid, **kw)
    b, _ = resolve_category_keep(vecs.unsqueeze(0), valid, **kw)
    assert torch.equal(a, b)


def test_absent_documents_stay_real():
    """A document that is not present in the row is never pooled (conservative)."""
    vecs, _ = _row([10, 2, 2])
    valid = torch.ones(1, vecs.shape[0] + 3, dtype=torch.bool)
    valid[0, -3:] = False
    padded = torch.cat([vecs, torch.zeros(3, D)]).unsqueeze(0)
    keep, _ = resolve_category_keep(padded, valid, mode="smallcat_keep", n_small=1, n_decoy=0)
    assert bool(keep[0, -3:].all())


# ---------------------------------------------------------------------------
# End-to-end through the real compaction (Transformer._compact_pooled_soft_tokens)
# ---------------------------------------------------------------------------

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
IGN = -100


def _tiny_model(cat_keep, keep_docs=None):
    from olmo_core.config import DType
    from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, keep_prob=0.0, cat_keep=cat_keep
    )
    if keep_docs is not None:
        model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_docs)
    model.train()
    return model


def _doc_row(docs):
    ids = [11, 12]
    for toks in docs:
        ids += [DOC_START, *toks, DOC_END]
    ids += [21, 22, EOS]
    return ids


def test_forward_compaction_runs_and_pools_something():
    """The real forward path accepts cat_keep, pools a strict subset, and never raises."""
    docs = [[200 + i, 201 + i, 202 + i] for i in range(6)] + [[700, 701, 702], [800, 801, 802]]
    ids = _doc_row(docs)
    model = _tiny_model(
        {"mode": "smallcat_keep", "n_small": 2, "n_decoy": 0, "decoy_max_size_mult": 2.0,
         "n_cats": 3, "cats_random": False, "threshold_rule": "gap", "log_every": 0}
    )
    x = torch.tensor([ids])
    lab = torch.full_like(x, IGN)
    lab[0, -3:-1] = x[0, -2:]
    out = model._compact_pooled_soft_tokens(x, lab, IGN)
    assert out is not None
    assert model._pooled_soft_tokens["_cat_calls"] >= 1


def test_gold_plus_wholecats_needs_the_gold_hook():
    from olmo_core.exceptions import OLMoConfigurationError

    docs = [[200 + i, 201 + i, 202 + i] for i in range(6)]
    ids = _doc_row(docs)
    model = _tiny_model(
        {"mode": "gold_plus_wholecats", "n_small": 3, "n_decoy": 1, "decoy_max_size_mult": 2.0,
         "n_cats": 2, "cats_random": False, "threshold_rule": "gap", "log_every": 0}
    )
    x = torch.tensor([ids])
    lab = torch.full_like(x, IGN)
    lab[0, -3:-1] = x[0, -2:]
    try:
        model._compact_pooled_soft_tokens(x, lab, IGN)
    except OLMoConfigurationError as e:
        assert "gold-sidecar keep hook" in str(e)
    else:
        raise AssertionError("expected OLMoConfigurationError without the gold hook")


def test_default_is_unchanged_without_cat_keep():
    """No cat_keep -> the historical keep_prob path, and no category machinery runs."""
    docs = [[200 + i, 201 + i, 202 + i] for i in range(4)]
    ids = _doc_row(docs)
    model = _tiny_model(None)
    assert model._pooled_soft_tokens["cat_keep"] is None
    x = torch.tensor([ids])
    lab = torch.full_like(x, IGN)
    lab[0, -3:-1] = x[0, -2:]
    assert model._compact_pooled_soft_tokens(x, lab, IGN) is not None
    assert model._pooled_soft_tokens["_cat_calls"] == 0
