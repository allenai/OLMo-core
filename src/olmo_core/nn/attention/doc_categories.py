"""
**Category-aware keep sets** for the soft-token / pooled-doc-KV feature: decide which documents
keep their real tokens from the *documents' own slot vectors*, one whole category at a time.

WHY WHOLE CATEGORIES. ``gold_plus_random`` -- keep every gold document real plus ``n_random``
random non-gold ones -- collapses on outlier, and the reason is a **completeness signature**: the
gold documents are the outlier category, and forcing all of them real made the gold category the
only category that was *entirely* real, while every other category showed up as one or two
scattered real documents. "Which category is complete?" is then a perfect, content-free detector of
the answer, so the gradient never has to read anything. Both policies here therefore keep
**whole categories only** -- a category is either fully real or fully pooled, never partial -- and
both deliberately keep more than one whole category so completeness alone names nothing:

* :func:`smallcat_keep` (**gold-blind**) keeps the ``C`` smallest categories plus ``D`` *decoy*
  large categories. The small categories are where an outlier lives, but the rule never looks at
  the label -- it is an inference the model could make for itself from the same vectors.
* :func:`gold_plus_wholecats` (**gold-aware**) keeps the gold document's category whole, as
  ``gold_plus_random`` did, **and** ``K`` other whole categories. The completeness signature is
  gone and "which of these complete categories is the small one?" is the real task.

THE CATEGORIES COME FROM THE SLOT VECTORS THE MODEL ACTUALLY SEES. The caller passes the same
``(B, n_docs, D)`` features the pooled slot carries -- with ``--st-slot-mode cent_cmean`` that is
the content-only, row-centred, renormalised vector (:func:`~olmo_core.nn.pooled_soft_token.apply_slot_mode`).
So the partition is computable from what a pooled document exposes, and a category the rule can see
is a category the model can see.

CLUSTERING RULE (cheap, deterministic, O(n_docs^2 * D)). Unit-normalise the document vectors, take
the full cosine matrix, and cut the row's **off-diagonal** cosines at the **midpoint of their
largest interior gap**; the categories are the connected components of that graph -- single
linkage. Three properties make this the right cheap rule here rather than k-means:

* it needs **no k**, which is the whole question ("how many outlier groups are there?");
* the threshold is **relative to the row**, so it does not depend on the absolute cosine scale of
  an embedding table, which differs per model and per slot mode;
* it puts the cut in genuinely EMPTY space, which single linkage requires. Two rules were tried
  and rejected first, both on a planted row of 20 near-duplicates (pairwise cosine ~0.98) plus
  three separated pairs (~0.0):

  - ``mean + sigma`` gives 0.75 + 0.47 = 1.22, above every cosine, so **every** document came back
    a singleton;
  - **Otsu** puts the cut at the top of the low mode rather than in the middle of the gap, and a
    single borderline pair 0.004 from the cut then chains two categories into one -- a merge is
    one edge under single linkage.

  The largest interior gap lands at ~0.51, halfway across the empty band, where one stray pair
  cannot reach.

"Interior" means the candidate cut is restricted to the middle ``1 - 2 * trim`` of the sorted
values, so a single isolated pair at either extreme cannot define the split. When the largest
interior gap is smaller than ``min_split_ratio`` of the trimmed range there is no structure to find
and the whole row is returned as ONE category rather than an arbitrary split.

Single linkage's known behaviour is chaining: a corpus of near-duplicates collapses into ONE large
component and genuine outliers fall out as singletons or tiny components. That is exactly the
partition outlier wants, and it is also why a decoy large category is often *unaffordable* -- see
:func:`resolve_category_keep`.

Everything is per-row, seeded and stable across layers, epochs and activation-checkpoint recompute,
and none of it touches the detach semantics: a kept document keeps its real per-token KV, a pooled
one gets its slot, exactly as before.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "CATEGORY_KEEP_MODES",
    "cosine_categories",
    "resolve_category_keep",
]

#: ``--st-keep-mode`` values handled by :func:`resolve_category_keep`.
CATEGORY_KEEP_MODES = ("smallcat_keep", "gold_plus_wholecats")


def cosine_categories(
    vecs: torch.Tensor,
    *,
    threshold_rule: str = "gap",
    threshold_lambda: float = 1.0,
    min_split_ratio: float = 0.2,
    trim: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Partition ``n`` document vectors into categories by single linkage on thresholded cosine.

    :param vecs: ``(n, D)`` document features (the slot vectors).
    :param threshold_rule: ``"gap"`` (default) cuts at the midpoint of the largest INTERIOR gap in
        the sorted off-diagonal cosines -- see the module docstring for the two rules this
        replaced and why. ``"mean_std"`` is the simpler ``mean + threshold_lambda * std`` cut, kept
        only as an ablation; it fails outright on this task's bimodal histogram.
    :param threshold_lambda: ``"mean_std"`` only.
    :param min_split_ratio: If the largest interior gap is smaller than this fraction of the
        trimmed value range, the row is treated as ONE category (no structure to find).
    :param trim: Fraction of the sorted values excluded at each end when looking for the gap, so a
        single isolated pair cannot define the split.
    :param eps: Normalisation floor.

    :returns: ``(n,)`` int64 category labels, canonicalised so a category's label is the smallest
        document index it contains (deterministic, order-independent).

    :raises ValueError: On an unknown ``threshold_rule``.
    """
    n = vecs.shape[0]
    if n <= 1:
        return torch.zeros(n, dtype=torch.long, device=vecs.device)
    v = vecs.float()
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)
    sim = v @ v.t()
    off = ~torch.eye(n, dtype=torch.bool, device=vecs.device)
    vals = sim[off]
    if threshold_rule == "mean_std":
        tau = float(vals.mean()) + threshold_lambda * float(vals.std(unbiased=False))
    elif threshold_rule == "gap":
        srt, _ = torch.sort(vals)
        N = srt.numel()
        lo_i = int(trim * N)
        hi_i = max(lo_i + 1, N - 1 - int(trim * N))
        if hi_i <= lo_i:
            return torch.zeros(n, dtype=torch.long, device=vecs.device)
        seg = srt[lo_i : hi_i + 1]
        if seg.numel() < 2:
            return torch.zeros(n, dtype=torch.long, device=vecs.device)
        diffs = seg[1:] - seg[:-1]
        j = int(torch.argmax(diffs))
        span = float(seg[-1] - seg[0])
        if span <= eps or float(diffs[j]) / span < min_split_ratio:
            return torch.zeros(n, dtype=torch.long, device=vecs.device)
        tau = float(seg[j] + seg[j + 1]) / 2.0
    else:
        raise ValueError(
            f"cosine_categories: unknown threshold_rule {threshold_rule!r} "
            "(expected 'gap' or 'mean_std')"
        )
    adj = (sim > tau) & off
    # Connected components by boolean transitive closure. n is the document count (<= a few
    # hundred), so log2(n) squarings is cheaper and far more predictable than a Python union-find.
    reach = adj | torch.eye(n, dtype=torch.bool, device=vecs.device)
    for _ in range(int(n).bit_length()):
        nxt = (reach.float() @ reach.float()) > 0
        if bool((nxt == reach).all()):
            break
        reach = nxt
    idx = torch.arange(n, device=vecs.device)
    return (
        torch.where(reach, idx.unsqueeze(0), torch.full_like(reach, n, dtype=torch.long))
        .min(dim=1)
        .values
    )


def _groups(labels: torch.Tensor) -> List[List[int]]:
    """Category id -> sorted member indices, ordered by (size, smallest member) so the caller's
    "the C smallest" is deterministic and independent of tensor iteration order."""
    buckets: Dict[int, List[int]] = {}
    for i, c in enumerate(labels.tolist()):
        buckets.setdefault(int(c), []).append(i)
    return sorted(buckets.values(), key=lambda g: (len(g), g[0]))


def _row_rng(seed: int, call: int, row_sig: int) -> random.Random:
    """Per (row, call) RNG: deterministic given the data order, and stable across layers and
    activation-checkpoint recompute within one forward."""
    return random.Random(f"cat:{seed}:{call}:{row_sig}")


def resolve_category_keep(
    doc_vecs: torch.Tensor,
    doc_valid: torch.Tensor,
    *,
    mode: str,
    gold_docs: Optional[torch.Tensor] = None,
    n_small: int = 3,
    n_decoy: int = 1,
    decoy_max_size_mult: float = 2.0,
    n_cats: int = 3,
    cats_random: bool = False,
    threshold_rule: str = "gap",
    threshold_lambda: float = 1.0,
    seed: int = 0,
    call: int = 0,
    doc_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    The ``(B, n_docs)`` bool keep mask for one forward, chosen **one whole category at a time**.

    :param doc_vecs: ``(B, n_docs, D)`` slot features -- the same vectors the pooled slot carries,
        so with ``--st-slot-mode cent_cmean`` the partition is over content-only, row-centred
        vectors.
    :param doc_valid: ``(B, n_docs)`` bool -- documents actually present in the row. A document
        that is not present stays ``True`` (conservative: never pool something we did not resolve).
    :param mode: ``"smallcat_keep"`` (gold-blind) or ``"gold_plus_wholecats"`` (gold-aware).
    :param gold_docs: ``(B, n_docs)`` bool gold mask; **required** for ``"gold_plus_wholecats"``,
        ignored otherwise. This is what the gold-sidecar forward pre-hook supplies.
    :param n_small: ``smallcat_keep``: how many of the SMALLEST categories to keep whole
        (``--st-keep-smallcat``).
    :param n_decoy: ``smallcat_keep``: how many additional LARGE categories to keep whole as decoys
        (``--st-keep-decoy-cats``), so "kept whole" does not by itself mean "small category".
    :param decoy_max_size_mult: A decoy category is only eligible if its size is at most this
        multiple of the largest kept small category. **A decoy is a real FLOP cost** -- it puts a
        whole category's tokens back in the forward -- and the single-linkage rule frequently
        leaves exactly one giant category, which this cap then makes ineligible. When nothing is
        eligible the row simply keeps no decoy and ``stats["decoy_skipped"]`` counts it; raise the
        multiplier to buy decoys with compute. (The alternative -- keeping only the first ``k``
        tokens of a large category's documents -- was rejected because it is a PARTIAL category,
        which is precisely the signature these policies exist to destroy.)
    :param n_cats: ``gold_plus_wholecats``: how many NON-gold categories to keep whole
        (``--st-keep-cats``).
    :param cats_random: ``gold_plus_wholecats``: choose those categories at random instead of
        taking the smallest (``--st-keep-cats-random``).
    :param threshold_rule: See :func:`cosine_categories` -- ``"gap"`` by default.
    :param threshold_lambda: See :func:`cosine_categories` (``"mean_std"`` rule only).
    :param seed: Base seed for the random choices.
    :param call: Forward index, so a row's decoy/random categories vary across epochs while staying
        deterministic given the data order.
    :param doc_counts: ``(B, n_docs)`` token counts per document; only used to report the fraction
        of context tokens that stay REAL, which is the FLOP cost of the policy.

    :returns: ``((B, n_docs) bool keep mask, stats dict)``. The stats carry ``n_cats_mean``,
        ``kept_sizes`` (a sample), ``real_token_frac``, ``decoy_skipped`` and
        ``saturated_rows`` -- everything needed to see what the rule did and what it cost.

    :raises ValueError: On an unknown ``mode``, or ``"gold_plus_wholecats"`` without ``gold_docs``.
    """
    if mode not in CATEGORY_KEEP_MODES:
        raise ValueError(
            f"resolve_category_keep: unknown mode {mode!r} (expected one of {CATEGORY_KEEP_MODES})"
        )
    if mode == "gold_plus_wholecats" and gold_docs is None:
        raise ValueError(
            "resolve_category_keep: mode 'gold_plus_wholecats' needs gold_docs -- install the "
            "gold-sidecar keep hook (make_fingerprint_keep_docs_fn) so the gold mask reaches here."
        )
    B, n_docs = doc_valid.shape
    keep = torch.ones(B, n_docs, dtype=torch.bool, device=doc_valid.device)
    n_cats_seen: List[int] = []
    kept_sizes: List[List[int]] = []
    decoy_skipped = 0
    saturated = 0
    real_tok = 0.0
    all_tok = 0.0
    for b in range(B):
        present = torch.nonzero(doc_valid[b], as_tuple=False).flatten()
        if present.numel() == 0:
            continue
        labels = cosine_categories(
            doc_vecs[b, present],
            threshold_rule=threshold_rule,
            threshold_lambda=threshold_lambda,
        )
        groups = _groups(labels)  # ascending (size, first member), indices into `present`
        n_cats_seen.append(len(groups))
        row_sig = int(present.numel()) * 1000003 + int(present[0]) + b
        rng = _row_rng(seed, call, row_sig)

        # At least one category must stay POOLED. Without this guard a row whose documents fall
        # into few categories keeps every one of them and the arm silently becomes dense -- with a
        # normal-looking loss curve and a FLOP meter that only shows it after the fact.
        budget = max(0, len(groups) - 1)
        if mode == "smallcat_keep":
            chosen = list(range(min(max(0, n_small), budget)))
            largest_small = max((len(groups[i]) for i in chosen), default=1)
            cap = max(1.0, decoy_max_size_mult * largest_small)
            eligible = [i for i in range(len(groups)) if i not in chosen and len(groups[i]) <= cap]
            rng.shuffle(eligible)
            picked = eligible[: max(0, min(n_decoy, budget - len(chosen)))]
            decoy_skipped += max(0, n_decoy) - len(picked)
            chosen += picked
        else:  # gold_plus_wholecats
            gold_local = {
                int(j)
                for j, d in enumerate(present.tolist())
                if bool(gold_docs[b, d])  # type: ignore[index]
            }
            gold_cats = [i for i, g in enumerate(groups) if gold_local & set(g)]
            others = [i for i in range(len(groups)) if i not in gold_cats]
            if cats_random:
                rng.shuffle(others)
            # `groups` is already sorted by (size, first member), so the default takes the SMALLEST
            # non-gold categories -- the ones a "which category is odd?" reader must rule out.
            chosen = gold_cats + others[: max(0, min(n_cats, budget - len(gold_cats)))]

        keep_local = torch.zeros(present.numel(), dtype=torch.bool, device=doc_valid.device)
        for i in chosen:
            keep_local[torch.tensor(groups[i], device=doc_valid.device)] = True
        keep[b, present] = keep_local
        if len(chosen) >= len(groups):
            saturated += 1
        kept_sizes.append(sorted(len(groups[i]) for i in chosen))
        if doc_counts is not None:
            c = doc_counts[b, present].float()
            all_tok += float(c.sum())
            real_tok += float(c[keep_local].sum())

    stats: Dict[str, object] = {
        "mode": mode,
        "rows": B,
        "n_cats_mean": (sum(n_cats_seen) / len(n_cats_seen)) if n_cats_seen else 0.0,
        "kept_sizes": kept_sizes[:4],
        "kept_docs_mean": (sum(sum(s) for s in kept_sizes) / len(kept_sizes)) if kept_sizes else 0.0,
        "decoy_skipped": decoy_skipped,
        # rows where every category ended up kept -- i.e. the row trained DENSE. A nonzero value
        # means the clustering found too little structure for the requested C/K.
        "saturated_rows": saturated,
        "real_token_frac": (real_tok / all_tok) if all_tok else None,
    }
    return keep, stats
