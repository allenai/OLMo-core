"""
Scoring primitives: a parsed answer plus a gold answer -> a number.

Two families, and they are not interchangeable:

* **QA metrics** (:func:`exact_match`, :func:`substring_match`, :func:`token_f1`) compare answer
  *text*, under HELMET-compatible normalization so our numbers are comparable with theirs.
* **Retrieval metrics** (:func:`retrieval_f1` and friends) compare *sets of document ids*.

Parsing lives in :mod:`ctc.format.parsing`, not here. That split matters: when a task scores near
zero the first question is always "did it parse?", and keeping the two apart makes that answerable
without re-running the model.

.. note::
   ``retrieval_f1`` is a **per-example** F1 over one example's id set, then averaged across the
   eval set by :func:`aggregate`. It is not a corpus-level F1, and the two differ. When quoting a
   number, also quote ``eval_size`` and its standard error -- at 488 examples the binomial SE is
   about ±0.021 at f1≈0.70 and ±0.010 at f1≈0.95, which is larger than many differences that have
   been reported as findings.

Ported from ``corpus_reasoning/lib/metrics.py``.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Sequence, Set, Union

__all__ = [
    "normalize_answer",
    "exact_match",
    "substring_match",
    "token_f1",
    "max_over_answers",
    "retrieval_exact_match",
    "retrieval_recall",
    "retrieval_precision",
    "retrieval_f1",
    "pair_metrics",
    "cycle_metrics",
    "set_metrics",
    "pairwise_metrics",
    "kendall_tau",
    "clustering_extras",
    "ordering_extras",
    "aggregate",
]


def normalize_answer(s: str) -> str:
    """
    Lowercase, strip articles and punctuation, and collapse whitespace.

    HELMET-compatible, deliberately -- changing it silently shifts every QA number and breaks
    comparability with previously reported results.

    :param s: Raw answer text.

    :returns: The normalized form used by every QA metric here.
    """
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    """
    :param pred: Predicted answer text.
    :param gold: Gold answer text.

    :returns: Whether the two match after normalization.
    """
    return normalize_answer(pred) == normalize_answer(gold)


def substring_match(pred: str, gold: str) -> bool:
    """
    :param pred: Predicted answer text.
    :param gold: Gold answer text.

    :returns: Whether the normalized gold appears anywhere in the normalized prediction. Lenient by
        design -- a rambling generation that contains the answer still counts.
    """
    return normalize_answer(gold) in normalize_answer(pred)


def token_f1(pred: str, gold: str) -> float:
    """
    Standard SQuAD-style token-overlap F1.

    :param pred: Predicted answer text.
    :param gold: Gold answer text.

    :returns: Harmonic mean of token precision and recall, ``0.0`` if they share no tokens.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    # Counter intersection takes the minimum count of each shared token, so repeating a word does
    # not earn credit more than once.
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_over_answers(
    metric_fn: Callable[[str, str], Union[bool, float]],
    prediction: str,
    answers: Union[str, Sequence],
) -> Union[bool, float]:
    """
    Score against every acceptable gold answer and keep the best.

    Many datasets list several valid answers (KILT supplies entity aliases), and SQuAD, HELMET and
    KILT all score this way.

    :param metric_fn: Any of the QA metrics above.
    :param prediction: Predicted answer text.
    :param answers: One answer, a list of answers, or a list-of-lists (some datasets add a wrapping
        layer).

    :returns: The best score across the gold answers.
    """
    if isinstance(answers, str):
        answers = [answers]
    elif answers and isinstance(answers[0], list):
        answers = [a for sublist in answers for a in sublist]
    return max(metric_fn(prediction, gt) for gt in answers)


def retrieval_exact_match(predicted_ids: Set[int], gold_ids: Set[int]) -> bool:
    """
    :param predicted_ids: Ids the model named.
    :param gold_ids: Gold ids.

    :returns: Whether the sets are equal. All-or-nothing; use :func:`retrieval_f1` for partial credit.
    """
    return predicted_ids == gold_ids


def retrieval_recall(predicted_ids: Set[int], gold_ids: Set[int]) -> float:
    """
    :param predicted_ids: Ids the model named.
    :param gold_ids: Gold ids. An empty gold set scores ``1.0`` -- there was nothing to miss.

    :returns: Fraction of gold documents retrieved.
    """
    if not gold_ids:
        return 1.0
    return len(predicted_ids & gold_ids) / len(gold_ids)


def retrieval_precision(predicted_ids: Set[int], gold_ids: Set[int]) -> float:
    """
    :param predicted_ids: Ids the model named. Naming none scores ``0.0``, so abstaining is not a
        way to score well.
    :param gold_ids: Gold ids.

    :returns: Fraction of named documents that are gold.
    """
    if not predicted_ids:
        return 0.0
    return len(predicted_ids & gold_ids) / len(predicted_ids)


def retrieval_f1(predicted_ids: Set[int], gold_ids: Set[int]) -> float:
    """
    :param predicted_ids: Ids the model named.
    :param gold_ids: Gold ids.

    :returns: Harmonic mean of :func:`retrieval_precision` and :func:`retrieval_recall`, for this
        one example. See the module note on per-example vs corpus-level F1.
    """
    p = retrieval_precision(predicted_ids, gold_ids)
    r = retrieval_recall(predicted_ids, gold_ids)
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)


def pair_metrics(
    predicted: Sequence[Sequence[int]], gold: Sequence[Sequence[int]]
) -> Dict[str, float]:
    """
    Set-level precision / recall / F1 / exact-match over pairs.

    Shared by every pair task: contradiction, redundancy, mathmatch, matching_ngram, strmatch.

    :param predicted: Parsed pairs. For unordered tasks these arrive already sorted from
        :func:`~ctc.format.parsing.parse_pairs`; passing unsorted pairs would make ``[4, 1]`` miss
        a gold ``[1, 4]``.
    :param gold: Gold pairs, under the same ordering convention.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``. Predicting nothing when there is
        nothing to find scores a perfect 1.0 -- correct, but it means a task whose examples mostly
        have no pairs can be gamed by always answering ``[]``, so check the positive rate before
        reading much into a high score.
    """
    pred_set = {tuple(p) for p in predicted}
    gold_set = {tuple(p) for p in gold}
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "exact_match": float(pred_set == gold_set)}


def cycle_metrics(
    predicted: Sequence[Sequence[int]], gold: Sequence[Sequence[int]]
) -> Dict[str, float]:
    """
    Set-of-sets scoring for the cycle family: cycle, groups4, textgroups.

    A predicted cycle counts as a true positive only if its id-set **exactly** equals a gold
    cycle's. That is deliberately strict -- a cycle with one member wrong is not a cycle -- so a
    softer ``claim_f1`` over the union of all cycle members is reported alongside it, giving partial
    credit for finding most of a cycle's items.

    Report both. Cycle-level F1 alone hides a model that is finding the right *items* but grouping
    them wrongly, which is a different failure from not finding them at all.

    :param predicted: Parsed cycles, each a sorted id list.
    :param gold: Gold cycles.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match`` (all cycle-level), and
        ``claim_f1`` (item-level).
    """
    pred_set = {frozenset(c) for c in predicted}
    gold_set = {frozenset(c) for c in gold}
    if not pred_set and not gold_set:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_match": 1.0,
            "claim_f1": 1.0,
        }
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    pred_ids = {i for c in predicted for i in c}
    gold_ids = {i for c in gold for i in c}
    ctp = len(pred_ids & gold_ids)
    cp = ctp / len(pred_ids) if pred_ids else 0.0
    cr = ctp / len(gold_ids) if gold_ids else 0.0
    claim_f1 = (2 * cp * cr / (cp + cr)) if (cp + cr) > 0 else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "exact_match": float(pred_set == gold_set),
        "claim_f1": claim_f1,
    }


def set_metrics(predicted: Set, gold: Set) -> Dict[str, float]:
    """
    Precision / recall / F1 / exact-match over two flat sets.

    Used by the absence family, where the answer is a set of item ids (or of normalized text
    snippets, for the Gutenberg variant -- the metric does not care which, as long as both sides
    were normalized the same way).

    :param predicted: What the model named.
    :param gold: The gold set.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``. Two empty sets score a perfect
        1.0, which is correct but means a task whose examples mostly have nothing missing can be
        gamed by always answering empty -- check the positive rate before reading much into a high
        score.
    """
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "exact_match": float(predicted == gold)}


def pairwise_metrics(pred_labels: Sequence[int], gold_labels: Sequence[int]) -> Dict[str, float]:
    """
    Pairwise clustering metrics for the grouping tasks.

    Compares which *pairs* of documents were put together, rather than comparing cluster labels --
    which is necessary because cluster identity is arbitrary: a model that finds exactly the right
    grouping but numbers the clusters differently is completely correct.

    Pure Python on purpose. The pre-migration scorer also reported ARI and NMI via sklearn, but the
    headline metric is this one, and keeping it dependency-free is what lets :mod:`ctc.format` be
    imported without sklearn. ARI/NMI live behind an optional extra.

    :param pred_labels: Predicted cluster label per document.
    :param gold_labels: Gold cluster label per document.

    :returns: ``pairwise_precision``, ``pairwise_recall``, ``pairwise_f1``.

    :raises ValueError: If the label arrays differ in length.
    """
    if len(pred_labels) != len(gold_labels):
        raise ValueError(
            f"label arrays differ in length ({len(pred_labels)} vs {len(gold_labels)}); "
            "both must cover every document"
        )
    n = len(pred_labels)
    pred_pairs = {(i, j) for i, j in combinations(range(n), 2) if pred_labels[i] == pred_labels[j]}
    gold_pairs = {(i, j) for i, j in combinations(range(n), 2) if gold_labels[i] == gold_labels[j]}
    tp = len(pred_pairs & gold_pairs)
    p = tp / len(pred_pairs) if pred_pairs else 0.0
    r = tp / len(gold_pairs) if gold_pairs else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"pairwise_precision": p, "pairwise_recall": r, "pairwise_f1": f1}


def kendall_tau(pred: Sequence[int], gold: Sequence[int]) -> float:
    """
    Kendall's tau between two orderings, for the reorder task.

    Both arguments are permutations, so there are no ties and tau-b reduces to tau-a -- a plain
    concordant-minus-discordant count over pairs. That equivalence is why this can be pure Python
    instead of pulling in scipy, and it is pinned against ``scipy.stats.kendalltau`` in the golden
    fixture rather than merely asserted here.

    :param pred: The predicted ordering.
    :param gold: The true ordering.

    :returns: Tau in ``[-1, 1]``; ``1.0`` for identical orderings, ``-1.0`` for exact reversal, and
        ``0.0`` for a sequence too short to have any pair.

    :raises ValueError: If the two differ in length.
    """
    if len(pred) != len(gold):
        raise ValueError(f"orderings differ in length ({len(pred)} vs {len(gold)})")
    n = len(pred)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i, j in combinations(range(n), 2):
        a = (pred[i] - pred[j]) * (gold[i] - gold[j])
        if a > 0:
            concordant += 1
        elif a < 0:
            discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def clustering_extras(pred_labels: Sequence[int], gold_labels: Sequence[int]) -> Dict[str, float]:
    """
    ARI and NMI for the grouping tasks, when scikit-learn is available.

    Secondary to :func:`pairwise_metrics`, which is the headline number. Kept separate and lazily
    imported so :mod:`ctc.format` stays importable on a bare install -- the data-generation side
    depends on this module and has no reason to carry scikit-learn.

    :param pred_labels: Predicted cluster label per document.
    :param gold_labels: Gold cluster label per document.

    :returns: ``{"ari": ..., "nmi": ...}``, or an empty dict when scikit-learn is absent. Empty
        rather than zeros -- a missing metric must not be recorded as a score of zero.
    """
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        return {}
    return {
        "ari": float(adjusted_rand_score(gold_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(gold_labels, pred_labels)),
    }


def ordering_extras(pred: Sequence[int], gold: Sequence[int]) -> Dict[str, float]:
    """
    Spearman's rho for the reorder task, when scipy is available.

    :param pred: The predicted ordering.
    :param gold: The true ordering.

    :returns: ``{"spearman_rho": ...}``, or an empty dict when scipy is absent -- never a zero,
        which would read as a real measurement.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return {}
    if len(pred) < 2:
        return {}
    return {"spearman_rho": float(spearmanr(pred, gold).correlation)}


def aggregate(results: List[Dict], keys: Iterable[str]) -> Dict[str, float]:
    """
    Average per-example metric values across an eval set.

    :param results: Per-example result dicts.
    :param keys: Metric names to average. Every result must carry every key -- a missing one raises
        rather than being skipped, since silently averaging over a subset misreports ``eval_size``.

    :returns: Mean value per key, or an empty dict when there are no results.
    """
    if not results:
        return {}
    return {k: sum(r[k] for r in results) / len(results) for k in keys}
