"""
Answer parsers: model generation -> the structured answer a scorer can grade.

Parsing is where grading bugs live. A parser that is slightly too strict does not raise -- it
returns ``None`` or an empty set, the scorer records a zero, and the run reads as "the model cannot
do this task." Three such bugs reached published numbers in this project, and each fix is preserved
below with the measurement that exposed it:

* :func:`parse_doc_ids` -- required both brackets, so a checkpoint completing a primed ``[`` with
  ``8]`` scored 0. niah collapsed to 0.16 while msmarco (which emits ``[15]``) scored 0.96.
* :func:`parse_partition` -- a greedy ``\\{[\\s\\S]*\\}`` regex spanned first-brace to last-brace
  across trailing ramble, failed to decode, and fell through to a digit scrape that lumped every id
  into one cluster. Roughly halved pairwise F1.
* :func:`parse_partition` again -- responses that begin mid-array because the opening brace landed
  ahead of a truncated ``</think>``. Chunked grouping @2k measured 0.44 broken vs 0.82 recovered.

The rule these share: **a near-zero score is a parser hypothesis until you have read the raw
generations.** That is why ``ctc-eval`` dumps generations by default.

Ported from ``corpus_reasoning/lib/eval_tasks.py`` and the ``parse_doc_ids`` in
``corpus_reasoning/lib/metrics.py``.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Set

__all__ = [
    "parse_doc_ids",
    "parse_outlier_ids",
    "parse_partition",
    "partition_to_labels",
    "parse_permutation",
]


# ── Document ids ────────────────────────────────────────────────────────────────────────────────

def parse_doc_ids(text: str) -> Set[int]:
    """
    Extract document ids from text like ``"[3], [7]"`` or ``"Document [3]"``.

    The opening bracket is **optional**. Eval prompts prime the answer with a trailing ``[``, so
    the model's generation is frequently the completion ``"8]"`` -- a closing bracket only. See the
    module docstring for what requiring both brackets cost.

    :param text: Raw model generation.

    :returns: The set of ids mentioned. Empty if none parse.
    """
    return {int(m) for m in re.findall(r"\[?\s*(\d+)\s*\]", text)}


# ── Outlier ─────────────────────────────────────────────────────────────────────────────────────

def parse_outlier_ids(text: str, n_docs: int) -> Optional[List[int]]:
    """
    Extract 1-indexed outlier document ids, in the order the model listed them.

    Unlike :func:`parse_doc_ids` this requires **both** brackets, and deliberately so: the outlier
    target is a complete ``"Outliers: [1], [3]"`` line rather than a primed continuation, so a bare
    ``[`` never appears at the start of a generation. Relaxing it here would instead start matching
    the bare integers in the preceding reasoning sentence, which names the majority and outlier
    attributes and often contains a star rating.

    :param text: Raw model generation.
    :param n_docs: Corpus size; ids outside ``1..n_docs`` are dropped as hallucinations.

    :returns: Unique ids in first-mention order, or ``None`` if nothing valid parsed.
    """
    if not text:
        return None
    m = re.search(r"Outliers?\s*:\s*(.+)", text, flags=re.IGNORECASE)
    scan = m.group(1) if m else text
    ids = [int(x) for x in re.findall(r"\[(\d+)\]", scan)]
    ids = [i for i in ids if 1 <= i <= n_docs]
    if not ids:
        return None
    seen: Set[int] = set()
    uniq: List[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


# ── Grouping ────────────────────────────────────────────────────────────────────────────────────

def _first_groups_object(text: str) -> Optional[List[List[int]]]:
    """
    Return the clusters from the **first complete** ``{"groups": ...}`` object in ``text``.

    Uses ``raw_decode``, which stops at the end of one object and therefore ignores any trailing
    ramble. A regex spanning first-brace to last-brace does not, and the resulting decode failure
    silently degraded to a digit scrape -- see the module docstring.

    :param text: Text that may contain a grouping object.

    :returns: One list of ids per group, or ``None`` if no such object decodes.
    """
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "groups" in obj:
            out = []
            for g in obj["groups"]:
                ids = g.get("doc_ids") if isinstance(g, dict) else g
                if isinstance(ids, list):
                    out.append([int(x) for x in ids])
            if out:
                return out
    return None


#: Prefixes re-attached to a response that begins mid-array. Longest first: the more specific
#: primer must be tried before the shorter one, or a response missing only ``[{"doc_ids": [`` would
#: decode against ``{"groups": [`` into the wrong shape.
_GROUPING_PRIMERS = ('{"groups": [{"doc_ids": [', '{"groups": [')


def parse_partition(text: str, n_docs: int) -> Optional[List[List[int]]]:
    """
    Extract a partition -- a list of clusters, each a list of 1-indexed document ids.

    Tries, in order: the first complete JSON grouping object; the same after re-attaching a primer
    for responses that begin mid-array; a bare list-of-lists; and finally a per-line digit scrape.
    The scrape is a last resort and is usually wrong -- if it fires often, read the generations.

    :param text: Raw model generation.
    :param n_docs: Corpus size, used to bound ids in the digit-scrape fallback.

    :returns: The clusters, or ``None`` if nothing parsed.
    """
    text = text.strip()
    out = _first_groups_object(text)
    if out is not None:
        return out

    # Primed-continuation recovery. Some checkpoints (notably the chunked -cmix models) place the
    # opening `{"groups": [{"doc_ids": [` ahead of a late `</think>` that the generation truncator
    # splits away, leaving a response that BEGINS mid-array, e.g. `2, 3, 4]}, {"doc_ids": [1, 6]}]}`.
    # Dense responses carry the full object and never reach this branch.
    if not text.startswith("{"):
        for primer in _GROUPING_PRIMERS:
            out = _first_groups_object(primer + text)
            if out is not None:
                return out

    m = re.search(r"\[\s*\[[\s\S]*\]\s*\]", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list) and all(isinstance(g, list) for g in obj):
                return [[int(x) for x in g] for g in obj]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    groups = []
    for line in text.splitlines():
        ids = [int(x) for x in re.findall(r"\d+", line)]
        ids = [i for i in ids if 1 <= i <= n_docs]
        if ids:
            groups.append(ids)
    return groups or None


def partition_to_labels(clusters: List[List[int]], n_docs: int) -> List[int]:
    """
    Convert 1-indexed clusters to a per-document label array, for clustering metrics.

    A document claimed by two clusters keeps its first assignment, and any document the model left
    out becomes its own singleton cluster -- so an under-specified answer is scored as
    under-specified rather than crashing the metric.

    :param clusters: Clusters of 1-indexed document ids.
    :param n_docs: Corpus size; the length of the returned array.

    :returns: A cluster label per document position.
    """
    labels = [-1] * n_docs
    for cid, cluster in enumerate(clusters):
        for d in cluster:
            idx = d - 1
            if 0 <= idx < n_docs and labels[idx] == -1:
                labels[idx] = cid
    next_label = max(labels) + 1 if any(l >= 0 for l in labels) else 0
    for i in range(n_docs):
        if labels[i] == -1:
            labels[i] = next_label
            next_label += 1
    return labels


# ── Reorder ─────────────────────────────────────────────────────────────────────────────────────

def parse_permutation(text: str, n: int) -> Optional[List[int]]:
    """
    Extract a permutation of ``1..n``.

    Prefers a JSON array; falls back to scanning every window of ``n`` consecutive integers in the
    text. Both paths require an exact permutation, so a near-miss returns ``None`` rather than a
    partially-credited answer -- reorder is graded on the whole ordering.

    :param text: Raw model generation.
    :param n: Expected permutation length.

    :returns: The permutation, or ``None`` if none is present.
    """
    for m in re.finditer(r"\[[^\[\]]*\]", text):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
                perm = [int(x) for x in obj]
                if len(perm) == n and sorted(perm) == list(range(1, n + 1)):
                    return perm
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    ints = [int(x) for x in re.findall(r"-?\d+", text)]
    for start in range(len(ints) - n + 1):
        cand = ints[start:start + n]
        if len(cand) == n and sorted(cand) == list(range(1, n + 1)):
            return cand
    return None
