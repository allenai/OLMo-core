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
    "parse_pairs",
    "parse_qd_pairs",
    "parse_cycles",
    "parse_id_set",
    "parse_snippet_list",
    "normalize_snippet",
    "ID_SET_ANCHORS",
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


# ── Pairs ───────────────────────────────────────────────────────────────────────────────────────
#
# Shared by every pair task -- contradiction, redundancy, mathmatch, matching_ngram, strmatch. One
# definition on purpose: five copies of a parser this fiddly is how the copies drift apart.


def parse_pairs(text: str) -> Optional[List[List[int]]]:
    """
    Extract unordered integer pairs, e.g. ``[[1, 4], [3, 7]]``.

    Each pair is **sorted**, because "1 contradicts 4" and "4 contradicts 1" are the same claim.
    Use :func:`parse_qd_pairs` where the two positions mean different things.

    The prompt primes the answer with ``[[``, so a generation is usually the completion with the
    opening brackets missing (``1, 37], [6, 60]]``). Requiring the leading ``[`` dropped the FIRST
    pair of every such example: contradiction exact-match read ~0.60 while the model was emitting
    the correct pairs and true EM was above 0.9. Bracket-reconstructed variants are therefore also
    tried, keeping whichever parse yields the most pairs.

    :param text: Raw model generation.

    :returns: The pairs; ``[]`` for an explicitly empty answer; ``None`` when nothing parsed.
        ``[]`` and ``None`` must not be conflated -- "the model said there are none" and "the model
        produced nothing usable" score the same on this task but mean opposite things about it.
    """
    text = text.strip()
    candidates = [text]
    if text[:1].isdigit():
        candidates.append("[[" + text)  # primed '[[' dropped, generation starts at a digit
    elif text.startswith("[") and not text.startswith("[["):
        candidates.append("[" + text)  # primed outer '[' dropped, starts at '[digit'

    best: Optional[List[List[int]]] = None
    for s in candidates:
        for candidate in [s, re.search(r"\[\[[\s\S]*\]\]", s) or re.search(r"\[[\s\S]*\]", s)]:
            if candidate is None:
                continue
            frag = candidate if isinstance(candidate, str) else candidate.group()
            try:
                parsed = json.loads(frag)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if isinstance(parsed, list):
                pairs = [
                    sorted([int(p[0]), int(p[1])])
                    for p in parsed
                    if isinstance(p, list) and len(p) == 2
                ]
                if best is None or len(pairs) > len(best):
                    best = pairs
    if best:
        return best

    matches = re.findall(r"[\[\(]?\s*(\d+)\s*,\s*(\d+)\s*[\]\)]", text)
    if matches:
        return [sorted([int(a), int(b)]) for a, b in matches]
    return [] if text in ("[]", "") else None


def parse_qd_pairs(text: str) -> Optional[List[List[int]]]:
    """
    Extract **order-preserving** pairs, for ``qdmatch``.

    Identical to :func:`parse_pairs` except that pairs are not sorted: a qdmatch pair is
    ``(query_id, document_id)`` over one shared index, so swapping the two makes a different claim.
    The regex fallback also requires a real opening bracket here, since without sorting there is no
    way to recover from a mis-split.

    :param text: Raw model generation.

    :returns: The pairs in the order given, ``[]`` for an empty answer, or ``None`` on failure.
    """
    text = text.strip()
    for candidate in [text, re.search(r"\[[\s\S]*\]", text)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [
                    [int(p[0]), int(p[1])] for p in parsed if isinstance(p, list) and len(p) == 2
                ]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    matches = re.findall(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]", text)
    if matches:
        return [[int(a), int(b)] for a, b in matches]
    return [] if text in ("[]", "") else None


# ── Id sets (absence family) ────────────────────────────────────────────────────────────────────

#: Answer-line anchors for the absence family. The **last** occurrence wins, so a model that
#: reasons aloud and revises itself is scored on its final answer rather than its first thought.
ID_SET_ANCHORS = ("Missing:", "Unmatched:")


def parse_id_set(text: str, n_docs: int) -> Optional[Set[int]]:
    """
    Extract a set of 1-indexed ids from an absence-style answer.

    Unlike :func:`parse_pairs`, ids **are** range-filtered to ``1..n_docs``. The difference is
    deliberate: this parser falls back to scraping bare integers when no bracketed ids are present,
    and without the range check that fallback would pick up any number in the surrounding prose.

    :param text: Raw model generation, e.g. ``"Missing: [3], [7]"``.
    :param n_docs: Corpus size, bounding valid ids.

    :returns: The ids; an empty set for an explicitly empty answer; ``None`` when nothing parsed.
        Empty and ``None`` differ -- "nothing is missing" is a real answer and can be correct.
    """
    for anchor in ID_SET_ANCHORS:
        if anchor in text:
            text = text.rsplit(anchor, 1)[1]
            break
    ids = re.findall(r"\[(\d+)\]", text)
    if not ids:
        ids = re.findall(r"\b(\d+)\b", text)
    out = {int(x) for x in ids if 1 <= int(x) <= n_docs}
    return out if (ids or text.strip() in ("", "Missing:", "[]")) else None


def normalize_snippet(s: str) -> str:
    """
    Normalize a first-four-words snippet for matching.

    :param s: A snippet.

    :returns: Lowercased, punctuation-stripped, whitespace-collapsed, truncated to four tokens --
        so a model that quotes slightly more or less of a sentence still matches.
    """
    s = re.sub(r"\s+", " ", str(s)).strip().strip("\"'").lower()
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return " ".join(s.split()[:4])


def parse_snippet_list(text: str) -> Optional[List[str]]:
    """
    Extract the JSON list of snippets used by the Gutenberg text-diff absence variant.

    The opening bracket is treated as optional, for the same reason as in :func:`parse_doc_ids`:
    this task's target is a bare JSON array with no natural-language lead-in, and some checkpoints
    -- notably under document-chunked attention -- drop the leading ``[`` (sometimes the whole
    ``["``) and start directly at the first snippet. Requiring it scored every such response as a
    parse failure: absence_gutenberg under chunked attention read ``parse_rate ~0.01`` while the
    snippets it emitted were coherent and correct.

    :param text: Raw model generation.

    :returns: The snippets, or ``None`` when nothing parsed.
    """
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        quoted = re.findall(r'"([^"]*)"', m.group(0))
        if quoted:
            return quoted

    # No bracketed array at all: rebuild one and retry, rather than scoring coherent output as a
    # parse failure.
    if "]" in text and "[" not in text:
        stripped = text.strip()
        rebuilt = ("[" + stripped) if stripped.startswith('"') else ('["' + stripped)
        m2 = re.search(r"\[.*\]", rebuilt, re.DOTALL)
        if m2:
            try:
                arr = json.loads(m2.group(0))
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            quoted = re.findall(r'"([^"]*)"', m2.group(0))
            if quoted:
                return quoted
    return None


# ── Cycles and groups ───────────────────────────────────────────────────────────────────────────
#
# Shared by cycle, groups4 and textgroups. The answer is a set of ID-*sets* of variable size, which
# is what distinguishes it from the pair tasks -- there, every group has exactly two members.


def parse_cycles(text: str) -> Optional[List[List[int]]]:
    """
    Extract a list of cycles/groups, each a list of item ids.

    Accepts a JSON list of lists, a single flat list (read as one cycle), or bracketed integer
    groups scraped from prose. Each cycle is normalized to a **sorted, de-duplicated** list, since
    a cycle is a set: the order the model happened to walk it in carries no information, and
    scoring compares sets.

    Groups of fewer than two ids are dropped -- a single item cannot form a cycle, and admitting
    them would let a model score by listing every id separately.

    :param text: Raw model generation.

    :returns: The cycles, ``[]`` for an explicitly empty answer, or ``None`` when nothing parsed.
    """
    text = text.strip()
    for candidate in [text, re.search(r"\[[\s\S]*\]", text)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, list):
            if parsed and all(isinstance(x, int) for x in parsed):
                return [sorted(set(parsed))]  # a single flat cycle
            out = []
            for c in parsed:
                if isinstance(c, list) and len(c) >= 2:
                    try:
                        out.append(sorted({int(x) for x in c}))
                    except (ValueError, TypeError):
                        pass
            return out

    groups = re.findall(r"\[([\d,\s]+)\]", text)
    if groups:
        out = []
        for g in groups:
            ids = [int(x) for x in re.findall(r"\d+", g)]
            if len(ids) >= 2:
                out.append(sorted(set(ids)))
        return out
    return [] if text in ("[]", "") else None


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
            obj, _ = decoder.raw_decode(text[m.start() :])
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
    next_label = max(labels) + 1 if any(label >= 0 for label in labels) else 0
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
        cand = ints[start : start + n]
        if len(cand) == n and sorted(cand) == list(range(1, n + 1)):
            return cand
    return None
