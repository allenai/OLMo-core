"""Shared parsers and small helpers for task-specific evaluation.

These are re-exported from `scripts/eval/evaluate.py` (which defines its own
private copies), and are kept here for offline tooling that needs to parse
saved evaluation outputs (e.g. `scripts/eval/recompute_grouping_rand.py`).
"""

from __future__ import annotations

import json
import re


# ── Outlier ──────────────────────────────────────────────────────────────

def parse_outlier_ids(text: str, n_docs: int) -> list[int] | None:
    """Extract 1-indexed outlier doc IDs from model output."""
    if not text:
        return None
    m = re.search(r"Outliers?\s*:\s*(.+)", text, flags=re.IGNORECASE)
    scan = m.group(1) if m else text
    ids = [int(x) for x in re.findall(r"\[(\d+)\]", scan)]
    ids = [i for i in ids if 1 <= i <= n_docs]
    if not ids:
        return None
    seen, uniq = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


# ── Grouping ─────────────────────────────────────────────────────────────

def _first_groups_object(text: str) -> list[list[int]] | None:
    """Return clusters from the FIRST complete ``{"groups": ...}`` JSON object in ``text``
    (via raw_decode, which stops at the end of one object -> ignores any trailing ramble),
    or None if there is no such object. An earlier greedy ``\\{[\\s\\S]*\\}`` regex spanned
    first-brace..last-brace across the ramble, failed to json-decode, and fell through to a
    digit-scrape fallback that lumped every doc_id on the first line into one giant cluster --
    silently deflating pairwise-F1 by ~0.5."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', text):
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


def parse_partition(text: str, n_docs: int) -> list[list[int]] | None:
    """Extract list of clusters (each a list of 1-indexed doc IDs)."""
    text = text.strip()
    out = _first_groups_object(text)
    if out is not None:
        return out
    # Primed-continuation recovery: some checkpoints (notably the chunked -cmix models) emit the
    # grouping answer such that its opening ``{"groups": [{"doc_ids": [`` is missing -- it lands
    # ahead of a late ``</think>`` that run_vllm_eval's truncate splits away, leaving a response that
    # BEGINS mid-array, e.g. ``2, 3, 4]}, {"doc_ids": [1, 6, 9]}]}``. Re-attach the standard grouping
    # primer and retry the first-object decode. Dense responses carry the full object (handled above)
    # and never reach this branch. Without this, EVERY such response falls to the digit-scrape
    # fallback and mis-grades (measured chunked grouping @2k: 0.44 broken vs 0.82 recovered).
    if not text.startswith('{'):
        for primer in ('{"groups": [{"doc_ids": [', '{"groups": ['):
            out = _first_groups_object(primer + text)
            if out is not None:
                return out
    # Non-JSON fallbacks (unchanged): list-of-lists, then bare ids-per-line.
    m = re.search(r'\[\s*\[[\s\S]*\]\s*\]', text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list) and all(isinstance(g, list) for g in obj):
                return [[int(x) for x in g] for g in obj]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    groups = []
    for line in text.splitlines():
        ids = re.findall(r'\d+', line)
        if ids:
            ids = [int(x) for x in ids if 1 <= int(x) <= n_docs]
            if ids:
                groups.append(ids)
    return groups or None


def partition_to_labels(clusters: list[list[int]], n_docs: int) -> list[int]:
    """Convert 1-indexed clusters to a label array of length n_docs."""
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


# ── Reorder ──────────────────────────────────────────────────────────────

def parse_permutation(text: str, n: int) -> list[int] | None:
    """Extract a permutation of 1..n from model output."""
    for m in re.finditer(r'\[[^\[\]]*\]', text):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
                perm = [int(x) for x in obj]
                if len(perm) == n and sorted(perm) == list(range(1, n + 1)):
                    return perm
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    ints = [int(x) for x in re.findall(r'-?\d+', text)]
    for start in range(len(ints) - n + 1):
        cand = ints[start:start + n]
        if len(cand) == n and sorted(cand) == list(range(1, n + 1)):
            return cand
    return None
