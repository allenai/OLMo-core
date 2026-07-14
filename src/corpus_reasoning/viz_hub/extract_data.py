"""
Stage 1 — Data Explorer source.

Samples a few real examples from each corpus-reasoning task at several points
along its context-length (or item-count) ladder, truncates them for display, and
writes ``outputs/data_examples.json`` for the renderer.

Reads only the first ``examples_per_length`` lines of each file, so it is cheap
even for the multi-megabyte long-context files.

Usage:
    python -m viz.extract_data
    CR_DATA_ROOT=/path/to/data python viz/extract_data.py
"""

import glob
import json
import os
import re

try:
    from . import config
except ImportError:  # run as a plain script
    import config

# How many example records to embed per ladder rung.
EXAMPLES_PER_LENGTH = 2
# For files larger than this (bytes), embed only one example (keeps HTML small).
BIG_FILE_BYTES = 6_000_000
# Truncation budget for each document/query/answer string shown in the UI.
DOC_PREVIEW_CHARS = 600
MAX_DOCS_SHOWN = 3
QUERY_PREVIEW_CHARS = 700
ANSWER_PREVIEW_CHARS = 300


# Each task: a glob over the eval/validation split, plus a regex that pulls the
# ladder value out of the filename. ``length_kind`` is "tokens" when the ladder
# value is a target context length, or "items" when it is a document/query/passage
# count (the O(N^k) toy tasks scale by item count, not raw tokens).
TASK_MANIFEST = [
    # ----- O(N): "scales fine with long context" -----------------------------
    {
        "key": "oolong",
        "title": "OOLONG (distributional aggregation)",
        "order": "O(N)",
        "substrate": "synthetic classification items",
        "description": "Aggregate reasoning over a labelled stream (count / timeline / "
        "per-user analysis). One built-in ladder from 1K to 4.2M tokens — the cleanest "
        "pure context-length sweep in the suite.",
        "glob": "oolong_validation_synth_ctx*.jsonl",
        "length_re": r"ctx(\d+)\.jsonl$",
        "length_kind": "tokens",
    },
    {
        "key": "narrativeqa",
        "title": "HELMET NarrativeQA (long-form QA)",
        "order": "O(N)",
        "substrate": "books / film scripts",
        "description": "Answer comprehension questions over a full book or screenplay. "
        "Real (non-synthetic) long documents at 8K / 32K / 131K tokens.",
        "glob": "helmet_narrativeqa_validation_L*.jsonl",
        "length_re": r"_L(\d+)\.jsonl$",
        "length_kind": "tokens",
    },
    {
        "key": "govreport",
        "title": "HELMET GovReport (summarization)",
        "order": "O(N)",
        "substrate": "government reports",
        "description": "Abstractive summarization of long government reports at 16K / 64K tokens.",
        "glob": "helmet_summ_govreport_validation_L*.jsonl",
        "length_re": r"_L(\d+)\.jsonl$",
        "length_kind": "tokens",
    },
    {
        "key": "ruler_niah",
        "title": "RULER NIAH (needle-in-a-haystack)",
        "order": "O(N)",
        "substrate": "synthetic haystack",
        "description": "Retrieve planted key/value needles from a synthetic haystack across a "
        "1K -> 32K+ token ladder. The canonical long-context retrieval probe.",
        "glob": "ruler_niah_single_L*_eval.jsonl",
        "length_re": r"_L(\d+)_eval\.jsonl$",
        "length_kind": "tokens",
    },
    {
        "key": "beir_scifact",
        "title": "BEIR SciFact (claim retrieval)",
        "order": "O(N)",
        "substrate": "scientific claims",
        "description": "Retrieve the abstracts supporting a scientific claim from a growing "
        "candidate pool (k = 20 / 50 / 100 passages).",
        "glob": "beir_scifact_test_k*.jsonl",
        "length_re": r"_k(\d+)_\d+\.jsonl$",
        "length_kind": "items",
    },
    # ----- O(N^2): "show the reasoning gap" ----------------------------------
    {
        "key": "absence",
        "title": "Absence (find what was removed)",
        "order": "O(N^2)",
        "substrate": "PubMed sentences",
        "description": "Given an original corpus and a perturbed copy, identify which items were "
        "deleted. Difficulty scales super-linearly with item count (n = 20 -> 1000).",
        "glob": "absence_eval_pubmed_n*_p01.jsonl",
        "length_re": r"_n(\d+)_p01\.jsonl$",
        "length_kind": "items",
    },
    {
        "key": "contradiction",
        "title": "Contradiction (find conflicting pairs)",
        "order": "O(N^2)",
        "substrate": "PubMed + SNLI claims",
        "description": "Find every pair of mutually contradictory claims in the corpus. All-pairs "
        "comparison — O(N^2) in the number of claims (n = 12 -> 1000).",
        "glob": "contradiction_eval_pubmed_both_n*_k3.jsonl",
        "length_re": r"_n(\d+)_k3\.jsonl$",
        "length_kind": "items",
    },
    {
        "key": "strmatch",
        "title": "String matching (shared runs)",
        "order": "O(N^2)",
        "substrate": "synthetic word lists",
        "description": "Find all pairs of strings sharing a run of >=3 consecutive words. "
        "Synthetic vocabulary (tokenizes ~2-3x its char count).",
        "glob": "strmatch_eval_n*_k3_sub3_l10_h3.jsonl",
        "length_re": r"_n(\d+)_k3",
        "length_kind": "items",
    },
    # ----- O(N^3+): "intractable in the extreme" -----------------------------
    {
        "key": "cycle",
        "title": "Cycle detection (comparative claims)",
        "order": "O(N^3+)",
        "substrate": "synthetic comparisons",
        "description": "Find sets of strict-comparison claims that form a logical cycle. "
        "Naively O(N^L) in the cycle length L (n = 20 -> 500).",
        "glob": "cycle_eval_n*_len3_k1.jsonl",
        "length_re": r"_n(\d+)_len3",
        "length_kind": "items",
    },
    {
        "key": "groups4",
        "title": "Groups-of-4 (tight numeric clusters)",
        "order": "O(N^3+)",
        "substrate": "math expressions",
        "description": "Identify the tight 4-element cluster among many numeric expressions. "
        "Combinatorial in the item count.",
        "glob": "groups4_eval_n*_g4_k1_x5.jsonl",
        "length_re": r"_n(\d+)_g4",
        "length_kind": "items",
    },
]


def _approx_tokens_from_text(total_chars: int) -> int:
    """Rough token estimate (chars/4). Synthetic-vocab tasks run higher; see task docs."""
    return round(total_chars / 4)


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n].rstrip() + " …"


def build_example(record: dict) -> dict:
    """Turn one raw task record into a compact, display-ready example."""
    docs = record.get("documents", []) or []
    queries = record.get("queries", []) or []
    answers = record.get("answers", []) or []

    # Full-corpus char count for the token estimate (before truncation).
    total_chars = 0
    for d in docs:
        total_chars += len(d.get("text", "") or "")
        total_chars += len(d.get("title", "") or "")
    for q in queries:
        total_chars += len(q or "")

    doc_previews = []
    for d in docs[:MAX_DOCS_SHOWN]:
        doc_previews.append(
            {
                "title": _truncate(d.get("title"), 120) if d.get("title") else None,
                "text": _truncate(d.get("text", ""), DOC_PREVIEW_CHARS),
            }
        )

    return {
        "source": record.get("source", ""),
        "n_docs": len(docs),
        "n_docs_shown": len(doc_previews),
        "approx_tokens": _approx_tokens_from_text(total_chars),
        "documents": doc_previews,
        "queries": [_truncate(q, QUERY_PREVIEW_CHARS) for q in queries[:2]],
        "answers": [_truncate(json.dumps(a) if not isinstance(a, str) else a, ANSWER_PREVIEW_CHARS)
                    for a in answers[:4]],
        "gold_doc_indices": record.get("gold_doc_indices", []),
        "meta": record.get("_meta", {}),
    }


def read_first_records(path: str, n: int) -> list:
    out = []
    with open(path) as fh:
        for _ in range(n):
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def collect_task(task: dict, data_root: str) -> dict:
    pattern = os.path.join(data_root, task["glob"])
    files = sorted(glob.glob(pattern))
    rungs = []
    for f in files:
        m = re.search(task["length_re"], os.path.basename(f))
        if not m:
            continue
        ladder_val = int(m.group(1))
        n_examples = 1 if os.path.getsize(f) > BIG_FILE_BYTES else EXAMPLES_PER_LENGTH
        records = read_first_records(f, n_examples)
        if not records:
            continue
        examples = [build_example(r) for r in records]
        # Prefer the measured token estimate from the first example; fall back to
        # the ladder value when it already denotes tokens.
        measured = examples[0]["approx_tokens"]
        rungs.append(
            {
                "ladder_value": ladder_val,
                "length_kind": task["length_kind"],
                "label": _rung_label(ladder_val, task["length_kind"]),
                "file": os.path.basename(f),
                "approx_tokens": measured,
                "examples": examples,
            }
        )
    rungs.sort(key=lambda r: r["ladder_value"])
    return {
        "key": task["key"],
        "title": task["title"],
        "order": task["order"],
        "substrate": task["substrate"],
        "description": task["description"],
        "rungs": rungs,
    }


def _rung_label(value: int, kind: str) -> str:
    if kind == "tokens":
        if value >= 1_000_000:
            return f"{value // 1_000_000}M ctx"
        if value >= 1000:
            return f"{value // 1000}K ctx"
        return f"{value} ctx"
    return f"n={value}"


def main():
    config.ensure_out_dir()
    data_root = config.CR_DATA_ROOT
    print(f"[extract_data] reading task data from {data_root}")
    if not os.path.isdir(data_root):
        print(f"[extract_data] WARNING: data root does not exist: {data_root}")

    tasks = []
    for task in TASK_MANIFEST:
        t = collect_task(task, data_root)
        n_rungs = len(t["rungs"])
        n_ex = sum(len(r["examples"]) for r in t["rungs"])
        print(f"  {task['key']:14s} {n_rungs:2d} rungs, {n_ex:2d} examples")
        if n_rungs:
            tasks.append(t)

    # Overview stats for the landing view.
    all_token_rungs = [
        r["approx_tokens"]
        for t in tasks
        for r in t["rungs"]
        if r["length_kind"] == "tokens"
    ]
    stats = {
        "n_tasks": len(tasks),
        "n_rungs": sum(len(t["rungs"]) for t in tasks),
        "max_ctx_tokens": max(all_token_rungs) if all_token_rungs else 0,
        "orders": sorted({t["order"] for t in tasks}),
        "data_root": data_root,
    }

    payload = {"tasks": tasks, "stats": stats}
    with open(config.DATA_EXAMPLES_JSON, "w") as f:
        json.dump(payload, f)
    print(f"[extract_data] wrote {config.DATA_EXAMPLES_JSON} "
          f"({stats['n_tasks']} tasks, {stats['n_rungs']} rungs)")


if __name__ == "__main__":
    main()
