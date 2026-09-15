"""
Rung -> eval-set file map for the v2/v3 multirung ladders.

This is a TRANSCRIPTION of the ``LADDERS`` table built inline in
``src/scripts/ctc_eval/eval/eval_lc_native.py`` (the ``elif args.ladder_version in ("v2", "v3")``
branch, plus the oolong short rungs and the ``--xlong`` glob block below it). It is copied rather
than imported because that table is a local variable inside ``main()``, not an importable symbol.

Copying means it can drift from the source. Two things keep drift loud rather than silent:

* :func:`resolve` reports every rung it cannot find, exactly as the eval's own resolve-check does.
* the caller cross-checks the gold it rebuilds through this map against the gold already recorded
  in the generations sidecar, and refuses to report any task where the two disagree. A stale path
  therefore surfaces as "MISSING rung" or "gold mismatch", never as a wrong number.

``loadtask`` mirrors ``LSPEC`` in the same file: it is the task name the *loader and output
builder* use, which is not always the ladder's own name (nq/fiqa/scifact all load as "retrieval",
outlier_review as "outlier", contra_fever as "contradiction").
"""

from __future__ import annotations

import glob
import os

# ladder task -> the task name load_unified_examples / _build_output expect (LSPEC, eval_lc_native.py)
LOAD_TASK = {
    "contradiction": "contradiction",
    "contra_fever": "contradiction",
    "nq": "retrieval",
    "fiqa": "retrieval",
    "scifact": "retrieval",
    "outlier": "outlier",
    "outlier_review": "outlier",
    "oolong": "oolong",
    "rerank": "rerank",
}


def base_ladders(root: str, ladder_version: str) -> dict[str, list[tuple[str, str]]]:
    """The 2k-32k rungs. ``_CM`` is the one token that separates v2 from v3 (contradiction only)."""
    cm = "realistic" if ladder_version == "v3" else "both"
    return {
        "contradiction": [
            ("2k", f"{root}/contra/contradiction_eval_pubmed_{cm}_n100_k3.jsonl"),
            ("8k", f"{root}/contra/contradiction_eval_pubmed_{cm}_n190_k3.jsonl"),
            ("16k", f"{root}/contra/contradiction_eval_pubmed_{cm}_n385_k3.jsonl"),
            ("32k", f"{root}/contra/contradiction_eval_pubmed_{cm}_n765_k3.jsonl"),
        ],
        "nq": [
            ("3k", f"{root}/nq/nq_validation_k20_600.jsonl"),
            ("8k", f"{root}/nq/nq_validation_k50_600.jsonl"),
            ("16k", f"{root}/nq/nq_validation_k100_600.jsonl"),
            ("32k", f"{root}/nq/nq_validation_k200_600.jsonl"),
        ],
        "outlier": [
            ("3k", f"{root}/outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
            ("8k", f"{root}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
            ("16k", f"{root}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
            ("32k", f"{root}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl"),
        ],
        "rerank": [
            ("3k", f"{root}/rerank/msmarco_trainhn_eval_k20_500.jsonl"),
            ("8k", f"{root}/rerank/msmarco_trainhn_eval_k50_500.jsonl"),
            ("16k", f"{root}/rerank/msmarco_trainhn_eval_k100_500.jsonl"),
        ],
        "oolong": [
            ("8k", f"{root}/oolong/oolong_test_synth_ctx8192_spliteval.jsonl"),
            ("16k", f"{root}/oolong/oolong_test_synth_ctx16384_spliteval.jsonl"),
            ("32k", f"{root}/oolong/oolong_test_synth_ctx32768_spliteval.jsonl"),
        ],
        "fiqa": [
            ("2k", f"{root}/beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
            ("4k", f"{root}/beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
            ("8k", f"{root}/beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
            ("16k", f"{root}/beir/beir_fiqa_ce_ladder_k80_648.jsonl"),
        ],
        "scifact": [
            ("4k", f"{root}/beir/beir_scifact_ladder_k11_299.jsonl"),
            ("8k", f"{root}/beir/beir_scifact_ladder_k22_299.jsonl"),
            ("16k", f"{root}/beir/beir_scifact_ladder_k44_299.jsonl"),
            ("32k", f"{root}/beir/beir_scifact_ladder_k88_299.jsonl"),
        ],
        "outlier_review": [
            ("3k", f"{root}/outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
            ("8k", f"{root}/outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
            ("16k", f"{root}/outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
            ("32k", f"{root}/outlier/outlier_review_matched_n300_k3_eval_600.jsonl"),
        ],
        "contra_fever": [
            ("2k", f"{root}/contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
            ("8k", f"{root}/contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
            ("16k", f"{root}/contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
            ("32k", f"{root}/contra/contradiction_eval_fever_plain_n1642_k3.jsonl"),
        ],
    }


# The ultra-long rungs are resolved by size-labelled GLOB, not a literal path, so the calibrated
# doc count in the filename can drift without editing this table (same reason as the eval's).
XLONG_GLOB = {
    "contradiction": ("contra", "contradiction_eval_pubmed_{cm}_n*_k3_xlong_{s}.jsonl"),
    "contra_fever": ("contra", "contradiction_eval_fever_plain_n*_k3_xlong_{s}.jsonl"),
    "nq": ("nq", "nq_validation_k*_xlong_{s}.jsonl"),
    "outlier": ("outlier", "outlier_wiki100w_n*_k3_eval_xlong_{s}.jsonl"),
    "rerank": ("rerank", "msmarco_trainhn_eval_k*_xlong_{s}.jsonl"),
}
XLONG_SIZES = ("64k", "128k", "256k", "512k", "1M", "2M")
# oolong is a packed item stream labelled by token budget, not a doc pool, so it has its own map.
XLONG_OOLONG = {
    "64k": 65536,
    "128k": 131072,
    "256k": 262144,
    "512k": 524288,
    "1M": 1048576,
    "2M": 2097152,
}
SHORT_OOLONG = (("2k", 2048), ("4k", 4096))


def resolve(root: str, ladder_version: str = "v2", xlong: bool = True):
    """``{(task, rung): path}`` for every rung file that exists, plus the list of missing ones.

    Mirrors the eval's own order of operations: base rungs, oolong short rungs prepended, then the
    opt-in xlong rungs appended where a file is actually present.
    """
    cm = "realistic" if ladder_version == "v3" else "both"
    ladders = base_ladders(root, ladder_version)

    for lab, ctx in SHORT_OOLONG:
        p = os.path.join(root, "oolong", f"oolong_test_synth_ctx{ctx}_spliteval.jsonl")
        if os.path.exists(p):
            ladders["oolong"].insert(0, (lab, p))

    if xlong:
        for task, (sub, pat) in XLONG_GLOB.items():
            if task not in ladders:
                continue
            for s in XLONG_SIZES:
                hits = sorted(glob.glob(os.path.join(root, sub, pat.format(cm=cm, s=s))))
                if hits:
                    ladders[task].append((s, hits[0]))
        for s, ctx in XLONG_OOLONG.items():
            p = os.path.join(root, "oolong", f"oolong_test_synth_ctx{ctx}_spliteval.jsonl")
            if os.path.exists(p):
                ladders["oolong"].append((s, p))

    found, missing = {}, []
    for task, rungs in ladders.items():
        for lab, path in rungs:
            if os.path.exists(path):
                found[(task, lab)] = path
            else:
                missing.append((task, lab, path))
    return found, missing
