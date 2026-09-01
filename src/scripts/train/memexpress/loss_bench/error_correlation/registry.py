"""
File manifest for per-example generation dumps (``*.generations.jsonl``) across the 8 checkpoints
that actually have them (``docchunk_bs128`` is the one model in ``models.MODELS`` with no
generations saved anywhere on weka -- confirmed by direct listing, see loss_bench/README.md).

Every path here was found by directly listing weka (gantry jobs ``01M1EN0R3WPPZ7TQZQ4CK58SNV`` for
the per-checkpoint ``eval*/`` directories, and ``01M1ENGQPXMT0HANRYACRG75B1`` for the summary-token
models' central ``_eval_results/`` collection -- the per-checkpoint directories for those 3 hold NO
generations; the plain-harness models' per-checkpoint directories DO). Do not regenerate this list
by guessing a naming convention -- re-verify on weka if a path here 404s.

Each entry is (task_short, ladder_version, source_tag, path):
  - task_short: the launcher's short task name embedded in the filename (contra/nq/outlier/oolong/
    rerank for in-distribution; contra_fever/fiqa/scifact/outlier_review for OOD). NOT the same
    string as the eval_lc_native.py-internal task key (contradiction/retrieval/...) -- see
    TASK_SHORT_TO_SCORER below for that mapping.
  - ladder_version: "v2" or "v3" (parsed from the "_v3" filename suffix; absence means v2).
  - source_tag: "base" | "xlong-native" | "xlong-yarn2" -- kept distinct because native-RoPE and
    YaRN-scaled xlong rungs are different measurement conditions even when they report the same
    nominal rung label (e.g. both can have a "256k" row).
  - path: absolute weka path.

model_key groupings (which models share IDENTICAL underlying eval examples, so idx aligns 1:1):
  - v2 in-distribution (contra/nq/outlier/oolong/rerank, both-mode contra / K-frozen outlier):
    sparselm_32k, fastlm_32k, dense_fixdata
  - v3 in-distribution (realistic-mode contra / scale-K outlier), base + xlong rungs:
    dense_xlong5_256k, fastlm_33344_datamatch, summtok_causal, summtok_decay, summtok_p50
  - v2 OOD (contra_fever/fiqa/scifact/outlier_review) -- shared by the 5 "plain harness" models
    (sparselm_32k, fastlm_32k, dense_xlong5_256k, fastlm_33344_datamatch, dense_fixdata); the 3
    summtok models' OOD rows are metric-only (no generations), so they're absent from OOD analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

AB = "/weka/oe-training-default/ai2-llm/checkpoints/amandab"
PR = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
CENTRAL = f"{PR}/_eval_results"

# task_short -> (scorer family, continuous score field, binary/exact_match field or None)
TASK_SHORT_TO_SCORER: Dict[str, Tuple[str, str, str]] = {
    "contra": ("contradiction", "f1", "exact_match"),
    "contra_fever": ("contradiction", "f1", "exact_match"),
    "nq": ("retrieval", "f1", "exact_match"),
    "fiqa": ("retrieval", "f1", "exact_match"),
    "scifact": ("retrieval", "f1", "exact_match"),
    "outlier": ("outlier", "f1", "exact_match"),
    "outlier_review": ("outlier", "f1", "exact_match"),
    "oolong": ("oolong", "score", "exact_match"),
    "rerank": ("rerank", "ndcg@10", None),  # falls back to mrr@10 if ndcg@10 absent (no ce_scores)
}

Entry = Tuple[str, str, str, str]  # (task_short, ladder_version, source_tag, path)


def _plain_entries(
    root: str,
    base_tasks: List[str],
    base_v3_tasks: List[str],
    xlong_native_v3_tasks: List[str],
    xlong_yarn2_v3_tasks: List[str],
    base_dir: str = "eval",
    xlong_v2_tasks: Optional[List[str]] = None,
) -> List[Entry]:
    entries: List[Entry] = []
    for t in base_tasks:
        entries.append((t, "v2", "base", f"{root}/{base_dir}/{t}_multirung.generations.jsonl"))
    for t in base_v3_tasks:
        entries.append((t, "v3", "base", f"{root}/{base_dir}/{t}_multirung_v3.generations.jsonl"))
    for t in xlong_v2_tasks or []:
        entries.append(
            (t, "v2", "xlong-native", f"{root}/eval_xlong/{t}_multirung.generations.jsonl")
        )
    for t in xlong_native_v3_tasks:
        entries.append(
            (
                t,
                "v3",
                "xlong-native",
                f"{root}/eval_xlong-native/{t}_multirung_v3.generations.jsonl",
            )
        )
    for t in xlong_yarn2_v3_tasks:
        entries.append(
            (t, "v3", "xlong-yarn2", f"{root}/eval_xlong-yarn2/{t}_multirung_v3.generations.jsonl")
        )
    return entries


def _central_entries(model_name: str) -> List[Entry]:
    entries: List[Entry] = []
    for t in ["contra", "nq", "oolong", "outlier", "rerank"]:
        entries.append(
            (t, "v3", "base", f"{CENTRAL}/{model_name}_v3c-base_{t}_multirung_v3.generations.jsonl")
        )
    for t in ["contra", "nq", "outlier"]:
        entries.append(
            (
                t,
                "v3",
                "xlong-native",
                f"{CENTRAL}/{model_name}_v3c-xlong-native_{t}_multirung_v3.generations.jsonl",
            )
        )
        entries.append(
            (
                t,
                "v3",
                "xlong-yarn2",
                f"{CENTRAL}/{model_name}_v3c-xlong-yarn2_{t}_multirung_v3.generations.jsonl",
            )
        )
    # v2-ood (contra_fever/fiqa/outlier_review/scifact) intentionally excluded: metric-only on weka,
    # no generations.jsonl. v3c-gencheck (nq-only rerun) and v3c-smoke (contra-only smoke test)
    # intentionally excluded: narrower/duplicate reruns of what v3c-base already covers.
    return entries


GENERATION_FILES: Dict[str, List[Entry]] = {
    "sparselm_32k": _plain_entries(
        root=f"{AB}/q35-4b-sparselm-5task-dolci25-32k-nocpt-cp4",
        base_tasks=[
            "contra_fever",
            "contra",
            "outlier",
            "nq",
            "oolong",
            "outlier_review",
            "fiqa",
            "scifact",
            "rerank",
        ],
        base_v3_tasks=[],
        xlong_native_v3_tasks=[],
        xlong_yarn2_v3_tasks=[],
        xlong_v2_tasks=["nq", "outlier", "contra", "rerank"],
    ),
    "fastlm_32k": _plain_entries(
        root=f"{AB}/q35-4b-fastlm-5task-dolci25-32k-nocpt",
        base_tasks=[
            "nq",
            "contra",
            "outlier_review",
            "rerank",
            "fiqa",
            "oolong",
            "outlier",
            "contra_fever",
            "scifact",
        ],
        base_v3_tasks=[],
        xlong_native_v3_tasks=[],
        xlong_yarn2_v3_tasks=[],
        xlong_v2_tasks=["contra", "rerank", "nq", "outlier"],
    ),
    "dense_fixdata": _plain_entries(
        root=f"{PR}/q4b-dense-5task-32k-nocpt-fixdata",
        base_tasks=[
            "outlier",
            "contra_fever",
            "oolong",
            "rerank",
            "fiqa",
            "nq",
            "contra",
            "scifact",
            "outlier_review",
        ],
        base_v3_tasks=[],
        xlong_native_v3_tasks=[],
        xlong_yarn2_v3_tasks=[],
    ),
    "dense_xlong5_256k": _plain_entries(
        root=f"{AB}/q35-4b-dense-xlong5-qboth-dolci25-256k",
        base_tasks=["fiqa", "scifact", "contra_fever"],
        base_v3_tasks=["oolong", "nq", "rerank", "outlier_review", "contra", "outlier"],
        xlong_native_v3_tasks=["rerank", "nq", "oolong", "outlier", "contra"],
        xlong_yarn2_v3_tasks=["rerank", "outlier", "contra", "oolong", "nq"],
        base_dir="eval_base",
    ),
    "fastlm_33344_datamatch": _plain_entries(
        root=f"{AB}/q35-4b-fastlm-5task-dolci25-33344-datamatch",
        base_tasks=["fiqa", "contra_fever", "scifact"],
        base_v3_tasks=["outlier", "nq", "oolong", "outlier_review", "contra", "rerank"],
        xlong_native_v3_tasks=["nq", "oolong", "contra", "rerank", "outlier"],
        xlong_yarn2_v3_tasks=["contra", "oolong", "outlier", "nq", "rerank"],
        base_dir="eval_base",
    ),
    "summtok_causal": _central_entries("q35-4b-summ-causal-5task-packed"),
    "summtok_decay": _central_entries("q35-4b-summ-decay-5task-packed"),
    "summtok_p50": _central_entries("q35-4b-summ-p50-5task-packed"),
    # docchunk_bs128: NO generations anywhere on weka (metric-only harness) -- intentionally absent.
}

# Which models share identical underlying eval examples (idx-aligned), by (ladder_version, in-dist-vs-ood).
SAME_DATA_GROUPS: Dict[str, List[str]] = {
    "v2_in_distribution": ["sparselm_32k", "fastlm_32k", "dense_fixdata"],
    "v3_in_distribution": [
        "dense_xlong5_256k",
        "fastlm_33344_datamatch",
        "summtok_causal",
        "summtok_decay",
        "summtok_p50",
    ],
    "v2_ood": [
        "sparselm_32k",
        "fastlm_32k",
        "dense_xlong5_256k",
        "fastlm_33344_datamatch",
        "dense_fixdata",
    ],
}
