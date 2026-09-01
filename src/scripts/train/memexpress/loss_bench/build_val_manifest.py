"""
Stage 1b: sample 50 example INDICES per (task, context-length rung) from the v3 eval bundle, ONCE,
shared across every model. Unlike train data this only needs example identity (file + row index) --
each model tokenizes its own copy at loss-compute time (query_position and tokenizer both differ by
model), so nothing is materialized here.

The rung table below is copied verbatim from the ladder construction in
``src/scripts/ctc_eval/eval/eval_lc_native.py`` (the ``--ladder-version v3`` branch, lines ~659-778
at the time of writing) -- same file paths, same task aliases, same xlong glob patterns -- rather
than re-derived, so it never drifts from what real evals actually score. v3 = contradiction realistic
mode ("_realistic_" filenames); nq/rerank/oolong/outlier come from the same paths v2 uses (see
records/v3-eval-howto.md: contra + outlier are the only genuinely-rebuilt v3 tasks, nq/rerank/oolong
are directory symlinks to v2_clean).

Needs weka mounted (same CPU gantry job as build_train_manifest.py). No GPU required; only line
counts are read here, not example content.

Usage:
    PYTHONPATH=src python build_val_manifest.py
"""

from __future__ import annotations

import glob
import json
import os
import random

from models import LENGTH_LADDER_LABELS, N_PER_BUCKET, SEED, V3_EVAL_BUNDLE_ROOT, VAL_MANIFEST_PATH

_CM = "realistic"  # v3's one difference from v2: contradiction rungs are realistic-mode gold

E5 = V3_EVAL_BUNDLE_ROOT

BASE_LADDERS = {
    "contradiction": [
        ("2k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n100_k3.jsonl"),
        ("8k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n190_k3.jsonl"),
        ("16k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n385_k3.jsonl"),
        ("32k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n765_k3.jsonl"),
    ],
    "nq": [
        ("3k", f"{E5}/nq/nq_validation_k20_600.jsonl"),
        ("8k", f"{E5}/nq/nq_validation_k50_600.jsonl"),
        ("16k", f"{E5}/nq/nq_validation_k100_600.jsonl"),
        ("32k", f"{E5}/nq/nq_validation_k200_600.jsonl"),
    ],
    "outlier": [
        ("3k", f"{E5}/outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
        ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
        ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
        ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl"),
    ],
    "rerank": [
        ("3k", f"{E5}/rerank/msmarco_trainhn_eval_k20_500.jsonl"),
        ("8k", f"{E5}/rerank/msmarco_trainhn_eval_k50_500.jsonl"),
        ("16k", f"{E5}/rerank/msmarco_trainhn_eval_k100_500.jsonl"),
        # no CE-graded pool above k100 exists -- rerank has no 32k rung.
    ],
    "oolong": [
        ("8k", f"{E5}/oolong/oolong_test_synth_ctx8192_spliteval.jsonl"),
        ("16k", f"{E5}/oolong/oolong_test_synth_ctx16384_spliteval.jsonl"),
        ("32k", f"{E5}/oolong/oolong_test_synth_ctx32768_spliteval.jsonl"),
    ],
}

# oolong short rungs (2k/4k), prepended if present.
for _lab, _ctx in (("2k", 2048), ("4k", 4096)):
    _p = os.path.join(E5, "oolong", f"oolong_test_synth_ctx{_ctx}_spliteval.jsonl")
    if os.path.exists(_p):
        BASE_LADDERS["oolong"] = [(_lab, _p)] + BASE_LADDERS["oolong"]

# xlong rungs (64k..256k -- 512k/1M/2M excluded: no model in this analysis trains that far).
_XL = {
    "contradiction": ("contra", f"contradiction_eval_pubmed_{_CM}_n*_k3_xlong_{{s}}.jsonl"),
    "nq": ("nq", "nq_validation_k*_xlong_{s}.jsonl"),
    "outlier": ("outlier", "outlier_wiki100w_n*_k3_eval_xlong_{s}.jsonl"),
    "rerank": ("rerank", "msmarco_trainhn_eval_k*_xlong_{s}.jsonl"),
}
_XL_OOLONG = {"64k": 65536, "128k": 131072, "256k": 262144}
for _t, (_sub, _pat) in _XL.items():
    for _s in ("64k", "128k", "256k"):
        hits = sorted(glob.glob(os.path.join(E5, _sub, _pat.format(s=_s))))
        if hits:
            BASE_LADDERS[_t].append((_s, hits[0]))
for _s, _ctx in _XL_OOLONG.items():
    _p = os.path.join(E5, "oolong", f"oolong_test_synth_ctx{_ctx}_spliteval.jsonl")
    if os.path.exists(_p):
        BASE_LADDERS["oolong"].append((_s, _p))


def count_jsonl_lines(path: str) -> int:
    n = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    rng = random.Random(SEED)
    manifest: dict = {"seed": SEED, "n_per_bucket": N_PER_BUCKET, "bundle_root": E5, "tasks": {}}

    missing = []
    for task, rungs in BASE_LADDERS.items():
        manifest["tasks"][task] = {}
        for label, path in rungs:
            if not os.path.exists(path):
                missing.append((task, label, path))
                continue
            n_total = count_jsonl_lines(path)
            k = min(N_PER_BUCKET, n_total)
            indices = sorted(rng.sample(range(n_total), k)) if n_total > 0 else []
            manifest["tasks"][task][label] = {"path": path, "n_total": n_total, "indices": indices}
            print(
                f"[val] {task}@{label}: n_total={n_total} sampled={len(indices)} path={path}",
                flush=True,
            )

    if missing:
        print(f"[val] WARNING: {len(missing)} rung file(s) missing under {E5}:")
        for t, lab, p in missing:
            print(f"    {t:>14} {lab:>5}  {p}")

    os.makedirs(os.path.dirname(VAL_MANIFEST_PATH), exist_ok=True)
    with open(VAL_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[val] wrote {VAL_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
