"""
Export a compact, browsable dataset for the compare-mt-style error-browser artifact: for every
SAME_DATA_GROUP, the full pairwise correlation stats (computed over ALL examples, via the same
logic as analyze.py) plus a per-example table capped at ``--max-per-bucket`` examples per
(task, source_tag) so the exported JSON stays a reasonable size for embedding in a web page --
compare-mt-style browsing is about reading individual examples, not needing all 20k+ at once, and
the aggregate stats (computed pre-cap) stay exact regardless of the browsing cap.

Needs weka mounted (run on a CPU gantry job). No GPU required.

Usage:
    PYTHONPATH=src python export_for_artifact.py [--max-per-bucket 200] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from itertools import combinations
from typing import Dict, List

from analyze import ExampleKey, compare_pair, load_model_scores
from registry import SAME_DATA_GROUPS

PROMPT_TAIL_CHARS = 350
PREDICTION_CHARS = 220


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-bucket", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="/weka/oe-training-default/ai2-llm/checkpoints/amandab/loss_bench_2026-08-31/error_correlation/artifact_data.json",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    all_scores: Dict[str, Dict[ExampleKey, dict]] = {}
    out_groups = {}

    for group_name, model_keys in SAME_DATA_GROUPS.items():
        print(f"\n=== {group_name}: {model_keys} ===", flush=True)
        group_scores = {}
        for m in model_keys:
            if m not in all_scores:
                all_scores[m] = load_model_scores(m)
            group_scores[m] = all_scores[m]

        # ---- full pairwise stats, computed over ALL examples (not capped) ----
        pairwise = {}
        for m1, m2 in combinations(model_keys, 2):
            tasks_in_group = sorted({k[1] for m in model_keys for k in group_scores[m]})
            pair_out = {"pooled": compare_pair(group_scores[m1], group_scores[m2])}
            for t in tasks_in_group:
                pair_out[t] = compare_pair(group_scores[m1], group_scores[m2], task_short=t)
            pairwise[f"{m1}__vs__{m2}"] = pair_out

        # ---- capped, browsable example table ----
        # UNION across the group's models, not intersection: a single model's gap (e.g. an empty
        # generations.jsonl for one task -- happens, see summtok_causal's contra dump, 0 bytes on
        # weka despite a real non-zero metric JSON) must not blank that task out for every OTHER
        # pair in the group. Each row only lists the models that actually have that key; the
        # client filters per the two runs currently selected, so a pair that both have the data
        # (e.g. summtok_decay vs summtok_p50) still gets to browse it.
        any_keys = set()
        for m in model_keys:
            any_keys |= set(group_scores[m].keys())
        by_bucket: Dict[tuple, List[ExampleKey]] = {}
        for k in any_keys:
            bucket = (k[1], k[2])  # (task_short, source_tag)
            by_bucket.setdefault(bucket, []).append(k)

        examples = []
        for bucket, keys in by_bucket.items():
            keys = sorted(keys, key=lambda k: k[4])  # stable order before sampling
            if len(keys) > args.max_per_bucket:
                keys = rng.sample(keys, args.max_per_bucket)
            for k in keys:
                ladder_version, task_short, source_tag, rung, idx = k
                present_models = [m for m in model_keys if k in group_scores[m]]
                any_v = group_scores[present_models[0]][k]
                row = {
                    "task": task_short,
                    "ladder_version": ladder_version,
                    "source_tag": source_tag,
                    "rung": rung,
                    "idx": idx,
                    "gold": any_v["gold"],
                    "prompt_tail": (any_v["prompt_tail"] or "")[-PROMPT_TAIL_CHARS:],
                    "models": {},
                }
                for m in present_models:
                    v = group_scores[m][k]
                    row["models"][m] = {
                        "binary": v["binary"],
                        "continuous": v["continuous"],
                        "prediction": (v["prediction"] or "")[:PREDICTION_CHARS],
                    }
                examples.append(row)
        print(
            f"  {len(examples)} browsable examples across {len(by_bucket)} (task,source_tag) buckets "
            f"(from {len(any_keys)} total keys, union across {len(model_keys)} models)",
            flush=True,
        )

        out_groups[group_name] = {"models": model_keys, "pairwise": pairwise, "examples": examples}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"groups": out_groups}, f, separators=(",", ":"))
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\n[done] wrote {args.out} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
