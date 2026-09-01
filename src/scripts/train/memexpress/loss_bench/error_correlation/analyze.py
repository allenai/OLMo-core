"""
Correlate per-example correctness across models that were scored on IDENTICAL eval data (see
``registry.SAME_DATA_GROUPS``), to answer questions like "do summtok_decay and summtok_p50 get the
same questions wrong?" -- i.e. do error patterns cluster by architecture/training choice, or are
they closer to independent noise.

For every data group (v2_in_distribution / v3_in_distribution / v2_ood), for every task within it,
aligns each model's per-example ``detail`` (already scored by the original eval run -- see
``registry.TASK_SHORT_TO_SCORER`` for which field is "correct") by (rung, idx), then for every model
PAIR in the group computes:
  - binary agreement rate (both right / both wrong / split), and Cohen's kappa (agreement corrected
    for the base rate -- two models that are both usually-right will have high raw agreement just by
    chance, kappa is the useful number for "do they fail on the SAME examples")
  - Jaccard overlap of their two error sets (of the examples where at least one is wrong, what
    fraction are wrong for both)
  - Pearson r of the continuous score (f1/ndcg/oolong-score), where defined for both

Also dumps a sample of "commonly wrong" examples (wrong for most/all models in a group, per task) so
the failure mode can be read directly, not just scored.

Needs weka mounted (run on a CPU gantry job). No GPU required.

Usage:
    PYTHONPATH=src python analyze.py [--groups v3_in_distribution,...] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from registry import GENERATION_FILES, SAME_DATA_GROUPS, TASK_SHORT_TO_SCORER

ExampleKey = Tuple[str, str, str, str, int]  # (ladder_version, task_short, source_tag, rung, idx)


def extract_score(task_short: str, detail: dict) -> Tuple[Optional[float], Optional[float]]:
    """Returns (continuous, binary) for one example's already-computed detail dict."""
    _, cont_field, bin_field = TASK_SHORT_TO_SCORER[task_short]
    continuous = detail.get(cont_field)
    if continuous is None and task_short == "rerank":
        continuous = detail.get("mrr@10")  # ndcg@10 absent when the pool has no ce_scores
    binary = detail.get(bin_field) if bin_field else None
    return continuous, binary


def load_model_scores(model_key: str) -> Dict[ExampleKey, dict]:
    """Returns key -> {"continuous", "binary", "gold", "prediction", "prompt_tail"}."""
    scores: Dict[ExampleKey, dict] = {}
    for task_short, ladder_version, source_tag, path in GENERATION_FILES.get(model_key, []):
        if not os.path.exists(path):
            print(f"[{model_key}] MISSING {path}", flush=True)
            continue
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                detail = rec.get("detail")
                if detail is None:
                    continue
                key: ExampleKey = (ladder_version, task_short, source_tag, rec["rung"], rec["idx"])
                continuous, binary = extract_score(task_short, detail)
                scores[key] = {
                    "continuous": continuous,
                    "binary": binary,
                    "gold": detail.get("gold") or detail.get("gold_pairs"),
                    "prediction": detail.get("prediction"),
                    "prompt_tail": rec.get("prompt_tail"),
                }
                n += 1
        if n == 0:
            # The file exists but parsed to nothing -- distinct from MISSING, and easy to miss in
            # a long log otherwise. Seen for real: summtok_causal's contra generations.jsonl is 0
            # bytes on weka despite a non-trivial metric JSON alongside it (the eval ran fine; only
            # the generation dump failed to write). Silently treating this the same as "no rows to
            # report" let one model's empty file blank an entire task out of a whole group's
            # browsable examples -- see export_for_artifact.py's union-not-intersection fix.
            print(f"[{model_key}] EMPTY (0 rows parsed, file is present): {path}", flush=True)
        else:
            print(
                f"[{model_key}] {task_short}/{source_tag}/{ladder_version}: {n} examples from {path}",
                flush=True,
            )
    return scores


def cohens_kappa(pairs: List[Tuple[float, float]]) -> Optional[float]:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    p1a = sum(1 for a, _ in pairs if a == 1) / n
    p1b = sum(1 for _, b in pairs if b == 1) / n
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def pearson(pairs: List[Tuple[float, float]]) -> Optional[float]:
    n = len(pairs)
    if n < 2:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def compare_pair(scores_a, scores_b, task_short: Optional[str] = None) -> dict:
    common_keys = [
        k for k in scores_a if k in scores_b and (task_short is None or k[1] == task_short)
    ]
    bin_pairs = [
        (scores_a[k]["binary"], scores_b[k]["binary"])
        for k in common_keys
        if scores_a[k]["binary"] is not None and scores_b[k]["binary"] is not None
    ]
    cont_pairs = [
        (scores_a[k]["continuous"], scores_b[k]["continuous"])
        for k in common_keys
        if scores_a[k]["continuous"] is not None and scores_b[k]["continuous"] is not None
    ]
    n_bin = len(bin_pairs)
    both_right = sum(1 for a, b in bin_pairs if a == 1 and b == 1)
    both_wrong = sum(1 for a, b in bin_pairs if a == 0 and b == 0)
    split = n_bin - both_right - both_wrong
    n_wrong_union = sum(1 for a, b in bin_pairs if a == 0 or b == 0)
    jaccard = (both_wrong / n_wrong_union) if n_wrong_union else None
    return {
        "n_common_examples": len(common_keys),
        "n_binary_scored": n_bin,
        "both_right_frac": both_right / n_bin if n_bin else None,
        "both_wrong_frac": both_wrong / n_bin if n_bin else None,
        "split_frac": split / n_bin if n_bin else None,
        "error_set_jaccard": jaccard,
        "cohens_kappa": cohens_kappa(bin_pairs),
        "pearson_r_continuous": pearson(cont_pairs),
        "n_continuous_scored": len(cont_pairs),
    }


def commonly_wrong_examples(
    group_scores: Dict[str, Dict[ExampleKey, dict]], task_short: str, top_n: int = 5
) -> List[dict]:
    """Examples where the largest fraction of the group's models scored binary=0, with per-model
    predictions attached so the actual failure mode can be read, not just counted."""
    models = list(group_scores.keys())
    all_keys = (
        set.intersection(*[{k for k in group_scores[m] if k[1] == task_short} for m in models])
        if models
        else set()
    )
    rows = []
    for k in all_keys:
        per_model = {m: group_scores[m][k] for m in models}
        binaries = [v["binary"] for v in per_model.values() if v["binary"] is not None]
        if not binaries:
            continue
        wrong_frac = sum(1 for b in binaries if b == 0) / len(binaries)
        rows.append((wrong_frac, k, per_model))
    rows.sort(key=lambda r: -r[0])
    out = []
    for wf, k, per_model in rows[:top_n]:
        any_v = next(iter(per_model.values()))
        out.append(
            {
                "key": {
                    "ladder_version": k[0],
                    "task": k[1],
                    "source_tag": k[2],
                    "rung": k[3],
                    "idx": k[4],
                },
                "wrong_frac": wf,
                "gold": any_v["gold"],
                "prompt_tail": (any_v["prompt_tail"] or "")[-400:],
                "predictions": {m: v["prediction"] for m, v in per_model.items()},
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--groups", default=None, help="comma list restricting which SAME_DATA_GROUPS to run"
    )
    ap.add_argument(
        "--out-dir",
        default="/weka/oe-training-default/ai2-llm/checkpoints/amandab/loss_bench_2026-08-31/error_correlation",
    )
    args = ap.parse_args()

    group_names = args.groups.split(",") if args.groups else list(SAME_DATA_GROUPS.keys())

    all_scores: Dict[str, Dict[ExampleKey, dict]] = {}

    out = {}
    for group_name in group_names:
        model_keys = SAME_DATA_GROUPS[group_name]
        print(f"\n=== group={group_name} models={model_keys} ===", flush=True)
        group_scores = {}
        for m in model_keys:
            if m not in all_scores:
                all_scores[m] = load_model_scores(m)
            group_scores[m] = all_scores[m]

        tasks_in_group = sorted({k[1] for m in model_keys for k in group_scores[m]})
        group_out = {"models": model_keys, "pairwise": {}, "commonly_wrong": {}}

        for m1, m2 in combinations(model_keys, 2):
            pair_key = f"{m1}__vs__{m2}"
            group_out["pairwise"][pair_key] = {
                "pooled": compare_pair(group_scores[m1], group_scores[m2])
            }
            print(f"  {pair_key} POOLED: {group_out['pairwise'][pair_key]['pooled']}", flush=True)
            for task_short in tasks_in_group:
                res = compare_pair(group_scores[m1], group_scores[m2], task_short=task_short)
                group_out["pairwise"][pair_key][task_short] = res
                print(
                    f"    {pair_key} / {task_short}: "
                    f"n={res['n_binary_scored']} kappa={res['cohens_kappa']} "
                    f"jaccard={res['error_set_jaccard']} pearson_r={res['pearson_r_continuous']}",
                    flush=True,
                )

        for task_short in tasks_in_group:
            group_out["commonly_wrong"][task_short] = commonly_wrong_examples(
                group_scores, task_short
            )

        out[group_name] = group_out

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/pairwise_correlations.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
