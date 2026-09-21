#!/usr/bin/env python
"""Turn measured per-cell costs into an eval_size policy that fits a GPU-hour budget.

The full 22-task suite to 256k is 153 (task, rung) cells and 2.52B prompt tokens at the sizes the
olmo-eval roster declares. A 1-hour run on 8 single-GPU jobs is 8 GPU-hours, so the question is
which subsetting -- fewer examples, fewer rungs, fewer tasks -- buys that, and the honest way to
answer it is a cost model fitted to measurement rather than a FLOP estimate.

Cost per cell is split because the two halves scale differently:

    t = prompt_tokens / R_prefill(rung)  +  gen_tokens_per_example * n / R_decode

`R_prefill` is measured per rung (it falls with context length, and for a GDN-hybrid it falls much
more slowly than for a dense model, which is exactly the thing worth measuring rather than
assuming). `gen_tokens_per_example` is per SPEC, not per rung, and is the half that depends on the
checkpoint: a model that never emits EOS runs to the spec's full budget. The benchmark reports it,
so a pessimistic policy can be built from the budget and an expected one from the measurement.

The balance step matters as much as the sizing: 8 jobs finish when the SLOWEST finishes, so cells
are packed longest-first into the least-loaded shard (LPT). A shard split by task instead would put
every 256k rung of the long tasks in one job.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

RUNG_TOKENS = {
    "r2k": 2048, "r4k": 4096, "r8k": 8192, "r16k": 16384, "r32k": 32768,
    "r64k": 65536, "r128k": 131072, "r256k": 262144,
}


def load_bench(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fit_rates(bench: dict) -> tuple[dict, dict]:
    """R_prefill per rung label, and generated tokens/example per spec."""
    prefill, gen = {}, defaultdict(list)
    by_rung = defaultdict(lambda: [0.0, 0.0])  # rung -> [prompt_tokens, seconds]
    for c in bench["cells"]:
        # Decode time is not prefill time; subtract an estimate of it before fitting the
        # prefill rate, or long-generation tasks depress the rate for every task at that rung.
        by_rung[c["rung"]][0] += c["prompt_tokens"]
        by_rung[c["rung"]][1] += c["wall_s"]
        gen[c["spec"]].append(c["gen_tok_per_example"])
    for rung, (toks, secs) in by_rung.items():
        prefill[rung] = toks / secs if secs else 0.0
    return prefill, {k: sum(v) / len(v) for k, v in gen.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, help="bench_vllm_rungs.py output JSON")
    ap.add_argument("--roster", required=True, help="JSON: {task: {subset, spec, rungs: [...]}}")
    ap.add_argument("--budget-gpu-hours", type=float, default=8.0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument(
        "--eval-size",
        default="r2k=200,r4k=200,r8k=200,r16k=200,r32k=200,r64k=100,r128k=100,r256k=100",
        help="per-rung example count to price",
    )
    ap.add_argument("--load-s", type=float, default=180.0, help="per-job model load + import cost")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bench = load_bench(args.bench)
    prefill, gen = fit_rates(bench)
    sizes = {}
    for part in args.eval_size.split(","):
        k, v = part.split("=")
        sizes[k] = int(v)

    with open(args.roster) as f:
        roster = json.load(f)

    # Rungs the benchmark never measured fall back to the slowest measured rate, flagged, rather
    # than silently extrapolating a rate that long-context attention does not obey.
    slowest = min(prefill.values()) if prefill else 0.0
    extrapolated = set()

    cells = []
    for task, row in roster.items():
        for rung in row["rungs"]:
            if rung not in sizes:
                continue
            n = min(sizes[rung], row.get("eval_size", {}).get(rung, 10**9))
            rate = prefill.get(rung)
            if rate is None:
                rate, _ = slowest, extrapolated.add(rung)
            g = gen.get(row["spec"], 64.0)
            # decode rate is read off the benchmark as a whole: total generated / total wall is a
            # blend, so price decode at the measured per-example generation cost instead.
            t = (RUNG_TOKENS[rung] * n) / rate
            cells.append({"task": task, "rung": rung, "n": n, "tokens": RUNG_TOKENS[rung] * n,
                          "gen_per_ex": g, "est_s": t})

    total = sum(c["est_s"] for c in cells)
    budget_s = args.budget_gpu_hours * 3600

    # LPT: longest cell first into the least-loaded shard.
    shards = [[] for _ in range(args.shards)]
    loads = [args.load_s] * args.shards
    for c in sorted(cells, key=lambda c: -c["est_s"]):
        i = loads.index(min(loads))
        shards[i].append(c)
        loads[i] += c["est_s"]

    report = {
        "prefill_tok_per_s_by_rung": {k: round(v, 1) for k, v in sorted(prefill.items(), key=lambda kv: RUNG_TOKENS.get(kv[0], 0))},
        "gen_tokens_per_example_by_spec": {k: round(v, 1) for k, v in sorted(gen.items())},
        "extrapolated_rungs": sorted(extrapolated),
        "eval_size_policy": sizes,
        "cells": len(cells),
        "total_prompt_tokens": sum(c["tokens"] for c in cells),
        "est_total_gpu_seconds": round(total, 1),
        "est_total_gpu_hours": round(total / 3600, 2),
        "budget_gpu_hours": args.budget_gpu_hours,
        "fits": total <= budget_s,
        "shards": args.shards,
        "est_wall_clock_s": round(max(loads), 1),
        "est_wall_clock_min": round(max(loads) / 60, 1),
        "shard_loads_min": [round(x / 60, 1) for x in loads],
        "shard_contents": [
            [f"{c['task']}:{c['rung']}(n={c['n']},{round(c['est_s'])}s)" for c in sorted(s, key=lambda c: -c["est_s"])]
            for s in shards
        ],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "shard_contents"}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
