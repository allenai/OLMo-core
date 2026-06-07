#!/usr/bin/env python
"""Copy eval-only backfill metrics onto their source training W&B runs."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from typing import Any

import wandb


PROJECT = "ai2-llm/jacobm-olmoe-ladder"


@dataclass(frozen=True)
class BackfillSpec:
    eval_prefix: str
    source_prefix: str


BACKFILLS = [
    BackfillSpec("eval-275m-275-01", "olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr1.2e-3-r2"),
    BackfillSpec("eval-275m-275-02", "olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr1.5e-3-r2"),
    BackfillSpec("eval-275m-275-03", "olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr1e-3-r2"),
    BackfillSpec("eval-275m-275-04", "olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2"),
    BackfillSpec("eval-275m-275-05", "olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr8e-4-r2"),
    BackfillSpec("eval-275m-275-06", "olmoe3-tiny-275m-cx16-b1m-gpu8-ep1mb16-lr1.2e-3-r2"),
    BackfillSpec("eval-275m-275-07", "olmoe3-tiny-275m-cx16-b1m-gpu8-ep1mb16-lr2.4e-3-r3"),
    BackfillSpec("eval-275m-275-08", "olmoe3-tiny-275m-cx16-b1m-gpu8-ep1mb16-lr2e-4-r2"),
    BackfillSpec("eval-275m-275-09", "olmoe3-tiny-275m-cx16-b1m-gpu8-ep1mb16-lr4e-4-r2"),
    BackfillSpec("eval-275m-275-10", "olmoe3-tiny-275m-cx16-b1m-gpu8-ep1mb16-lr6e-3-sentinel"),
    BackfillSpec("eval-275m-275-11", "olmoe3-tiny-275m-cx16-b1m-gpu8-ep1mb16-lr6e-4-r2"),
    BackfillSpec("eval-275m-275-12", "olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr1e-3"),
    BackfillSpec("eval-275m-275-13", "olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr6e-4-r2"),
    BackfillSpec("eval-275m-275-14", "olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr8e-4-r2"),
    BackfillSpec("eval-275m-275-15", "olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1.5e-3"),
    BackfillSpec("eval-275m-275-16", "olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1e-3"),
    BackfillSpec("eval-275m-275-17", "olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr2.5e-3"),
    BackfillSpec("eval-275m-275-18", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-2-sentinel"),
    BackfillSpec("eval-275m-275-19", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3-r2"),
    BackfillSpec("eval-275m-275-20", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr2e-4-r2"),
    BackfillSpec("eval-275m-275-21", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr3.2e-3-r3"),
    BackfillSpec("eval-275m-275-22", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr4e-4-r2"),
    BackfillSpec("eval-275m-275-23", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr6.4e-3-r3"),
    BackfillSpec("eval-275m-275-24", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr6e-4-r2"),
    BackfillSpec("eval-275m-275-25", "olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr8e-4-r2"),
    BackfillSpec("eval-810m-810-01", "olmoe3-810m-cx1-b256k-gpu8-ep1mb4-lr5e-5-cs-r2"),
    BackfillSpec("eval-810m-810-02", "olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr1.2e-3-r1"),
    BackfillSpec("eval-810m-810-03", "olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr1.5e-4-cold-r1"),
    BackfillSpec("eval-810m-810-04", "olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr2.4e-3-r1"),
    BackfillSpec("eval-810m-810-05", "olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr3e-4-cold-r1"),
    BackfillSpec("eval-810m-810-06", "olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr6e-3-r1"),
    BackfillSpec("eval-810m-810-07", "olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr6e-4-r1"),
    BackfillSpec("eval-810m-cx4-lr2e-4-r1", "olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr2e-4-r1"),
    BackfillSpec("eval-810m-cx4-lr4e-4-r1", "olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr4e-4-r1"),
    BackfillSpec("eval-810m-cx4-lr8e-4-r1", "olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr8e-4-r1"),
    BackfillSpec("eval-810m-cx4-lr1.6e-3-r1", "olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr1.6e-3-r1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", help="Only process eval runs whose names contain this substring")
    return parser.parse_args()


def run_sort_key(run: Any) -> tuple[int, int, str]:
    finished = 1 if run.state == "finished" else 0
    tokens = int(run.summary.get("throughput/total tokens") or 0)
    return finished, tokens, str(getattr(run, "created_at", ""))


def find_one_run(api: wandb.Api, project: str, prefix: str) -> Any | None:
    matches = [
        run
        for run in api.runs(project, filters={"display_name": {"$regex": f"^{prefix}"}})
        if run.name.startswith(prefix)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        chosen = max(matches, key=run_sort_key)
        print(
            f"ambiguous prefix {prefix!r}; chose {chosen.id} "
            f"state={chosen.state} tokens={chosen.summary.get('throughput/total tokens')}",
            file=sys.stderr,
        )
        return chosen
    return None


def metric_items(summary: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in summary.items():
        if key.startswith("eval/") or key.startswith("throughput/in-loop eval"):
            metrics[key] = value
    return metrics


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    copied = 0
    skipped = 0

    for spec in BACKFILLS:
        if args.only and args.only not in spec.eval_prefix and args.only not in spec.source_prefix:
            continue

        eval_run = find_one_run(api, args.project, spec.eval_prefix)
        source_run = find_one_run(api, args.project, spec.source_prefix)
        if eval_run is None or source_run is None:
            print(f"missing eval/source for {spec.eval_prefix} -> {spec.source_prefix}")
            skipped += 1
            continue
        if eval_run.state != "finished":
            print(f"skip unfinished {eval_run.name}: state={eval_run.state}")
            skipped += 1
            continue

        metrics = metric_items(dict(eval_run.summary))
        if not metrics:
            print(f"skip {eval_run.name}: no eval metrics in summary")
            skipped += 1
            continue

        already = [
            key
            for key in metrics
            if key in source_run.summary and source_run.summary.get(key) is not None
        ]
        if already and not args.overwrite:
            print(f"skip {source_run.name}: {len(already)} eval metrics already present")
            skipped += 1
            continue

        print(f"copy {len(metrics)} metrics: {eval_run.name} -> {source_run.name}")
        if not args.dry_run:
            for key, value in metrics.items():
                source_run.summary[key] = value
            source_run.summary["eval_backfill/eval_run_id"] = eval_run.id
            source_run.summary["eval_backfill/eval_run_name"] = eval_run.name
            source_run.summary["eval_backfill/copied_metric_count"] = len(metrics)
            source_run.summary["eval_backfill/copied_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
            source_run.summary.update()
        copied += 1

    print(f"done: copied={copied} skipped={skipped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
