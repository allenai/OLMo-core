#!/usr/bin/env python
"""Write a latest-step in-loop eval dashboard from W&B summaries.

The script caches W&B summaries for runs whose state is ``finished``. Running,
queued, failed, and preempted runs are queried live each time so the dashboard
can reflect current state without treating partial data as immutable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb

from ladder_run_metadata import family_label_from_name, model_label_from_name

LADDER_DIR = Path(__file__).parent
RESULTS_DIR = LADDER_DIR / "results"
DEFAULT_CACHE_DIR = RESULTS_DIR / "cache" / "wandb_summaries"
DEFAULT_PROJECT = "ai2-llm/jacobm-olmoe-ladder"
DEFAULT_NAME_REGEX = r"olmoe3|q3-|int-|sp-|eg-|dense|se0m9|shared|mt-"
DEFAULT_METRIC_REGEX = r"^eval/|^throughput/in-loop eval"
CACHE_VERSION = 1


def cache_key(project: str, run_id: str) -> str:
    return f"{project.replace('/', '__')}__{run_id}.json"


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def load_summary(run: Any, *, project: str, cache_dir: Path, refresh_cache: bool) -> dict[str, Any]:
    cache_path = cache_dir / cache_key(project, run.id)
    if not refresh_cache and cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        meta = cached.get("metadata", {})
        if meta.get("cache_version") == CACHE_VERSION and meta.get("state") == "finished" and run.state == "finished":
            return cached

    summary = {key: jsonable(value) for key, value in dict(run.summary).items()}
    payload = {
        "metadata": {
            "cache_version": CACHE_VERSION,
            "project": project,
            "run_id": run.id,
            "state": run.state,
            "display_name": run.display_name,
            "url": run.url,
            "updated_at": str(getattr(run, "updated_at", "")),
            "cached_at_utc": datetime.now(UTC).isoformat(),
        },
        "summary": summary,
    }
    if run.state == "finished":
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp.replace(cache_path)
    return payload


def parse_cx(name: str) -> str:
    match = re.search(r"cx([0-9]+)", name)
    return f"Cx{match.group(1)}" if match else "Cx?"


def short_run_label(name: str, run_id: str) -> str:
    stable = name.split("_")[0]
    stable = re.sub(r"-gpu[0-9]+-ep[0-9]+mb[0-9]+", "", stable)
    stable = stable.replace("olmoe3-moe-a0-", "").replace("olmoe3-tiny-", "")
    if len(stable) > 54:
        stable = stable[:51] + "..."
    return f"{stable}<br>`{run_id}`"


def direction_for_metric(metric: str) -> str:
    lowered = metric.lower()
    if any(token in lowered for token in ("accuracy", "acc", "pass@", "exact_match", "f1")):
        return "higher"
    if any(token in lowered for token in ("bpb", "loss", "ppl", "perplexity")):
        return "lower"
    return "see metric"


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return ""
    if abs(f) >= 100:
        return f"{f:.1f}"
    if abs(f) >= 10:
        return f"{f:.2f}"
    if abs(f) >= 1:
        return f"{f:.4f}"
    return f"{f:.5f}"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name-regex", default=DEFAULT_NAME_REGEX)
    parser.add_argument("--metric-regex", default=DEFAULT_METRIC_REGEX)
    parser.add_argument("--states", nargs="+", default=["finished"], help="W&B states to include; use 'all' to include all states.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "in_loop_evals.md")
    parser.add_argument("--json-output", type=Path, default=RESULTS_DIR / "in_loop_evals.json")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-runs-per-table", type=int, default=24)
    args = parser.parse_args()

    api = wandb.Api(timeout=90)
    name_re = re.compile(args.name_regex)
    metric_re = re.compile(args.metric_regex)
    states = None if args.states == ["all"] else set(args.states)

    records: list[dict[str, Any]] = []
    skipped_no_metrics = 0
    for run in api.runs(args.project, filters={"display_name": {"$regex": args.name_regex}}):
        if states is not None and run.state not in states:
            continue
        if not name_re.search(run.display_name):
            continue
        payload = load_summary(run, project=args.project, cache_dir=args.cache_dir, refresh_cache=args.refresh_cache)
        summary = payload["summary"]
        metrics = {key: value for key, value in summary.items() if metric_re.search(key) and isinstance(value, (int, float))}
        if not metrics:
            skipped_no_metrics += 1
            continue
        name = run.display_name
        records.append(
            {
                "run_id": run.id,
                "name": name,
                "url": run.url,
                "state": run.state,
                "model": model_label_from_name(name),
                "cx": parse_cx(name),
                "family": family_label_from_name(name),
                "tokens": summary.get("throughput/total tokens") or summary.get("optim/total tokens"),
                "step": summary.get("_step"),
                "metrics": metrics,
            }
        )

    records.sort(key=lambda r: (r["model"], r["cx"], r["name"], r["run_id"]))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w") as f:
        json.dump({"generated_at_utc": datetime.now(UTC).isoformat(), "records": records}, f, indent=2, sort_keys=True)

    lines = [
        "# In-Loop Eval Results",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Values are the latest W&B summary values for each selected run. Higher is better for accuracy/F1/pass-style metrics; lower is better for BPB/loss/perplexity-style metrics. Values marked `see metric` need manual interpretation.",
        "",
        f"Selected states: `{', '.join(args.states)}`. Cached finished-run summaries under `{args.cache_dir}`.",
        f"Runs with no matching eval metrics skipped: {skipped_no_metrics}.",
        "",
    ]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["model"], record["cx"])].append(record)

    for (model, cx), group in sorted(groups.items()):
        lines.extend([f"## {model} {cx}", ""])
        if len(group) > args.max_runs_per_table:
            lines.append(f"Showing first {args.max_runs_per_table} of {len(group)} runs in this table. Use `--name-regex` to narrow the view.")
            lines.append("")
            group = group[: args.max_runs_per_table]
        metric_names = sorted({metric for record in group for metric in record["metrics"]})
        headers = ["metric", "direction"] + [short_run_label(record["name"], record["run_id"]) for record in group]
        rows = []
        for metric in metric_names:
            rows.append([metric.replace("|", "\\|"), direction_for_metric(metric)] + [fmt(record["metrics"].get(metric)) for record in group])
        lines.extend(md_table(headers, rows))
        lines.append("")
        meta_rows = [
            [short_run_label(record["name"], record["run_id"]), record["state"], record["family"], fmt(record["tokens"]), str(record.get("step") or ""), f"[W&B]({record['url']})"]
            for record in group
        ]
        lines.extend(md_table(["run", "state", "family", "tokens", "step", "link"], meta_rows))
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
