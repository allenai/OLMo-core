#!/usr/bin/env python3
"""Merge finalized Beaker throughput-smoke metrics into the comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "275m_gdn_gdn2_swa_large_batch_parallelism.csv"
FIELDS = (
    "family",
    "batch_tokens",
    "gpus",
    "path",
    "rank_microbatch_sequences",
    "final10_median_tflops_gpu",
    "final10_median_tps_gpu",
    "aggregate_tps",
    "stable_step_seconds",
    "final10_median_mfu_pct",
    "peak_active_gib",
    "peak_reserved_gib",
    "max_skipped_steps",
    "beaker_job",
    "wandb_url",
)
STEP_RE = re.compile(r"\[step=(\d+)")
METRIC_RE = re.compile(r"^\s+([^=]+)=(.+?)\s*$")
WANDB_RE = re.compile(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9]+")


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def parse_number(raw: str) -> float:
    value = raw.strip().replace(",", "")
    suffixes = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    if value and value[-1] in suffixes:
        return float(value[:-1]) * suffixes[value[-1]]
    return float(value)


def parse_steps(logs: str) -> tuple[list[dict[str, float]], str]:
    steps: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in logs.splitlines():
        if match := STEP_RE.search(line):
            current = {"step": float(match.group(1))}
            steps.append(current)
            continue
        if current is None or (match := METRIC_RE.match(line)) is None:
            continue
        try:
            current[match.group(1).strip()] = parse_number(match.group(2))
        except ValueError:
            continue
    wandb = WANDB_RE.search(logs)
    return steps, wandb.group(0) if wandb else ""


def median_tail(steps: list[dict[str, float]], key: str, count: int = 10) -> float:
    values = [step[key] for step in steps if key in step]
    if len(values) < count:
        raise ValueError(f"found only {len(values)} samples for {key!r}; expected at least {count}")
    return statistics.median(values[-count:])


def load_experiment(experiment_id: str) -> dict[str, Any]:
    payload = json.loads(run("beaker", "experiment", "get", experiment_id, "--format", "json"))
    if len(payload) != 1:
        raise ValueError(f"expected one experiment for {experiment_id}, found {len(payload)}")
    return payload[0]


def env_values(job: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["value"]
        for item in job["execution"]["spec"].get("envVars", [])
        if "value" in item
    }


def path_label(env: dict[str, str], ep_size: int) -> str:
    if env.get("OLMOE3_HYBRID_EP_USE_CODE_DEFAULTS") == "1":
        return f"EP{ep_size}/code-default"
    if ep_size == 1:
        if env.get("OLMOE3_HYBRID_DP_USE_REDUCE_SCATTER") == "1":
            return "EP1/reduce-scatter"
        return "EP1/all-reduce"
    return f"EP{ep_size}/{env.get('OLMOE3_HYBRID_EP_PATH', 'unknown')}"


def collect(experiment_id: str, family: str, task_pattern: re.Pattern[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    experiment = load_experiment(experiment_id)
    for job in experiment["jobs"]:
        if task_pattern.search(job["name"]) is None:
            continue
        status = job["status"]
        if "finalized" not in status:
            print(f"pending: {job['name']} ({job['id']})")
            continue
        if status.get("exitCode") != 0:
            print(f"failed: {job['name']} ({job['id']}), exit={status.get('exitCode')}")
            continue

        env = env_values(job)
        logs = run("beaker", "job", "logs", job["id"], "--no-timestamps")
        steps, wandb_url = parse_steps(logs)
        tflops = median_tail(steps, "throughput/device/TFLOPs_per_GPU")
        tps = median_tail(steps, "throughput/device/TPS")
        mfu = median_tail(steps, "throughput/device/MFU")
        active = max(step["gpu_memory/GPU active mem (GiB)"] for step in steps)
        reserved = max(step["gpu_memory/GPU reserved mem (GiB)"] for step in steps)
        skipped = max(step.get("optim/step skipped", 0.0) for step in steps)
        batch_tokens = int(env["OLMOE3_HYBRID_GLOBAL_BATCH_SIZE"])
        gpus = int(env["OLMOE3_HYBRID_WORLD_SIZE"])
        ep_size = int(env["OLMOE3_HYBRID_EP_SIZE"])
        aggregate_tps = tps * gpus
        rows.append(
            {
                "family": family,
                "batch_tokens": str(batch_tokens),
                "gpus": str(gpus),
                "path": path_label(env, ep_size),
                "rank_microbatch_sequences": env["OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES"],
                "final10_median_tflops_gpu": f"{tflops:.2f}",
                "final10_median_tps_gpu": f"{tps:.1f}",
                "aggregate_tps": f"{aggregate_tps:.1f}",
                "stable_step_seconds": f"{batch_tokens / aggregate_tps:.4f}",
                "final10_median_mfu_pct": f"{mfu:.3f}",
                "peak_active_gib": f"{active:.2f}",
                "peak_reserved_gib": f"{reserved:.2f}",
                "max_skipped_steps": f"{skipped:g}",
                "beaker_job": job["id"],
                "wandb_url": wandb_url,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="FAMILY=EXPERIMENT_ID",
        help="Collect one experiment; repeat for capacity and parallelism work",
    )
    parser.add_argument("--task-regex", default=".*")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    collected: list[dict[str, str]] = []
    task_pattern = re.compile(args.task_regex)
    for source in args.source:
        family, separator, experiment_id = source.partition("=")
        if not separator or not family or not experiment_id:
            parser.error(f"invalid --source {source!r}; expected FAMILY=EXPERIMENT_ID")
        collected.extend(collect(experiment_id, family, task_pattern))

    existing: list[dict[str, str]] = []
    if args.output.is_file():
        with args.output.open(newline="") as file:
            existing = list(csv.DictReader(file))
    by_job = {row["beaker_job"]: row for row in existing}
    by_job.update({row["beaker_job"]: row for row in collected})
    rows = sorted(
        by_job.values(),
        key=lambda row: (
            row["family"],
            int(row["batch_tokens"]),
            int(row["gpus"]),
            row["path"],
            int(row["rank_microbatch_sequences"]),
        ),
    )
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(collected)} finalized rows; {len(rows)} total rows in {args.output}")


if __name__ == "__main__":
    main()
