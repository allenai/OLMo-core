#!/usr/bin/env python
"""Analyze tiny MoE ladder W&B runs with matched-token comparisons.

This is a slightly more opinionated companion to ``summarize_wandb_losses.py``:

- it can select only the current relaunch family, avoiding canceled/requeued
  predecessors when making live decisions;
- it compares in-flight runs at a shared token count;
- it can print final-window summaries for completed runs without pretending
  partial runs are final.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import wandb

from wandb_cache import DEFAULT_CACHE_DIR, scan_history_cached


WANDB_PATH = "ai2-llm/jacobm-olmoe-ladder"
LOSS_KEY = "train/CE loss"
TOKENS_KEY = "throughput/total tokens"
FIELDS = ["_step", LOSS_KEY, TOKENS_KEY]

CURRENT_FAMILY_MARKERS = ("gpu2-ep1mb16", "gpu4-ep1mb8", "gpu4-ep1mb16")
LR_TAG_RE = re.compile(r"lr([0-9]+(?:\.[0-9]+)?e-[0-9]+)")


@dataclass(frozen=True)
class RunSpec:
    cx: int
    batch_label: str
    batch_tokens: int
    lr_tag: str
    lr: float
    current_family: bool


@dataclass(frozen=True)
class HistoryPoint:
    tokens: int
    step: int
    loss: float


@dataclass
class RunRow:
    name: str
    state: str
    run_id: str
    url: str
    spec: RunSpec
    history: list[HistoryPoint]


def parse_run_spec(name: str) -> RunSpec | None:
    if "olmoe3-tiny-275m-cx" not in name:
        return None

    cx_match = re.search(r"cx([0-9]+)", name)
    lr_match = LR_TAG_RE.search(name)
    if cx_match is None or lr_match is None:
        return None

    if "b128k" in name:
        batch_label, batch_tokens = "128k", 131_072
    elif "b256k" in name:
        batch_label, batch_tokens = "256k", 262_144
    elif "b512k" in name:
        batch_label, batch_tokens = "512k", 524_288
    elif "b768k" in name:
        batch_label, batch_tokens = "768k", 786_432
    elif "b1m" in name:
        batch_label, batch_tokens = "1M", 1_048_576
    elif "cx1-lr" in name:
        batch_label, batch_tokens = "2M", 2_097_152
    else:
        return None

    lr_tag = lr_match.group(1)
    return RunSpec(
        cx=int(cx_match.group(1)),
        batch_label=batch_label,
        batch_tokens=batch_tokens,
        lr_tag=lr_tag,
        lr=float(lr_tag),
        current_family=any(marker in name for marker in CURRENT_FAMILY_MARKERS),
    )


def mean_loss_in_window(history: list[HistoryPoint], end_tokens: int, window_tokens: int) -> tuple[float, int]:
    start_tokens = end_tokens - window_tokens
    vals = [point.loss for point in history if start_tokens <= point.tokens <= end_tokens]
    return (statistics.mean(vals), len(vals)) if vals else (math.nan, 0)


def matched_point(history: list[HistoryPoint], target_tokens: int) -> HistoryPoint | None:
    eligible = [point for point in history if point.tokens <= target_tokens]
    return eligible[-1] if eligible else None


def load_rows(args: argparse.Namespace) -> list[RunRow]:
    api = wandb.Api()
    name_re = re.compile(args.name_regex)
    rows: list[RunRow] = []

    for run in api.runs(args.project):
        name = run.display_name
        if not name_re.search(name):
            continue

        spec = parse_run_spec(name)
        if spec is None:
            continue
        if args.current_family and not spec.current_family:
            continue
        if args.exclude_current_family and spec.current_family:
            continue
        if args.states and run.state not in args.states:
            continue

        history: list[HistoryPoint] = []
        for item in scan_history_cached(
            run,
            project=args.project,
            keys=FIELDS,
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
        ):
            loss = item.get(LOSS_KEY)
            step = item.get("_step")
            tokens = item.get(TOKENS_KEY)
            if loss is None or step is None:
                continue
            if tokens is None:
                tokens = int(step) * spec.batch_tokens
            history.append(HistoryPoint(tokens=int(tokens), step=int(step), loss=float(loss)))

        history.sort(key=lambda point: point.tokens)
        rows.append(
            RunRow(
                name=name,
                state=run.state,
                run_id=run.id,
                url=run.url,
                spec=spec,
                history=history,
            )
        )

    rows.sort(key=lambda row: (row.spec.cx, row.spec.batch_tokens, row.spec.lr, row.name))
    return rows


def infer_target_tokens(rows: Iterable[RunRow], explicit_target: float | None) -> int:
    if explicit_target is not None:
        return int(explicit_target * 1_000_000_000)
    latest = [row.history[-1].tokens for row in rows if row.history]
    if not latest:
        raise SystemExit("No W&B history found for selected runs")
    return min(latest)


def print_matched(rows: list[RunRow], args: argparse.Namespace) -> None:
    target_tokens = infer_target_tokens(rows, args.target_tokens_b)
    windows = [window_m * 1_000_000 for window_m in args.windows_m]
    headers = ["cx", "batch", "lr", "state", "latestB", "matched_step", "matched_loss"]
    for window_m in args.windows_m:
        headers.extend([f"avg{window_m}M", f"n{window_m}M"])
    headers.extend(["run_id", "name", "url"])
    print(f"# target_tokensB\t{target_tokens / 1e9:.3f}")
    print("\t".join(headers))

    for row in rows:
        if not row.history:
            continue
        point = matched_point(row.history, target_tokens)
        if point is None:
            continue

        values = [
            str(row.spec.cx),
            row.spec.batch_label,
            row.spec.lr_tag,
            row.state,
            f"{row.history[-1].tokens / 1e9:.3f}",
            str(point.step),
            f"{point.loss:.4f}",
        ]
        for window in windows:
            avg, count = mean_loss_in_window(row.history, target_tokens, window)
            values.extend([f"{avg:.4f}", str(count)])
        values.extend([row.run_id, row.name, row.url])
        print("\t".join(values))


def print_final(rows: list[RunRow], args: argparse.Namespace) -> None:
    windows = [window_m * 1_000_000 for window_m in args.windows_m]
    headers = ["cx", "batch", "lr", "state", "step", "tokensB", "final"]
    for window_m in args.windows_m:
        headers.extend([f"avg{window_m}M", f"n{window_m}M"])
    headers.extend(["run_id", "name", "url"])
    print("\t".join(headers))

    for row in rows:
        if not row.history:
            continue
        if args.finished_only and row.state != "finished":
            continue
        final = row.history[-1]
        values = [
            str(row.spec.cx),
            row.spec.batch_label,
            row.spec.lr_tag,
            row.state,
            str(final.step),
            f"{final.tokens / 1e9:.3f}",
            f"{final.loss:.4f}",
        ]
        for window in windows:
            avg, count = mean_loss_in_window(row.history, final.tokens, window)
            values.extend([f"{avg:.4f}", str(count)])
        values.extend([row.run_id, row.name, row.url])
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=WANDB_PATH)
    parser.add_argument("--name-regex", default="olmoe3-tiny-275m-cx")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--windows-m", type=int, nargs="+", default=[100, 250, 500])
    parser.add_argument(
        "--mode",
        choices=["matched", "final"],
        default="matched",
        help="Matched compares selected runs at a shared token count; final summarizes each run at its latest point.",
    )
    parser.add_argument(
        "--target-tokens-b",
        type=float,
        default=None,
        help="Matched-token target in billions. Defaults to the least-progressed selected run.",
    )
    parser.add_argument("--current-family", action="store_true", help="Keep only gpu*/ep1mb16 relaunches.")
    parser.add_argument("--exclude-current-family", action="store_true", help="Drop gpu*/ep1mb16 relaunches.")
    parser.add_argument("--states", nargs="+", default=None, help="Optional W&B states to keep, e.g. running finished.")
    parser.add_argument("--finished-only", action="store_true", help="In final mode, print only finished runs.")
    args = parser.parse_args()

    if args.current_family and args.exclude_current_family:
        raise SystemExit("--current-family and --exclude-current-family are mutually exclusive")

    rows = load_rows(args)
    if args.mode == "matched":
        print_matched(rows, args)
    else:
        print_final(rows, args)


if __name__ == "__main__":
    main()
