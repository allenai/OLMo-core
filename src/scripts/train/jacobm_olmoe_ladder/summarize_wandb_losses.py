#!/usr/bin/env python
"""Summarize W&B train/CE loss by final-token windows.

This is intended for quick U-plot tables while comparing runs with different
global batch sizes. It uses token windows instead of step windows so 2M-token and
256k-token batches are compared over the same amount of training data.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass

import wandb


WANDB_PATH = "ai2-llm/jacobm-olmoe-ladder"
LOSS_KEY = "train/CE loss"
TOKENS_KEY = "throughput/total tokens"


@dataclass(frozen=True)
class RunSpec:
    chinchilla: str
    batch_label: str
    batch_tokens: int
    lr: str


def parse_run_spec(name: str) -> RunSpec | None:
    if "olmoe3-tiny-275m-cx" not in name:
        return None

    chinchilla_match = re.search(r"cx([0-9]+)", name)
    if chinchilla_match is None:
        return None

    if "b128k" in name:
        batch_label, batch_tokens = "128k", 131_072
    elif "b256k" in name:
        batch_label, batch_tokens = "256k", 262_144
    elif "b512k" in name:
        batch_label, batch_tokens = "512k", 524_288
    elif "cx1-lr" in name:
        batch_label, batch_tokens = "2M", 2_097_152
    else:
        return None

    match = re.search(
        r"lr(3\.5e-3|2\.5e-3|1\.2e-3|1\.5e-3|2e-3|1e-3|8e-4|7e-4|6e-4|5e-4|4e-4|3e-4|1e-4)",
        name,
    )
    if match is None:
        return None

    return RunSpec(
        chinchilla=chinchilla_match.group(1),
        batch_label=batch_label,
        batch_tokens=batch_tokens,
        lr=match.group(1),
    )


def lr_sort_key(lr: str) -> float:
    return float(lr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=WANDB_PATH)
    parser.add_argument("--name-regex", default="olmoe3-tiny-275m-cx1")
    parser.add_argument(
        "--windows-m",
        type=int,
        nargs="+",
        default=[100, 250, 500],
        help="Final-token averaging windows in millions of tokens.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = list(api.runs(args.project))
    name_re = re.compile(args.name_regex)
    fields = ["_step", LOSS_KEY, TOKENS_KEY]
    windows = [window_m * 1_000_000 for window_m in args.windows_m]
    rows: list[dict[str, object]] = []

    for run in runs:
        if not name_re.search(run.display_name):
            continue

        spec = parse_run_spec(run.display_name)
        if spec is None:
            continue

        history: list[tuple[int, int, float]] = []
        for row in run.scan_history(keys=fields, page_size=1000):
            loss = row.get(LOSS_KEY)
            step = row.get("_step")
            tokens = row.get(TOKENS_KEY)
            if loss is None or step is None:
                continue
            if tokens is None:
                tokens = int(step) * spec.batch_tokens
            history.append((int(step), int(tokens), float(loss)))

        if not history:
            continue

        history.sort(key=lambda item: item[1])
        final_step, final_tokens, final_loss = history[-1]
        out: dict[str, object] = {
            "cx": spec.chinchilla,
            "batch": spec.batch_label,
            "lr": spec.lr,
            "state": run.state,
            "step": final_step,
            "tokens": final_tokens,
            "final": final_loss,
            "run_id": run.id,
            "url": run.url,
        }

        for window in windows:
            cutoff = final_tokens - window
            vals = [loss for _, tokens, loss in history if tokens >= cutoff]
            window_m = window // 1_000_000
            out[f"avg_last_{window_m}M"] = statistics.mean(vals) if vals else math.nan
            out[f"n_last_{window_m}M"] = len(vals)

        rows.append(out)

    rows.sort(key=lambda row: (float(row["cx"]), str(row["batch"]), lr_sort_key(str(row["lr"]))))

    window_headers = []
    for window in windows:
        window_m = window // 1_000_000
        window_headers.append(f"avg{window_m}M")
        window_headers.append(f"n{window_m}M")

    print("\t".join(["cx", "batch", "lr", "state", "step", "tokensB", "final", *window_headers, "run_id", "url"]))
    for row in rows:
        values = [
            str(row["cx"]),
            str(row["batch"]),
            str(row["lr"]),
            str(row["state"]),
            str(row["step"]),
            f"{int(row['tokens']) / 1e9:.3f}",
            f"{float(row['final']):.4f}",
        ]
        for window in windows:
            window_m = window // 1_000_000
            values.append(f"{float(row[f'avg_last_{window_m}M']):.4f}")
            values.append(str(row[f"n_last_{window_m}M"]))
        values.extend([str(row["run_id"]), str(row["url"])])
        print("\t".join(values))


if __name__ == "__main__":
    main()
