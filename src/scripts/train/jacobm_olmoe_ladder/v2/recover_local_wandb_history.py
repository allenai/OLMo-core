#!/usr/bin/env python
"""Recover a strictly verified loss tail from a local W&B binary run file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

LOSS_KEY = "train/CE loss"
TOKENS_KEY = "throughput/total tokens"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_row(record: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for item in record.history.item:
        key = item.key or "/".join(item.nested_key)
        if key in {"_step", TOKENS_KEY, LOSS_KEY}:
            row[key] = json.loads(item.value_json)
    return row


def _mean_tail_loss(rows: list[dict[str, Any]], window_tokens: int) -> float:
    latest_loss_by_tokens: dict[float, float] = {}
    for row in sorted(rows, key=lambda item: float(item["_step"])):
        tokens = float(row[TOKENS_KEY])
        loss = float(row[LOSS_KEY])
        if math.isfinite(tokens) and math.isfinite(loss):
            latest_loss_by_tokens[tokens] = loss
    samples = sorted(latest_loss_by_tokens.items())
    end_tokens = samples[-1][0]
    if end_tokens - samples[0][0] < window_tokens:
        raise ValueError("retained history does not cover the requested final-token window")
    losses = [loss for tokens, loss in samples if tokens >= end_tokens - window_tokens]
    return statistics.mean(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--expected-final-tokens", type=int, required=True)
    parser.add_argument("--window-tokens", type=int, default=250_000_000)
    parser.add_argument("--retain-tokens", type=int, default=1_000_000_000)
    parser.add_argument(
        "--verified-training-complete",
        action="store_true",
        help="Assert that trainer logs and the final checkpoint independently prove completion.",
    )
    args = parser.parse_args()
    if not args.verified_training_complete:
        parser.error("--verified-training-complete is required before publishing a recovery")
    if args.retain_tokens < args.window_tokens:
        parser.error("--retain-tokens must be at least --window-tokens")

    store = DataStore()
    store.open_for_scan(str(args.source))
    total_records = 0
    history_records = 0
    paired_rows: list[dict[str, Any]] = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        total_records += 1
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        history_records += 1
        row = _history_row(record)
        if all(key in row for key in ("_step", TOKENS_KEY, LOSS_KEY)):
            paired_rows.append(row)

    if not paired_rows:
        raise ValueError("no paired step/token/loss history rows found")
    paired_rows.sort(key=lambda row: float(row["_step"]))
    final_row = paired_rows[-1]
    final_step = int(final_row["_step"])
    final_tokens = int(final_row[TOKENS_KEY])
    if final_step != args.expected_final_step:
        raise ValueError(f"final step {final_step} != expected {args.expected_final_step}")
    if final_tokens != args.expected_final_tokens:
        raise ValueError(f"final tokens {final_tokens} != expected {args.expected_final_tokens}")

    retain_after = final_tokens - args.retain_tokens
    first_retained_index = next(
        index
        for index, row in enumerate(paired_rows)
        if float(row[TOKENS_KEY]) >= retain_after
    )
    # Keep the preceding sample so the retained span is guaranteed to cover
    # the requested interval despite discrete token increments.
    first_retained_index = max(0, first_retained_index - 1)
    retained_rows = paired_rows[first_retained_index:]
    mean_loss = _mean_tail_loss(retained_rows, args.window_tokens)

    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "display_name": args.display_name,
        "source_file": str(args.source),
        "source_file_size": args.source.stat().st_size,
        "source_sha256": _sha256(args.source),
        "remote_state_at_recovery": "crashed",
        "verified_training_complete": True,
        "final_step": final_step,
        "final_tokens": final_tokens,
        "window_tokens": args.window_tokens,
        "mean_final_window_ce": mean_loss,
        "total_records": total_records,
        "history_records": history_records,
        "paired_history_records": len(paired_rows),
        "retained_history_records": len(retained_rows),
        "rows": retained_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"recovered {len(retained_rows)} tail rows; final step={final_step}, "
        f"tokens={final_tokens}, avg{args.window_tokens / 1e6:.0f}M={mean_loss:.12f}"
    )


if __name__ == "__main__":
    main()
