#!/usr/bin/env python
"""Write a midtraining validation/progress dashboard from W&B summaries.

The first 275M midtraining grid was launched with ``--eval-task-set=fast`` but
without enabling ladder eval callbacks, so the runs currently do not log
``eval/*`` validation metrics. This dashboard still tracks the grid separately
from the pretraining/in-loop eval dashboards and will surface validation metrics
if later midtraining runs log them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb

LADDER_DIR = Path(__file__).parent
RESULTS_DIR = LADDER_DIR / "results"
DEFAULT_CACHE_DIR = RESULTS_DIR / "cache" / "midtraining_wandb_summaries"
DEFAULT_PROJECT = "ai2-llm/jacobm-olmoe-ladder"
CACHE_VERSION = 1


@dataclass(frozen=True)
class MidtrainTarget:
    source: str
    source_cx: str
    lr: str
    beaker_id: str
    checkpoint: str

    @property
    def semantic_name(self) -> str:
        return f"mt-275m-baseline-{self.source_cx.lower()}-lr{self.lr}-r1"


TARGETS = [
    MidtrainTarget(
        source="275M baseline Cx1",
        source_cx="Cx1",
        lr="2e-4",
        beaker_id="01KWWM1043JEC9MC3PV7PXQ745",
        checkpoint="olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365",
    ),
    MidtrainTarget(
        source="275M baseline Cx1",
        source_cx="Cx1",
        lr="4e-4",
        beaker_id="01KWWM1AVQXMJ3JBJQ1W2G8YAV",
        checkpoint="olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365",
    ),
    MidtrainTarget(
        source="275M baseline Cx1",
        source_cx="Cx1",
        lr="8e-4",
        beaker_id="01KWWM1N0EQDMCFER90NVP9QW0",
        checkpoint="olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365",
    ),
    MidtrainTarget(
        source="275M baseline Cx1",
        source_cx="Cx1",
        lr="1.6e-3",
        beaker_id="01KWWM1ZXN5R5XWK00GH0WA36G",
        checkpoint="olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365",
    ),
    MidtrainTarget(
        source="275M baseline Cx8",
        source_cx="Cx8",
        lr="2e-4",
        beaker_id="01KWWM10ANMMW2YTNN6RKJBGE7",
        checkpoint="olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3-r2/step40971",
    ),
    MidtrainTarget(
        source="275M baseline Cx8",
        source_cx="Cx8",
        lr="4e-4",
        beaker_id="01KWWM1AK89SKDA1KGCX5D8SMM",
        checkpoint="olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3-r2/step40971",
    ),
    MidtrainTarget(
        source="275M baseline Cx8",
        source_cx="Cx8",
        lr="8e-4",
        beaker_id="01KWWM1P9TXRSMDV5DH9EF4KXM",
        checkpoint="olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3-r2/step40971",
    ),
    MidtrainTarget(
        source="275M baseline Cx8",
        source_cx="Cx8",
        lr="1.6e-3",
        beaker_id="01KWWM213KMG47KADPK5Q67GJP",
        checkpoint="olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3-r2/step40971",
    ),
]


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


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return ""
    if abs(f) >= 1_000_000_000:
        return f"{f / 1_000_000_000:.2f}B"
    if abs(f) >= 1_000_000:
        return f"{f / 1_000_000:.2f}M"
    if abs(f) >= 100:
        return f"{f:.1f}"
    if abs(f) >= 10:
        return f"{f:.2f}"
    if abs(f) >= 1:
        return f"{f:.4f}"
    return f"{f:.5f}"


def metric_direction(metric: str) -> str:
    lowered = metric.lower()
    if any(token in lowered for token in ("acc", "accuracy", "exact_match", "f1", "pass@")):
        return "higher"
    if any(token in lowered for token in ("loss", "ppl", "perplexity", "bpb")):
        return "lower"
    return "see metric"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def lr_sort_key(lr: str) -> float:
    return float(lr)


def parse_record(target: MidtrainTarget, run: Any | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    summary = payload["summary"] if payload else {}
    train_loss = summary.get("train/CE loss")
    return {
        "source": target.source,
        "source_cx": target.source_cx,
        "lr": target.lr,
        "semantic_name": target.semantic_name,
        "beaker_id": target.beaker_id,
        "checkpoint": target.checkpoint,
        "wandb_id": getattr(run, "id", None),
        "wandb_name": getattr(run, "display_name", None),
        "wandb_url": getattr(run, "url", None),
        "state": getattr(run, "state", "missing"),
        "step": summary.get("_step"),
        "tokens": summary.get("throughput/total tokens") or summary.get("optim/total tokens"),
        "train_ce_loss": train_loss,
        "train_ppl": summary.get("train/PPL") or (math.exp(train_loss) if isinstance(train_loss, (int, float)) else None),
        "z_loss": summary.get("train/Z loss"),
        "router_z_loss": summary.get("train/router Z loss"),
        "load_balancing_loss": summary.get("train/load balancing loss"),
        "mfu": summary.get("throughput/device/MFU") or summary.get("throughput/device/MFU (actual avg)"),
        "tps": summary.get("throughput/device/TPS") or summary.get("throughput/device/TPS (actual avg)"),
        "eval_metrics": {
            key: value
            for key, value in summary.items()
            if key.startswith("eval/") and isinstance(value, (int, float))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "midtraining_validation.md")
    parser.add_argument("--json-output", type=Path, default=RESULTS_DIR / "midtraining_validation.json")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    api = wandb.Api(timeout=90)
    name_regex = r"^mt-275m-baseline-cx(1|8)-lr(2e-4|4e-4|8e-4|1\.6e-3)-r1_"
    runs = {
        re.sub(r"_.*$", "", run.display_name): run
        for run in api.runs(args.project, filters={"tags": {"$in": ["exp_midtraining"]}})
        if re.search(name_regex, run.display_name)
    }

    records: list[dict[str, Any]] = []
    for target in TARGETS:
        run = runs.get(target.semantic_name)
        payload = (
            load_summary(run, project=args.project, cache_dir=args.cache_dir, refresh_cache=args.refresh_cache)
            if run is not None
            else None
        )
        records.append(parse_record(target, run, payload))

    records.sort(key=lambda r: (r["source_cx"], lr_sort_key(r["lr"])))
    generated_at = datetime.now(UTC)
    payload = {"generated_at_utc": generated_at.isoformat(), "records": records}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    eval_metric_names = sorted({metric for record in records for metric in record["eval_metrics"]})
    lines = [
        "# Midtraining Validation Results",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        (
            "Interpretation: lower is better for CE loss, PPL, Z loss, router Z loss, "
            "and load-balancing loss; higher is better for MFU/TPS. Accuracy-style "
            "validation metrics are higher-is-better when present."
        ),
        "",
        (
            "Note: the current 275M midtraining grid was launched with "
            "`--eval-task-set=fast`, but `midtraining_ladder.py` sets "
            "`ladder_evals=False`, so no `eval/*` validation metrics are present "
            "in W&B for these runs yet. The tables below therefore show final or "
            "latest training/progress metrics and will include validation metrics "
            "automatically once future midtraining runs log them."
        ),
        "",
        "Settings: 100B midtraining tokens, sequence length 8192, global batch seq 128 "
        "(1,048,576 tokens), 1 node, 4 GPUs, EP1, microbatch 8, fresh optimizer state, "
        "2000-step warmup then constant LR.",
        "",
    ]

    summary_rows = []
    for source_cx in ("Cx1", "Cx8"):
        group = [record for record in records if record["source_cx"] == source_cx]
        finished = [record for record in group if record["state"] == "finished"]
        best = min(
            (record for record in finished if record["train_ce_loss"] is not None),
            key=lambda record: float(record["train_ce_loss"]),
            default=None,
        )
        summary_rows.append(
            [
                source_cx,
                f"{len(finished)}/{len(group)}",
                best["lr"] if best else "",
                fmt(best["train_ce_loss"] if best else None),
                fmt(best["train_ppl"] if best else None),
                ", ".join(record["lr"] for record in group if record["state"] != "finished") or "",
            ]
        )
    lines.extend(md_table(["source", "finished", "best finished LR", "best CE", "best PPL", "still running"], summary_rows))
    lines.append("")

    for source_cx in ("Cx1", "Cx8"):
        lines.extend([f"## {source_cx} Source", ""])
        group = [record for record in records if record["source_cx"] == source_cx]
        rows = []
        for record in group:
            link = f"[W&B]({record['wandb_url']})" if record["wandb_url"] else ""
            beaker = f"[Beaker](https://beaker.org/ex/{record['beaker_id']})"
            rows.append(
                [
                    record["lr"],
                    record["state"],
                    fmt(record["tokens"]),
                    fmt(record["train_ce_loss"]),
                    fmt(record["train_ppl"]),
                    fmt(record["z_loss"]),
                    fmt(record["router_z_loss"]),
                    fmt(record["load_balancing_loss"]),
                    fmt(record["mfu"]),
                    fmt(record["tps"]),
                    f"{link} {beaker}".strip(),
                ]
            )
        lines.extend(
            md_table(
                [
                    "LR",
                    "state",
                    "tokens",
                    "train CE",
                    "train PPL",
                    "Z loss",
                    "router Z",
                    "load balance",
                    "MFU",
                    "TPS/GPU",
                    "links",
                ],
                rows,
            )
        )
        lines.append("")

    lines.extend(["## Validation Metrics", ""])
    if not eval_metric_names:
        lines.append("No `eval/*` validation metrics were found for this midtraining grid.")
        lines.append("")
    else:
        headers = ["metric", "direction"] + [f"{record['source_cx']} {record['lr']}" for record in records]
        rows = []
        for metric in eval_metric_names:
            rows.append(
                [metric.replace("|", "\\|"), metric_direction(metric)]
                + [fmt(record["eval_metrics"].get(metric)) for record in records]
            )
        lines.extend(md_table(headers, rows))
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
