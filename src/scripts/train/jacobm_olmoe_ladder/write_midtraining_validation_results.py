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
SOURCE_CX_ORDER = ("Cx1", "Cx2", "Cx4", "Cx8")
MODEL_ORDER = ("275M", "480M", "810M", "1.2B")


@dataclass(frozen=True)
class MidtrainTarget:
    source: str
    source_cx: str
    lr: str
    beaker_id: str
    checkpoint: str
    run_name: str | None = None

    @property
    def semantic_name(self) -> str:
        return self.run_name or f"mt-275m-baseline-{self.source_cx.lower()}-lr{self.lr}-r1"

    @property
    def model_size(self) -> str:
        return self.source.split()[0]


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
        source="275M baseline Cx2",
        source_cx="Cx2",
        lr="1.8e-4",
        beaker_id="01KWZ8T9BZ3B869VZ878FNNQ8T",
        checkpoint="olmoe3-tiny-275m-cx2-b384k-gpu2-ep1mb8-lr1.8e-3-r3/step20486",
    ),
    MidtrainTarget(
        source="275M baseline Cx4",
        source_cx="Cx4",
        lr="1.5e-4",
        beaker_id="01KWZ8T9DZN8Y4VD63AP1AN387",
        checkpoint="olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1.5e-3/step30729",
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
    MidtrainTarget(
        source="480M baseline Cx1",
        source_cx="Cx1",
        lr="1.2e-4",
        beaker_id="01KWZARWH7XAS4MD2238VRKP0Y",
        checkpoint="m480-cx1-b256k-gpu4-ep1mb8-lr1.2e-3-r1/step29022",
        run_name="mt-480m-baseline-cx1-lr1.2e-4-r1",
    ),
    MidtrainTarget(
        source="480M baseline Cx8",
        source_cx="Cx8",
        lr="8e-5",
        beaker_id="01KWZARZ2T7FZDH42Q2VS92WXN",
        checkpoint="m480-cx8-b768k-gpu8-ep1mb4-lr8e-4-r1/step77392",
        run_name="mt-480m-baseline-cx8-lr8e-5-r1",
    ),
    MidtrainTarget(
        source="810M baseline Cx1",
        source_cx="Cx1",
        lr="6e-5",
        beaker_id="01KWZAT4PF87PYA96JT7ZXSQKX",
        checkpoint="olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr6e-4-r1/step52648",
        run_name="mt-810m-baseline-cx1-lr6e-5-r1",
    ),
    MidtrainTarget(
        source="810M baseline Cx8",
        source_cx="Cx8",
        lr="4e-5",
        beaker_id="01KWZAT4PQ0NWD20VS0XT12ZF5",
        checkpoint="olmoe3-moe-a0-810m-cx8-b768k-gpu8-ep1mb4-lr4e-4-r1/step140394",
        run_name="mt-810m-baseline-cx8-lr4e-5-r1",
    ),
    MidtrainTarget(
        source="1.2B baseline Cx1",
        source_cx="Cx1",
        lr="4e-5",
        beaker_id="01KWZAVZFS1FMR59ASRVH7VD4X",
        checkpoint="olmoe3-moe-a0-1p2b-cx1-b256k-gpu8-ep1mb2-lr4e-4-r1/step81190",
        run_name="mt-1p2b-baseline-cx1-lr4e-5-r1",
    ),
    MidtrainTarget(
        source="1.2B baseline Cx8",
        source_cx="Cx8",
        lr="4e-5",
        beaker_id="01KWZAV9AGDSD3PTS5RAM7G5N2",
        checkpoint="olmoe3-moe-a0-1p2b-cx8-b768k-gpu32-ep1mb1-lr4e-4-r1/step216505",
        run_name="mt-1p2b-baseline-cx8-lr4e-5-r1",
    ),
    MidtrainTarget(
        source="275M integration wide Cx1",
        source_cx="Cx1",
        lr="2e-4",
        beaker_id="01KX1WZ551YCZYTHXRQEDX1WK1",
        checkpoint="integration/int-275m-cx1-intw256e8k-lr1.6e-3-r1/step15499",
        run_name="mt-275m-intw256e8k-cx1-lr2e-4-r3",
    ),
    MidtrainTarget(
        source="275M integration wide Cx2",
        source_cx="Cx2",
        lr="1.8e-4",
        beaker_id="01KX1WZGKE5K5GF4SZ98G7SVJD",
        checkpoint="integration/int-275m-cx2-intw256e8k-lr1.6e-3-r1/step20665",
        run_name="mt-275m-intw256e8k-cx2-lr1p8e-4-r3",
    ),
    MidtrainTarget(
        source="275M integration wide Cx4",
        source_cx="Cx4",
        lr="1.5e-4",
        beaker_id="01KX1WZXCBX8VFNFFZK2DT50JB",
        checkpoint="integration/int-275m-cx4-intw256e8k-lr8e-4-r1/step30997",
        run_name="mt-275m-intw256e8k-cx4-lr1p5e-4-r3",
    ),
    MidtrainTarget(
        source="275M integration wide Cx8",
        source_cx="Cx8",
        lr="1.6e-4",
        beaker_id="01KX1X0AJAMCEE5MSXBP33PHNG",
        checkpoint="integration/int-275m-cx8-intw256e8k-lr8e-4-r1/step41329",
        run_name="mt-275m-intw256e8k-cx8-lr1p6e-4-r3",
    ),
    MidtrainTarget(
        source="275M integration deep Cx1",
        source_cx="Cx1",
        lr="2e-4",
        beaker_id="01KX1X0QVHZ7X0HSF0YNJWEBBH",
        checkpoint="integration/int-275m-cx1-intd256e8k-lr1.6e-3-r1/step15130",
        run_name="mt-275m-intd256e8k-cx1-lr2e-4-r3",
    ),
    MidtrainTarget(
        source="275M integration deep Cx2",
        source_cx="Cx2",
        lr="1.8e-4",
        beaker_id="01KX1X14K49GJN4KS9QQ3JEQ1E",
        checkpoint="integration/int-275m-cx2-intd256e8k-lr1.6e-3-r1/step20173",
        run_name="mt-275m-intd256e8k-cx2-lr1p8e-4-r3",
    ),
    MidtrainTarget(
        source="275M integration deep Cx4",
        source_cx="Cx4",
        lr="1.5e-4",
        beaker_id="01KX1X1GPC3N51C7E8AW2YXN63",
        checkpoint="integration/int-275m-cx4-intd256e8k-lr1.6e-3-r1/step30259",
        run_name="mt-275m-intd256e8k-cx4-lr1p5e-4-r3",
    ),
    MidtrainTarget(
        source="275M integration deep Cx8",
        source_cx="Cx8",
        lr="1.6e-4",
        beaker_id="01KX1X203AABJ29GHKST48J5E9",
        checkpoint="integration/int-275m-cx8-intd256e8k-lr1.6e-3-r1/step40345",
        run_name="mt-275m-intd256e8k-cx8-lr1p6e-4-r3",
    ),
    MidtrainTarget(
        source="480M integration wide Cx8",
        source_cx="Cx8",
        lr="8e-5",
        beaker_id="01KX6K2NTZZRVE98HCQWE2DR64",
        checkpoint="integration/int-480m-cx8-intw256e8k-lr8e-4-r1/step78042",
        run_name="mt-480m-intw256e8k-cx8-lr8e-5-r1",
    ),
    MidtrainTarget(
        source="480M integration deep Cx8",
        source_cx="Cx8",
        lr="8e-5",
        beaker_id="01KX6K41XG75V1DQ9NWZB988GQ",
        checkpoint="integration/int-480m-cx8-intd256e8k-lr8e-4-r1/step78659",
        run_name="mt-480m-intd256e8k-cx8-lr8e-5-r1",
    ),
    MidtrainTarget(
        source="810M integration wide Cx8",
        source_cx="Cx8",
        lr="4e-5",
        beaker_id="01KX6K330J3Z5CSDY518BGJ05B",
        checkpoint="integration/int-810m-cx8-intw256e8k-lr4e-4-r1/step141423",
        run_name="mt-810m-intw256e8k-cx8-lr4e-5-r1",
    ),
    MidtrainTarget(
        source="810M integration deep Cx8",
        source_cx="Cx8",
        lr="4e-5",
        beaker_id="01KX6K4GKDHQS05727Q9DH0NF4",
        checkpoint="integration/int-810m-cx8-intd256e8k-lr4e-4-r1/step138619",
        run_name="mt-810m-intd256e8k-cx8-lr4e-5-r1",
    ),
    MidtrainTarget(
        source="1.2B integration wide Cx8",
        source_cx="Cx8",
        lr="4e-5",
        beaker_id="01KX7032K1GXDKQR2D9FAA0CBF",
        checkpoint="integration/int-1p2b-cx8-intw256e8k-lr4e-4-r2/step217870",
        run_name="mt-1p2b-intw256e8k-cx8-lr4e-5-r1",
    ),
    MidtrainTarget(
        source="1.2B integration deep Cx8",
        source_cx="Cx8",
        lr="4e-5",
        beaker_id="01KX703FE8T96HJM4EFG90KHVF",
        checkpoint="integration/int-1p2b-cx8-intd256e8k-lr4e-4-r2/step218156",
        run_name="mt-1p2b-intd256e8k-cx8-lr4e-5-r1",
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


def metric_task_and_measure(metric: str) -> tuple[str, str]:
    downstream = re.match(r"^eval/downstream/(.*?) \((.*?)\)$", metric)
    if downstream:
        return downstream.group(1), downstream.group(2)
    if metric.startswith("eval/lm/"):
        rest = metric.removeprefix("eval/lm/")
        if "/" in rest:
            task, measure = rest.rsplit("/", 1)
            return f"lm/{task}", measure
    return metric, metric


def metric_category(metric: str) -> str:
    if metric.startswith("eval/lm/"):
        return "lm"
    task, _ = metric_task_and_measure(metric)
    if task.startswith("arc_"):
        return "arc"
    if task.startswith("basic_skills_"):
        return "basic_skills"
    if task.startswith("mmlu_"):
        return "mmlu"
    return task.split("_")[0]


def normalized_measure(measure: str) -> str:
    return re.sub(r" v2$", "", measure)


def deduplicated_metrics(metrics: list[str]) -> list[str]:
    """Collapse v2/non-v2 repeats for the same task and score family.

    When both variants are present, keep v2 because it is the newer metric name
    and avoids counting the same score family twice in summary win counts.
    """
    chosen: dict[tuple[str, str], str] = {}
    for metric in sorted(metrics):
        task, measure = metric_task_and_measure(metric)
        key = (task, normalized_measure(measure))
        previous = chosen.get(key)
        if previous is None or measure.endswith(" v2"):
            chosen[key] = metric
    return sorted(chosen.values(), key=lambda metric: (*metric_task_and_measure(metric), metric))


def record_key(record: dict[str, Any]) -> str:
    return f"{record['source']}:{record['lr']}"


def source_sort_key(source: str) -> tuple[int, int, str]:
    parts = source.split()
    model = parts[0] if parts else source
    cx = parts[-1] if parts else ""
    model_idx = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    cx_idx = SOURCE_CX_ORDER.index(cx) if cx in SOURCE_CX_ORDER else len(SOURCE_CX_ORDER)
    return (model_idx, cx_idx, source)


def normalize_run_name(name: str) -> str:
    base = re.sub(r"_.*$", "", name)
    for old, new in (
        ("lr1p2e-4", "lr1.2e-4"),
        ("lr1p5e-4", "lr1.5e-4"),
        ("lr1p6e-4", "lr1.6e-4"),
        ("lr1p8e-4", "lr1.8e-4"),
    ):
        base = base.replace(old, new)
    return base


def metric_winners(records: list[dict[str, Any]], metric: str) -> set[str]:
    values: list[tuple[dict[str, Any], float]] = []
    for record in records:
        value = record["eval_metrics"].get(metric)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append((record, float(value)))
    if not values:
        return set()
    direction = metric_direction(metric)
    if direction == "higher":
        best = max(value for _, value in values)
    elif direction == "lower":
        best = min(value for _, value in values)
    else:
        return set()
    return {record_key(record) for record, value in values if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12)}


def fmt_eval_cell(record: dict[str, Any], metric: str, winners: set[str]) -> str:
    value = fmt(record["eval_metrics"].get(metric))
    if value and record_key(record) in winners:
        return f"**{value}**"
    return value


def win_count_tables(records: list[dict[str, Any]], metrics: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    by_lr = {record["lr"]: 0 for record in records}
    by_category: dict[str, dict[str, int]] = {}
    category_totals: dict[str, int] = {}
    for metric in metrics:
        winners = metric_winners(records, metric)
        category = metric_category(metric)
        category_totals[category] = category_totals.get(category, 0) + 1
        by_category.setdefault(category, {record["lr"]: 0 for record in records})
        for record in records:
            if record_key(record) in winners:
                by_lr[record["lr"]] += 1
                by_category[category][record["lr"]] += 1
    total = len(metrics)
    count_rows = [[lr, f"{by_lr[lr]}/{total}"] for lr in sorted(by_lr, key=lr_sort_key)]
    category_rows = []
    for category in sorted(by_category):
        row = [category, str(category_totals[category])]
        row.extend(str(by_category[category][record["lr"]]) for record in sorted(records, key=lambda r: lr_sort_key(r["lr"])))
        category_rows.append(row)
    return count_rows, category_rows


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
    target_names = {target.semantic_name for target in TARGETS}
    runs = {
        normalized_name: run
        for run in api.runs(args.project, filters={"tags": {"$in": ["exp_midtraining"]}})
        if (normalized_name := normalize_run_name(run.display_name)) in target_names
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

    records.sort(key=lambda r: (*source_sort_key(r["source"]), lr_sort_key(r["lr"])))
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
            "Selection rule: use only `eval/*` validation metrics for midtraining "
            "checkpoint/LR selection. Training loss on the midtraining mixture is "
            "shown only as run-health metadata and must not be used to choose LRs."
        ),
        "",
        (
            "Backfill note: the first 275M grid did not run in-loop evals during "
            "training, so final-checkpoint eval-only backfills are required. Once "
            "those eval jobs finish and `copy_eval_backfills_to_wandb.py` copies "
            "their metrics back, this table will populate the `eval/*` section."
        ),
        "",
        "Settings: 100B midtraining tokens, sequence length 8192, global batch seq 128 "
        "(1,048,576 tokens), 1 node, 4 GPUs, EP1, microbatch 8, fresh optimizer state, "
        "2000-step warmup then constant LR.",
        "",
    ]

    summary_rows = []
    present_sources = sorted({record["source"] for record in records}, key=source_sort_key)

    for source in present_sources:
        group = [record for record in records if record["source"] == source]
        finished = [record for record in group if record["state"] == "finished"]
        with_eval = [record for record in group if record["eval_metrics"]]
        summary_rows.append(
            [
                source,
                f"{len(finished)}/{len(group)}",
                f"{len(with_eval)}/{len(group)}",
                ", ".join(record["lr"] for record in group if record["eval_metrics"]) or "",
                ", ".join(record["lr"] for record in group if record["state"] != "finished") or "",
            ]
        )
    lines.extend(md_table(["source", "training finished", "eval metrics present", "LRs with evals", "still running"], summary_rows))
    lines.append("")

    if eval_metric_names:
        raw_metric_names = eval_metric_names
        dedup_metric_names = deduplicated_metrics(eval_metric_names)
        lines.extend([
            "## Eval Win Summary",
            "",
            (
                "Wins are computed separately within each source checkpoint group. "
                "Raw counts include every logged eval metric. De-duplicated counts "
                "collapse `v2`/non-`v2` repeats for the same task and score family, "
                "preferring `v2` when both are present. Ties, if any, count for every tied LR."
            ),
            "",
        ])
        for source in present_sources:
            group = [record for record in records if record["source"] == source]
            group_metric_names = sorted({metric for record in group for metric in record["eval_metrics"]})
            lines.extend([f"### {source} Win Counts", ""])
            if not group_metric_names:
                lines.extend(["No `eval/*` validation metrics are present for this source group yet.", ""])
                continue
            group_dedup_metric_names = deduplicated_metrics(group_metric_names)
            raw_rows, _ = win_count_tables(group, group_metric_names)
            dedup_rows, category_rows = win_count_tables(group, group_dedup_metric_names)
            dedup_by_lr = {row[0]: row[1] for row in dedup_rows}
            combined_rows = [[lr, raw_count, dedup_by_lr.get(lr, "")] for lr, raw_count in raw_rows]
            lines.extend(md_table(["LR", f"raw wins / {len(group_metric_names)}", f"dedup wins / {len(group_dedup_metric_names)}"], combined_rows))
            lines.append("")
            category_headers = ["category", "dedup metrics"] + [record["lr"] for record in sorted(group, key=lambda r: lr_sort_key(r["lr"]))]
            lines.extend(md_table(category_headers, category_rows))
            lines.append("")

    for source in present_sources:
        lines.extend([f"## {source} Source", ""])
        group = [record for record in records if record["source"] == source]
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
        lines.append("No `eval/*` validation metrics have been copied onto these midtraining runs yet.")
        lines.append("")
    else:
        headers = ["metric", "direction"] + [f"{record['source']} {record['lr']}" for record in records]
        rows = []
        records_by_source = {source: [record for record in records if record["source"] == source] for source in present_sources}
        for metric in eval_metric_names:
            winners = set()
            for group in records_by_source.values():
                winners.update(metric_winners(group, metric))
            rows.append(
                [metric.replace("|", "\\|"), metric_direction(metric)]
                + [fmt_eval_cell(record, metric, winners) for record in records]
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
