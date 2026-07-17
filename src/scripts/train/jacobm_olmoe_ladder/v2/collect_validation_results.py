#!/usr/bin/env python3
"""Collect v2 post-training validation summaries from W&B."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb
import yaml


V2_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFESTS = (
    V2_DIR / "launchers" / "validation" / "manifests" / "275m_hybrid_geometry_full.yaml",
    V2_DIR / "launchers" / "validation" / "manifests" / "275m_geometry_cx4_cx8_full.yaml",
    V2_DIR / "launchers" / "validation" / "manifests" / "hybrid_scale_completed_full.yaml",
)
DEFAULT_OUTPUT_BASE = V2_DIR / "results" / "validation" / "hybrid_full"
DEFAULT_PROJECT = "ai2-llm/jacobm-olmoe-ladder"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    try:
        return jsonable(value.item())
    except AttributeError:
        return str(value)


def load_targets(manifests: list[Path]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in manifests:
        manifest = yaml.safe_load(manifest_path.read_text())
        checkpoint_root = Path(str(manifest["experiment"]["checkpoint_root"]))
        for target in manifest["targets"]:
            source_run = str(target["source_run"])
            if source_run in seen:
                raise ValueError(f"Duplicate source run across manifests: {source_run}")
            seen.add(source_run)
            targets.append(
                {
                    "source_run": source_run,
                    "eval_run": f"val-{source_run}",
                    "model_size": str(target.get("model_size", "275m")),
                    "variant": str(target["variant"]),
                    "cx": int(target["cx"]),
                    "lr": str(target["lr"]),
                    "checkpoint": str(checkpoint_root / source_run / str(target["step"])),
                    "manifest": str(manifest_path),
                }
            )
    return targets


def collect(targets: list[dict[str, Any]], project: str) -> list[dict[str, Any]]:
    api = wandb.Api(timeout=90)
    eval_names = [target["eval_run"] for target in targets]
    runs = list(api.runs(project, filters={"display_name": {"$in": eval_names}}))
    by_name: dict[str, Any] = {}
    for run in runs:
        previous = by_name.get(run.display_name)
        if previous is None or str(getattr(run, "updated_at", "")) > str(
            getattr(previous, "updated_at", "")
        ):
            by_name[run.display_name] = run

    records: list[dict[str, Any]] = []
    for target in targets:
        run = by_name.get(target["eval_run"])
        summary = dict(run.summary) if run is not None else {}
        eval_metrics = {
            key: jsonable(value)
            for key, value in sorted(summary.items())
            if key.startswith("eval/")
        }
        records.append(
            {
                **target,
                "state": run.state if run is not None else "not_started",
                "wandb_id": run.id if run is not None else None,
                "wandb_url": run.url if run is not None else None,
                "eval_metric_count": len(eval_metrics),
                "eval_metrics": eval_metrics,
            }
        )
    return records


def write_outputs(
    records: list[dict[str, Any]], *, project: str, manifests: list[Path], output_base: Path
) -> tuple[Path, Path]:
    generated_at = datetime.now(UTC)
    states: dict[str, int] = {}
    for record in records:
        states[record["state"]] = states.get(record["state"], 0) + 1
    payload = {
        "generated_at_utc": generated_at.isoformat(),
        "project": project,
        "manifests": [str(path) for path in manifests],
        "target_count": len(records),
        "states": states,
        "records": records,
    }
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# V2 Post-Training Validation Results",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Each finished W&B run used the full post-training validation task set. "
        "The complete metric dictionary is retained in the adjacent JSON file.",
        "",
        "Coverage: " + ", ".join(f"{state}={count}" for state, count in sorted(states.items())),
        "",
        "| Model | Variant | Cx | LR | Source checkpoint | State | Eval metrics | W&B |",
        "|---|---|---:|---:|---|---|---:|---|",
    ]
    for record in sorted(
        records,
        key=lambda item: (item["model_size"], item["variant"], item["cx"], float(item["lr"])),
    ):
        wandb_link = (
            f"[{record['wandb_id']}]({record['wandb_url']})" if record["wandb_url"] else "—"
        )
        lines.append(
            f"| {record['model_size']} | `{record['variant']}` | Cx{record['cx']} | "
            f"{record['lr']} | `{Path(record['checkpoint']).name}` | {record['state']} | "
            f"{record['eval_metric_count']} | {wandb_link} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()

    manifests = [path.resolve() for path in (args.manifest or DEFAULT_MANIFESTS)]
    targets = load_targets(manifests)
    records = collect(targets, args.project)
    paths = write_outputs(
        records,
        project=args.project,
        manifests=manifests,
        output_base=args.output_base,
    )
    print(f"Collected {len(records)} target(s):")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
