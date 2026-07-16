#!/usr/bin/env python3
"""Build compact long-context result files from cached olmo-eval metrics."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LADDER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LADDER_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache" / "ruler"
JSON_OUTPUT = RESULTS_DIR / "long_context_evals.json"
MD_OUTPUT = RESULTS_DIR / "long_context_evals.md"


def model_size(run_name: str) -> str:
    match = re.search(r"(?:^|-)(275m|480m|810m|1p2b)(?:-|$)", run_name)
    return match.group(1) if match else "unknown"


def variant(run_name: str) -> str:
    lowered = run_name.lower()
    if "baseline" in lowered:
        return "baseline"
    if "wide" in lowered or "intw" in lowered:
        return "integration_wide"
    if "deep" in lowered or "intd" in lowered:
        return "integration_deep"
    return "unknown"


def read_record(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text())
    model = Path(payload["config"]["provider"]["model"])
    beaker_experiment_id = metrics_path.parents[1].name
    tasks = {
        task["task"]: task["metrics"]["recall"]["substring_recall"]
        for task in payload["tasks"]
    }
    num_instances = sum(task["num_instances"] for task in payload["tasks"])
    score_seconds = max(task["duration_seconds"] for task in payload["tasks"])
    aggregate = payload["summary"]["ruler_all__65536"]["score"]
    provider_init = max(payload.get("provider_init_seconds", {}).values(), default=None)
    return {
        "beaker_experiment_id": beaker_experiment_id,
        "beaker_url": f"https://beaker.org/ex/{beaker_experiment_id}",
        "experiment_name": payload["experiment_name"],
        "run_name": model.parent.name,
        "step": model.name,
        "model_size": model_size(model.parent.name),
        "variant": variant(model.parent.name),
        "hf_checkpoint": str(model),
        "backend": payload["config"]["provider"]["kind"],
        "task_suite": "ruler_all__65536",
        "context_length": 65536,
        "num_tasks": len(tasks),
        "num_instances": num_instances,
        "aggregate_recall": aggregate,
        "task_recall": dict(sorted(tasks.items())),
        "provider_init_seconds": provider_init,
        "scoring_seconds": score_seconds,
        "items_per_second": num_instances / score_seconds,
        "experiment_duration_seconds": payload["experiment_duration_seconds"],
        "errors": payload["errors"],
        "raw_metrics": str(metrics_path),
    }


def format_number(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(records: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# Long-Context Evaluation Results",
        "",
        f"Generated: {generated_at}",
        "",
        "RULER uses 13 tasks with 100 examples per task at 65,536 tokens. "
        "Higher recall is better. Raw metrics and predictions are cached under "
        "`results/cache/ruler/`.",
        "",
        "The canonical inference path is converted HF checkpoints with vLLM on "
        "one Jupiter H100. The current OLMo-core provider does not load OLMo-DDP "
        "checkpoints whose distributed state keys use `module.*.main`.",
        "",
        "| size | variant | checkpoint | backend | aggregate recall | examples | scoring time | examples/s | Beaker |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            "| {model_size} | {variant} | `{run_name}/{step}` | {backend} | "
            "{score} | {num_instances} | {seconds:.1f}s | {rate:.2f} | "
            "[{experiment_id}]({url}) |".format(
                model_size=record["model_size"],
                variant=record["variant"],
                run_name=record["run_name"],
                step=record["step"],
                backend=record["backend"],
                score=format_number(record["aggregate_recall"]),
                num_instances=record["num_instances"],
                seconds=record["scoring_seconds"],
                rate=record["items_per_second"],
                experiment_id=record["beaker_experiment_id"],
                url=record["beaker_url"],
            )
        )

    task_names = sorted({name for record in records for name in record["task_recall"]})
    for record in records:
        lines.extend(
            [
                "",
                f"## {record['model_size']} {record['variant']}",
                "",
                "| task | recall |",
                "| --- | ---: |",
            ]
        )
        for task_name in task_names:
            score = record["task_recall"].get(task_name)
            lines.append(
                f"| `{task_name}` | {format_number(score) if score is not None else ''} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    metrics_paths = sorted(CACHE_DIR.glob("*/main/metrics.json"))
    if not metrics_paths:
        raise RuntimeError(f"No cached RULER metrics found under {CACHE_DIR}")

    records = sorted(
        (read_record(path) for path in metrics_paths),
        key=lambda record: (record["model_size"], record["variant"], record["run_name"]),
    )
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    output = {
        "protocol": "jacobm_olmoe_long_context_evals_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
    }
    JSON_OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    MD_OUTPUT.write_text(render_markdown(records, generated_at))
    print(f"Wrote {len(records)} record(s) to {JSON_OUTPUT} and {MD_OUTPUT}")


if __name__ == "__main__":
    main()
