#!/usr/bin/env python
"""Write an OLMoBase eval dashboard from Beaker results.

Completed Beaker experiment results are downloaded once and cached under
``results/cache/olmobase``. Incomplete experiments are listed live but not cached
as final results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LADDER_DIR = Path(__file__).parent
RESULTS_DIR = LADDER_DIR / "results"
DEFAULT_GROUPS = [
    "01KWNA4QFKYY6K87BBJXNRDAMK",  # ai2/olmo-instruct small-model eval group
    "01KWNAAV94Z28846JBYK29ZH3T",  # normal workspace larger-model eval group
    "01KWV53TWPXFD7KSM9BFVEFK19",  # 1.2B integration Cx1/Cx2 eval group
]
CACHE_VERSION = 1
HIGH_LEVEL_SUITES = [
    ("olmobase:mcqa_stem", "mcqa_stem"),
    ("olmobase:mcqa_non_stem", "mcqa_non_stem"),
    ("olmobase:gen", "gen"),
    ("olmobase:math", "math"),
    ("olmobase:easy:qa:rc", "easy_qa_rc"),
    ("olmobase:easy:qa:bpb", "easy_qa_bpb"),
    ("olmobase:easy:math:bpb", "easy_math_bpb"),
    ("olmobase:easy:code:bpb", "easy_code_bpb"),
]


def run_json(cmd: list[str]) -> Any:
    return json.loads(subprocess.check_output(cmd, text=True))


def latest_job(experiment: dict[str, Any]) -> dict[str, Any]:
    jobs = experiment.get("jobs") or []
    return jobs[-1] if jobs else {}


def status_label(job: dict[str, Any]) -> str:
    status = job.get("status", {})
    for key in ("failed", "canceled", "finalized", "started", "scheduled", "created"):
        if key in status:
            return key
    return "unknown"


def is_successfully_final(job: dict[str, Any]) -> bool:
    status = job.get("status", {})
    return "finalized" in status and "failed" not in status and "canceled" not in status and not status.get("message")


def normalize_name(name: str) -> str:
    return re.sub(r"-[0-9a-f]{4}$", "", name)


def model_sort_key(name: str) -> tuple[int, str]:
    for idx, marker in enumerate(("275m", "480m", "810m", "1p2b")):
        if marker in name:
            return idx, name
    return 99, name


def size_sort_key(size: str) -> int:
    return {"275M": 0, "480M": 1, "810M": 2, "1.2B": 3}.get(size, 99)


def direction_for_metric(suite: str, metric: str) -> str:
    metric_lower = metric.lower()
    suite_lower = suite.lower()
    if any(token in metric_lower for token in ("bits_per_byte", "bpb", "loss", "perplexity")):
        return "lower"
    if any(token in metric_lower for token in ("accuracy", "exact_match", "pass@", "pass_at", "f1", "recall")):
        return "higher"
    if "primary_score" in metric_lower:
        if any(token in suite_lower for token in ("bpb", "loss", "perplexity")):
            return "lower"
        return "higher"
    if any(token in suite_lower for token in ("bpb", "loss", "perplexity")):
        return "lower"
    return "see metric"


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
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


def suite_table_rows(suites: list[str], records: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for suite in suites:
        metric = next((record["suites"][suite]["metric"] for record in records if suite in record["suites"]), "")
        if not metric:
            continue
        rows.append([suite, metric, direction_for_metric(suite, metric)] + [fmt(record["suites"].get(suite, {}).get("score")) for record in records])
    return rows


def display_size(name: str) -> str:
    match = re.search(r"(?:^|-)(275m|480m|810m|1p2b)(?:-|$)", name)
    if not match:
        return ""
    return {"275m": "275M", "480m": "480M", "810m": "810M", "1p2b": "1.2B"}[match.group(1)]


def display_cx(name: str) -> str:
    match = re.search(r"(?:^|-)(cx\d+)(?:-|$)", name)
    return match.group(1).replace("cx", "Cx") if match else ""


def display_intervention(name: str) -> str:
    labels = (
        ("baseline", "baseline"),
        ("int-wide", "integration 1 wide"),
        ("intw256e8k", "integration 1 wide"),
        ("int-deep", "integration 1 deep"),
        ("intd256e8k", "integration 1 deep"),
        ("q3am", "qwen-like 4.5d"),
        ("q3td", "qwen-like 3.0d"),
        ("se0m9", "shared expert"),
        ("sp192e4k", "sparsity 192E/top4"),
        ("sp96e4k", "sparsity 96E/top4"),
        ("eg96e8k", "granularity 96E/top8"),
        ("eg24e2k", "granularity 24E/top2"),
        ("dense", "dense schedule"),
    )
    for token, label in labels:
        if token in name:
            return label
    cleaned = normalize_name(name)
    cleaned = re.sub(r"^olmoe3-", "", cleaned)
    cleaned = re.sub(r"-olmobase$", "", cleaned)
    cleaned = re.sub(r"(?:^|-)(275m|480m|810m|1p2b)(?:-|$)", "-", cleaned)
    cleaned = re.sub(r"-?cx\d+", "", cleaned)
    cleaned = cleaned.strip("-")
    return cleaned or name


def high_level_table(records: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    headers = ["size", "Cx", "intervention"]
    suite_directions: list[str] = []
    for suite, label in HIGH_LEVEL_SUITES:
        metric = next((record["suites"][suite]["metric"] for record in records if suite in record["suites"]), "")
        direction = direction_for_metric(suite, metric) if metric else ""
        suite_directions.append(direction)
        arrow = "down" if direction == "lower" else "up" if direction == "higher" else ""
        headers.append(f"{label} {arrow}".strip())

    row_records = []
    for record in records:
        values: list[float | None] = []
        for suite, _label in HIGH_LEVEL_SUITES:
            score = record["suites"].get(suite, {}).get("score")
            try:
                values.append(float(score))
            except (TypeError, ValueError):
                values.append(None)
        if any(value is not None for value in values):
            row_records.append(
                {
                    "size": display_size(record["name"]),
                    "cx": display_cx(record["name"]),
                    "intervention": display_intervention(record["name"]),
                    "values": values,
                }
            )
    row_records.sort(key=lambda row: (size_sort_key(row["size"]), row["cx"], row["intervention"]))

    best_by_size: dict[str, list[float | None]] = defaultdict(lambda: [None] * len(HIGH_LEVEL_SUITES))
    for row in row_records:
        for idx, value in enumerate(row["values"]):
            if value is None:
                continue
            current = best_by_size[row["size"]][idx]
            direction = suite_directions[idx]
            if current is None:
                best_by_size[row["size"]][idx] = value
            elif direction == "lower" and value < current:
                best_by_size[row["size"]][idx] = value
            elif direction != "lower" and value > current:
                best_by_size[row["size"]][idx] = value

    rows = []
    previous_size = None
    for row in row_records:
        if previous_size is not None and row["size"] != previous_size:
            rows.append([""] * len(headers))
        previous_size = row["size"]
        values = []
        for idx, value in enumerate(row["values"]):
            formatted = fmt(value)
            best = best_by_size[row["size"]][idx]
            if value is not None and best is not None and math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12):
                formatted = f"**{formatted}**"
            values.append(formatted)
        rows.append([row["size"], row["cx"], row["intervention"], *values])
    return headers, rows


def find_result_json(download_dir: Path) -> dict[str, Any] | None:
    candidates = []
    for path in download_dir.rglob("*.json"):
        if path.stat().st_size > 200_000_000:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        score = 0
        if isinstance(data, dict):
            score += 5 if "suites" in data else 0
            score += 3 if "summary" in data else 0
            score += 1 if "tasks" in data else 0
        if score:
            candidates.append((score, path, data))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][2]


def cache_path(cache_dir: Path, experiment_id: str) -> Path:
    return cache_dir / f"{experiment_id}.json"


def load_or_download_result(experiment: dict[str, Any], *, cache_dir: Path, refresh_cache: bool) -> dict[str, Any] | None:
    experiment_id = experiment["id"]
    path = cache_path(cache_dir, experiment_id)
    if not refresh_cache and path.exists():
        with path.open() as f:
            cached = json.load(f)
        if cached.get("metadata", {}).get("cache_version") == CACHE_VERSION:
            return cached

    work_dir = cache_dir / "downloads" / experiment_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["beaker", "experiment", "results", experiment_id, "--prefix", "metrics.json", "--output", str(work_dir)], check=True)
    result = find_result_json(work_dir)
    if result is None:
        return None
    payload = {
        "metadata": {
            "cache_version": CACHE_VERSION,
            "experiment_id": experiment_id,
            "experiment_name": experiment["name"],
            "cached_at_utc": datetime.now(UTC).isoformat(),
        },
        "result": result,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)
    return payload


def suite_scores(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = payload.get("result", {})
    suites = result.get("suites") or {}
    out = {}
    for suite_name, suite_data in suites.items():
        metric_key = suite_data.get("primary_metric")
        metrics = suite_data.get("metrics", {})
        score = None
        if metric_key and ":" in metric_key:
            metric, scorer = metric_key.split(":", 1)
            score = metrics.get(metric, {}).get(scorer)
        if score is None and "primary_score" in metrics:
            metric_key = "primary_score:average"
            score = metrics.get("primary_score", {}).get("average")
        if score is not None:
            out[suite_name] = {"metric": metric_key or "", "score": score}
    if out:
        return out

    summary = result.get("summary") or {}
    for suite_name, suite_data in summary.items():
        if not isinstance(suite_data, dict):
            continue
        metric = suite_data.get("metric")
        score = suite_data.get("score")
        if metric is not None and score is not None:
            out[suite_name] = {"metric": metric, "score": score}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", default=[], help="Beaker group ID/name. Defaults to current Cx8 OLMoBase groups.")
    parser.add_argument("--cache-dir", type=Path, default=RESULTS_DIR / "cache" / "olmobase")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "olmobase_evals.md")
    parser.add_argument("--json-output", type=Path, default=RESULTS_DIR / "olmobase_evals.json")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--skip-uncached-final",
        action="store_true",
        help="List newly finalized evals without downloading their result artifacts. Cached finalized evals are still included.",
    )
    args = parser.parse_args()

    groups = args.group or DEFAULT_GROUPS
    by_name: dict[str, dict[str, Any]] = {}
    for group in groups:
        try:
            experiments = run_json(["beaker", "group", "experiments", group, "--format", "json"])
        except subprocess.CalledProcessError as exc:
            print(f"warning: failed to read group {group}: {exc}")
            continue
        for experiment in experiments:
            base = normalize_name(experiment["name"])
            previous = by_name.get(base)
            if previous is None or experiment.get("created", "") > previous.get("created", ""):
                by_name[base] = experiment

    records = []
    for name, experiment in sorted(by_name.items(), key=lambda item: model_sort_key(item[0])):
        job = latest_job(experiment)
        status = status_label(job)
        record = {
            "experiment_id": experiment["id"],
            "name": name,
            "workspace": experiment.get("workspaceRef", {}).get("fullName", ""),
            "status": status,
            "url": f"https://beaker.org/ex/{experiment['id']}",
            "suites": {},
        }
        if is_successfully_final(job):
            cached_path = cache_path(args.cache_dir, experiment["id"])
            if args.skip_uncached_final and not cached_path.exists():
                record["message"] = "finalized; result artifact not cached yet"
            else:
                payload = load_or_download_result(experiment, cache_dir=args.cache_dir, refresh_cache=args.refresh_cache)
                if payload is not None:
                    record["suites"] = suite_scores(payload)
        else:
            message = job.get("status", {}).get("message") or job.get("status", {}).get("canceledFor") or ""
            record["message"] = message
        records.append(record)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w") as f:
        json.dump({"generated_at_utc": datetime.now(UTC).isoformat(), "records": records}, f, indent=2, sort_keys=True)

    lines = [
        "# OLMoBase Eval Results",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Values are suite-level aggregates emitted by `olmo-eval`. Higher is better for accuracy/F1/pass-style metrics; lower is better for BPB/loss/perplexity-style metrics. The `direction` column is a heuristic based on suite/metric names, so treat `see metric` rows literally.",
        "",
        f"Completed result caches live under `{args.cache_dir}`.",
        "",
    ]
    high_level_headers, high_level_rows = high_level_table(records)
    if high_level_rows:
        lines.extend(["## High-Level Aggregates", ""])
        lines.extend(md_table(high_level_headers, high_level_rows))
        lines.append("")

    lines.extend([
        "## Status",
        "",
    ])
    lines.extend(md_table(["model", "status", "workspace", "link", "message"], [[r["name"], r["status"], r["workspace"], f"[beaker]({r['url']})", r.get("message", "")] for r in records]))
    lines.append("")

    suites = sorted({suite for record in records for suite in record["suites"]})
    if suites:
        lines.extend(["## Suite Aggregates", ""])
        headers = ["suite", "metric", "direction"] + [record["name"] for record in records]
        lines.extend(md_table(headers, suite_table_rows(suites, records)))
    else:
        lines.extend(["## Suite Aggregates", "", "No completed OLMoBase result files found yet."])

    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
