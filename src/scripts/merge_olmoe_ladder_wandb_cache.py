#!/usr/bin/env python3
"""Merge legacy OLMoE ladder W&B history caches without downloading runs.

The cache filename is the stable run identity.  When several source directories
contain the same filename, keep the most complete valid entry.  Source order is
the final tie breaker, so put the preferred/current cache first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def cache_score(path: Path, source_priority: int) -> tuple[Any, ...]:
    with path.open() as f:
        payload = json.load(f)
    metadata = payload.get("metadata", {})
    history = payload.get("history", [])
    if not isinstance(metadata, dict) or not isinstance(history, list):
        raise ValueError("expected object metadata and list history")

    keys = metadata.get("history_keys") or []
    return (
        int(metadata.get("cache_version") == 2),
        int(metadata.get("state") == "finished"),
        float(metadata.get("summary_total_tokens") or -1),
        float(metadata.get("summary_step") or -1),
        len(keys),
        len(history),
        -source_priority,
    )


def copy_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def merge(sources: list[Path], destination: Path) -> dict[str, Any]:
    candidates: dict[str, list[tuple[int, Path]]] = {}
    for source_priority, source in enumerate(sources):
        if not source.is_dir():
            raise FileNotFoundError(f"cache source is not a directory: {source}")
        for path in source.glob("*.json"):
            candidates.setdefault(path.name, []).append((source_priority, path))

    destination.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    selected_by_source: Counter[str] = Counter()

    for name, paths in sorted(candidates.items()):
        valid: list[tuple[tuple[Any, ...], int, Path]] = []
        for source_priority, path in paths:
            try:
                valid.append((cache_score(path, source_priority), source_priority, path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                counts["invalid_candidates"] += 1
                errors.append({"path": str(path), "error": str(error)})
        if not valid:
            counts["unresolved_files"] += 1
            continue

        _, source_priority, selected = max(valid, key=lambda item: item[0])
        output = destination / name
        if output.exists():
            try:
                if cache_score(output, -1) >= cache_score(selected, source_priority):
                    counts["already_current"] += 1
                    selected_by_source[str(output.parent)] += 1
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                counts["replaced_invalid_destination"] += 1

        copy_atomically(selected, output)
        counts["copied"] += 1
        selected_by_source[str(sources[source_priority])] += 1

    report = {
        "protocol": "olmoe_ladder_wandb_cache_merge_v1",
        "sources": [str(path.resolve()) for path in sources],
        "destination": str(destination.resolve()),
        "candidate_filenames": len(candidates),
        "destination_json_files": len(list(destination.glob("*.json"))),
        "counts": dict(sorted(counts.items())),
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "errors": errors,
    }
    report_path = destination / "migration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.source, args.destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["errors"] or report["counts"].get("unresolved_files", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
