#!/usr/bin/env python3
"""Audit the durable receipts for the Jacob OLMoE DDP migration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REQUIRED_MODEL_CHECKS = (
    "schema",
    "conversion",
    "strict_tensors",
    "exact_logits",
    "local_publication_marker",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit(
    manifest_path: Path,
    family_status_root: Path,
    artifact_status_root: Path,
    artifact_labels: list[str],
) -> dict[str, Any]:
    problems: list[str] = []
    manifest = load_json(manifest_path)
    models = manifest.get("models")
    if not isinstance(models, list):
        raise TypeError("publication manifest models must be a list")

    expected_by_family: dict[str, set[str]] = defaultdict(set)
    for model in models:
        model_id = model.get("id")
        family = model.get("family")
        if not isinstance(model_id, str) or not isinstance(family, str):
            problems.append(f"invalid manifest model identity: {model!r}")
            continue
        if model.get("optimizer_state_included") is not False:
            problems.append(f"manifest includes optimizer state: {model_id}")
        if model.get("trainer_state_included") is not False:
            problems.append(f"manifest includes trainer state: {model_id}")
        expected_by_family[family].add(model_id)

    family_results: dict[str, Any] = {}
    completed_model_ids: set[str] = set()
    for family, expected_ids in sorted(expected_by_family.items()):
        summary_path = family_status_root / family / "family_summary.json"
        if not summary_path.is_file():
            problems.append(f"missing family summary: {family}")
            family_results[family] = {"status": "MISSING", "expected": len(expected_ids)}
            continue
        summary = load_json(summary_path)
        rows = summary.get("models")
        if not isinstance(rows, list):
            problems.append(f"family summary models is not a list: {family}")
            rows = []
        received_ids = [row.get("model_id") for row in rows if isinstance(row, dict)]
        received_set = {model_id for model_id in received_ids if isinstance(model_id, str)}
        if len(received_ids) != len(received_set):
            problems.append(f"duplicate model receipt in family: {family}")
        if summary.get("status") != "FAMILY_COMPLETE":
            problems.append(f"family is not complete: {family} ({summary.get('status')})")
        if summary.get("model_count") != len(expected_ids):
            problems.append(f"family model_count mismatch: {family}")
        if received_set != expected_ids:
            missing = sorted(expected_ids - received_set)
            extra = sorted(received_set - expected_ids)
            problems.append(f"family identity mismatch: {family}; missing={missing}; extra={extra}")
        for row in rows:
            if not isinstance(row, dict):
                problems.append(f"invalid model receipt in family: {family}")
                continue
            model_id = row.get("model_id")
            failed_checks = [key for key in REQUIRED_MODEL_CHECKS if row.get(key) is not True]
            if failed_checks:
                problems.append(f"model checks failed: {model_id}; {failed_checks}")
            elif isinstance(model_id, str):
                completed_model_ids.add(model_id)
        family_results[family] = {
            "status": summary.get("status"),
            "expected": len(expected_ids),
            "received": len(received_set),
            "completed_at": summary.get("completed_at"),
        }

    artifact_results: dict[str, Any] = {}
    for label in artifact_labels:
        receipt_path = artifact_status_root / label / "_SUCCESS.json"
        if not receipt_path.is_file():
            problems.append(f"missing artifact receipt: {label}")
            artifact_results[label] = {"status": "MISSING"}
            continue
        receipt = load_json(receipt_path)
        if receipt.get("status") != "COMPLETE":
            problems.append(f"artifact is not complete: {label}")
        if receipt.get("file_count", 0) <= 0 or receipt.get("total_bytes", 0) <= 0:
            problems.append(f"artifact inventory is empty: {label}")
        if "no changes" not in str(receipt.get("verification", "")):
            problems.append(f"artifact lacks no-change checksum verification: {label}")
        artifact_results[label] = receipt

    expected_model_ids = set().union(*expected_by_family.values()) if expected_by_family else set()
    if completed_model_ids != expected_model_ids:
        problems.append(
            f"completed checkpoint set mismatch: expected={len(expected_model_ids)} "
            f"completed={len(completed_model_ids)}"
        )

    return {
        "protocol": "jacobm_olmoe_ddp_migration_audit_v1",
        "status": "COMPLETE" if not problems else "INCOMPLETE",
        "audited_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "expected_checkpoints": len(models),
        "completed_checkpoints": len(completed_model_ids),
        "families": family_results,
        "artifacts": artifact_results,
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("JACOBM_DDP_PUBLICATION_MANIFEST.json"))
    parser.add_argument("--family-status-root", type=Path, required=True)
    parser.add_argument("--artifact-status-root", type=Path, required=True)
    parser.add_argument("--artifact-label", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.manifest,
        args.family_status_root,
        args.artifact_status_root,
        args.artifact_label,
    )
    write_json_atomically(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
