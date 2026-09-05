"""Read-only CPU sign-off for the matched integration smoke and uploader manifests."""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

MOUNT = Path("/weka/olmo-3p5-checkpoints")


def snapshot(name: str) -> tuple[dict, dict]:
    """Read only small audit/checkpoint metadata, never model or optimizer tensors."""
    root = MOUNT / "production-integration" / name
    audit = root / "audit"
    state = MOUNT / "uploader/state"
    registration = MOUNT / "uploader/control/registrations" / f"{name}.json"
    result = {"run": name, "sessions": [], "checkpoints": [], "metrics": []}
    evidence = {"weights": None, "inputs": {}}
    if registration.is_file():
        reg = json.loads(registration.read_text())
        assert reg["deletion_mode"] == "report_only" and reg["enabled"]
        assert reg["checkpoint_root"] == str(root)
        result["registration"] = reg
    for path in sorted(audit.glob("session-*.json")):
        result["sessions"].append(json.loads(path.read_text()))
    weights = audit / "initial-weights-sha256.json"
    if weights.is_file():
        evidence["weights"] = json.loads(weights.read_text())
    for path in sorted(audit.glob("input-step*-rank*.sha256")):
        evidence["inputs"][path.name] = path.read_text()
    metrics = audit / "metrics.jsonl"
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live final append can be incomplete; retry on next poll.
            selected = {
                key: value
                for key, value in record.items()
                if key in ("step", "train/CE loss", "optim/total grad norm", "optim/step skipped")
                or key.startswith("eval/")
            }
            for key, value in selected.items():
                if isinstance(value, (int, float)):
                    assert math.isfinite(value), (name, record["step"], key, value)
            result["metrics"].append(selected)
    for step in (0, 4, 8):
        checkpoint = root / f"step{step}"
        record_path = state / "checkpoints" / name / f"step-{step:012d}.json"
        item = {
            "step": step,
            "source_complete": all(
                (checkpoint / rel).is_file()
                for rel in (".metadata.json", "model_and_optim/.metadata", "train/rank0.pt")
            ),
            "remote_verified": False,
        }
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            assert not record["local_deleted"] and record["deletion_attempts"] == 0
            item.update(
                {
                    key: record.get(key)
                    for key in (
                        "status",
                        "remote_verified",
                        "remote_verified_at",
                        "attempts",
                        "last_error",
                        "receipt_sha256",
                        "local_deleted",
                    )
                }
            )
            inventory = record.get("source_inventory") or {}
            item["bytes"] = inventory.get("total_bytes")
            item["files"] = inventory.get("file_count")
        result["checkpoints"].append(item)
    result["manifest_published"] = (state / "manifests" / f"{name}.published.sha256").is_file()
    return result, evidence


def compare(reports: list[dict], evidence: list[dict]) -> dict:
    """Fail closed on differing initialization/data and require both successful restores."""
    expected_inputs = {
        f"input-step{step}-rank{rank}.sha256" for step in (1, 5) for rank in range(64)
    }
    weights_ready = all(item["weights"] is not None for item in evidence)
    if weights_ready:
        assert all(len(item["weights"]) == 64 for item in evidence)
        assert evidence[0]["weights"] == evidence[1]["weights"], "Initial weights differ"
    shared_inputs = evidence[0]["inputs"].keys() & evidence[1]["inputs"].keys()
    assert all(
        evidence[0]["inputs"][key] == evidence[1]["inputs"][key] for key in shared_inputs
    ), "Matched token batches differ"
    gates = {
        "initial_weights_match": weights_ready,
        "first_and_resumed_batches_match": all(
            expected_inputs <= item["inputs"].keys() for item in evidence
        ),
        "both_restored_step4": all(
            any(session["start_step"] == 4 for session in report["sessions"]) for report in reports
        ),
        "both_trained_through_step8": all(
            any(row.get("step") == 8 and "train/CE loss" in row for row in report["metrics"])
            for report in reports
        ),
        "both_evaluated": all(
            any(any(key.startswith("eval/") for key in row) for row in report["metrics"])
            for report in reports
        ),
        "all_six_checkpoints_complete": all(
            item["source_complete"] for report in reports for item in report["checkpoints"]
        ),
        "all_six_uploads_verified": all(
            item["remote_verified"] for report in reports for item in report["checkpoints"]
        ),
        "both_manifests_published": all(report["manifest_published"] for report in reports),
    }
    return {"gates": gates, "passed": all(gates.values()), "runs": reports}


def main():
    """Poll the two smoke lineages, exporting only small diagnostics to Beaker results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.startswith("olmoe3-small-"):
        raise ValueError("Expected an integration smoke run name, not a path")
    assert MOUNT.is_mount(), "Required Weka mount is absent"
    output = Path(os.environ.get("RESULTS_DIR", "/results")) / "integration-smoke"
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_seconds
    previous = None
    while True:
        reports, evidence = zip(
            *(snapshot(f"{args.name}-{arm}") for arm in ("reference", "optimized"))
        )
        report = compare(list(reports), list(evidence))
        report["free_bytes"] = shutil.disk_usage(MOUNT).free
        (output / "signoff.json").write_text(json.dumps(report, indent=2))
        (output / "fingerprints.json").write_text(json.dumps(evidence, indent=2))
        status = {"gates": report["gates"], "checkpoints": [r["checkpoints"] for r in reports]}
        if status != previous:
            print("INTEGRATION_SMOKE_AUDIT", json.dumps(status), flush=True)
            previous = status
        else:
            print("INTEGRATION_SMOKE_WAITING", json.dumps(report["gates"]), flush=True)
        if report["passed"]:
            print("INTEGRATION_SMOKE_SIGNOFF_PASSED", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Incomplete smoke gates: {report['gates']}")
        time.sleep(30)


if __name__ == "__main__":
    main()
