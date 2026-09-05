"""Read-only launch validation for the two matched 100B integration arms."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from olmoe3_integration_collect import MOUNT, snapshot

RUNS = (
    "olmoe3-small-16mi-100b-reference-r1",
    "olmoe3-small-16mi-100b-optimized-r1",
)


def compare_starts(reports, evidence, minimum_step):
    """Require matched initialization/data and finite observed training in both arms."""
    weights_ready = all(item["weights"] is not None for item in evidence)
    if weights_ready:
        assert all(len(item["weights"]) == 64 for item in evidence)
        assert evidence[0]["weights"] == evidence[1]["weights"], "Initial weights differ"
    expected_inputs = {f"input-step1-rank{rank}.sha256" for rank in range(64)}
    shared = evidence[0]["inputs"].keys() & evidence[1]["inputs"].keys()
    assert all(evidence[0]["inputs"][key] == evidence[1]["inputs"][key] for key in shared)
    for arm, report in zip(("reference", "optimized"), reports):
        for session in report["sessions"]:
            assert session["arm"] == arm
            for key, expected in {
                "policy": "core-docpool-top16-wgrad-rs",
                "stop_step": 6000,
                "global_batch_tokens": 16_777_216,
                "total_tokens": 100_663_296_000,
                "init_seed": 12536,
                "data_seed": 928543231,
                "gpus": 64,
                "mb_sequences": 4,
                "ga": 8,
                "lr": 0.00185,
            }.items():
                assert session[key] == expected, (arm, key, session[key], expected)
    commits = {session["source_commit"] for report in reports for session in report["sessions"]}
    assert len(commits) <= 1, "Integration arms use different source commits"
    initial_checkpoints = [report["checkpoints"][0] for report in reports]
    gates = {
        "matching_initial_weights": weights_ready,
        "matching_first_batches": all(
            expected_inputs <= item["inputs"].keys() for item in evidence
        ),
        "both_sessions_recorded": all(report["sessions"] for report in reports),
        "source_commit_recorded": bool(commits) and None not in commits,
        "both_have_minimum_successful_updates": all(
            len(
                {
                    row["step"]
                    for row in report["metrics"]
                    if "train/CE loss" in row
                    and "optim/total grad norm" in row
                    and row.get("optim/step skipped") == 0
                }
            )
            >= minimum_step
            for report in reports
        ),
        "both_step0_complete": all(item["source_complete"] for item in initial_checkpoints),
        "both_step0_remote_verified": all(item["remote_verified"] for item in initial_checkpoints),
    }
    return {"gates": gates, "passed": all(gates.values()), "runs": reports}


def main():
    """Poll small metadata only; never write to the checkpoint mount."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-step", type=int, default=5)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    assert args.minimum_step > 0 and args.wait_seconds > 0
    assert MOUNT.is_mount(), "Required Weka mount is absent"
    output = Path(os.environ.get("RESULTS_DIR", "/results")) / "integration-start"
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        reports, evidence = zip(*(snapshot(name) for name in RUNS))
        report = compare_starts(reports, evidence, args.minimum_step)
        report["free_bytes"] = shutil.disk_usage(MOUNT).free
        (output / "signoff.json").write_text(json.dumps(report, indent=2))
        (output / "fingerprints.json").write_text(json.dumps(evidence, indent=2))
        print(
            "INTEGRATION_START_CHECK",
            json.dumps(
                {
                    "gates": report["gates"],
                    "latest_steps": [
                        max(
                            (r["step"] for r in run["metrics"] if "train/CE loss" in r),
                            default=None,
                        )
                        for run in reports
                    ],
                    "free_bytes": report["free_bytes"],
                }
            ),
            flush=True,
        )
        if report["passed"]:
            print("INTEGRATION_START_CHECK_PASSED", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Launch gates still pending: {report['gates']}")
        time.sleep(30)


if __name__ == "__main__":
    main()
