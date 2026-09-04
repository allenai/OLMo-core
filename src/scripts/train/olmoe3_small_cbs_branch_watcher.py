"""Launch the 16 Mi-token CBS branch when reference step 4,000 is finalized.

This is intended to run as a tiny, restartable Beaker task with the checkpoint
Weka bucket mounted. It treats the checkpoint-ready event as the trigger, then
independently validates the finalized checkpoint before launching. Exact
Beaker-name checks and a durable receipt make retries idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHECKPOINT_MOUNT = Path("/weka/olmo-3p5-checkpoints")
REFERENCE_RUN_ID = "olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
BRANCH_RUN_ID = "olmoe3-small-cbs-16mi-from-step4000-lr1p85em3-uploader-r1"
REFERENCE_STEP = 4_000
REFERENCE_CHECKPOINT = (
    CHECKPOINT_MOUNT / "production-cbs" / REFERENCE_RUN_ID / f"step{REFERENCE_STEP}"
)
READY_EVENT = (
    CHECKPOINT_MOUNT
    / "uploader/control/inbox"
    / REFERENCE_RUN_ID
    / f"step-{REFERENCE_STEP:012d}.ready.json"
)
AUTOMATION_STATE = CHECKPOINT_MOUNT / "uploader/automation" / f"{BRANCH_RUN_ID}.json"
TARGET_WORKSPACE = "ai2/olmo3p5-training"
TARGET_CLUSTER = "ai2/holmes"
TRAINING_SCRIPT = "src/examples/olmo_ddp/olmoe3_small_cbs.py"


def log(message: str, **fields: Any) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "message": message,
        **fields,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


def _metadata_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready_checkpoint() -> dict[str, Any] | None:
    if not READY_EVENT.is_file():
        return None
    try:
        event = json.loads(READY_EVENT.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "event": "checkpoint_ready",
            "run_id": REFERENCE_RUN_ID,
            "lineage_id": REFERENCE_RUN_ID,
            "step": REFERENCE_STEP,
        }
        for field, value in expected.items():
            if event.get(field) != value:
                raise RuntimeError(
                    f"ready event {field!r}={event.get(field)!r}, expected {value!r}"
                )
        event_checkpoint = Path(event["checkpoint_path"])
        if event_checkpoint.resolve() != REFERENCE_CHECKPOINT.resolve():
            raise RuntimeError(
                f"ready event points to {event_checkpoint}, expected {REFERENCE_CHECKPOINT}"
            )
        required = (
            REFERENCE_CHECKPOINT / ".metadata.json",
            REFERENCE_CHECKPOINT / "model_and_optim/.metadata",
            REFERENCE_CHECKPOINT / "train/rank0.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"checkpoint is not complete; missing {missing}")
        digest = _metadata_digest(required[0])
        if digest != event.get("checkpoint_metadata_sha256"):
            raise RuntimeError("checkpoint metadata digest differs from the published ready event")
        return event
    except Exception as exc:
        log("checkpoint trigger exists but failed validation", error=repr(exc))
        return None


def existing_experiment() -> dict[str, Any] | None:
    # The lightweight Gantry runtime provides the Beaker Python SDK but does
    # not necessarily include the separate Go CLI binary.
    from beaker import Beaker

    beaker = Beaker.from_env(
        default_workspace=TARGET_WORKSPACE,
        check_for_upgrades=False,
    )
    workspace = beaker.workspace.get(TARGET_WORKSPACE)
    prefix = f"{BRANCH_RUN_ID}-train-"
    matches: list[dict[str, Any]] = []
    for workload in beaker.workload.list(
        workspace=workspace,
        name_or_description=BRANCH_RUN_ID,
        limit=100,
    ):
        if not workload.HasField("experiment"):
            continue
        experiment = workload.experiment
        if experiment.name == f"{BRANCH_RUN_ID}-train" or experiment.name.startswith(prefix):
            matches.append(
                {
                    "id": experiment.id,
                    "name": experiment.name,
                    "created": experiment.created.ToDatetime().isoformat(),
                }
            )
    if not matches:
        return None
    return max(matches, key=lambda experiment: experiment.get("created", ""))


def write_receipt(experiment: dict[str, Any], event: dict[str, Any]) -> None:
    AUTOMATION_STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_event": str(READY_EVENT),
        "source_checkpoint": str(REFERENCE_CHECKPOINT),
        "source_metadata_sha256": event["checkpoint_metadata_sha256"],
        "target_run_id": BRANCH_RUN_ID,
        "target_workspace": TARGET_WORKSPACE,
        "target_experiment_id": experiment.get("id"),
        "target_experiment_name": experiment.get("name"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=AUTOMATION_STATE.parent,
        prefix=f".{AUTOMATION_STATE.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, AUTOMATION_STATE)


def launch_branch() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env.update(
        {
            "OLMOE3_SMALL_CBS_PHASE": "16mi",
            "OLMOE3_BEAKER_WORKSPACE": TARGET_WORKSPACE,
            "OLMOE3_BEAKER_PRIORITY": "urgent",
            "PYTHONPATH": "src",
        }
    )
    command = [
        sys.executable,
        TRAINING_SCRIPT,
        "launch",
        BRANCH_RUN_ID,
        TARGET_CLUSTER,
    ]
    log("launching 16Mi branch", command=command, cwd=str(repo_root))
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def run(poll_seconds: float) -> int:
    last_heartbeat = 0.0
    while True:
        try:
            if experiment := existing_experiment():
                event = ready_checkpoint()
                if event is not None:
                    write_receipt(experiment, event)
                log(
                    "16Mi branch already submitted",
                    experiment_id=experiment.get("id"),
                    experiment_name=experiment.get("name"),
                )
                return 0
        except Exception as exc:
            log("Beaker duplicate check failed; launch suppressed", error=repr(exc))
            time.sleep(poll_seconds)
            continue

        event = ready_checkpoint()
        if event is not None:
            try:
                launch_branch()
                for _ in range(12):
                    if experiment := existing_experiment():
                        write_receipt(experiment, event)
                        log(
                            "16Mi branch submitted",
                            experiment_id=experiment.get("id"),
                            experiment_name=experiment.get("name"),
                            receipt=str(AUTOMATION_STATE),
                        )
                        return 0
                    time.sleep(10)
                raise RuntimeError("launch returned successfully but Beaker did not index the job")
            except Exception as exc:
                log("16Mi branch launch failed; will retry", error=repr(exc))
                time.sleep(poll_seconds)
                continue

        now = time.monotonic()
        if now - last_heartbeat >= 300:
            log(
                "waiting for finalized reference checkpoint",
                event=str(READY_EVENT),
                checkpoint=str(REFERENCE_CHECKPOINT),
            )
            last_heartbeat = now
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return run(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
