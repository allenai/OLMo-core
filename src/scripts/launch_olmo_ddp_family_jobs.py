#!/usr/bin/env python3
"""Validate or explicitly submit a prepared batch of OLMoDDP family jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    REPO_ROOT
    / "src/scripts/beaker/generated/olmo_ddp_conversion_families/launch_plan.json"
)
DEFAULT_RECEIPT = Path(
    "/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/converted-checkpoints/"
    "_family_status/_launches/remaining_families_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--workspace")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--experiment-name-suffix",
        default="",
        help="Suffix appended to prepared experiment names, useful for retries.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually create experiments. Without this flag the script is read-only.",
    )
    parser.add_argument(
        "--confirm-gpu-count",
        type=int,
        help="Required with --submit and must equal the selected aggregate GPU count.",
    )
    parser.add_argument(
        "--allow-resubmit",
        action="store_true",
        help="Submit families already recorded as submitted in the receipt.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def find_experiment_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("id", "experimentId", "experiment_id"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        for child in value.values():
            if (candidate := find_experiment_id(child)) is not None:
                return candidate
    elif isinstance(value, list):
        for child in value:
            if (candidate := find_experiment_id(child)) is not None:
                return candidate
    return None


def main() -> None:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    plan = load_json(plan_path)
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Launch plan has no jobs")

    requested = set(args.families or [])
    known = {job["family"] for job in jobs}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Families are not in the launch plan: {', '.join(unknown)}")
    selected = [job for job in jobs if not requested or job["family"] in requested]
    workspace = args.workspace or plan["workspace"]
    receipt_path = args.receipt.expanduser().resolve()
    receipt: dict[str, Any]
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
    else:
        receipt = {
            "schema_version": 1,
            "plan": str(plan_path),
            "workspace": workspace,
            "submissions": {},
        }
    submissions = receipt.setdefault("submissions", {})

    commands: list[tuple[dict[str, Any], list[str]]] = []
    for job in selected:
        family = job["family"]
        experiment_name = job["experiment_name"] + args.experiment_name_suffix
        spec = (plan_path.parent / job["spec"]).resolve()
        if not spec.is_file():
            raise FileNotFoundError(f"Missing prepared spec for {family}: {spec}")
        if family in submissions and not args.allow_resubmit:
            print(f"SKIP already submitted: {family} ({submissions[family].get('experiment_id')})")
            continue
        command = [
            "beaker",
            "experiment",
            "create",
            str(spec),
            "--workspace",
            workspace,
            "--name",
            experiment_name,
            "--format",
            "json",
        ]
        commands.append(({**job, "experiment_name": experiment_name}, command))

    pending_gpu_count = sum(int(job["gpu_count"]) for job, _ in commands)
    pending_model_count = sum(int(job["model_count"]) for job, _ in commands)
    print(
        f"Selected {len(selected)} jobs; {len(commands)} pending submissions / "
        f"{pending_gpu_count} GPUs / {pending_model_count} models."
    )
    for _, command in commands:
        print(shlex.join(command))

    if not args.submit:
        print("DRY RUN: no Beaker experiments were created. Add --submit and "
              f"--confirm-gpu-count {pending_gpu_count} to launch this selection.")
        return
    if not commands:
        print("Nothing to submit; every selected family is already in the receipt.")
        return
    if args.confirm_gpu_count != pending_gpu_count:
        raise ValueError(
            "Refusing submission: --confirm-gpu-count must be exactly "
            f"{pending_gpu_count}"
        )

    for job, command in commands:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Beaker returned non-JSON output for {job['family']}: {completed.stdout!r}"
            ) from exc
        experiment_id = find_experiment_id(response)
        if experiment_id is None:
            raise RuntimeError(f"Could not find experiment ID in Beaker response: {response!r}")
        submissions[job["family"]] = {
            "experiment_id": experiment_id,
            "experiment_name": job["experiment_name"],
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "beaker_response": response,
        }
        atomic_write_json(receipt_path, receipt)
        print(f"SUBMITTED {job['family']}: {experiment_id}")

    print(f"Submission receipt: {receipt_path}")


if __name__ == "__main__":
    main()
