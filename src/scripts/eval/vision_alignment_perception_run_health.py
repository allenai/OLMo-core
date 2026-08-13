"""Build one immutable run-health receipt for a finished perception arm.

This producer is deliberately read-only with respect to Beaker.  It consumes a saved raw
``beaker experiment get --format json`` response, or can fetch that response when explicitly
requested, then binds it to the native checkpoint trainer states and the local W&B artifacts.
It never prints environment variables or secret values and never submits a job.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import types
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from olmo_core.eval.vision_alignment_perception_promotion import (
    CONTROL_ARM,
    EXPECTED_EXPERIMENT_IDS,
    EXPECTED_TREATMENT_SKIP_STEPS,
    PRIMARY_STEP,
    RECEIPT_VERSION,
    ROLLING_INTERVAL_LENGTH,
    RUN_HEALTH_RECEIPT_FORMAT,
    SIGMA_FACTOR,
    TREATMENT_ARM,
    PromotionValidationError,
    artifact_reference,
    candidate_from_outcome_receipt,
    canonical_sha256,
    sha256_file,
    validate_perception_run_health_receipt,
)

WORKSPACE = "ai2/molmofication"
WORKSPACE_ID = "01KSTRJHG4A32N7GDM82KY8J3E"
EXPECTED_GIT_REF = "d8ec4f57cf026424ccd13f20452365b6b1df34e5"
EXPECTED_EXPERIMENT_NAMES = {
    CONTROL_ARM: "vision-alignment-perception-frozen-vision-control-v1-1bb58fbf",
    TREATMENT_ARM: "vision-alignment-perception-treatment-v1-e620ec1f",
}
EXPECTED_LAUNCH_ARGUMENTS = {
    CONTROL_ARM: (
        "python",
        "src/scripts/train/Vision-Alignment.py",
        "train",
        "vision-alignment-perception-frozen-vision-control-v1",
        "--profile=configs/vision_moe/vision_alignment/perception/frozen_vision_control_v1.yaml",
    ),
    TREATMENT_ARM: (
        "python",
        "src/scripts/train/Vision-Alignment.py",
        "train",
        "vision-alignment-perception-treatment-v1",
        "--profile=configs/vision_moe/vision_alignment/perception/treatment_v1.yaml",
    ),
}
EXPECTED_SUCCESSFUL_JOBS = {
    CONTROL_ARM: {
        0: "01KZWCJV6S16TGHY4DV3C72BYP",
        1: "01KZWCSN0ZC975N5X4KZAVX9QA",
    },
    TREATMENT_ARM: {
        0: "01KZWCJWGZAV4M0F7KDJJVDB1E",
        1: "01KZWCJWM9XSFEWCCH4GRX7MT4",
    },
}
CONTROL_PRESTART_FAILURE = {
    "job_id": "01KZWCJVA6TQJVRNW2DX9V9968",
    "replica_rank": 1,
    "canceled_code": 10,
    "reason": "healthcheck for job 01KZWCJVA6TQJVRNW2DX9V9968 failed",
}
PERMANENT_STEPS = (0, 1000, 2000, 3000, 4000)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STEP_RE = re.compile(r"\[step=(\d+)/4000(?:,|\])")
_METRIC_RE = re.compile(
    r"^\s+(?P<name>[^=\n]+?)=(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|"
    r"[-+]?(?:nan|inf(?:inity)?))\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SKIP_RE = re.compile(r"^\s+optim/step skipped=(?P<value>\S+)\s*$", re.MULTILINE)
_FATAL_PATTERNS = (
    "Traceback (most recent call last):",
    " CRITICAL ",
    "Unhandled exception",
    "NCCL watchdog caught collective operation timeout",
    "ChildFailedError",
    "CUDA out of memory",
)
_CHECKPOINT_IDENTITY_FIELDS = frozenset(
    {
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
        "identity_sha256",
    }
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--perception-outcome", type=Path, required=True)
    parser.add_argument("--expected-perception-outcome-sha256", required=True)
    parser.add_argument("--arm", choices=(CONTROL_ARM, TREATMENT_ARM), required=True)
    parser.add_argument("--beaker-experiment-id", required=True)
    parser.add_argument("--beaker-experiment-json", type=Path, required=True)
    parser.add_argument(
        "--fetch-beaker-if-missing",
        action="store_true",
        help="Read-only fetch the raw experiment JSON when the requested file does not exist.",
    )
    parser.add_argument(
        "--successful-job-log",
        action="append",
        default=[],
        metavar="JOB_ID,REPLICA_RANK,PATH",
        help="Pinned raw log for one successful Beaker replica (exactly two required).",
    )
    parser.add_argument(
        "--prestart-failure",
        action="append",
        default=[],
        metavar="JOB_ID,REPLICA_RANK,EXIT_CODE,REASON,EVIDENCE_PATH",
        help="Control-only failed job that provably stopped before user training.",
    )
    parser.add_argument("--wandb-output", type=Path, required=True)
    parser.add_argument("--wandb-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def _write_bytes_once(path: Path, raw: bytes, *, label: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite immutable {label} {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_once(path: Path, payload: Mapping[str, Any], *, label: str = "receipt") -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _write_bytes_once(path, raw, label=label)


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise PromotionValidationError(f"JSON repeats key {key!r}")
        output[key] = value
    return output


def _strict_json_bytes(raw: bytes, *, name: str) -> Any:
    """Strictly decode the exact byte buffer whose digest becomes evidence."""

    def reject_constant(value: str) -> Any:
        raise PromotionValidationError(f"{name} contains non-finite JSON constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionValidationError(f"Could not parse {name}: {error}") from error


def _read_bytes_reference(path: Path, *, name: str) -> tuple[Path, bytes, dict[str, str]]:
    """Read one artifact once and bind its semantics and reference to that byte buffer."""
    path = path.expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PromotionValidationError(f"Could not read {name} {path}: {error}") from error
    reference = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return path, raw, reference


def _read_json_reference(
    path: Path, *, name: str, expected_sha256: str | None = None
) -> tuple[Path, Any, dict[str, str]]:
    """Read, hash, and strictly decode one immutable JSON byte snapshot."""
    path, raw, reference = _read_bytes_reference(path, name=name)
    if expected_sha256 is not None and reference["sha256"] != expected_sha256:
        raise PromotionValidationError(
            f"{name} differs from its explicit SHA-256 pin: "
            f"expected {expected_sha256}, got {reference['sha256']}"
        )
    return path, _strict_json_bytes(raw, name=name), reference


def _load_checkpoint_identity_helper() -> tuple[types.ModuleType, dict[str, str]]:
    """Load and pin the canonical stable checkpoint-identity implementation bytes."""
    path = Path(__file__).resolve().with_name("vision_alignment_perception_matched_wrong.py")
    path, raw, reference = _read_bytes_reference(path, name="checkpoint identity helper")
    module = types.ModuleType("_vision_alignment_perception_identity_for_run_health")
    module.__file__ = str(path)
    module.__package__ = None
    module.__spec__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    if not callable(getattr(module, "_checkpoint_identity", None)):
        raise PromotionValidationError("Canonical evaluator omits its checkpoint identity helper")
    return module, reference


def _materialize_beaker_json(path: Path, experiment_id: str, *, fetch: bool) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not fetch:
        raise FileNotFoundError(
            f"Saved Beaker experiment JSON is missing: {path}; use --fetch-beaker-if-missing"
        )
    completed = subprocess.run(
        ["beaker", "experiment", "get", experiment_id, "--format", "json"],
        check=True,
        capture_output=True,
    )
    # Parse these same bytes before persisting so an authentication/error banner can never become
    # evidence and duplicate/non-finite JSON cannot be accepted by the producer alone.
    _strict_json_bytes(completed.stdout, name="fetched Beaker experiment snapshot")
    _write_bytes_once(path, completed.stdout, label="Beaker experiment snapshot")
    return path


def _walk_json(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _string_values(value: Any) -> set[str]:
    return {item for item in _walk_json(value) if isinstance(item, str)}


def _workspace_names(value: Any) -> set[str]:
    names: set[str] = set()
    for item in _walk_json(value):
        if not isinstance(item, Mapping):
            continue
        for key in ("fullName", "full_name", "workspace", "workspaceName"):
            candidate = item.get(key)
            if isinstance(candidate, str) and "/" in candidate:
                names.add(candidate)
        owner = item.get("owner")
        name = item.get("name")
        if (
            isinstance(owner, Mapping)
            and isinstance(owner.get("name"), str)
            and isinstance(name, str)
        ):
            names.add(f"{owner['name']}/{name}")
    return names


def _job_objects(value: Any, job_id: str) -> list[Mapping[str, Any]]:
    """Return only objects whose own identifier equals ``job_id``."""
    matches: list[Mapping[str, Any]] = []
    for item in _walk_json(value):
        if not isinstance(item, Mapping):
            continue
        identifiers = {item.get(key) for key in ("id", "jobId", "job_id")}
        if job_id in identifiers:
            matches.append(item)
    return matches


def _integer_values(value: Any, keys: Sequence[str]) -> set[int]:
    output: set[int] = set()
    for item in _walk_json(value):
        if not isinstance(item, Mapping):
            continue
        for key in keys:
            candidate = item.get(key)
            if type(candidate) is int:
                output.add(candidate)
    return output


def _verify_job_claim(
    experiment: Any, job_id: str, *, exit_code: int, canceled: bool = False
) -> None:
    if type(exit_code) is not int or type(canceled) is not bool:
        raise PromotionValidationError(
            "Beaker job claim codes and cancellation flags are malformed"
        )
    matches = _job_objects(experiment, job_id)
    if not matches:
        raise PromotionValidationError(f"Beaker snapshot omits job {job_id}")
    code_keys = (
        ("canceledCode", "canceled_code")
        if canceled
        else ("exitCode", "exit_code", "exitStatus", "exit_status")
    )
    exact_matches = [match for match in matches if exit_code in _integer_values(match, code_keys)]
    if len(exact_matches) != 1:
        raise PromotionValidationError(
            f"Beaker snapshot does not uniquely attest {'cancellation' if canceled else 'exit'} "
            f"code {exit_code} for job {job_id}"
        )
    if canceled:
        status = exact_matches[0].get("status")
        if not isinstance(status, Mapping) or any(
            status.get(field) is not None for field in ("started", "exited", "exitCode")
        ):
            raise PromotionValidationError(
                f"Canceled Beaker job {job_id} does not attest a pre-start failure"
            )


def _env_value(spec: Mapping[str, Any], name: str) -> str:
    env_vars = spec.get("envVars")
    if not isinstance(env_vars, list):
        raise PromotionValidationError("Beaker job spec omits its environment inventory")
    values = [
        item.get("value")
        for item in env_vars
        if isinstance(item, Mapping) and item.get("name") == name and "value" in item
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise PromotionValidationError(f"Beaker job spec does not uniquely attest {name}")
    return values[0]


def _locked_experiment(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise PromotionValidationError("Beaker snapshot must contain exactly one experiment")
    return value[0]


def _verify_locked_job(
    job: Mapping[str, Any], *, arm: str, job_id: str, replica_rank: int, canceled: bool
) -> None:
    if type(replica_rank) is not int or type(canceled) is not bool:
        raise PromotionValidationError(f"Locked Beaker job {job_id} inputs are malformed")
    execution = job.get("execution")
    if not isinstance(execution, Mapping):
        raise PromotionValidationError(f"Beaker job {job_id} omits its execution")
    spec = execution.get("spec")
    status = job.get("status")
    if not isinstance(spec, Mapping) or not isinstance(status, Mapping):
        raise PromotionValidationError(f"Beaker job {job_id} omits its spec or status")
    expected_experiment_id = EXPECTED_EXPERIMENT_IDS[arm]
    expected_environment = {
        "GIT_REF": EXPECTED_GIT_REF,
        "BEAKER_JOB_ID": job_id,
        "BEAKER_WORKLOAD_ID": expected_experiment_id,
        "BEAKER_EXPERIMENT_ID": expected_experiment_id,
        "BEAKER_WORKSPACE_ID": WORKSPACE_ID,
        "BEAKER_REPLICA_RANK": str(replica_rank),
        "BEAKER_REPLICA_COUNT": "2",
    }
    if (
        job.get("id") != job_id
        or job.get("name") != f"train-replica-{replica_rank}"
        or job.get("workspace") != WORKSPACE_ID
        or execution.get("experiment") != expected_experiment_id
        or execution.get("workspace") != WORKSPACE_ID
        or type(execution.get("replicaRank")) is not int
        or execution.get("replicaRank") != replica_rank
        or spec.get("command") != ["bash", "/gantry/entrypoint.sh"]
        or tuple(spec.get("arguments", ())) != EXPECTED_LAUNCH_ARGUMENTS[arm]
        or type(spec.get("replicas")) is not int
        or spec.get("replicas") != 2
        or spec.get("leaderSelection") is not True
        or spec.get("hostNetworking") is not True
    ):
        raise PromotionValidationError(
            f"Beaker job {job_id} differs from its locked rank, workspace, or profile command"
        )
    for name, expected in expected_environment.items():
        if _env_value(spec, name) != expected:
            raise PromotionValidationError(f"Beaker job {job_id} {name} differs from its pin")
    if canceled:
        expected_reason = str(CONTROL_PRESTART_FAILURE["reason"])
        if (
            type(status.get("canceledCode")) is not int
            or status.get("canceledCode") != CONTROL_PRESTART_FAILURE["canceled_code"]
            or status.get("canceledFor") != expected_reason
            or not isinstance(status.get("canceled"), str)
            or not isinstance(status.get("finalized"), str)
            or any(status.get(field) is not None for field in ("started", "exited", "exitCode"))
        ):
            raise PromotionValidationError(
                "Locked control healthcheck cancellation is not a no-start canceledCode=10 job"
            )
    elif (
        type(status.get("exitCode")) is not int
        or status.get("exitCode") != 0
        or not isinstance(status.get("started"), str)
        or not isinstance(status.get("exited"), str)
        or not isinstance(status.get("finalized"), str)
        or status.get("canceledCode") is not None
    ):
        raise PromotionValidationError(f"Locked successful Beaker job {job_id} did not exit 0")


def _verify_locked_experiment_snapshot(value: Any, *, arm: str) -> None:
    experiment = _locked_experiment(value)
    workspace = experiment.get("workspaceRef")
    jobs = experiment.get("jobs")
    if (
        experiment.get("id") != EXPECTED_EXPERIMENT_IDS[arm]
        or experiment.get("name") != EXPECTED_EXPERIMENT_NAMES[arm]
        or not isinstance(workspace, Mapping)
        or workspace.get("id") != WORKSPACE_ID
        or workspace.get("fullName") != WORKSPACE
        or not isinstance(jobs, list)
        or any(not isinstance(job, Mapping) for job in jobs)
    ):
        raise PromotionValidationError("Beaker experiment identity or workspace differs")
    expected_jobs = dict(EXPECTED_SUCCESSFUL_JOBS[arm])
    if arm == CONTROL_ARM:
        expected_ids = {*expected_jobs.values(), str(CONTROL_PRESTART_FAILURE["job_id"])}
    else:
        expected_ids = set(expected_jobs.values())
    indexed = {str(job.get("id")): job for job in jobs}
    if len(indexed) != len(jobs) or set(indexed) != expected_ids:
        raise PromotionValidationError(
            "Beaker experiment job inventory differs from the locked run"
        )
    for replica_rank, job_id in expected_jobs.items():
        _verify_locked_job(
            indexed[job_id],
            arm=arm,
            job_id=job_id,
            replica_rank=replica_rank,
            canceled=False,
        )
    if arm == CONTROL_ARM:
        failure_id = str(CONTROL_PRESTART_FAILURE["job_id"])
        failure_rank = CONTROL_PRESTART_FAILURE["replica_rank"]
        if isinstance(failure_rank, bool) or not isinstance(failure_rank, int):
            raise PromotionValidationError("Locked control failure rank is malformed")
        _verify_locked_job(
            indexed[failure_id],
            arm=arm,
            job_id=failure_id,
            replica_rank=failure_rank,
            canceled=True,
        )


def _parse_success_spec(raw: str) -> tuple[str, int, Path]:
    fields = raw.split(",", 2)
    if len(fields) != 3 or not fields[0]:
        raise ValueError("--successful-job-log must be JOB_ID,REPLICA_RANK,PATH")
    return fields[0], int(fields[1]), Path(fields[2]).expanduser().resolve()


def _parse_failure_spec(raw: str) -> tuple[str, int, int, str, Path]:
    fields = raw.split(",", 4)
    if len(fields) != 5 or not fields[0] or not fields[3]:
        raise ValueError(
            "--prestart-failure must be JOB_ID,REPLICA_RANK,EXIT_CODE,REASON,EVIDENCE_PATH"
        )
    return fields[0], int(fields[1]), int(fields[2]), fields[3], Path(fields[4]).resolve()


def _audit_log_text(text: str) -> dict[str, Any]:
    """Audit the exact decoded bytes of a W&B output log."""
    text = _ANSI_RE.sub("", text)
    steps = {int(match.group(1)) for match in _STEP_RE.finditer(text)}
    if steps != set(range(1, PRIMARY_STEP + 1)):
        missing = sorted(set(range(1, PRIMARY_STEP + 1)) - steps)
        raise PromotionValidationError(f"W&B output omits training metric steps {missing[:10]}")
    numeric_count = 0
    nonfinite_count = 0
    finite_recovery_metric_steps: set[int] = set()
    skip_steps: list[int] = []
    current_step: int | None = None
    for line in text.splitlines():
        step_match = _STEP_RE.search(line)
        if step_match is not None:
            current_step = int(step_match.group(1))
        metric_match = _METRIC_RE.match(line)
        if metric_match is not None:
            numeric_count += 1
            try:
                metric_value = float(metric_match.group("value"))
            except ValueError:
                nonfinite_count += 1
            else:
                is_finite = math.isfinite(metric_value)
                nonfinite_count += int(not is_finite)
                if (
                    current_step is not None
                    and is_finite
                    and metric_match.group("name").strip() != "optim/step skipped"
                ):
                    finite_recovery_metric_steps.add(current_step)
        skip_match = _SKIP_RE.match(line)
        if skip_match is not None:
            if current_step is None:
                raise PromotionValidationError("Guarded-skip metric precedes its step marker")
            try:
                value = float(skip_match.group("value"))
            except ValueError as error:
                raise PromotionValidationError("Optimizer skip metric is not numeric") from error
            if not math.isfinite(value):
                nonfinite_count += 1
            elif value == 1.0:
                skip_steps.append(current_step)
            elif value != 0.0:
                raise PromotionValidationError("Optimizer skip metric is not binary")
    if len(skip_steps) != len(set(skip_steps)):
        raise PromotionValidationError("W&B output repeats a guarded-skip step")
    anomaly_count = sum(text.count(pattern) for pattern in _FATAL_PATTERNS)
    terminal = "Finalizing successful W&B run" in text
    if numeric_count <= 0 or nonfinite_count or anomaly_count or not terminal:
        raise PromotionValidationError("W&B output does not attest a clean finite completion")
    missing_recovery = [
        step + 1
        for step in skip_steps
        if step < PRIMARY_STEP and step + 1 not in finite_recovery_metric_steps
    ]
    if missing_recovery:
        raise PromotionValidationError(
            f"Guarded-skip recovery steps lack actual finite metrics: {missing_recovery}"
        )
    every_next_finite = not missing_recovery
    return {
        "metric_step_count": len(steps),
        "numeric_metric_count": numeric_count,
        "nonfinite_metric_count": nonfinite_count,
        "unexpected_anomaly_count": anomaly_count,
        "guarded_skip_steps": skip_steps,
        "successful_terminal_marker": terminal,
        "every_next_step_finite": every_next_finite,
        "started_training": bool(steps),
        "completed_training": PRIMARY_STEP in steps and terminal,
    }


def _audit_log(path: Path) -> dict[str, Any]:
    return _audit_log_text(path.read_text(errors="replace"))


def _audit_job_log_text(text: str, *, name: str = "job log") -> tuple[bool, bool]:
    """Audit the exact decoded bytes of one successful replica log."""
    text = _ANSI_RE.sub("", text)
    started = "olmo_core.train" in text or "[step=" in text
    completed = (
        "Finalizing successful W&B run" in text
        or "Training complete" in text
        or "step=4000/4000" in text
    )
    if not started or not completed or any(pattern in text for pattern in _FATAL_PATTERNS):
        raise PromotionValidationError(f"Successful {name} does not prove completion")
    return started, completed


def _audit_job_log(path: Path) -> tuple[bool, bool]:
    return _audit_job_log_text(path.read_text(errors="replace"), name=str(path))


def _audit_summary_value(summary: Any, *, expected_run_id: str) -> None:
    """Audit an already decoded W&B summary snapshot."""
    if (
        not isinstance(summary, Mapping)
        or type(summary.get("_step")) is not int
        or summary.get("_step") != PRIMARY_STEP
    ):
        raise PromotionValidationError("W&B summary is not at step4000")
    for value in _walk_json(summary):
        if isinstance(value, float) and not math.isfinite(value):
            raise PromotionValidationError("W&B summary contains a non-finite metric")
    # The run ID is checkpoint-owned. When present in the summary metadata it must agree.
    for key in ("run_id", "wandb/run_id"):
        value = summary.get(key)
        if value is not None and value != expected_run_id:
            raise PromotionValidationError("W&B summary run ID differs from checkpoint")


def _audit_summary(path: Path, *, expected_run_id: str) -> None:
    _path, summary, _reference = _read_json_reference(path, name="W&B summary")
    _audit_summary_value(summary, expected_run_id=expected_run_id)


def _exact_rank_state_paths(checkpoint: Path) -> list[Path]:
    train = checkpoint.expanduser().resolve() / "train"
    expected = [train / f"rank{rank}.pt" for rank in range(16)]
    observed = sorted(train.glob("rank*.pt"), key=lambda path: path.name)
    if set(observed) != set(expected) or any(path.is_symlink() for path in observed):
        raise PromotionValidationError(
            "Perception rank states must be the exact checkpoint/train/rank{0..15}.pt files"
        )
    return expected


def _load_trainer_state_and_sha256(path: Path) -> tuple[Mapping[str, Any], str]:
    """Safely decode and hash one trainer state from the same immutable byte buffer."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PromotionValidationError(f"Could not read trainer state {path}: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    allowed_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype("uint32")),
        type(np.dtype("int64")),
        type(np.dtype("float64")),
        type(np.dtype("bool")),
    ]
    try:
        with torch.serialization.safe_globals(allowed_globals):
            value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PromotionValidationError(
            f"Could not safely load trainer state {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise PromotionValidationError(f"Trainer state {path} must be an object")
    return value, digest


def _rank_states(checkpoint: Path) -> tuple[list[dict[str, Any]], str, int]:
    paths = _exact_rank_state_paths(checkpoint)
    if len(paths) != 16:  # Defensive: keep the invariant local if the helper changes.
        raise PromotionValidationError("Perception checkpoint must contain 16 rank states")
    inventory: list[dict[str, Any]] = []
    leader_run_id: str | None = None
    total_errors = 0
    for expected_rank, path in enumerate(paths):
        rank = int(path.stem.removeprefix("rank"))
        if rank != expected_rank:
            raise PromotionValidationError("Trainer rank-state inventory is not contiguous")
        payload, digest = _load_trainer_state_and_sha256(path)
        loader = payload.get("data_loader") if isinstance(payload, Mapping) else None
        callbacks = payload.get("callbacks") if isinstance(payload, Mapping) else None
        wandb = callbacks.get("wandb") if isinstance(callbacks, Mapping) else None
        if not isinstance(loader, Mapping) or not isinstance(wandb, Mapping):
            raise PromotionValidationError(f"Rank{rank} trainer state is incomplete")
        run_id = wandb.get("run_id")
        if rank == 0:
            if not isinstance(run_id, str) or not run_id:
                raise PromotionValidationError("Rank0 trainer state lacks a W&B run ID")
            leader_run_id = run_id
        elif run_id is not None:
            raise PromotionValidationError(f"Non-leader rank{rank} unexpectedly owns a run ID")
        errors = loader.get("total_data_errors")
        if isinstance(errors, bool) or not isinstance(errors, int):
            raise PromotionValidationError(f"Rank{rank} data-error count is invalid")
        total_errors += errors
        inventory.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "sha256": digest,
                "global_step": payload.get("global_step"),
                "batches_processed": loader.get("batches_processed"),
                "total_data_errors": errors,
                "run_id": run_id,
            }
        )
    assert leader_run_id is not None
    return inventory, leader_run_id, total_errors


def _permanent_checkpoints(
    checkpoint: Path, *, identity_helper: types.ModuleType
) -> list[dict[str, Any]]:
    """Build exact full identities for the fixed permanent-checkpoint sequence."""
    checkpoint = checkpoint.expanduser().resolve()
    output: list[dict[str, Any]] = []
    for step in PERMANENT_STEPS:
        root = checkpoint.parent / f"step{step}"
        marker = root / ".metadata.json"
        try:
            raw_identity = identity_helper._checkpoint_identity(
                root, root / "config.json", hash_workers=8
            )
        except Exception as error:
            raise PromotionValidationError(
                f"Could not build stable identity for permanent step{step}: {error}"
            ) from error
        if (
            not isinstance(raw_identity, Mapping)
            or set(raw_identity) != _CHECKPOINT_IDENTITY_FIELDS
        ):
            raise PromotionValidationError(f"Step{step} stable checkpoint identity is malformed")
        identity = dict(raw_identity)
        if identity["root"] != str(root.resolve()):
            raise PromotionValidationError(f"Step{step} stable checkpoint identity has wrong root")
        _marker_path, raw_marker, marker_reference = _read_json_reference(
            marker, name=f"permanent step{step} marker"
        )
        if marker_reference["sha256"] != identity["checkpoint_marker_sha256"]:
            raise PromotionValidationError(
                f"Step{step} marker bytes differ from its stable checkpoint identity"
            )
        if not isinstance(raw_marker, Mapping) or raw_marker.get("ephemeral") is not False:
            raise PromotionValidationError(f"Step{step} is missing its permanent marker")
        output.append({"step": step, "identity": identity})
    return output


def _candidate_fields(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": candidate["checkpoint"],
        "global_step": candidate["global_step"],
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Audit one completed arm and write an immutable canonical v1 receipt."""
    args = _parse_args(argv)
    if args.beaker_experiment_id != EXPECTED_EXPERIMENT_IDS[args.arm]:
        raise PromotionValidationError("Beaker experiment ID differs from the locked arm run")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable receipt {output}")
    outcome_path = args.perception_outcome.expanduser().resolve()
    _outcome_path, outcome, _outcome_reference = _read_json_reference(
        outcome_path,
        name="perception outcome",
        expected_sha256=args.expected_perception_outcome_sha256,
    )
    if not isinstance(outcome, Mapping):
        raise PromotionValidationError("Perception outcome must be a JSON object")
    role: Literal["control", "treatment"] = "control" if args.arm == CONTROL_ARM else "treatment"
    candidate = candidate_from_outcome_receipt(args.checkpoint, outcome, role=role)

    beaker_path = _materialize_beaker_json(
        args.beaker_experiment_json,
        args.beaker_experiment_id,
        fetch=args.fetch_beaker_if_missing,
    )
    beaker_path, beaker_payload, beaker_reference = _read_json_reference(
        beaker_path, name="Beaker experiment snapshot"
    )
    _verify_locked_experiment_snapshot(beaker_payload, arm=args.arm)
    if args.beaker_experiment_id not in _string_values(beaker_payload):
        raise PromotionValidationError("Beaker snapshot names a different experiment")
    if WORKSPACE not in _workspace_names(beaker_payload):
        raise PromotionValidationError("Beaker snapshot is not from ai2/molmofication")

    success_specs = sorted(
        (_parse_success_spec(value) for value in args.successful_job_log), key=lambda item: item[1]
    )
    if [item[1] for item in success_specs] != [0, 1]:
        raise PromotionValidationError("Exactly one successful job log is required per replica")
    if {rank: job_id for job_id, rank, _ in success_specs} != EXPECTED_SUCCESSFUL_JOBS[args.arm]:
        raise PromotionValidationError("Successful job IDs/ranks differ from the locked arm run")
    successful_jobs = []
    for job_id, replica_rank, log_path in success_specs:
        _verify_job_claim(beaker_payload, job_id, exit_code=0)
        log_path, log_raw, log_reference = _read_bytes_reference(
            log_path, name=f"successful replica{replica_rank} log"
        )
        started, completed = _audit_job_log_text(
            log_raw.decode("utf-8", errors="replace"), name=str(log_path)
        )
        successful_jobs.append(
            {
                "job_id": job_id,
                "replica_rank": replica_rank,
                "exit_code": 0,
                "started_training": started,
                "completed_training": completed,
                "log": log_reference,
            }
        )

    failures = []
    for raw in args.prestart_failure:
        job_id, rank, exit_code, reason, evidence_path = _parse_failure_spec(raw)
        _verify_job_claim(beaker_payload, job_id, exit_code=exit_code, canceled=True)
        evidence_path, evidence_raw, evidence_reference = _read_bytes_reference(
            evidence_path, name=f"pre-start failure {job_id} evidence"
        )
        evidence_text = _ANSI_RE.sub("", evidence_raw.decode("utf-8", errors="replace"))
        if "[step=" in evidence_text or "Finalizing successful W&B run" in evidence_text:
            raise PromotionValidationError("Claimed pre-start failure reached user training")
        failures.append(
            {
                "job_id": job_id,
                "replica_rank": rank,
                "exit_code": exit_code,
                "started_training": False,
                "reason": reason,
                "evidence": evidence_reference,
            }
        )
    expected_failure = []
    if args.arm == CONTROL_ARM:
        expected_failure = [
            {
                "job_id": CONTROL_PRESTART_FAILURE["job_id"],
                "replica_rank": CONTROL_PRESTART_FAILURE["replica_rank"],
                "exit_code": CONTROL_PRESTART_FAILURE["canceled_code"],
                "reason": CONTROL_PRESTART_FAILURE["reason"],
            }
        ]
    observed_failure = [
        {
            "job_id": item["job_id"],
            "replica_rank": item["replica_rank"],
            "exit_code": item["exit_code"],
            "reason": item["reason"],
        }
        for item in failures
    ]
    if observed_failure != expected_failure:
        raise PromotionValidationError("Pre-start failures differ from the locked arm run")

    wandb_output = args.wandb_output.expanduser().resolve()
    wandb_summary = args.wandb_summary.expanduser().resolve()
    wandb_output, wandb_raw, wandb_output_reference = _read_bytes_reference(
        wandb_output, name="W&B output"
    )
    log_audit = _audit_log_text(wandb_raw.decode("utf-8", errors="replace"))
    rank_inventory, run_id, total_errors = _rank_states(Path(candidate["checkpoint"]))
    wandb_summary, summary_payload, wandb_summary_reference = _read_json_reference(
        wandb_summary, name="W&B summary"
    )
    _audit_summary_value(summary_payload, expected_run_id=run_id)
    identity_helper, identity_helper_reference = _load_checkpoint_identity_helper()
    permanent = _permanent_checkpoints(
        Path(candidate["checkpoint"]), identity_helper=identity_helper
    )
    final_identity = permanent[-1]["identity"]
    if (
        final_identity["config_sha256"] != candidate["checkpoint_config_sha256"]
        or final_identity["identity_sha256"] != candidate["checkpoint_identity_sha256"]
    ):
        raise PromotionValidationError("Permanent step4000 identity differs from the candidate")
    expected_skips = [] if args.arm == CONTROL_ARM else list(EXPECTED_TREATMENT_SKIP_STEPS)
    if log_audit["guarded_skip_steps"] != expected_skips:
        raise PromotionValidationError("Observed guarded skips differ from the locked arm evidence")
    spacings = [right - left for left, right in pairwise(expected_skips)]
    minimum_spacing = min(spacings) if spacings else PRIMARY_STEP
    clean_final = PRIMARY_STEP - expected_skips[-1] if expected_skips else PRIMARY_STEP

    receipt: dict[str, Any] = {
        "format": RUN_HEALTH_RECEIPT_FORMAT,
        "version": RECEIPT_VERSION,
        "status": "passed",
        "created_at": args.created_at or datetime.now(timezone.utc).isoformat(),
        "producer": artifact_reference(Path(__file__).resolve()),
        "checkpoint_identity_helper": identity_helper_reference,
        "arm": args.arm,
        "candidate": _candidate_fields(candidate),
        "launch": {
            "workspace": WORKSPACE,
            "experiment_id": args.beaker_experiment_id,
            "successful_jobs": successful_jobs,
            "prestart_failures": failures,
        },
        "run": {
            "run_id": run_id,
            "global_steps": PRIMARY_STEP,
            "exit_code": 0,
            "rank_state_count": len(rank_inventory),
            "permanent_checkpoint_steps": list(PERMANENT_STEPS),
            "metric_step_count": log_audit["metric_step_count"],
            "numeric_metric_count": log_audit["numeric_metric_count"],
            "nonfinite_metric_count": log_audit["nonfinite_metric_count"],
            "unexpected_anomaly_count": log_audit["unexpected_anomaly_count"],
            "total_data_errors": total_errors,
            "successful_terminal_marker": log_audit["successful_terminal_marker"],
        },
        "rank_state_inventory": rank_inventory,
        "permanent_checkpoints": permanent,
        "optimizer_guard": {
            "rolling_interval_length": ROLLING_INTERVAL_LENGTH,
            "sigma_factor": SIGMA_FACTOR,
            "observed_steps": expected_skips,
            "count": len(expected_skips),
            "rate": len(expected_skips) / PRIMARY_STEP,
            "minimum_spacing": minimum_spacing,
            "clean_final_steps": clean_final,
            "every_next_step_finite": log_audit["every_next_step_finite"],
        },
        "evidence": {
            "beaker_experiment": beaker_reference,
            "wandb_output": wandb_output_reference,
            "wandb_summary": wandb_summary_reference,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    validate_perception_run_health_receipt(receipt, candidate=candidate, expected_arm=args.arm)
    _write_json_once(output, receipt)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "arm": args.arm,
                "guarded_skip_steps": expected_skips,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
