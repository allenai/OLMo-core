"""Launch a reviewed distributed SSMax vision-evidence producer on Beaker.

This is deliberately narrower than the generic Beaker launcher. It fixes the workspace,
budget, Holmes placement, 2x8 topology, urgent priority, eight-hour minimum runtime, source
checkout, and Weka mount used by the paired SSMax experiment. The selected evaluator remains
responsible for reopening its immutable manifest and checkpoint identities.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
)
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.utils import prepare_cli_environment

BEAKER_WORKSPACE = "ai2/scaling-ladders"
BEAKER_BUDGET = "ai2/oe-other"
BEAKER_CLUSTER = "ai2/holmes"
MIN_RUNTIME = "8h"
NUM_NODES = 2
GPUS_PER_NODE = 8
DIRECT_EVIDENCE_GIT_HISTORY_POST_SETUP = 'git fetch --no-tags --depth 3 origin "$GIT_REF"'
EVIDENCE_ROOT = Path(
    "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
    "vision-alignment/evidence"
)
_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_EVALUATORS: Mapping[str, str] = {
    "bridge": "src/scripts/eval/vision_alignment_ssmax_bridge.py",
    "perception": "src/scripts/eval/vision_alignment_ssmax_perception.py",
    "perception_direct": "src/scripts/eval/vision_alignment_ssmax_perception_direct.py",
    "joint": "src/scripts/eval/vision_alignment_ssmax_joint.py",
}
_COMMON_FLAGS = frozenset(
    {
        "--manifest",
        "--expected-manifest-sha256",
        "--step",
        "--output",
        "--work-dir",
        "--checkpoint-load-threads",
        "--checkpoint-hash-workers",
    }
)
_REQUIRED_FLAGS = frozenset(
    {"--manifest", "--expected-manifest-sha256", "--step", "--output", "--work-dir"}
)
_STAGE_FLAGS = {
    "bridge": _COMMON_FLAGS,
    "perception": _COMMON_FLAGS | {"--arm"},
    "perception_direct": _COMMON_FLAGS,
    "joint": _COMMON_FLAGS,
}
_STAGE_REQUIRED_FLAGS = {
    "bridge": _REQUIRED_FLAGS,
    "perception": _REQUIRED_FLAGS | {"--arm"},
    "perception_direct": _REQUIRED_FLAGS,
    "joint": _REQUIRED_FLAGS,
}
_STEPS = {
    "bridge": frozenset({0, 100, 200, 250, 300, 400, 500}),
    "perception": frozenset({0, 3000, 4000}),
    "perception_direct": frozenset({0, 3000, 4000}),
    "joint": frozenset({0, 4000, 8000, 12000, 16000}),
}


class EvidenceLaunchError(ValueError):
    """Raised when an evidence launch differs from the reviewed operational contract."""


def _parse_evaluator_arguments(stage: str, values: Sequence[str]) -> dict[str, str]:
    if not values:
        raise EvidenceLaunchError("evaluator arguments must not be empty")
    if values[0] == "--":
        values = values[1:]
    if not values or len(values) % 2:
        raise EvidenceLaunchError("evaluator arguments must be --flag value pairs")
    allowed = _STAGE_FLAGS[stage]
    parsed: dict[str, str] = {}
    for index in range(0, len(values), 2):
        flag, value = values[index : index + 2]
        if flag not in allowed:
            raise EvidenceLaunchError(f"unsupported {stage} evaluator flag {flag!r}")
        if flag in parsed:
            raise EvidenceLaunchError(f"duplicate {stage} evaluator flag {flag}")
        if not value or value.startswith("--") or "\n" in value or "\x00" in value:
            raise EvidenceLaunchError(f"invalid value for {flag}")
        parsed[flag] = value
    missing = _STAGE_REQUIRED_FLAGS[stage] - parsed.keys()
    if missing:
        raise EvidenceLaunchError(f"missing required evaluator flags {sorted(missing)}")
    return parsed


def _validate_evaluator_arguments(
    stage: str,
    parsed: Mapping[str, str],
    *,
    verify_manifest_bytes: bool,
) -> None:
    digest = parsed["--expected-manifest-sha256"]
    if _SHA256.fullmatch(digest) is None:
        raise EvidenceLaunchError("expected manifest identity must be a SHA-256 digest")
    try:
        step = int(parsed["--step"])
    except ValueError as error:
        raise EvidenceLaunchError("step must be a canonical non-negative integer") from error
    if str(step) != parsed["--step"] or step not in _STEPS[stage]:
        raise EvidenceLaunchError(f"unsupported {stage} evidence step {parsed['--step']!r}")
    if stage == "perception" and parsed["--arm"] not in {
        "treatment",
        "frozen_vision_control",
    }:
        raise EvidenceLaunchError("perception arm must be treatment or frozen_vision_control")
    for flag in ("--checkpoint-load-threads", "--checkpoint-hash-workers"):
        if flag in parsed:
            try:
                count = int(parsed[flag])
            except ValueError as error:
                raise EvidenceLaunchError(f"{flag} must be a positive integer") from error
            if str(count) != parsed[flag] or count <= 0:
                raise EvidenceLaunchError(f"{flag} must be a positive integer")

    manifest = Path(parsed["--manifest"])
    output = Path(parsed["--output"])
    work_dir = Path(parsed["--work-dir"])
    for name, path in (("manifest", manifest), ("output", output), ("work directory", work_dir)):
        if not path.is_absolute() or not path.is_relative_to(EVIDENCE_ROOT):
            raise EvidenceLaunchError(f"{name} must be an absolute path below {EVIDENCE_ROOT}")
    if output.suffix != ".json":
        raise EvidenceLaunchError("evidence output must be a JSON path")
    if output.exists():
        raise EvidenceLaunchError(f"refusing to overwrite existing evidence output {output}")
    if verify_manifest_bytes:
        if not manifest.is_file():
            raise EvidenceLaunchError(f"manifest does not exist: {manifest}")
        actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if actual != digest:
            raise EvidenceLaunchError(
                f"manifest bytes differ from --expected-manifest-sha256: {actual}"
            )


def build_launch_config(
    *,
    name: str,
    stage: str,
    evaluator_arguments: Sequence[str],
    verify_manifest_bytes: bool = True,
) -> BeakerLaunchConfig:
    """Build the fixed distributed Beaker configuration for one evidence receipt."""

    if _SAFE_NAME.fullmatch(name) is None:
        raise EvidenceLaunchError(
            "name must use lowercase letters, digits, underscores, or hyphens and start "
            "with a letter or digit"
        )
    if stage not in _EVALUATORS:
        raise EvidenceLaunchError(f"unsupported evidence stage {stage!r}")
    parsed = _parse_evaluator_arguments(stage, evaluator_arguments)
    _validate_evaluator_arguments(
        stage,
        parsed,
        verify_manifest_bytes=verify_manifest_bytes,
    )
    arguments = list(evaluator_arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]

    preset = get_preset("olmo-ddp")
    if preset.beaker_image is None:
        raise RuntimeError("olmo-ddp launch preset does not define a Beaker image")
    env = dict(preset.env_vars)
    env.pop("OLMO_SYMM_VDEV2D_AUTO_BUILD", None)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    return BeakerLaunchConfig(
        name=name,
        task_name=f"{stage}-evidence",
        description=f"Immutable paired SSMax {stage} evidence producer",
        cmd=[_EVALUATORS[stage], *arguments],
        torchrun=True,
        budget=BEAKER_BUDGET,
        workspace=BEAKER_WORKSPACE,
        beaker_image=preset.beaker_image,
        num_nodes=NUM_NODES,
        num_gpus=GPUS_PER_NODE,
        clusters=[BEAKER_CLUSTER],
        hostnames=None,
        shared_filesystem=True,
        priority="urgent",
        preemptible=True,
        min_runtime=MIN_RUNTIME,
        post_setup=(
            DIRECT_EVIDENCE_GIT_HISTORY_POST_SETUP if stage == "perception_direct" else None
        ),
        allow_dirty=False,
        follow=False,
        slack_notifications=False,
        launch_timeout=60 * 60,
        step_timeout=None,
        step_soft_timeout=None,
        env_vars=[BeakerEnvVar(name=key, value=value) for key, value in env.items()],
        env_secrets=[
            BeakerEnvSecret(
                name="BEAKER_TOKEN",
                secret="RUSTINS_BEAKER_TOKEN",
                required=True,
            )
        ],
        google_credentials_secret=None,
        aws_config_secret=None,
        aws_credentials_secret=None,
        weka_buckets=[
            BeakerWekaBucket(
                bucket="oe-training-default",
                mount="/weka/oe-training-default",
            )
        ],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry_run", "launch"))
    parser.add_argument("stage", choices=tuple(_EVALUATORS))
    parser.add_argument("name")
    parser.add_argument("evaluator_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Validate and either render or launch one fixed evidence job."""

    args = _parser().parse_args(argv)
    prepare_cli_environment()
    config = build_launch_config(
        name=args.name,
        stage=args.stage,
        evaluator_arguments=args.evaluator_arguments,
    )
    if args.action == "dry_run":
        config.dry_run(follow=False, torchrun=True)
    else:
        config.launch(follow=False, torchrun=True)


if __name__ == "__main__":
    main()
