"""Launch the reviewed distributed SSMax perception runtime preflight on Beaker.

This wrapper is deliberately narrower than the generic Beaker launcher. It derives the exact
treatment profile and v3 pair-receipt path from one of the two reviewed SSMax lineages, verifies
all caller-pinned bytes, and fixes the scaling-ladders/Holmes 2x8 operational contract. It cannot
launch training and accepts no arbitrary worker arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gantry.api import GitRepoState

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
GIT_BRANCH = "rustin/vision-ssmax-molmofication"
MIN_RUNTIME = "8h"
NUM_NODES = 2
GPUS_PER_NODE = 8
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECIPE_PATH = "src/scripts/train/Vision-Alignment.py"
PREFLIGHT_PATH = "src/scripts/eval/vision_alignment_perception_runtime_preflight.py"
ARTIFACT_ROOT = Path(
    "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
    "vision-alignment/artifacts"
)
CHECKPOINT_ROOT = ARTIFACT_ROOT.parent / "checkpoints"
MODEL_VARIANTS = ("ssmax_head_qknorm", "ssmax_no_qknorm")
PROFILE_PATHS = {
    "ssmax_head_qknorm": (
        "configs/vision_moe/vision_alignment/perception/"
        "ssmax_head_qknorm_1p4b_cx8_treatment_v1.yaml"
    ),
    "ssmax_no_qknorm": (
        "configs/vision_moe/vision_alignment/perception/"
        "ssmax_no_qknorm_1p4b_cx8_treatment_v1.yaml"
    ),
}
PROFILE_NAMES = {
    "ssmax_head_qknorm": {
        "frozen_vision_control": (
            "vision-ssmax-head-qknorm-1p4b-cx8-perception-frozen-vision-control-v1"
        ),
        "treatment": "vision-ssmax-head-qknorm-1p4b-cx8-perception-treatment-v1",
    },
    "ssmax_no_qknorm": {
        "frozen_vision_control": (
            "vision-ssmax-no-qknorm-1p4b-cx8-perception-frozen-vision-control-v1"
        ),
        "treatment": "vision-ssmax-no-qknorm-1p4b-cx8-perception-treatment-v1",
    },
}
PAIR_RECEIPT_NAMES = {
    "ssmax_head_qknorm": "ssmax-head-qknorm-perception-profile-pair-v3.json",
    "ssmax_no_qknorm": "ssmax-no-qknorm-perception-profile-pair-v3.json",
}
EXPERIMENT_NAMES = {
    "ssmax_head_qknorm": "ssmax-head-qknorm-perception-runtime-preflight-v1",
    "ssmax_no_qknorm": "ssmax-no-qknorm-perception-runtime-preflight-v1",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REF_RE = re.compile(r"[0-9a-f]{40}")
_PAIR_FORMAT = "vision_alignment_perception_profile_pair_audit"
_PAIR_VERSION = 3
_LAUNCH_CONTRACT = {
    "num_nodes": NUM_NODES,
    "num_gpus": GPUS_PER_NODE,
    "workspace": BEAKER_WORKSPACE,
    "cluster": BEAKER_CLUSTER,
    "budget": BEAKER_BUDGET,
    "priority": "urgent",
    "min_runtime": MIN_RUNTIME,
}


class PerceptionPreflightLaunchError(ValueError):
    """Raised when a preflight launch differs from the reviewed contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PerceptionPreflightLaunchError(f"Could not hash required input {path}: {error}")
    return digest.hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _require_sha256(value: str, *, name: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise PerceptionPreflightLaunchError(f"{name} must be a lowercase SHA-256")
    return value


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlink_components(path: Path, *, name: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise PerceptionPreflightLaunchError(f"{name} may not contain symlinks: {current}")


def _pinned_file(path: Path, expected_sha256: str, *, name: str) -> Path:
    _reject_symlink_components(path, name=name)
    if not path.is_file() or path.is_symlink():
        raise PerceptionPreflightLaunchError(f"{name} is not a regular non-symlink file: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PerceptionPreflightLaunchError(
            f"{name} SHA-256 differs: expected {expected_sha256}, got {actual_sha256}"
        )
    return path


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PerceptionPreflightLaunchError(f"{name} must be an object")
    return value


def _is_exact_mapping(value: Any, expected: Mapping[str, Any]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(expected)
        and all(
            type(value[field_name]) is type(expected_value) and value[field_name] == expected_value
            for field_name, expected_value in expected.items()
        )
    )


def _validate_pair_receipt(
    path: Path,
    *,
    model_variant: str,
    expected_recipe_sha256: str,
    expected_profile_sha256: str,
    expected_git_ref: str,
    checkpoint_root: Path,
) -> None:
    try:
        receipt = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PerceptionPreflightLaunchError(f"Invalid profile-pair receipt {path}: {error}")
    root = _mapping(receipt, name="profile-pair receipt")
    if (
        root.get("format") != _PAIR_FORMAT
        or type(root.get("version")) is not int
        or root.get("version") != _PAIR_VERSION
        or root.get("status") != "passed"
        or root.get("model_variant") != model_variant
        or not _is_exact_mapping(root.get("launch_contract"), _LAUNCH_CONTRACT)
    ):
        raise PerceptionPreflightLaunchError(
            "Profile-pair receipt identity or launch contract differs"
        )
    recipe = _mapping(root.get("recipe"), name="profile-pair recipe")
    git = _mapping(root.get("git"), name="profile-pair git")
    if recipe.get("sha256") != expected_recipe_sha256:
        raise PerceptionPreflightLaunchError("Profile-pair receipt recipe SHA-256 differs")
    if git.get("branch") != GIT_BRANCH or git.get("ref") != expected_git_ref:
        raise PerceptionPreflightLaunchError("Profile-pair receipt Git identity differs")

    profiles = _mapping(root.get("profiles"), name="profile-pair profiles")
    for arm in ("frozen_vision_control", "treatment"):
        profile = _mapping(profiles.get(arm), name=f"profile-pair {arm} profile")
        if profile.get("name") != PROFILE_NAMES[model_variant][arm]:
            raise PerceptionPreflightLaunchError(f"Profile-pair {arm} name differs")
    treatment = _mapping(profiles.get("treatment"), name="profile-pair treatment profile")
    if (
        treatment.get("repository_path") != PROFILE_PATHS[model_variant]
        or treatment.get("sha256") != expected_profile_sha256
    ):
        raise PerceptionPreflightLaunchError("Profile-pair treatment profile identity differs")

    save_folders = _mapping(root.get("save_folders"), name="profile-pair save folders")
    if save_folders.get("status") != "verified_absent_and_distinct":
        raise PerceptionPreflightLaunchError("Profile-pair save-folder status differs")
    observed: list[Path] = []
    for arm in ("frozen_vision_control", "treatment"):
        expected = checkpoint_root / PROFILE_NAMES[model_variant][arm]
        actual = _absolute_lexical(str(save_folders.get(arm, "")))
        _reject_symlink_components(actual, name=f"{arm} save folder")
        if actual != expected or os.path.lexists(actual):
            raise PerceptionPreflightLaunchError(
                f"Profile-pair {arm} save folder is not the absent production path"
            )
        observed.append(actual)
    if len(set(observed)) != 2:
        raise PerceptionPreflightLaunchError("Profile-pair save folders are not distinct")


def _verify_remote_ref(git: GitRepoState, expected_git_ref: str) -> None:
    remote_ref = f"refs/heads/{GIT_BRANCH}"
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "--heads", git.repo_url, remote_ref],
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        raise PerceptionPreflightLaunchError("Could not verify the reviewed remote Git branch")
    records = [line.split("\t", 1) for line in output.splitlines() if line]
    if records != [[expected_git_ref, remote_ref]]:
        raise PerceptionPreflightLaunchError(
            "Expected Git revision is not the exact tip of the reviewed remote branch"
        )


def build_launch_config(
    *,
    model_variant: str,
    expected_recipe_sha256: str,
    expected_profile_sha256: str,
    profile_pair_receipt: str | Path,
    expected_profile_pair_receipt_sha256: str,
    expected_git_ref: str,
    verify_remote_ref: bool = True,
    git_state: GitRepoState | None = None,
) -> BeakerLaunchConfig:
    """Build the fixed two-node runtime-preflight launch after validating every input."""

    if model_variant not in MODEL_VARIANTS:
        raise PerceptionPreflightLaunchError(f"Unsupported SSMax model variant {model_variant!r}")
    expected_recipe_sha256 = _require_sha256(expected_recipe_sha256, name="expected recipe SHA-256")
    expected_profile_sha256 = _require_sha256(
        expected_profile_sha256, name="expected profile SHA-256"
    )
    expected_profile_pair_receipt_sha256 = _require_sha256(
        expected_profile_pair_receipt_sha256,
        name="expected profile-pair receipt SHA-256",
    )
    if _GIT_REF_RE.fullmatch(expected_git_ref) is None:
        raise PerceptionPreflightLaunchError("expected Git ref must be lowercase 40-hex")

    repository_root = _absolute_lexical(REPOSITORY_ROOT)
    artifact_root = _absolute_lexical(ARTIFACT_ROOT)
    checkpoint_root = _absolute_lexical(CHECKPOINT_ROOT)
    recipe_path = _pinned_file(
        repository_root / RECIPE_PATH,
        expected_recipe_sha256,
        name="training recipe",
    )
    profile_path = _pinned_file(
        repository_root / PROFILE_PATHS[model_variant],
        expected_profile_sha256,
        name="treatment profile",
    )
    receipt_path = _absolute_lexical(profile_pair_receipt)
    expected_receipt_path = artifact_root / PAIR_RECEIPT_NAMES[model_variant]
    if receipt_path != expected_receipt_path:
        raise PerceptionPreflightLaunchError(
            f"Profile-pair receipt must be exactly {expected_receipt_path}"
        )
    _pinned_file(
        receipt_path,
        expected_profile_pair_receipt_sha256,
        name="profile-pair receipt",
    )
    _validate_pair_receipt(
        receipt_path,
        model_variant=model_variant,
        expected_recipe_sha256=expected_recipe_sha256,
        expected_profile_sha256=expected_profile_sha256,
        expected_git_ref=expected_git_ref,
        checkpoint_root=checkpoint_root,
    )

    preset = get_preset("olmo-ddp")
    if preset.beaker_image is None:
        raise RuntimeError("olmo-ddp launch preset does not define a Beaker image")
    env = dict(preset.env_vars)
    env.pop("OLMO_SYMM_VDEV2D_AUTO_BUILD", None)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    command = [
        PREFLIGHT_PATH,
        f"--model-variant={model_variant}",
        f"--recipe={RECIPE_PATH}",
        f"--expected-recipe-sha256={expected_recipe_sha256}",
        f"--profile={PROFILE_PATHS[model_variant]}",
        f"--expected-profile-sha256={expected_profile_sha256}",
        f"--profile-pair-receipt={receipt_path}",
        ("--expected-profile-pair-receipt-sha256=" f"{expected_profile_pair_receipt_sha256}"),
        f"--expected-git-ref={expected_git_ref}",
    ]
    config = BeakerLaunchConfig(
        name=EXPERIMENT_NAMES[model_variant],
        task_name="perception-preflight",
        description=f"Reviewed {model_variant} perception runtime preflight",
        cmd=command,
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
        allow_dirty=False,
        follow=False,
        slack_notifications=False,
        launch_timeout=60 * 60,
        step_timeout=None,
        step_soft_timeout=None,
        env_vars=[BeakerEnvVar(name=name, value=value) for name, value in env.items()],
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret="RUSTINS_BEAKER_TOKEN", required=True)
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
        git=git_state if git_state is not None else GitRepoState.from_env(),
    )
    if config.git.repo != "allenai/OLMo-core" or config.git.ref != expected_git_ref:
        raise PerceptionPreflightLaunchError("Local launch Git identity differs from its pin")
    config.git.branch = GIT_BRANCH
    if verify_remote_ref:
        _verify_remote_ref(config.git, expected_git_ref)
    # Rehash after all parsing and optional remote verification to close input races.
    for path, expected, name in (
        (recipe_path, expected_recipe_sha256, "training recipe"),
        (profile_path, expected_profile_sha256, "treatment profile"),
        (receipt_path, expected_profile_pair_receipt_sha256, "profile-pair receipt"),
    ):
        _pinned_file(path, expected, name=name)
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry_run", "launch"))
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, required=True)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--profile-pair-receipt", type=Path, required=True)
    parser.add_argument("--expected-profile-pair-receipt-sha256", required=True)
    parser.add_argument("--expected-git-ref", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Validate and either render or launch one fixed runtime-preflight job."""

    args = _parser().parse_args(argv)
    prepare_cli_environment()
    config = build_launch_config(
        model_variant=args.model_variant,
        expected_recipe_sha256=args.expected_recipe_sha256,
        expected_profile_sha256=args.expected_profile_sha256,
        profile_pair_receipt=args.profile_pair_receipt,
        expected_profile_pair_receipt_sha256=args.expected_profile_pair_receipt_sha256,
        expected_git_ref=args.expected_git_ref,
    )
    if args.action == "dry_run":
        config.dry_run(follow=False, torchrun=True)
    else:
        config.launch(follow=False, torchrun=True)


if __name__ == "__main__":
    main()
