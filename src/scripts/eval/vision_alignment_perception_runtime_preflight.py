"""Run the production perception provenance check on the exact 2x8 Holmes topology.

This program is deliberately read-only. It loads one allowlisted production perception profile,
builds it through the exact SHA-256-pinned training recipe in runtime mode, and therefore executes
the recipe's rank-zero full provenance validation plus its rank-local lightweight validation. It
does not launch training, construct a trainer, create a checkpoint folder, or write a receipt.

Run this under the same two-node Gantry/torchrun topology as production::

    python src/scripts/eval/vision_alignment_perception_runtime_preflight.py \
      --recipe=src/scripts/train/Vision-Alignment.py \
      --expected-recipe-sha256=<sha256> \
      --profile=configs/vision_moe/vision_alignment/perception/treatment_v1.yaml \
      --expected-profile-sha256=<sha256> \
      --expected-git-ref=<40-character-commit>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import re
import sys
import types
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from git import Repo

from olmo_core.train import prepare_training_environment, teardown_training_environment

WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
NUM_NODES = 2
CANONICAL_WORKSPACE = "ai2/molmofication"
CANONICAL_WORKSPACE_ID = "01KSTRJHG4A32N7GDM82KY8J3E"
CANONICAL_CLUSTER = "ai2/holmes"
CANONICAL_BUDGET = "ai2/oe-other"
CANONICAL_GIT_BRANCH = "vision-moe"
RECIPE_REPOSITORY_PATH = "src/scripts/train/Vision-Alignment.py"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REF_RE = re.compile(r"[0-9a-f]{40}")
_HOLMES_HOSTNAME_RE = re.compile(r"holmes-[a-z0-9][a-z0-9.-]*")
_FORBIDDEN_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_CONFIG_FILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "GANTRY_AWS_CONFIG",
        "GANTRY_AWS_CREDENTIALS",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)
_T = TypeVar("_T")


class PerceptionRuntimePreflightError(RuntimeError):
    """Raised when the production perception runtime preflight fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PerceptionRuntimePreflightError(
            f"Could not hash required input {path}: {error}"
        ) from error
    return digest.hexdigest()


def _sha256_arg(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be exactly 64 lowercase hexadecimal characters")
    return value


def _git_ref_arg(value: str) -> str:
    if _GIT_REF_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be exactly 40 lowercase hexadecimal characters")
    return value


def _pinned_file(path_value: str | Path, expected_sha256: str, *, name: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise PerceptionRuntimePreflightError(f"{name} is not a regular file: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PerceptionRuntimePreflightError(
            f"{name} SHA-256 differs: expected {expected_sha256}, got {actual_sha256}"
        )
    return path


def _absolute_lexical_path(path_value: str | Path) -> Path:
    """Make a path absolute without following a final symlink."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))


def _repository_root(recipe_path: Path) -> Path:
    try:
        root = recipe_path.parents[3]
    except IndexError as error:
        raise PerceptionRuntimePreflightError(
            f"Recipe must use the repository path {RECIPE_REPOSITORY_PATH!r}"
        ) from error
    try:
        relative = recipe_path.relative_to(root).as_posix()
    except ValueError as error:
        raise PerceptionRuntimePreflightError("Could not resolve recipe repository path") from error
    if relative != RECIPE_REPOSITORY_PATH:
        raise PerceptionRuntimePreflightError(
            f"Recipe must use the repository path {RECIPE_REPOSITORY_PATH!r}, got {relative!r}"
        )
    return root


def _load_pinned_recipe(path: Path, expected_sha256: str) -> types.ModuleType:
    """Dynamically execute the exact caller-pinned training recipe bytes."""
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PerceptionRuntimePreflightError(
            f"Recipe changed before import: expected {expected_sha256}, got {actual_sha256}"
        )
    module_name = f"_vision_alignment_perception_preflight_recipe_{actual_sha256[:20]}"
    module = types.ModuleType(module_name)
    loader = importlib.machinery.SourceFileLoader(module_name, str(path))
    module.__file__ = str(path)
    module.__loader__ = loader
    module.__package__ = ""
    module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader, origin=str(path))
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    required = (
        "_load_profile",
        "build_config",
        "_apply_profile_launch",
        "_validate_phase_contract",
        "_perception_provenance",
    )
    for symbol in required:
        if not callable(getattr(module, symbol, None)):
            raise PerceptionRuntimePreflightError(
                f"Pinned recipe does not expose callable {symbol}"
            )
    cache = getattr(module, "_PERCEPTION_PROVENANCE_RUNTIME_CACHE", None)
    if not isinstance(cache, dict) or cache:
        raise PerceptionRuntimePreflightError(
            "Pinned recipe must begin with an empty perception provenance runtime cache"
        )
    return module


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise PerceptionRuntimePreflightError(f"Required runtime metadata {name} is missing")
    return value


def _integer_env(name: str) -> int:
    value = _required_env(name)
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise PerceptionRuntimePreflightError(
            f"Runtime metadata {name} must be a canonical non-negative integer, got {value!r}"
        )
    return int(value)


def _credential_env_names(environment: Mapping[str, str]) -> list[str]:
    """Return credential-bearing environment variable names without reading their values."""
    return sorted(
        name
        for name in environment
        if name in _FORBIDDEN_CREDENTIAL_ENV_NAMES or name.startswith("RUSTINS_AWS_")
    )


def _runtime_metadata_packet(repository_root: Path) -> dict[str, Any]:
    rank = dist.get_rank()
    try:
        checkout = Repo(repository_root)
        return {
            "rank": rank,
            "env_rank": _integer_env("RANK"),
            "local_rank": _integer_env("LOCAL_RANK"),
            "world_size": _integer_env("WORLD_SIZE"),
            "local_world_size": _integer_env("LOCAL_WORLD_SIZE"),
            "num_nodes": _integer_env("NUM_NODES"),
            "replica_count": _integer_env("BEAKER_REPLICA_COUNT"),
            "replica_rank": _integer_env("BEAKER_REPLICA_RANK"),
            "assigned_gpu_count": _integer_env("BEAKER_ASSIGNED_GPU_COUNT"),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device": torch.cuda.current_device(),
            "workspace_id": _required_env("BEAKER_WORKSPACE_ID"),
            "experiment_id": _required_env("BEAKER_EXPERIMENT_ID"),
            "workload_id": _required_env("BEAKER_WORKLOAD_ID"),
            "task_id": _required_env("BEAKER_TASK_ID"),
            "job_id": _required_env("BEAKER_JOB_ID"),
            "job_kind": _required_env("BEAKER_JOB_KIND"),
            "node_id": _required_env("BEAKER_NODE_ID"),
            "hostname": _required_env("BEAKER_NODE_HOSTNAME"),
            "leader_node_id": _required_env("BEAKER_LEADER_REPLICA_NODE_ID"),
            "leader_hostname": _required_env("BEAKER_LEADER_REPLICA_HOSTNAME"),
            "git_branch": _required_env("GIT_BRANCH"),
            "git_ref": _required_env("GIT_REF"),
            "checkout_ref": checkout.head.commit.hexsha,
            "tracked_checkout_dirty": checkout.is_dirty(untracked_files=False),
            "credential_env_names": _credential_env_names(os.environ),
        }
    except Exception as error:  # noqa: BLE001 - the exact rank failure must reach every rank.
        return {"rank": rank, "error": f"{type(error).__name__}: {error}"}


def _one_value(packets: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = {packet.get(field) for packet in packets}
    if len(values) != 1:
        raise PerceptionRuntimePreflightError(
            f"Runtime metadata {field} differs across ranks: {sorted(map(repr, values))}"
        )
    return next(iter(values))


def _validate_rank_metadata(
    packets: Sequence[Mapping[str, Any]], *, expected_git_ref: str
) -> dict[str, Any]:
    """Validate authoritative Beaker metadata for exactly two eight-GPU Holmes nodes."""
    if len(packets) != WORLD_SIZE:
        raise PerceptionRuntimePreflightError(
            f"Metadata collective returned {len(packets)} ranks, expected {WORLD_SIZE}"
        )
    failures = []
    for index, packet in enumerate(packets):
        if not isinstance(packet, Mapping):
            failures.append(f"slot {index}: malformed packet {packet!r}")
        elif packet.get("error") is not None:
            failures.append(f"rank {packet.get('rank', index)}: {packet['error']}")
    if failures:
        raise PerceptionRuntimePreflightError(
            "Rank-local runtime metadata validation failed: " + "; ".join(failures)
        )

    ranks: list[int] = []
    for packet in packets:
        rank = packet.get("rank")
        if type(rank) is not int:
            raise PerceptionRuntimePreflightError(f"Distributed rank is malformed: {rank!r}")
        ranks.append(rank)
    if sorted(ranks) != list(range(WORLD_SIZE)) or len(set(ranks)) != WORLD_SIZE:
        raise PerceptionRuntimePreflightError(f"Distributed ranks differ: {ranks}")
    for packet in packets:
        rank = packet["rank"]
        local_rank = packet.get("local_rank")
        replica_rank = packet.get("replica_rank")
        exact_fields = {
            "env_rank": rank,
            "world_size": WORLD_SIZE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "num_nodes": NUM_NODES,
            "replica_count": NUM_NODES,
            "assigned_gpu_count": LOCAL_WORLD_SIZE,
            "cuda_device_count": LOCAL_WORLD_SIZE,
            "cuda_device": local_rank,
            "workspace_id": CANONICAL_WORKSPACE_ID,
            "job_kind": "batch",
            "git_branch": CANONICAL_GIT_BRANCH,
            "git_ref": expected_git_ref,
            "checkout_ref": expected_git_ref,
            "tracked_checkout_dirty": False,
            "credential_env_names": [],
        }
        for field, expected in exact_fields.items():
            if type(packet.get(field)) is not type(expected) or packet.get(field) != expected:
                raise PerceptionRuntimePreflightError(
                    f"Rank {rank} runtime metadata {field} differs: expected {expected!r}, "
                    f"got {packet.get(field)!r}"
                )
        if type(local_rank) is not int or local_rank not in range(LOCAL_WORLD_SIZE):
            raise PerceptionRuntimePreflightError(
                f"Rank {rank} has invalid local rank {local_rank!r}"
            )
        if type(replica_rank) is not int or replica_rank not in range(NUM_NODES):
            raise PerceptionRuntimePreflightError(
                f"Rank {rank} has invalid replica rank {replica_rank!r}"
            )
        if rank != replica_rank * LOCAL_WORLD_SIZE + local_rank:
            raise PerceptionRuntimePreflightError(
                f"Rank {rank} is inconsistent with replica {replica_rank} and local rank {local_rank}"
            )
        hostname = packet.get("hostname")
        if not isinstance(hostname, str) or _HOLMES_HOSTNAME_RE.fullmatch(hostname) is None:
            raise PerceptionRuntimePreflightError(
                f"Rank {rank} is not assigned authoritative Holmes metadata: {hostname!r}"
            )
        for field in ("experiment_id", "workload_id", "task_id", "job_id", "node_id"):
            if not isinstance(packet.get(field), str) or not packet[field]:
                raise PerceptionRuntimePreflightError(
                    f"Rank {rank} lacks authoritative Beaker {field} metadata"
                )
        if packet["experiment_id"] != packet["workload_id"]:
            raise PerceptionRuntimePreflightError(
                f"Rank {rank} Beaker experiment/workload identities differ"
            )

    experiment_id = _one_value(packets, "experiment_id")
    _one_value(packets, "workload_id")
    leader_node_id = _one_value(packets, "leader_node_id")
    leader_hostname = _one_value(packets, "leader_hostname")
    by_node: dict[str, list[Mapping[str, Any]]] = {}
    for packet in packets:
        by_node.setdefault(packet["node_id"], []).append(packet)
    if len(by_node) != NUM_NODES:
        raise PerceptionRuntimePreflightError(
            f"Ranks occupy {len(by_node)} distinct Beaker nodes, expected {NUM_NODES}"
        )
    node_summaries = []
    for node_id, node_packets in sorted(by_node.items()):
        if len(node_packets) != LOCAL_WORLD_SIZE:
            raise PerceptionRuntimePreflightError(
                f"Beaker node {node_id} contains {len(node_packets)} ranks, "
                f"expected {LOCAL_WORLD_SIZE}"
            )
        hostnames = {packet["hostname"] for packet in node_packets}
        replica_ranks = {packet["replica_rank"] for packet in node_packets}
        local_ranks = {packet["local_rank"] for packet in node_packets}
        job_ids = {packet["job_id"] for packet in node_packets}
        task_ids = {packet["task_id"] for packet in node_packets}
        if (
            len(hostnames) != 1
            or len(replica_ranks) != 1
            or local_ranks != set(range(LOCAL_WORLD_SIZE))
            or len(job_ids) != 1
            or len(task_ids) != 1
        ):
            raise PerceptionRuntimePreflightError(
                f"Beaker node {node_id} does not contain one coherent eight-rank replica"
            )
        node_summaries.append(
            {
                "node_id": node_id,
                "hostname": next(iter(hostnames)),
                "replica_rank": next(iter(replica_ranks)),
                "job_id": next(iter(job_ids)),
                "task_id": next(iter(task_ids)),
            }
        )
    if Counter(item["replica_rank"] for item in node_summaries) != Counter(range(NUM_NODES)):
        raise PerceptionRuntimePreflightError(
            "Beaker nodes do not map one-to-one to replicas 0 and 1"
        )
    if len({item["hostname"] for item in node_summaries}) != NUM_NODES:
        raise PerceptionRuntimePreflightError("Beaker node hostnames are not distinct")
    if len({item["job_id"] for item in node_summaries}) != NUM_NODES:
        raise PerceptionRuntimePreflightError("Beaker replica job IDs are not distinct")
    if len({item["task_id"] for item in node_summaries}) != NUM_NODES:
        raise PerceptionRuntimePreflightError("Beaker replica task IDs are not distinct")
    leader = next(item for item in node_summaries if item["replica_rank"] == 0)
    if leader["node_id"] != leader_node_id or leader["hostname"] != leader_hostname:
        raise PerceptionRuntimePreflightError("Beaker leader metadata differs from replica zero")
    return {
        "world_size": WORLD_SIZE,
        "nodes": NUM_NODES,
        "gpus_per_node": LOCAL_WORLD_SIZE,
        "workspace": CANONICAL_WORKSPACE,
        "workspace_id": CANONICAL_WORKSPACE_ID,
        "cluster": CANONICAL_CLUSTER,
        "experiment_id": experiment_id,
        "node_hostnames": sorted(item["hostname"] for item in node_summaries),
    }


def _collective_stage(name: str, operation: Callable[[], _T]) -> _T:
    value: _T | None = None
    error_message: str | None = None
    try:
        value = operation()
    except Exception as error:  # noqa: BLE001 - every rank must receive rank-local failures.
        error_message = f"{type(error).__name__}: {error}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error_message)
    failures = [(rank, error) for rank, error in enumerate(errors) if error is not None]
    if failures:
        details = "; ".join(f"rank {rank}: {error}" for rank, error in failures)
        raise PerceptionRuntimePreflightError(f"{name} failed: {details}")
    return value  # type: ignore[return-value]


def _load_reviewed_profile(
    recipe: types.ModuleType,
    profile_path: Path,
    expected_profile_sha256: str,
) -> tuple[dict[str, Any], list[str], str, dict[str, str]]:
    profile, overrides = recipe._load_profile([f"--profile={profile_path}"])
    if not isinstance(profile, dict) or not isinstance(overrides, list):
        raise PerceptionRuntimePreflightError("Pinned recipe returned malformed profile data")
    if not all(isinstance(value, str) for value in overrides):
        raise PerceptionRuntimePreflightError("Pinned recipe returned malformed profile overrides")
    if profile.get("phase") != "perception":
        raise PerceptionRuntimePreflightError("Runtime preflight accepts perception profiles only")
    if profile.get("__reviewed_sha256__") != expected_profile_sha256:
        raise PerceptionRuntimePreflightError("Reviewed profile identity differs from its CLI pin")
    run_name = profile.get("name")
    if not isinstance(run_name, str) or not run_name:
        raise PerceptionRuntimePreflightError("Reviewed perception profile lacks a run name")
    review_fields = {
        "reviewed_profile_path": profile.get("__reviewed_path__"),
        "reviewed_profile_sha256": profile.get("__reviewed_sha256__"),
        "reviewed_profile_allowlist_path": profile.get("__reviewed_allowlist_path__"),
        "reviewed_profile_allowlist_sha256": profile.get("__reviewed_allowlist_sha256__"),
    }
    if not all(isinstance(value, str) and value for value in review_fields.values()):
        raise PerceptionRuntimePreflightError(
            "Pinned recipe omitted reviewed profile or allowlist metadata"
        )
    return profile, overrides, run_name, review_fields  # type: ignore[return-value]


def _expected_cache_key(config: Any) -> tuple[str, str]:
    path = config.data.perception_provenance_path
    sha256 = config.data.perception_provenance_sha256
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise PerceptionRuntimePreflightError("Runtime config lacks pinned perception provenance")
    return str(Path(path).expanduser().resolve()), sha256


def _validate_runtime_config(
    recipe: types.ModuleType,
    config: Any,
    *,
    run_name: str,
    expected_git_ref: str,
    expected_profile_sha256: str,
    expected_save_folder: Path,
) -> None:
    phase = getattr(config.phase, "value", config.phase)
    if phase != "perception":
        raise PerceptionRuntimePreflightError("Runtime config does not select perception")
    if (
        getattr(recipe, "BEAKER_WORKSPACE", None) != CANONICAL_WORKSPACE
        or getattr(recipe, "BEAKER_CLUSTER", None) != CANONICAL_CLUSTER
        or getattr(recipe, "BEAKER_BUDGET", None) != CANONICAL_BUDGET
    ):
        raise PerceptionRuntimePreflightError(
            "Pinned recipe launch constants differ from molmofication/Holmes"
        )
    launch = config.launch
    if (
        launch.workspace != CANONICAL_WORKSPACE
        or launch.clusters != [CANONICAL_CLUSTER]
        or launch.budget != CANONICAL_BUDGET
        or launch.num_nodes != NUM_NODES
        or launch.num_gpus != LOCAL_WORLD_SIZE
        or launch.priority != "urgent"
        or launch.min_runtime != "8h"
        or launch.hostnames is not None
        or launch.allow_dirty is not False
    ):
        raise PerceptionRuntimePreflightError(
            "Reviewed config does not use the exact molmofication/Holmes 2x8 launch"
        )
    if launch.git is None or launch.git.ref != expected_git_ref:
        raise PerceptionRuntimePreflightError(
            "Runtime config git revision differs from its CLI pin"
        )
    if config.required_run_name != run_name or config.vision_alignment.lineage_id != run_name:
        raise PerceptionRuntimePreflightError(
            "Runtime config run identity differs from its profile"
        )
    if config.reviewed_profile_sha256 != expected_profile_sha256:
        raise PerceptionRuntimePreflightError("Runtime config profile SHA-256 differs from its pin")
    profile_path = config.reviewed_profile_path
    expected_command = [
        RECIPE_REPOSITORY_PATH,
        "train",
        run_name,
        f"--profile={profile_path}",
    ]
    if config.expected_launch_command != expected_command or launch.cmd != expected_command:
        raise PerceptionRuntimePreflightError(
            "Runtime config is not bound to one profile-only command"
        )
    save_folder = _absolute_lexical_path(config.trainer.save_folder)
    if save_folder != expected_save_folder or os.path.lexists(save_folder):
        raise PerceptionRuntimePreflightError(
            f"Runtime preflight requires an absent production save folder: {save_folder}"
        )
    if any(
        secret is not None
        for secret in (
            launch.aws_config_secret,
            launch.aws_credentials_secret,
            launch.google_credentials_secret,
        )
    ):
        raise PerceptionRuntimePreflightError(
            "Runtime config unexpectedly exposes cloud credentials"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--expected-recipe-sha256", type=_sha256_arg, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", type=_sha256_arg, required=True)
    parser.add_argument("--expected-git-ref", type=_git_ref_arg, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the read-only, distributed production perception provenance preflight."""
    args = _parse_args(argv)
    if os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
        raise PerceptionRuntimePreflightError(
            f"Runtime preflight requires torchrun WORLD_SIZE={WORLD_SIZE} before initialization"
        )
    if os.environ.get("LOCAL_WORLD_SIZE") != str(LOCAL_WORLD_SIZE):
        raise PerceptionRuntimePreflightError(
            f"Runtime preflight requires LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE} before initialization"
        )
    if os.environ.get("NUM_NODES") != str(NUM_NODES):
        raise PerceptionRuntimePreflightError(
            f"Runtime preflight requires NUM_NODES={NUM_NODES} before initialization"
        )

    recipe_path = Path(args.recipe).expanduser().resolve()
    repository_root = _repository_root(recipe_path)
    prepare_training_environment(timeout=timedelta(minutes=60))
    try:
        if dist.get_world_size() != WORLD_SIZE:
            raise PerceptionRuntimePreflightError(
                f"Initialized process group has {dist.get_world_size()} ranks, expected {WORLD_SIZE}"
            )
        packets: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(packets, _runtime_metadata_packet(repository_root))
        topology = _validate_rank_metadata(packets, expected_git_ref=args.expected_git_ref)

        recipe_path, profile_path = _collective_stage(
            "input identity pin",
            lambda: (
                _pinned_file(recipe_path, args.expected_recipe_sha256, name="recipe"),
                _pinned_file(args.profile, args.expected_profile_sha256, name="profile"),
            ),
        )

        recipe = _collective_stage(
            "pinned recipe import",
            lambda: _load_pinned_recipe(recipe_path, args.expected_recipe_sha256),
        )
        profile_data = _collective_stage(
            "reviewed profile load",
            lambda: _load_reviewed_profile(recipe, profile_path, args.expected_profile_sha256),
        )
        profile, overrides, run_name, review_fields = profile_data

        def require_absent_output() -> Path:
            root = getattr(recipe, "VISION_ALIGNMENT_ROOT", None)
            if not isinstance(root, str) or not root:
                raise PerceptionRuntimePreflightError(
                    "Pinned recipe lacks its production vision-alignment root"
                )
            save_folder = _absolute_lexical_path(Path(root) / "checkpoints" / run_name)
            if os.path.lexists(save_folder):
                raise PerceptionRuntimePreflightError(
                    f"Runtime preflight requires an absent production save folder: {save_folder}"
                )
            return save_folder

        expected_save_folder = _collective_stage("production output absence", require_absent_output)

        def build_runtime_config() -> Any:
            config = recipe.build_config(
                RECIPE_REPOSITORY_PATH,
                run_name,
                overrides,
                runtime=True,
                **review_fields,
            )
            config = recipe._apply_profile_launch(config, profile, run_name=run_name)
            recipe._validate_phase_contract(config, run_name, runtime=True)
            _validate_runtime_config(
                recipe,
                config,
                run_name=run_name,
                expected_git_ref=args.expected_git_ref,
                expected_profile_sha256=args.expected_profile_sha256,
                expected_save_folder=expected_save_folder,
            )
            return config

        config = _collective_stage("runtime profile construction", build_runtime_config)
        cache = recipe._PERCEPTION_PROVENANCE_RUNTIME_CACHE

        def require_exact_runtime_snapshot() -> tuple[str, str]:
            expected_cache_key = _expected_cache_key(config)
            if set(cache) != {expected_cache_key}:
                raise PerceptionRuntimePreflightError(
                    "Pinned runtime build did not execute exactly one perception provenance snapshot"
                )
            return expected_cache_key

        expected_cache_key = _collective_stage(
            "runtime provenance cache verification", require_exact_runtime_snapshot
        )

        def confirm_exact_provenance_call() -> Any:
            manifest = recipe._perception_provenance(config)
            if manifest is not cache[expected_cache_key]:
                raise PerceptionRuntimePreflightError(
                    "Explicit provenance confirmation did not return the runtime-validated snapshot"
                )
            if _sha256_file(recipe_path) != args.expected_recipe_sha256:
                raise PerceptionRuntimePreflightError("Recipe changed during runtime preflight")
            if _sha256_file(profile_path) != args.expected_profile_sha256:
                raise PerceptionRuntimePreflightError("Profile changed during runtime preflight")
            if os.path.lexists(expected_save_folder):
                raise PerceptionRuntimePreflightError(
                    f"Production save folder appeared during preflight: {expected_save_folder}"
                )
            return manifest

        _collective_stage("exact provenance confirmation", confirm_exact_provenance_call)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "mode": "read_only_no_receipt",
                        "recipe": {
                            "path": RECIPE_REPOSITORY_PATH,
                            "sha256": args.expected_recipe_sha256,
                        },
                        "profile": {
                            "path": config.reviewed_profile_path,
                            "sha256": args.expected_profile_sha256,
                            "name": run_name,
                        },
                        "git_ref": args.expected_git_ref,
                        "perception_provenance_sha256": config.data.perception_provenance_sha256,
                        "topology": topology,
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
