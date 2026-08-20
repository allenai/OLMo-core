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
      --profile-pair-receipt=<path> \
      --expected-profile-pair-receipt-sha256=<sha256> \
      --expected-git-ref=<40-character-commit>

For either SSMax lineage, additionally pass its explicit ``--model-variant`` selector. This binds
the scaling-ladders workspace, SSMax branch, v3 pair receipt/name, dense HSDP topology, and fixed
checkpoint cadence without changing the historical default s002 checks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.machinery
import json
import os
import re
import stat
import sys
import types
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from git import Repo

import olmo_core
from olmo_core.data.multimodal import vision_alignment_perception_provenance
from olmo_core.train import prepare_training_environment, teardown_training_environment

WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
NUM_NODES = 2
CANONICAL_WORKSPACE = "ai2/molmofication"
CANONICAL_WORKSPACE_ID = "01KSTRJHG4A32N7GDM82KY8J3E"
SSMAX_CANONICAL_WORKSPACE = "ai2/scaling-ladders"
SSMAX_CANONICAL_WORKSPACE_ID = "01KSTRR20XQE9V505A61SW3EBS"
CANONICAL_CLUSTER = "ai2/holmes"
CANONICAL_BUDGET = "ai2/oe-other"
CANONICAL_GIT_BRANCH = "vision-moe"
SSMAX_CANONICAL_GIT_BRANCH = "rustin/vision-ssmax-molmofication"
RECIPE_REPOSITORY_PATH = "src/scripts/train/Vision-Alignment.py"
PROFILE_PAIR_FORMAT = "vision_alignment_perception_profile_pair_audit"
PROFILE_PAIR_VERSION = 2
SSMAX_PROFILE_PAIR_VERSION = 3
S002_MODEL_VARIANT = "s002"
SSMAX_MODEL_VARIANTS = ("ssmax_head_qknorm", "ssmax_no_qknorm")
PROFILE_ARMS = ("frozen_vision_control", "treatment")
PROFILE_NAMES = {
    "frozen_vision_control": "vision-alignment-perception-frozen-vision-control-v1",
    "treatment": "vision-alignment-perception-treatment-v1",
}
PROFILE_PAIR_NAME = "perception-profile-pair-v2.json"
SSMAX_PROFILE_NAMES = {
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
SSMAX_PROFILE_PAIR_NAMES = {
    "ssmax_head_qknorm": "ssmax-head-qknorm-perception-profile-pair-v3.json",
    "ssmax_no_qknorm": "ssmax-no-qknorm-perception-profile-pair-v3.json",
}
_PROFILE_PAIR_ROOT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "recipe_execution_module",
        "producer",
        "recipe",
        "review_allowlist",
        "profiles",
        "launch_contract",
        "comparison",
        "data",
        "git",
        "initialization",
        "perception_contract",
        "save_folders",
    }
)
_SSMAX_PROFILE_PAIR_ROOT_FIELDS = frozenset({*_PROFILE_PAIR_ROOT_FIELDS, "model_variant"})
_PATH_IDENTITY_FIELDS = frozenset({"path", "repository_path", "sha256"})
_PROFILE_FIELDS = frozenset({"name", "path", "repository_path", "sha256"})
_DATA_FIELDS = frozenset(
    {
        "config_sha256",
        "data_contract_sha256",
        "evaluation_config_sha256",
        "perception_provenance_sha256",
        "source_audit_fingerprint",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REF_RE = re.compile(r"[0-9a-f]{40}")
_HOLMES_HOSTNAME_RE = re.compile(r"holmes-[a-z0-9][a-z0-9.-]*")
_SSMAX_FIXED_STEPS = [500, 1000, 2000, 3000, 4000]
_SSMAX_TRAIN_MODULE_CLASS = (
    "olmo_core.train.train_module.transformer.multimodal_train_module."
    "MultimodalTransformerTrainModuleConfig"
)
_SSMAX_DP_CONFIG_CLASS = (
    "olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig"
)
_ALLOWED_IDENTITY_CONFIG_PATHS = (
    "/expected_launch_command",
    "/launch/cmd",
    "/launch/description",
    "/launch/name",
    "/required_run_name",
    "/reviewed_profile_path",
    "/reviewed_profile_sha256",
    "/trainer/callbacks/wandb/name",
    "/trainer/save_folder",
    "/vision_alignment/lineage_id",
)
_ALLOWED_ARM_CONFIG_PATHS = (
    "/perception_trainability_arm",
    "/train_module/freeze_params",
    "/train_module/optim/group_overrides/<vision>/opts/lr",
    "/vision_alignment/trainable_contract_sha256",
)
_LEGACY_PERCEPTION_CONTRACT: dict[str, Any] = {
    "duration": {"unit": "steps", "value": 4000},
    "evaluation": {
        "interval": 500,
        "examples_per_source": 512,
        "rank_batch_instances": 4,
        "seed": 6198,
        "eval_on_startup": True,
        "eval_on_finish": True,
    },
    "checkpointer": {
        "save_interval": 1000,
        "ephemeral_save_interval": 400,
        "fixed_steps": None,
        "max_checkpoints": 6,
        "save_async": False,
        "pre_train_checkpoint": True,
    },
    "data_sequence_length": 2560,
    "global_batch_size": 327680,
    "rank_microbatch_size": 10240,
    "expert_parallel_degree": 8,
    "data_seed": 95818,
    "init_seed": 6198,
    "checkpoint_load_threads": 8,
    "router_lb_loss_weight": 0.015,
}
_SSMAX_PERCEPTION_CONTRACT: dict[str, Any] = {
    "duration": {"unit": "steps", "value": 4000},
    "evaluation": {
        "interval": 500,
        "examples_per_source": 512,
        "rank_batch_instances": 4,
        "seed": 6198,
        "eval_on_startup": True,
        "eval_on_finish": True,
    },
    "checkpointer": {
        "save_interval": None,
        "ephemeral_save_interval": 400,
        "fixed_steps": _SSMAX_FIXED_STEPS,
        "max_checkpoints": 6,
        "save_async": False,
        "pre_train_checkpoint": True,
    },
    "data_sequence_length": 2560,
    "global_batch_size": 327680,
    "rank_microbatch_size": 10240,
    "data_seed": 95818,
    "init_seed": 6198,
    "checkpoint_load_threads": 8,
    "router_lb_loss_weight": None,
    "parallelism": {
        "train_module_class": _SSMAX_TRAIN_MODULE_CLASS,
        "data_parallel": {
            "class": _SSMAX_DP_CONFIG_CLASS,
            "name": "hsdp",
            "param_dtype": "bfloat16",
            "reduce_dtype": "float32",
        },
        "expert_parallel": None,
    },
}
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


def _model_variant_policy(model_variant: str) -> dict[str, Any]:
    """Return the closed runtime policy for one supported perception lineage."""

    if model_variant == S002_MODEL_VARIANT:
        return {
            "model_variant": model_variant,
            "workspace": CANONICAL_WORKSPACE,
            "workspace_id": CANONICAL_WORKSPACE_ID,
            "workspace_recipe_constant": "BEAKER_WORKSPACE",
            "git_branch": CANONICAL_GIT_BRANCH,
            "profile_pair_version": PROFILE_PAIR_VERSION,
            "profile_pair_name": PROFILE_PAIR_NAME,
            "profile_pair_root_fields": _PROFILE_PAIR_ROOT_FIELDS,
            "profile_names": PROFILE_NAMES,
            "perception_contract": _LEGACY_PERCEPTION_CONTRACT,
            "experiment_root_constant": "VISION_ALIGNMENT_ROOT",
        }
    if model_variant in SSMAX_MODEL_VARIANTS:
        return {
            "model_variant": model_variant,
            "workspace": SSMAX_CANONICAL_WORKSPACE,
            "workspace_id": SSMAX_CANONICAL_WORKSPACE_ID,
            "workspace_recipe_constant": "SSMAX_BEAKER_WORKSPACE",
            "git_branch": SSMAX_CANONICAL_GIT_BRANCH,
            "profile_pair_version": SSMAX_PROFILE_PAIR_VERSION,
            "profile_pair_name": SSMAX_PROFILE_PAIR_NAMES[model_variant],
            "profile_pair_root_fields": _SSMAX_PROFILE_PAIR_ROOT_FIELDS,
            "profile_names": SSMAX_PROFILE_NAMES[model_variant],
            "perception_contract": _SSMAX_PERCEPTION_CONTRACT,
            "experiment_root_constant": "SSMAX_VISION_ALIGNMENT_ROOT",
        }
    raise PerceptionRuntimePreflightError(f"Unsupported perception model variant {model_variant!r}")


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


def _reject_symlink_components(path: Path, *, name: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise PerceptionRuntimePreflightError(f"{name} may not contain symlinks: {current}")


def _pinned_profile_pair_receipt(
    path_value: str | Path,
    expected_sha256: str,
    *,
    model_variant: str = S002_MODEL_VARIANT,
) -> Path:
    policy = _model_variant_policy(model_variant)
    path = _absolute_lexical_path(path_value)
    if path.name != policy["profile_pair_name"] or path.parent.name != "artifacts":
        raise PerceptionRuntimePreflightError(
            f"Profile-pair receipt must be artifacts/{policy['profile_pair_name']}"
        )
    _reject_symlink_components(path, name="profile-pair receipt")
    try:
        info = path.lstat()
    except OSError as error:
        raise PerceptionRuntimePreflightError(
            f"Profile-pair receipt is unavailable: {path}"
        ) from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise PerceptionRuntimePreflightError(
            f"Profile-pair receipt is not a regular non-symlink file: {path}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PerceptionRuntimePreflightError(
            "profile-pair receipt SHA-256 differs: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return path


def _runtime_source_identity(repository_root: Path) -> dict[str, str]:
    source_root = (repository_root / "src").resolve()
    if os.environ.get("PYTHONPATH") != str(source_root):
        raise PerceptionRuntimePreflightError(
            f"Runtime PYTHONPATH must be exactly the checkout source root {source_root}"
        )
    expected = {
        "olmo_core": source_root / "olmo_core" / "__init__.py",
        "perception_provenance": (
            source_root
            / "olmo_core"
            / "data"
            / "multimodal"
            / "vision_alignment_perception_provenance.py"
        ),
    }
    actual = {
        "olmo_core": Path(olmo_core.__file__).resolve(),
        "perception_provenance": Path(vision_alignment_perception_provenance.__file__).resolve(),
    }
    if actual != expected:
        raise PerceptionRuntimePreflightError(
            f"Runtime imports do not resolve to the exact checkout source: {actual}"
        )
    return {name: path.relative_to(repository_root).as_posix() for name, path in actual.items()}


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


def _is_main_guard(node: ast.stmt) -> bool:
    """Return whether an AST node is exactly ``if __name__ == "__main__"``."""
    if not isinstance(node, ast.If) or node.orelse:
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _recipe_import_tree(raw: bytes, *, path: Path) -> ast.Module:
    """Parse a recipe and remove only its final, conventional CLI entrypoint guard."""
    try:
        tree = ast.parse(raw, filename=str(path))
    except (SyntaxError, ValueError) as error:
        raise PerceptionRuntimePreflightError(
            f"Could not parse pinned recipe {path}: {error}"
        ) from error
    guards = [index for index, node in enumerate(tree.body) if _is_main_guard(node)]
    if guards != [len(tree.body) - 1]:
        raise PerceptionRuntimePreflightError(
            "Pinned recipe must contain exactly one final if __name__ == '__main__' CLI guard"
        )
    tree.body.pop()
    return tree


@contextmanager
def _recipe_main_module(recipe: types.ModuleType) -> Iterator[None]:
    """Temporarily expose a pinned recipe under its real script module identity."""
    previous = sys.modules.get("__main__")
    sys.modules["__main__"] = recipe
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = previous


def _load_pinned_recipe(path: Path, expected_sha256: str) -> types.ModuleType:
    """Execute the exact pinned recipe definitions with their real ``__main__`` identity."""
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PerceptionRuntimePreflightError(
            f"Recipe changed before import: expected {expected_sha256}, got {actual_sha256}"
        )
    tree = _recipe_import_tree(raw, path=path)
    module_name = "__main__"
    module = types.ModuleType(module_name)
    loader = importlib.machinery.SourceFileLoader(module_name, str(path))
    module.__file__ = str(path)
    module.__loader__ = loader
    module.__package__ = None
    module.__spec__ = None
    with _recipe_main_module(module):
        exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102

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


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key!r}")
        value[key] = item
    return value


def _required_mapping(
    value: Any,
    *,
    name: str,
    fields: frozenset[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PerceptionRuntimePreflightError(f"Profile-pair receipt {name} must be an object")
    if fields is not None and set(value) != fields:
        raise PerceptionRuntimePreflightError(
            f"Profile-pair receipt {name} fields differ: expected {sorted(fields)}, "
            f"got {sorted(value)}"
        )
    return value


def _receipt_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PerceptionRuntimePreflightError(
            f"Profile-pair receipt {name} must be a lowercase SHA-256"
        )
    return value


def _load_profile_pair_receipt(
    path: Path,
    expected_sha256: str,
    *,
    repository_root: Path,
    recipe_sha256: str,
    profile_path: Path,
    profile_sha256: str,
    git_ref: str,
    model_variant: str = S002_MODEL_VARIANT,
) -> dict[str, Any]:
    """Load and bind the exact lineage-specific pair receipt to the pinned runtime inputs."""
    policy = _model_variant_policy(model_variant)
    receipt_path = _pinned_profile_pair_receipt(path, expected_sha256, model_variant=model_variant)
    try:
        raw = receipt_path.read_bytes()
        receipt = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PerceptionRuntimePreflightError(
            f"Profile-pair receipt is not strict JSON: {error}"
        ) from error
    root = _required_mapping(receipt, name="root", fields=policy["profile_pair_root_fields"])
    canonical = (
        json.dumps(
            root,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise PerceptionRuntimePreflightError("Profile-pair receipt bytes are not canonical JSON")
    if (
        root.get("format") != PROFILE_PAIR_FORMAT
        or root.get("version") != policy["profile_pair_version"]
        or isinstance(root.get("version"), bool)
        or root.get("status") != "passed"
        or root.get("recipe_execution_module") != "__main__"
    ):
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt identity, version, or passed status differs"
        )
    if model_variant in SSMAX_MODEL_VARIANTS and root.get("model_variant") != model_variant:
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt SSMax model variant differs from the requested lineage"
        )

    producer = _required_mapping(
        root.get("producer"), name="producer", fields=_PATH_IDENTITY_FIELDS
    )
    review_allowlist = _required_mapping(
        root.get("review_allowlist"),
        name="review_allowlist",
        fields=_PATH_IDENTITY_FIELDS,
    )
    for name, record in (("producer", producer), ("review_allowlist", review_allowlist)):
        _receipt_sha256(record.get("sha256"), name=f"{name}.sha256")
        if not all(isinstance(record.get(field), str) and record[field] for field in record):
            raise PerceptionRuntimePreflightError(
                f"Profile-pair receipt {name} path identity is malformed"
            )
    recipe = _required_mapping(
        root.get("recipe"),
        name="recipe",
        fields=frozenset({"path", "command_path", "sha256"}),
    )
    if (
        recipe.get("command_path") != RECIPE_REPOSITORY_PATH
        or recipe.get("sha256") != recipe_sha256
    ):
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt recipe differs from the pinned runtime recipe"
        )
    _receipt_sha256(recipe.get("sha256"), name="recipe.sha256")
    git = _required_mapping(root.get("git"), name="git", fields=frozenset({"branch", "ref"}))
    if git.get("branch") != policy["git_branch"] or git.get("ref") != git_ref:
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt git identity differs from the pinned checkout"
        )
    launch = _required_mapping(
        root.get("launch_contract"),
        name="launch_contract",
        fields=frozenset(
            {"workspace", "cluster", "budget", "num_nodes", "num_gpus", "priority", "min_runtime"}
        ),
    )
    expected_launch = {
        "workspace": policy["workspace"],
        "cluster": CANONICAL_CLUSTER,
        "budget": CANONICAL_BUDGET,
        "num_nodes": NUM_NODES,
        "num_gpus": LOCAL_WORLD_SIZE,
        "priority": "urgent",
        "min_runtime": "8h",
    }
    if dict(launch) != expected_launch:
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt launch contract differs from production"
        )

    try:
        profile_repository_path = profile_path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise PerceptionRuntimePreflightError(
            "Pinned profile must be inside the recipe repository"
        ) from error
    profiles = _required_mapping(
        root.get("profiles"), name="profiles", fields=frozenset(PROFILE_ARMS)
    )
    if set(profiles) != set(PROFILE_ARMS):
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt does not contain exactly the canonical two arms"
        )
    selected: list[tuple[str, Mapping[str, Any]]] = []
    for arm in PROFILE_ARMS:
        record = _required_mapping(profiles[arm], name=f"profiles.{arm}", fields=_PROFILE_FIELDS)
        if (
            record.get("name") != policy["profile_names"][arm]
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or not isinstance(record.get("repository_path"), str)
            or not record["repository_path"]
        ):
            raise PerceptionRuntimePreflightError(
                f"Profile-pair receipt profiles.{arm} identity is malformed"
            )
        _receipt_sha256(record.get("sha256"), name=f"profiles.{arm}.sha256")
        if (
            record.get("repository_path") == profile_repository_path
            and record.get("sha256") == profile_sha256
        ):
            selected.append((arm, record))
    if len(selected) != 1:
        raise PerceptionRuntimePreflightError(
            "Pinned profile is not exactly one arm in the profile-pair receipt"
        )
    arm, profile_record = selected[0]
    profile_name = profile_record.get("name")
    if not isinstance(profile_name, str) or not profile_name:
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt selected profile lacks a run name"
        )

    data = _required_mapping(root.get("data"), name="data", fields=_DATA_FIELDS)
    for field in _DATA_FIELDS:
        _receipt_sha256(data.get(field), name=f"data.{field}")
    data_contract_sha256 = _receipt_sha256(
        data.get("data_contract_sha256"), name="data.data_contract_sha256"
    )
    provenance_sha256 = _receipt_sha256(
        data.get("perception_provenance_sha256"),
        name="data.perception_provenance_sha256",
    )
    comparison = _required_mapping(
        root.get("comparison"),
        name="comparison",
        fields=frozenset(
            {
                "allowed_identity_config_paths",
                "allowed_arm_config_paths",
                "arm_config_sha256",
                "shared_config_sha256",
                "trainable_contract_sha256",
            }
        ),
    )
    if comparison.get("allowed_identity_config_paths") != list(
        _ALLOWED_IDENTITY_CONFIG_PATHS
    ) or comparison.get("allowed_arm_config_paths") != list(_ALLOWED_ARM_CONFIG_PATHS):
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt does not declare the exact identity and causal-arm difference"
        )
    _receipt_sha256(comparison.get("shared_config_sha256"), name="comparison.shared_config_sha256")
    for field in ("arm_config_sha256", "trainable_contract_sha256"):
        values = _required_mapping(
            comparison.get(field), name=f"comparison.{field}", fields=frozenset(PROFILE_ARMS)
        )
        for arm in PROFILE_ARMS:
            _receipt_sha256(values.get(arm), name=f"comparison.{field}.{arm}")
    initialization = _required_mapping(
        root.get("initialization"),
        name="initialization",
        fields=frozenset(
            {"checkpoint", "parent_config_sha256", "parent_gate_path", "parent_gate_sha256"}
        ),
    )
    for field in ("parent_config_sha256", "parent_gate_sha256"):
        _receipt_sha256(initialization.get(field), name=f"initialization.{field}")
    perception_contract = _required_mapping(
        root.get("perception_contract"), name="perception_contract"
    )
    actual_contract = json.dumps(
        perception_contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    expected_contract = json.dumps(
        policy["perception_contract"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if actual_contract != expected_contract:
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt perception contract differs from the exact lineage policy"
        )
    save_folders = _required_mapping(
        root.get("save_folders"),
        name="save_folders",
        fields=frozenset({"status", *PROFILE_ARMS}),
    )
    if save_folders.get("status") != "verified_absent_and_distinct":
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt save-folder status is not verified"
        )
    save_folder_paths: dict[str, Path] = {}
    for save_arm in PROFILE_ARMS:
        value = save_folders.get(save_arm)
        if not isinstance(value, str) or not value:
            raise PerceptionRuntimePreflightError(
                f"Profile-pair receipt save_folders.{save_arm} is malformed"
            )
        save_path = _absolute_lexical_path(value)
        _reject_symlink_components(save_path, name=f"save_folders.{save_arm}")
        if os.path.lexists(save_path):
            raise PerceptionRuntimePreflightError(
                f"Profile-pair receipt production save folder already exists: {save_path}"
            )
        save_folder_paths[save_arm] = save_path
    if len(set(save_folder_paths.values())) != len(PROFILE_ARMS):
        raise PerceptionRuntimePreflightError(
            "Profile-pair receipt production save folders are not distinct"
        )
    return {
        "path": str(receipt_path),
        "sha256": expected_sha256,
        "version": policy["profile_pair_version"],
        "model_variant": model_variant,
        "arm": arm,
        "profile_name": profile_name,
        "data_contract_sha256": data_contract_sha256,
        "perception_provenance_sha256": provenance_sha256,
        "control_save_folder": str(save_folder_paths["frozen_vision_control"]),
        "treatment_save_folder": str(save_folder_paths["treatment"]),
    }


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
    packets: Sequence[Mapping[str, Any]],
    *,
    expected_git_ref: str,
    model_variant: str = S002_MODEL_VARIANT,
) -> dict[str, Any]:
    """Validate authoritative Beaker metadata for exactly two eight-GPU Holmes nodes."""
    policy = _model_variant_policy(model_variant)
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
            "workspace_id": policy["workspace_id"],
            "job_kind": "batch",
            "git_branch": policy["git_branch"],
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
        "workspace": policy["workspace"],
        "workspace_id": policy["workspace_id"],
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


def _collective_identical(name: str, value: _T) -> _T:
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    if any(item != values[0] for item in values[1:]):
        raise PerceptionRuntimePreflightError(f"{name} differs across ranks")
    return values[0]


def _load_reviewed_profile(
    recipe: types.ModuleType,
    profile_path: Path,
    expected_profile_sha256: str,
) -> tuple[dict[str, Any], list[str], str, dict[str, str]]:
    with _recipe_main_module(recipe):
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


def _serialized_at(value: Any, *path: str) -> Any:
    current = value
    for field in path:
        if not isinstance(current, Mapping) or field not in current:
            raise PerceptionRuntimePreflightError(
                f"Runtime config is missing required field {'.'.join(path)}"
            )
        current = current[field]
    return current


def _runtime_perception_contract(config: Any, *, model_variant: str) -> dict[str, Any]:
    """Reconstruct the receipt contract independently from the runtime config."""

    try:
        raw = config.as_config_dict()
    except Exception as error:  # noqa: BLE001 - convert config failures to a closed preflight.
        raise PerceptionRuntimePreflightError(
            f"Runtime config could not be serialized for contract validation: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise PerceptionRuntimePreflightError("Runtime config serialization must be an object")

    duration = _serialized_at(raw, "trainer", "max_duration")
    evaluation = _serialized_at(raw, "evaluation")
    checkpointer = _serialized_at(raw, "trainer", "callbacks", "checkpointer")
    train_module = _serialized_at(raw, "train_module")
    if not isinstance(checkpointer, Mapping) or not isinstance(train_module, Mapping):
        raise PerceptionRuntimePreflightError(
            "Runtime checkpointer and train module serializations must be objects"
        )
    contract: dict[str, Any] = {
        "duration": {field: _serialized_at(duration, field) for field in ("unit", "value")},
        "evaluation": {
            field: _serialized_at(evaluation, field)
            for field in (
                "interval",
                "examples_per_source",
                "rank_batch_instances",
                "seed",
                "eval_on_startup",
                "eval_on_finish",
            )
        },
        "checkpointer": {
            "save_interval": checkpointer.get("save_interval"),
            "ephemeral_save_interval": _serialized_at(checkpointer, "ephemeral_save_interval"),
            "fixed_steps": checkpointer.get("fixed_steps"),
            "max_checkpoints": _serialized_at(checkpointer, "max_checkpoints"),
            "save_async": _serialized_at(checkpointer, "save_async"),
            "pre_train_checkpoint": _serialized_at(checkpointer, "pre_train_checkpoint"),
        },
        "data_sequence_length": _serialized_at(raw, "data", "sequence_length"),
        "global_batch_size": _serialized_at(raw, "global_batch_size"),
        "rank_microbatch_size": _serialized_at(train_module, "rank_microbatch_size"),
        "data_seed": _serialized_at(raw, "data_seed"),
        "init_seed": _serialized_at(raw, "init_seed"),
        "checkpoint_load_threads": _serialized_at(raw, "checkpoint_load_threads"),
    }
    if model_variant == S002_MODEL_VARIANT:
        contract["expert_parallel_degree"] = _serialized_at(train_module, "ep_config", "degree")
        contract["router_lb_loss_weight"] = _serialized_at(raw, "router_lb_loss_weight")
        return contract

    if "save_interval" in checkpointer:
        raise PerceptionRuntimePreflightError(
            "SSMax runtime checkpointer save_interval must be omitted (serialized None)"
        )
    if "ep_config" in train_module:
        raise PerceptionRuntimePreflightError(
            "SSMax runtime ep_config must be omitted for dense generic HSDP"
        )
    if "router_lb_loss_weight" in raw:
        raise PerceptionRuntimePreflightError(
            "SSMax runtime router_lb_loss_weight must be omitted (serialized None)"
        )
    if (
        raw.get("model_variant") != model_variant
        or _serialized_at(raw, "vision_alignment", "model_variant") != model_variant
    ):
        raise PerceptionRuntimePreflightError(
            "SSMax runtime root and metadata model variants differ from the requested lineage"
        )
    dp_config = _serialized_at(train_module, "dp_config")
    contract["router_lb_loss_weight"] = None
    contract["parallelism"] = {
        "train_module_class": _serialized_at(train_module, "_CLASS_"),
        "data_parallel": {
            "class": _serialized_at(dp_config, "_CLASS_"),
            "name": _serialized_at(dp_config, "name"),
            "param_dtype": _serialized_at(dp_config, "param_dtype"),
            "reduce_dtype": _serialized_at(dp_config, "reduce_dtype"),
        },
        "expert_parallel": None,
    }
    return contract


def _validate_runtime_perception_contract(config: Any, *, model_variant: str) -> None:
    policy = _model_variant_policy(model_variant)
    actual = json.dumps(
        _runtime_perception_contract(config, model_variant=model_variant),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    expected = json.dumps(
        policy["perception_contract"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if actual != expected:
        raise PerceptionRuntimePreflightError(
            "Runtime perception cadence, scale, router, or parallelism contract differs"
        )


def _validate_runtime_config(
    recipe: types.ModuleType,
    config: Any,
    *,
    run_name: str,
    expected_git_ref: str,
    expected_profile_sha256: str,
    expected_data_contract_sha256: str,
    expected_provenance_sha256: str,
    expected_save_folder: Path,
    model_variant: str = S002_MODEL_VARIANT,
) -> None:
    policy = _model_variant_policy(model_variant)
    phase = getattr(config.phase, "value", config.phase)
    if phase != "perception":
        raise PerceptionRuntimePreflightError("Runtime config does not select perception")
    actual_model_variant = getattr(config.model_variant, "value", config.model_variant)
    if actual_model_variant != model_variant:
        raise PerceptionRuntimePreflightError(
            "Runtime config model variant differs from the requested lineage"
        )
    if (
        getattr(recipe, policy["workspace_recipe_constant"], None) != policy["workspace"]
        or getattr(recipe, "BEAKER_CLUSTER", None) != CANONICAL_CLUSTER
        or getattr(recipe, "BEAKER_BUDGET", None) != CANONICAL_BUDGET
    ):
        raise PerceptionRuntimePreflightError(
            f"Pinned recipe launch constants differ from {policy['workspace']}/Holmes"
        )
    launch = config.launch
    if (
        launch.workspace != policy["workspace"]
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
            f"Reviewed config does not use the exact {policy['workspace']}/Holmes 2x8 launch"
        )
    if (
        launch.git is None
        or launch.git.ref != expected_git_ref
        or launch.git.branch != policy["git_branch"]
    ):
        raise PerceptionRuntimePreflightError(
            "Runtime config git revision differs from its CLI pin"
        )
    if config.required_run_name != run_name or config.vision_alignment.lineage_id != run_name:
        raise PerceptionRuntimePreflightError(
            "Runtime config run identity differs from its profile"
        )
    if config.reviewed_profile_sha256 != expected_profile_sha256:
        raise PerceptionRuntimePreflightError("Runtime config profile SHA-256 differs from its pin")
    if (
        config.vision_alignment.data_contract_sha256 != expected_data_contract_sha256
        or config.data.perception_provenance_sha256 != expected_provenance_sha256
    ):
        raise PerceptionRuntimePreflightError(
            "Runtime data contract differs from the pinned profile-pair receipt"
        )
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
    _validate_runtime_perception_contract(config, model_variant=model_variant)
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
    launch_pythonpaths = [item.value for item in launch.env_vars if item.name == "PYTHONPATH"]
    if launch_pythonpaths != [str(Path("/gantry-runtime/src"))]:
        raise PerceptionRuntimePreflightError(
            "Runtime launch must contain exactly PYTHONPATH=/gantry-runtime/src"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-variant",
        choices=(S002_MODEL_VARIANT, *SSMAX_MODEL_VARIANTS),
        default=S002_MODEL_VARIANT,
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--expected-recipe-sha256", type=_sha256_arg, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", type=_sha256_arg, required=True)
    parser.add_argument("--profile-pair-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-profile-pair-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--expected-git-ref", type=_git_ref_arg, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the read-only, distributed production perception provenance preflight."""
    args = _parse_args(argv)
    policy = _model_variant_policy(args.model_variant)
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
    receipt_path = _absolute_lexical_path(args.profile_pair_receipt)
    prepare_training_environment(timeout=timedelta(minutes=60))
    try:
        if dist.get_world_size() != WORLD_SIZE:
            raise PerceptionRuntimePreflightError(
                f"Initialized process group has {dist.get_world_size()} ranks, expected {WORLD_SIZE}"
            )
        packets: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(packets, _runtime_metadata_packet(repository_root))
        topology = _validate_rank_metadata(
            packets,
            expected_git_ref=args.expected_git_ref,
            model_variant=args.model_variant,
        )
        source_identity = _collective_stage(
            "runtime source identity", lambda: _runtime_source_identity(repository_root)
        )
        source_identity = _collective_identical("Runtime source identity", source_identity)

        recipe_path, profile_path, receipt_path = _collective_stage(
            "input identity pin",
            lambda: (
                _pinned_file(recipe_path, args.expected_recipe_sha256, name="recipe"),
                _pinned_file(args.profile, args.expected_profile_sha256, name="profile"),
                _pinned_profile_pair_receipt(
                    receipt_path,
                    args.expected_profile_pair_receipt_sha256,
                    model_variant=args.model_variant,
                ),
            ),
        )
        receipt = _collective_stage(
            "profile-pair receipt validation",
            lambda: _load_profile_pair_receipt(
                receipt_path,
                args.expected_profile_pair_receipt_sha256,
                repository_root=repository_root,
                recipe_sha256=args.expected_recipe_sha256,
                profile_path=profile_path,
                profile_sha256=args.expected_profile_sha256,
                git_ref=args.expected_git_ref,
                model_variant=args.model_variant,
            ),
        )
        receipt = _collective_identical("Profile-pair receipt summary", receipt)

        recipe = _collective_stage(
            "pinned recipe import",
            lambda: _load_pinned_recipe(recipe_path, args.expected_recipe_sha256),
        )
        profile_data = _collective_stage(
            "reviewed profile load",
            lambda: _load_reviewed_profile(recipe, profile_path, args.expected_profile_sha256),
        )
        profile, overrides, run_name, review_fields = profile_data
        _collective_stage(
            "profile-pair arm selection",
            lambda: (
                None
                if receipt["profile_name"] == run_name
                else (_ for _ in ()).throw(
                    PerceptionRuntimePreflightError(
                        "Reviewed profile run name differs from the profile-pair receipt"
                    )
                )
            ),
        )

        def require_absent_output() -> Path:
            root = getattr(recipe, policy["experiment_root_constant"], None)
            if not isinstance(root, str) or not root:
                raise PerceptionRuntimePreflightError(
                    "Pinned recipe lacks its production lineage-specific vision-alignment root"
                )
            save_folder = _absolute_lexical_path(Path(root) / "checkpoints" / run_name)
            if os.path.lexists(save_folder):
                raise PerceptionRuntimePreflightError(
                    f"Runtime preflight requires an absent production save folder: {save_folder}"
                )
            return save_folder

        expected_save_folder = _collective_stage("production output absence", require_absent_output)

        def build_runtime_config() -> Any:
            with _recipe_main_module(recipe):
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
                    expected_data_contract_sha256=receipt["data_contract_sha256"],
                    expected_provenance_sha256=receipt["perception_provenance_sha256"],
                    expected_save_folder=expected_save_folder,
                    model_variant=args.model_variant,
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
            with _recipe_main_module(recipe):
                manifest = recipe._perception_provenance(config)
            if manifest is not cache[expected_cache_key]:
                raise PerceptionRuntimePreflightError(
                    "Explicit provenance confirmation did not return the runtime-validated snapshot"
                )
            if _sha256_file(recipe_path) != args.expected_recipe_sha256:
                raise PerceptionRuntimePreflightError("Recipe changed during runtime preflight")
            if _sha256_file(profile_path) != args.expected_profile_sha256:
                raise PerceptionRuntimePreflightError("Profile changed during runtime preflight")
            if _sha256_file(receipt_path) != args.expected_profile_pair_receipt_sha256:
                raise PerceptionRuntimePreflightError(
                    "Profile-pair receipt changed during runtime preflight"
                )
            for field in ("control_save_folder", "treatment_save_folder"):
                save_folder = _absolute_lexical_path(receipt[field])
                _reject_symlink_components(save_folder, name=field)
                if os.path.lexists(save_folder):
                    raise PerceptionRuntimePreflightError(
                        f"Production save folder appeared during preflight: {save_folder}"
                    )
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
                        "profile_pair_receipt": {
                            "path": receipt["path"],
                            "sha256": receipt["sha256"],
                            "version": receipt["version"],
                            "model_variant": receipt["model_variant"],
                            "arm": receipt["arm"],
                            "recipe_execution_module": "__main__",
                            "data_contract_sha256": receipt["data_contract_sha256"],
                            "perception_provenance_sha256": receipt["perception_provenance_sha256"],
                        },
                        "git_ref": args.expected_git_ref,
                        "perception_provenance_sha256": config.data.perception_provenance_sha256,
                        "vision_alignment": {
                            "data_contract_sha256": (config.vision_alignment.data_contract_sha256),
                        },
                        "runtime_source": source_identity,
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
