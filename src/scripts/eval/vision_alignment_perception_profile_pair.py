"""Audit a reviewed Vision Alignment perception control/treatment profile pair.

This tool builds both profiles through the exact, SHA-256-pinned training recipe that would
launch them. It never submits a job. The resulting receipt proves that the two reviewed
profiles use identical data and configuration except for run identity and the intended
vision-trainability intervention.

Example::

    PYTHONPATH=src python src/scripts/eval/vision_alignment_perception_profile_pair.py \
      --recipe=src/scripts/train/Vision-Alignment.py \
      --expected-recipe-sha256=<sha256> \
      --control-profile=configs/vision_moe/vision_alignment/perception/control.yaml \
      --expected-control-profile-sha256=<sha256> \
      --treatment-profile=configs/vision_moe/vision_alignment/perception/treatment.yaml \
      --expected-treatment-profile-sha256=<sha256> \
      --output=/path/to/profile-pair-audit.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.machinery
import json
import os
import re
import sys
import tempfile
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

FORMAT = "vision_alignment_perception_profile_pair_audit"
VERSION = 2
SSMAX_VERSION = 3
CONTROL_ARM = "frozen_vision_control"
TREATMENT_ARM = "treatment"
ARM_OVERRIDE_PREFIX = "--perception_trainability_arm="
MODEL_VARIANT_OVERRIDE_PREFIX = "--model_variant="
S002_MODEL_VARIANT = "s002"
SSMAX_MODEL_VARIANTS = ("ssmax_head_qknorm", "ssmax_no_qknorm")
CANONICAL_WORKSPACE = "ai2/molmofication"
SSMAX_CANONICAL_WORKSPACE = "ai2/scaling-ladders"
CANONICAL_CLUSTER = "ai2/holmes"
CANONICAL_BUDGET = "ai2/oe-other"
CANONICAL_GIT_BRANCH = "vision-moe"
SSMAX_CANONICAL_GIT_BRANCH = "rustin/vision-ssmax-molmofication"
PRODUCER_REPOSITORY_PATH = "src/scripts/eval/vision_alignment_perception_profile_pair.py"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REF_RE = re.compile(r"[0-9a-f]{40}")

_TREATMENT_FREEZE_PARAMS = [
    "lm.embedding_norm.*",
    "lm.blocks.*",
    "lm.lm_head.*",
]

_PERCEPTION_CONSTANTS: dict[str, Any] = {
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

_SSMAX_FIXED_STEPS = [500, 1000, 2000, 3000, 4000]
_SSMAX_TRAIN_MODULE_CLASS = (
    "olmo_core.train.train_module.transformer.multimodal_train_module."
    "MultimodalTransformerTrainModuleConfig"
)
_SSMAX_DP_CONFIG_CLASS = (
    "olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig"
)
_SSMAX_PERCEPTION_CONSTANTS: dict[str, Any] = {
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

_REQUIRED_LAUNCH: dict[str, Any] = {
    "num_nodes": 2,
    "num_gpus": 8,
    "workspace": CANONICAL_WORKSPACE,
    "cluster": CANONICAL_CLUSTER,
    "budget": CANONICAL_BUDGET,
    "priority": "urgent",
    "min_runtime": "8h",
}

_SSMAX_REQUIRED_LAUNCH: dict[str, Any] = {
    **_REQUIRED_LAUNCH,
    "workspace": SSMAX_CANONICAL_WORKSPACE,
}

_IDENTITY_CONFIG_PATHS = (
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

_SSMAX_IDENTITY_CONFIG_PATHS = (
    *_IDENTITY_CONFIG_PATHS,
    "/trainer/callbacks/ssmax_health_ledger/run_name",
)

_ARM_CONFIG_PATHS = (
    "/perception_trainability_arm",
    "/train_module/freeze_params",
    "/train_module/optim/group_overrides/<vision>/opts/lr",
    "/vision_alignment/trainable_contract_sha256",
)


class ProfilePairAuditError(ValueError):
    """Raised when a perception control/treatment pair is not causally matched."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ProfilePairAuditError(f"Could not hash required input {path}: {error}") from error
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProfilePairAuditError(f"Value is not canonical-JSON serializable: {error}") from error


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProfilePairAuditError(f"{name} must be a lowercase SHA-256")
    return value


def _pinned_file(path_value: str | Path, expected_sha256: str, *, name: str) -> Path:
    expected_sha256 = _require_sha256(expected_sha256, name=f"expected {name} SHA-256")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ProfilePairAuditError(f"{name} is not a regular file: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ProfilePairAuditError(
            f"{name} SHA-256 differs: expected {expected_sha256}, got {actual_sha256}"
        )
    return path


def _absolute_lexical_path(path_value: str | Path) -> Path:
    """Make a path absolute without following its final or parent symlinks."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))


def _reject_symlink_components(path: Path, *, name: str) -> None:
    """Reject redirectable path components, including a dangling final symlink."""
    for component in [path, *path.parents]:
        if component.is_symlink():
            raise ProfilePairAuditError(f"{name} may not contain symlinks: {component}")


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
        raise ProfilePairAuditError(f"Could not parse pinned recipe {path}: {error}") from error
    guards = [index for index, node in enumerate(tree.body) if _is_main_guard(node)]
    if guards != [len(tree.body) - 1]:
        raise ProfilePairAuditError(
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
    """Execute caller-pinned recipe definitions with their real ``__main__`` identity."""
    raw = path.read_bytes()
    actual_sha256 = _sha256_bytes(raw)
    if actual_sha256 != expected_sha256:
        raise ProfilePairAuditError(
            f"recipe SHA-256 changed before import: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    tree = _recipe_import_tree(raw, path=path)
    module_name = "__main__"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__loader__ = importlib.machinery.SourceFileLoader(module_name, str(path))
    # Match ``python path/to/recipe.py`` rather than import-module semantics. In particular,
    # Config.as_config_dict() records recipe-local classes as ``__main__.ClassName``.
    module.__package__ = None
    module.__spec__ = None
    try:
        with _recipe_main_module(module):
            exec(compile(tree, str(path), "exec"), module.__dict__)
    except Exception:
        raise

    for symbol in (
        "prepare_cli_environment",
        "_load_profile",
        "build_config",
        "_apply_profile_launch",
        "_validate_phase_contract",
    ):
        if not callable(getattr(module, symbol, None)):
            raise ProfilePairAuditError(f"Pinned recipe does not expose callable {symbol}")
    return module


def _repository_root(recipe_path: Path) -> Path:
    try:
        root = recipe_path.parents[3]
    except IndexError as error:
        raise ProfilePairAuditError(
            "Recipe must have the repository layout src/scripts/train/<recipe>.py"
        ) from error
    expected_parent = root / "src" / "scripts" / "train"
    if recipe_path.parent != expected_parent.resolve():
        raise ProfilePairAuditError(
            "Recipe must be located directly under src/scripts/train in its repository"
        )
    return root


def _recipe_command_path(recipe_path: Path, repository_root: Path) -> str:
    return recipe_path.relative_to(repository_root).as_posix()


def _at(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for field_name in path:
        if not isinstance(current, Mapping) or field_name not in current:
            raise ProfilePairAuditError(f"Canonical config lacks {'/'.join(path)}")
        current = current[field_name]
    return current


def _is_exact_mapping(actual: Any, expected: Mapping[str, Any]) -> bool:
    """Compare a mapping without allowing bool/int or other equality aliases."""
    return (
        isinstance(actual, Mapping)
        and set(actual) == set(expected)
        and all(
            type(actual[field_name]) is type(expected_value)
            and actual[field_name] == expected_value
            for field_name, expected_value in expected.items()
        )
    )


def _require_exact_value(actual: Any, expected: Any, *, name: str) -> Any:
    """Require equal values with identical JSON-facing Python types."""
    if type(actual) is not type(expected) or actual != expected:
        raise ProfilePairAuditError(
            f"{name} differs: expected {expected!r} ({type(expected).__name__}), "
            f"got {actual!r} ({type(actual).__name__})"
        )
    return actual


def _model_variant_policy(model_variant: str) -> dict[str, Any]:
    """Return the closed launch and receipt policy for one supported model lineage."""

    if model_variant == S002_MODEL_VARIANT:
        return {
            "model_variant": model_variant,
            "receipt_version": VERSION,
            "workspace": CANONICAL_WORKSPACE,
            "workspace_recipe_constant": "BEAKER_WORKSPACE",
            "git_branch": CANONICAL_GIT_BRANCH,
            "launch": _REQUIRED_LAUNCH,
        }
    if model_variant in SSMAX_MODEL_VARIANTS:
        return {
            "model_variant": model_variant,
            "receipt_version": SSMAX_VERSION,
            "workspace": SSMAX_CANONICAL_WORKSPACE,
            "workspace_recipe_constant": "SSMAX_BEAKER_WORKSPACE",
            "git_branch": SSMAX_CANONICAL_GIT_BRANCH,
            "launch": _SSMAX_REQUIRED_LAUNCH,
        }
    raise ProfilePairAuditError(f"Unsupported perception model variant {model_variant!r}")


def _clear_perception_provenance_cache(recipe: types.ModuleType) -> None:
    cache = getattr(recipe, "_PERCEPTION_PROVENANCE_RUNTIME_CACHE", None)
    if not isinstance(cache, dict):
        raise ProfilePairAuditError(
            "Pinned recipe does not expose its perception provenance runtime cache"
        )
    cache.clear()


def _load_reviewed_profile(
    recipe: types.ModuleType,
    profile_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    try:
        with _recipe_main_module(recipe):
            profile, overrides = recipe._load_profile([f"--profile={profile_path}"])
    except Exception as error:
        raise ProfilePairAuditError(
            f"Pinned recipe rejected reviewed profile {profile_path}: {error}"
        ) from error
    if not isinstance(profile, dict):
        raise ProfilePairAuditError(f"Recipe did not load {profile_path} as a reviewed profile")
    if not isinstance(overrides, list) or not all(isinstance(value, str) for value in overrides):
        raise ProfilePairAuditError("Recipe profile loader returned malformed overrides")
    return profile, overrides


def _build_profile_config(
    recipe: types.ModuleType,
    *,
    command_path: str,
    profile: dict[str, Any],
    overrides: list[str],
) -> tuple[Any, dict[str, Any]]:
    run_name = profile.get("name")
    if not isinstance(run_name, str) or not run_name:
        raise ProfilePairAuditError("Every reviewed profile must have a non-empty name")
    review_fields = {
        "reviewed_profile_path": profile.get("__reviewed_path__"),
        "reviewed_profile_sha256": profile.get("__reviewed_sha256__"),
        "reviewed_profile_allowlist_path": profile.get("__reviewed_allowlist_path__"),
        "reviewed_profile_allowlist_sha256": profile.get("__reviewed_allowlist_sha256__"),
    }
    if not all(isinstance(value, str) and value for value in review_fields.values()):
        raise ProfilePairAuditError("Recipe profile loader omitted reviewed profile metadata")
    try:
        with _recipe_main_module(recipe):
            config = recipe.build_config(
                command_path,
                run_name,
                overrides,
                runtime=False,
                **review_fields,
            )
            config = recipe._apply_profile_launch(config, profile, run_name=run_name)
            recipe._validate_phase_contract(config, run_name, runtime=False)
            config_dict = config.as_config_dict()
    except Exception as error:
        raise ProfilePairAuditError(
            f"Pinned recipe could not build and validate profile {run_name!r}: {error}"
        ) from error
    if not isinstance(config_dict, dict):
        raise ProfilePairAuditError("Recipe config did not serialize to a mapping")
    # A canonical round trip both rejects non-finite/non-JSON values and detaches mutable values.
    return config, json.loads(_canonical_json_bytes(config_dict))


def _profile_arm(profile: Mapping[str, Any], *, expected_arm: str) -> list[str]:
    overrides = profile.get("overrides")
    if not isinstance(overrides, list) or not all(isinstance(value, str) for value in overrides):
        raise ProfilePairAuditError("Reviewed profile overrides must be a list of strings")
    arm_overrides = [value for value in overrides if value.startswith(ARM_OVERRIDE_PREFIX)]
    expected = f"{ARM_OVERRIDE_PREFIX}{expected_arm}"
    if arm_overrides != [expected]:
        raise ProfilePairAuditError(
            f"Profile arm must be declared exactly once as {expected!r}; got {arm_overrides}"
        )
    return [value for value in overrides if not value.startswith(ARM_OVERRIDE_PREFIX)]


def _profile_model_variant(profile: Mapping[str, Any], *, arm: str) -> str:
    overrides = profile.get("overrides")
    if not isinstance(overrides, list) or not all(isinstance(value, str) for value in overrides):
        raise ProfilePairAuditError(f"{arm} reviewed profile overrides must be strings")
    selectors = [
        value.split("=", 1)[1]
        for value in overrides
        if value.startswith(MODEL_VARIANT_OVERRIDE_PREFIX)
    ]
    if not selectors:
        return S002_MODEL_VARIANT
    if len(selectors) != 1:
        raise ProfilePairAuditError(
            f"{arm} profile must select its model variant at most once; got {selectors}"
        )
    model_variant = selectors[0]
    _model_variant_policy(model_variant)
    if model_variant in SSMAX_MODEL_VARIANTS:
        expected = f"{MODEL_VARIANT_OVERRIDE_PREFIX}{model_variant}"
        if [value for value in overrides if value.startswith(MODEL_VARIANT_OVERRIDE_PREFIX)] != [
            expected
        ]:
            raise ProfilePairAuditError(
                f"{arm} SSMax profile must select {model_variant!r} exactly once"
            )
    return model_variant


def _reviewed_relative_path(
    profile: Mapping[str, Any], *, repository_root: Path, input_path: Path
) -> str:
    value = profile.get("__reviewed_path__")
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise ProfilePairAuditError("Reviewed profile path metadata must be repository-relative")
    if (repository_root / value).resolve() != input_path:
        raise ProfilePairAuditError("Reviewed profile path metadata differs from its pinned input")
    return value


def _audit_profiles(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    repository_root: Path,
    control_path: Path,
    control_sha256: str,
    treatment_path: Path,
    treatment_sha256: str,
) -> dict[str, Any]:
    control_model_variant = _profile_model_variant(control, arm=CONTROL_ARM)
    treatment_model_variant = _profile_model_variant(treatment, arm=TREATMENT_ARM)
    if control_model_variant != treatment_model_variant:
        raise ProfilePairAuditError("Reviewed profiles select different model variants")
    model_variant = control_model_variant
    policy = _model_variant_policy(model_variant)

    allowed_internal = {
        "__reviewed_path__",
        "__reviewed_sha256__",
        "__reviewed_allowlist_path__",
        "__reviewed_allowlist_sha256__",
    }
    for arm, profile, expected_sha256 in (
        (CONTROL_ARM, control, control_sha256),
        (TREATMENT_ARM, treatment, treatment_sha256),
    ):
        internal = {key for key in profile if key.startswith("__")}
        if internal != allowed_internal:
            raise ProfilePairAuditError(
                f"{arm} profile reviewed metadata fields differ: {sorted(internal)}"
            )
        if profile.get("__reviewed_sha256__") != expected_sha256:
            raise ProfilePairAuditError(f"{arm} reviewed SHA-256 differs from its caller pin")
        if profile.get("phase") != "perception":
            raise ProfilePairAuditError(f"{arm} profile must select the perception phase")
        launch = profile.get("launch")
        if not _is_exact_mapping(launch, policy["launch"]):
            raise ProfilePairAuditError(
                f"{arm} profile must use the exact "
                f"{policy['workspace'].removeprefix('ai2/')}/Holmes "
                "2x8 urgent/8h launch"
            )

    control_base_overrides = _profile_arm(control, expected_arm=CONTROL_ARM)
    treatment_base_overrides = _profile_arm(treatment, expected_arm=TREATMENT_ARM)
    if control_base_overrides != treatment_base_overrides:
        raise ProfilePairAuditError("Reviewed profiles differ outside their exact arm selector")

    control_public = {
        key: copy.deepcopy(value) for key, value in control.items() if not key.startswith("__")
    }
    treatment_public = {
        key: copy.deepcopy(value) for key, value in treatment.items() if not key.startswith("__")
    }
    for normalized, base_overrides in (
        (control_public, control_base_overrides),
        (treatment_public, treatment_base_overrides),
    ):
        normalized["name"] = "<run-identity>"
        normalized.pop("description", None)
        normalized["overrides"] = base_overrides
    if _canonical_json_bytes(control_public) != _canonical_json_bytes(treatment_public):
        raise ProfilePairAuditError("Reviewed profile documents differ outside identity and arm")

    control_name = control.get("name")
    treatment_name = treatment.get("name")
    if (
        not isinstance(control_name, str)
        or not control_name
        or not isinstance(treatment_name, str)
        or not treatment_name
        or control_name == treatment_name
    ):
        raise ProfilePairAuditError("Control and treatment must have distinct non-empty names")

    control_relative = _reviewed_relative_path(
        control, repository_root=repository_root, input_path=control_path
    )
    treatment_relative = _reviewed_relative_path(
        treatment, repository_root=repository_root, input_path=treatment_path
    )
    if control_relative == treatment_relative:
        raise ProfilePairAuditError("Control and treatment must be distinct reviewed files")

    allowlist_path = control.get("__reviewed_allowlist_path__")
    allowlist_sha256 = control.get("__reviewed_allowlist_sha256__")
    if (
        allowlist_path != treatment.get("__reviewed_allowlist_path__")
        or allowlist_sha256 != treatment.get("__reviewed_allowlist_sha256__")
        or not isinstance(allowlist_path, str)
        or not allowlist_path
        or Path(allowlist_path).is_absolute()
        or ".." in Path(allowlist_path).parts
    ):
        raise ProfilePairAuditError("Reviewed profiles do not share one repository allowlist")
    allowlist_sha256 = _require_sha256(allowlist_sha256, name="profile allowlist SHA-256")
    allowlist_absolute = (repository_root / allowlist_path).resolve()
    if _sha256_file(allowlist_absolute) != allowlist_sha256:
        raise ProfilePairAuditError("Reviewed profile allowlist bytes differ from their pin")

    return {
        "model_variant": model_variant,
        "allowlist_path": allowlist_absolute,
        "allowlist_relative_path": allowlist_path,
        "allowlist_sha256": allowlist_sha256,
        "control_name": control_name,
        "control_relative_path": control_relative,
        "treatment_name": treatment_name,
        "treatment_relative_path": treatment_relative,
    }


def _find_vision_group(config: Mapping[str, Any], *, arm: str) -> tuple[int, Mapping[str, Any]]:
    groups = _at(config, "train_module", "optim", "group_overrides")
    if not isinstance(groups, list):
        raise ProfilePairAuditError(f"{arm} optimizer group_overrides must be a list")
    matches = [
        (index, group)
        for index, group in enumerate(groups)
        if isinstance(group, Mapping) and group.get("params") == ["*vision.*"]
    ]
    if len(matches) != 1:
        raise ProfilePairAuditError(f"{arm} config must have exactly one *vision.* optimizer group")
    return matches[0]


def _validate_config_identity(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    arm: str,
    command_path: str,
    profile_relative_path: str,
    profile_sha256: str,
    model_variant: str,
) -> Path:
    policy = _model_variant_policy(model_variant)
    run_name = profile["name"]
    expected_command = [
        command_path,
        "train",
        run_name,
        f"--profile={profile_relative_path}",
    ]
    if config.get("required_run_name") != run_name:
        raise ProfilePairAuditError(f"{arm} required_run_name differs from profile identity")
    if _at(config, "vision_alignment", "lineage_id") != run_name:
        raise ProfilePairAuditError(f"{arm} lineage differs from profile identity")
    if _at(config, "trainer", "callbacks", "wandb", "name") != run_name:
        raise ProfilePairAuditError(f"{arm} W&B name differs from profile identity")
    if config.get("expected_launch_command") != expected_command:
        raise ProfilePairAuditError(f"{arm} expected launch command is not profile-only")
    if _at(config, "launch", "cmd") != expected_command:
        raise ProfilePairAuditError(f"{arm} Beaker command differs from the validated command")
    launch_name = _at(config, "launch", "name")
    if (
        not isinstance(launch_name, str)
        or re.fullmatch(rf"{re.escape(run_name)}-[0-9a-f]{{8}}", launch_name) is None
    ):
        raise ProfilePairAuditError(f"{arm} Beaker launch name is not exact run identity plus UUID")
    description = profile.get("description")
    if _at(config, "launch").get("description") != description:
        raise ProfilePairAuditError(f"{arm} launch description differs from its profile")
    if config.get("reviewed_profile_path") != profile_relative_path:
        raise ProfilePairAuditError(f"{arm} config reviewed path differs from its profile")
    if config.get("reviewed_profile_sha256") != profile_sha256:
        raise ProfilePairAuditError(f"{arm} config reviewed SHA-256 differs from its pin")

    launch = _at(config, "launch")
    expected_final_launch = {
        "num_nodes": launch.get("num_nodes"),
        "num_gpus": launch.get("num_gpus"),
        "workspace": launch.get("workspace"),
        "cluster": (launch.get("clusters") or [None])[0],
        "budget": launch.get("budget"),
        "priority": launch.get("priority"),
        "min_runtime": launch.get("min_runtime"),
    }
    if not _is_exact_mapping(expected_final_launch, policy["launch"]) or launch.get("clusters") != [
        CANONICAL_CLUSTER
    ]:
        raise ProfilePairAuditError(
            f"{arm} config must use the exact "
            f"{policy['workspace'].removeprefix('ai2/')}/Holmes "
            "2x8 urgent/8h launch"
        )
    if launch.get("hostnames"):
        raise ProfilePairAuditError(f"{arm} config may not select exact hosts")

    save_folder = _at(config, "trainer", "save_folder")
    if not isinstance(save_folder, str) or not save_folder:
        raise ProfilePairAuditError(f"{arm} save folder must be a non-empty path")
    save_path = _absolute_lexical_path(save_folder)
    _reject_symlink_components(save_path, name=f"{arm} save folder")
    return save_path


def _normalize_identity(
    config: Mapping[str, Any], vision_group_index: int, *, model_variant: str
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    run_name = _at(normalized, "required_run_name")
    callbacks = _at(normalized, "trainer", "callbacks")
    health_ledger = callbacks.get("ssmax_health_ledger")
    if model_variant in SSMAX_MODEL_VARIANTS:
        if not isinstance(health_ledger, dict):
            raise ProfilePairAuditError("SSMax health-ledger callback is missing")
        _require_exact_value(
            health_ledger.get("model_variant"),
            model_variant,
            name="SSMax health-ledger model variant",
        )
        _require_exact_value(
            health_ledger.get("phase"),
            "perception",
            name="SSMax health-ledger phase",
        )
        _require_exact_value(
            health_ledger.get("run_name"),
            run_name,
            name="SSMax health-ledger run name",
        )
        health_ledger["run_name"] = "<run>"
    normalized["expected_launch_command"] = ["<recipe>", "train", "<run>", "--profile=<profile>"]
    normalized["required_run_name"] = "<run>"
    normalized["reviewed_profile_path"] = "<profile>"
    normalized["reviewed_profile_sha256"] = "<profile-sha256>"
    normalized["launch"]["cmd"] = ["<recipe>", "train", "<run>", "--profile=<profile>"]
    normalized["launch"]["name"] = "<run>-<uuid>"
    normalized["launch"].pop("description", None)
    normalized["trainer"]["save_folder"] = "<save-folder>"
    normalized["trainer"]["callbacks"]["wandb"]["name"] = "<run>"
    normalized["vision_alignment"]["lineage_id"] = "<run>"
    # Assert the optimizer shape used by the caller before returning a mutable normalized copy.
    _ = normalized["train_module"]["optim"]["group_overrides"][vision_group_index]
    return normalized


def _normalize_arm(config: dict[str, Any], vision_group_index: int) -> None:
    config["perception_trainability_arm"] = "<arm>"
    config["train_module"]["freeze_params"] = "<arm-freeze-params>"
    config["train_module"]["optim"]["group_overrides"][vision_group_index]["opts"][
        "lr"
    ] = "<arm-vision-lr>"
    config["vision_alignment"]["trainable_contract_sha256"] = "<arm-contract-sha256>"


def _data_identity(config: Mapping[str, Any]) -> dict[str, str]:
    data_contract_sha256 = _require_sha256(
        _at(config, "vision_alignment", "data_contract_sha256"),
        name="data contract SHA-256",
    )
    provenance_sha256 = _require_sha256(
        _at(config, "data", "perception_provenance_sha256"),
        name="perception provenance SHA-256",
    )
    source_audit_sha256 = _require_sha256(
        _at(config, "data", "source_audit_fingerprint"),
        name="perception source-audit fingerprint",
    )
    return {
        "config_sha256": _canonical_sha256(_at(config, "data")),
        "data_contract_sha256": data_contract_sha256,
        "evaluation_config_sha256": _canonical_sha256(_at(config, "evaluation")),
        "perception_provenance_sha256": provenance_sha256,
        "source_audit_fingerprint": source_audit_sha256,
    }


def _perception_contract(
    config: Mapping[str, Any], *, arm: str, model_variant: str = S002_MODEL_VARIANT
) -> dict[str, Any]:
    """Independently require every frozen perception cadence, scale, and topology constant."""
    constants = (
        _SSMAX_PERCEPTION_CONSTANTS
        if model_variant in SSMAX_MODEL_VARIANTS
        else _PERCEPTION_CONSTANTS
    )
    duration = _at(config, "trainer", "max_duration")
    duration_contract = {
        field_name: _require_exact_value(
            _at(duration, field_name), expected, name=f"{arm} duration {field_name}"
        )
        for field_name, expected in constants["duration"].items()
    }

    evaluation = _at(config, "evaluation")
    evaluation_contract = {
        field_name: _require_exact_value(
            _at(evaluation, field_name), expected, name=f"{arm} evaluation {field_name}"
        )
        for field_name, expected in constants["evaluation"].items()
    }

    checkpointer = _at(config, "trainer", "callbacks", "checkpointer")
    checkpointer_contract: dict[str, Any] = {}
    for field_name, expected in constants["checkpointer"].items():
        # Config.as_config_dict() excludes None fields. With the pinned producer, absence of
        # an optional field is its exact serialized representation of None; every non-None value
        # is serialized and therefore rejected here.
        if expected is None:
            if model_variant in SSMAX_MODEL_VARIANTS and field_name in checkpointer:
                raise ProfilePairAuditError(
                    f"{arm} SSMax checkpointer {field_name} must be omitted (serialized None)"
                )
            actual = checkpointer.get(field_name)
        else:
            actual = _at(checkpointer, field_name)
        checkpointer_contract[field_name] = _require_exact_value(
            actual, expected, name=f"{arm} checkpointer {field_name}"
        )

    scalar_paths: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("data_sequence_length", ("data", "sequence_length")),
        ("global_batch_size", ("global_batch_size",)),
        ("rank_microbatch_size", ("train_module", "rank_microbatch_size")),
        ("data_seed", ("data_seed",)),
        ("init_seed", ("init_seed",)),
        ("checkpoint_load_threads", ("checkpoint_load_threads",)),
    )
    contract: dict[str, Any] = {
        "duration": duration_contract,
        "evaluation": evaluation_contract,
        "checkpointer": checkpointer_contract,
    }
    for field_name, path in scalar_paths:
        expected = constants[field_name]
        contract[field_name] = _require_exact_value(
            _at(config, *path), expected, name=f"{arm} {field_name}"
        )

    if model_variant == S002_MODEL_VARIANT:
        contract["expert_parallel_degree"] = _require_exact_value(
            _at(config, "train_module", "ep_config", "degree"),
            _PERCEPTION_CONSTANTS["expert_parallel_degree"],
            name=f"{arm} expert_parallel_degree",
        )
        contract["router_lb_loss_weight"] = _require_exact_value(
            _at(config, "router_lb_loss_weight"),
            _PERCEPTION_CONSTANTS["router_lb_loss_weight"],
            name=f"{arm} router_lb_loss_weight",
        )
        return contract

    train_module = _at(config, "train_module")
    train_module_class = _require_exact_value(
        _at(train_module, "_CLASS_"),
        _SSMAX_TRAIN_MODULE_CLASS,
        name=f"{arm} SSMax train module class",
    )
    if "ep_config" in train_module:
        raise ProfilePairAuditError(
            f"{arm} SSMax ep_config must be omitted for the dense generic train module"
        )
    dp_config = _at(train_module, "dp_config")
    data_parallel = {
        "class": _require_exact_value(
            _at(dp_config, "_CLASS_"),
            _SSMAX_DP_CONFIG_CLASS,
            name=f"{arm} SSMax data-parallel config class",
        ),
        "name": _require_exact_value(
            _at(dp_config, "name"), "hsdp", name=f"{arm} SSMax data parallelism"
        ),
        "param_dtype": _require_exact_value(
            _at(dp_config, "param_dtype"),
            "bfloat16",
            name=f"{arm} SSMax parameter dtype",
        ),
        "reduce_dtype": _require_exact_value(
            _at(dp_config, "reduce_dtype"),
            "float32",
            name=f"{arm} SSMax reduction dtype",
        ),
    }
    if "router_lb_loss_weight" in config:
        raise ProfilePairAuditError(
            f"{arm} SSMax router_lb_loss_weight must be omitted (serialized None)"
        )
    contract["router_lb_loss_weight"] = None
    contract["parallelism"] = {
        "train_module_class": train_module_class,
        "data_parallel": data_parallel,
        "expert_parallel": None,
    }
    return contract


def _git_and_parent_identity(
    config: Mapping[str, Any], *, arm: str, model_variant: str = S002_MODEL_VARIANT
) -> dict[str, Any]:
    """Require and collect the shared code revision and parent-quality lineage."""
    git = _at(config, "launch", "git")
    policy = _model_variant_policy(model_variant)
    branch = _require_exact_value(
        _at(git, "branch"), policy["git_branch"], name=f"{arm} git branch"
    )
    git_ref = _at(git, "ref")
    if not isinstance(git_ref, str) or _GIT_REF_RE.fullmatch(git_ref) is None:
        raise ProfilePairAuditError(f"{arm} git ref must be an exact lowercase 40-hex revision")

    initialization = _at(config, "initialization")
    checkpoint = _at(initialization, "checkpoint")
    parent_gate_path = _at(initialization, "parent_gate_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ProfilePairAuditError(f"{arm} initialization checkpoint must be non-empty")
    if not isinstance(parent_gate_path, str) or not parent_gate_path:
        raise ProfilePairAuditError(f"{arm} parent gate path must be non-empty")
    parent_config_sha256 = _require_sha256(
        _at(initialization, "parent_config_sha256"),
        name=f"{arm} parent config SHA-256",
    )
    parent_gate_sha256 = _require_sha256(
        _at(initialization, "parent_gate_sha256"),
        name=f"{arm} parent gate SHA-256",
    )
    _require_exact_value(
        _at(initialization, "mode"), "checkpoint", name=f"{arm} initialization mode"
    )
    _require_exact_value(
        _at(initialization, "expected_parent_phase"),
        "bridge",
        name=f"{arm} expected parent phase",
    )

    metadata = _at(config, "vision_alignment")
    _require_exact_value(
        _at(metadata, "parent_checkpoint"),
        checkpoint,
        name=f"{arm} mirrored parent checkpoint",
    )
    _require_exact_value(
        _at(metadata, "parent_config_sha256"),
        parent_config_sha256,
        name=f"{arm} mirrored parent config SHA-256",
    )
    _require_exact_value(
        _at(metadata, "parent_gate_sha256"),
        parent_gate_sha256,
        name=f"{arm} mirrored parent gate SHA-256",
    )
    return {
        "git": {"branch": branch, "ref": git_ref},
        "initialization": {
            "checkpoint": checkpoint,
            "parent_config_sha256": parent_config_sha256,
            "parent_gate_path": parent_gate_path,
            "parent_gate_sha256": parent_gate_sha256,
        },
    }


def _validate_config_model_variant(
    config: Mapping[str, Any], *, arm: str, model_variant: str
) -> None:
    root_variant = config.get("model_variant")
    metadata_variant = _at(config, "vision_alignment").get("model_variant")
    if model_variant == S002_MODEL_VARIANT:
        for location, value in (("root", root_variant), ("metadata", metadata_variant)):
            if value not in (None, S002_MODEL_VARIANT):
                raise ProfilePairAuditError(
                    f"{arm} {location} model variant differs from historical s002"
                )
        return
    if root_variant != model_variant or metadata_variant != model_variant:
        raise ProfilePairAuditError(
            f"{arm} SSMax root and metadata model variants must both equal {model_variant!r}"
        )


def _audit_configs(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    control_profile: Mapping[str, Any],
    treatment_profile: Mapping[str, Any],
    command_path: str,
    control_relative_path: str,
    control_sha256: str,
    treatment_relative_path: str,
    treatment_sha256: str,
    model_variant: str = S002_MODEL_VARIANT,
) -> dict[str, Any]:
    if control.get("phase") != "perception" or treatment.get("phase") != "perception":
        raise ProfilePairAuditError("Both canonical configs must select perception")
    if control.get("perception_trainability_arm") != CONTROL_ARM:
        raise ProfilePairAuditError("Control config does not select frozen_vision_control")
    if treatment.get("perception_trainability_arm") != TREATMENT_ARM:
        raise ProfilePairAuditError("Treatment config does not select treatment")
    _validate_config_model_variant(control, arm=CONTROL_ARM, model_variant=model_variant)
    _validate_config_model_variant(treatment, arm=TREATMENT_ARM, model_variant=model_variant)

    control_save = _validate_config_identity(
        control,
        control_profile,
        arm=CONTROL_ARM,
        command_path=command_path,
        profile_relative_path=control_relative_path,
        profile_sha256=control_sha256,
        model_variant=model_variant,
    )
    treatment_save = _validate_config_identity(
        treatment,
        treatment_profile,
        arm=TREATMENT_ARM,
        command_path=command_path,
        profile_relative_path=treatment_relative_path,
        profile_sha256=treatment_sha256,
        model_variant=model_variant,
    )
    if control_save == treatment_save:
        raise ProfilePairAuditError("Control and treatment save folders must be distinct")
    for arm, save_folder in ((CONTROL_ARM, control_save), (TREATMENT_ARM, treatment_save)):
        if os.path.lexists(save_folder):
            raise ProfilePairAuditError(f"{arm} save folder already exists: {save_folder}")

    control_freeze = _at(control, "train_module", "freeze_params")
    treatment_freeze = _at(treatment, "train_module", "freeze_params")
    _require_exact_value(
        treatment_freeze,
        _TREATMENT_FREEZE_PARAMS,
        name="treatment freeze list",
    )
    _require_exact_value(
        control_freeze,
        ["vision.*", *_TREATMENT_FREEZE_PARAMS],
        name="control freeze list",
    )

    control_vision_index, control_vision_group = _find_vision_group(control, arm=CONTROL_ARM)
    treatment_vision_index, treatment_vision_group = _find_vision_group(
        treatment, arm=TREATMENT_ARM
    )
    if control_vision_index != treatment_vision_index:
        raise ProfilePairAuditError("Vision optimizer group position differs between arms")
    control_lr = _at(control_vision_group, "opts", "lr")
    treatment_lr = _at(treatment_vision_group, "opts", "lr")
    _require_exact_value(control_lr, 0.0, name="control vision LR")
    _require_exact_value(treatment_lr, 3e-6, name="treatment vision LR")

    control_trainable_contract_sha256 = _require_sha256(
        _at(control, "vision_alignment", "trainable_contract_sha256"),
        name="control trainable contract SHA-256",
    )
    treatment_trainable_contract_sha256 = _require_sha256(
        _at(treatment, "vision_alignment", "trainable_contract_sha256"),
        name="treatment trainable contract SHA-256",
    )
    if control_trainable_contract_sha256 == treatment_trainable_contract_sha256:
        raise ProfilePairAuditError("Arm trainable-contract SHA-256 values must be distinct")

    control_identity_normalized = _normalize_identity(
        control, control_vision_index, model_variant=model_variant
    )
    treatment_identity_normalized = _normalize_identity(
        treatment, treatment_vision_index, model_variant=model_variant
    )
    control_arm_config_sha256 = _canonical_sha256(control_identity_normalized)
    treatment_arm_config_sha256 = _canonical_sha256(treatment_identity_normalized)
    _normalize_arm(control_identity_normalized, control_vision_index)
    _normalize_arm(treatment_identity_normalized, treatment_vision_index)
    control_shared_bytes = _canonical_json_bytes(control_identity_normalized)
    treatment_shared_bytes = _canonical_json_bytes(treatment_identity_normalized)
    if control_shared_bytes != treatment_shared_bytes:
        raise ProfilePairAuditError(
            "Canonical configs differ outside the reviewed identity and exact arm fields"
        )

    control_data = _data_identity(control)
    treatment_data = _data_identity(treatment)
    if control_data != treatment_data:
        raise ProfilePairAuditError("Control and treatment data hashes are not identical")

    control_perception_contract = _perception_contract(
        control, arm=CONTROL_ARM, model_variant=model_variant
    )
    treatment_perception_contract = _perception_contract(
        treatment, arm=TREATMENT_ARM, model_variant=model_variant
    )
    if _canonical_json_bytes(control_perception_contract) != _canonical_json_bytes(
        treatment_perception_contract
    ):
        raise ProfilePairAuditError("Control and treatment perception constants differ")

    control_lineage = _git_and_parent_identity(
        control, arm=CONTROL_ARM, model_variant=model_variant
    )
    treatment_lineage = _git_and_parent_identity(
        treatment, arm=TREATMENT_ARM, model_variant=model_variant
    )
    if _canonical_json_bytes(control_lineage) != _canonical_json_bytes(treatment_lineage):
        raise ProfilePairAuditError("Control and treatment git or parent lineage differs")

    return {
        "arm_config_sha256": {
            CONTROL_ARM: control_arm_config_sha256,
            TREATMENT_ARM: treatment_arm_config_sha256,
        },
        "data": control_data,
        "git_and_parent": control_lineage,
        "perception_contract": control_perception_contract,
        "save_folders": {
            CONTROL_ARM: str(control_save),
            TREATMENT_ARM: str(treatment_save),
        },
        "shared_config_sha256": _sha256_bytes(control_shared_bytes),
        "trainable_contract_sha256": {
            CONTROL_ARM: control_trainable_contract_sha256,
            TREATMENT_ARM: treatment_trainable_contract_sha256,
        },
    }


def _verify_inputs(expected_inputs: Mapping[Path, str]) -> None:
    for path, expected_sha256 in expected_inputs.items():
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ProfilePairAuditError(
                f"Input changed during audit: {path}; expected {expected_sha256}, "
                f"got {actual_sha256}"
            )


def _verify_save_folders_absent(save_folders: Sequence[Path]) -> None:
    for save_folder in save_folders:
        _reject_symlink_components(save_folder, name="save folder")
        if os.path.lexists(save_folder):
            raise ProfilePairAuditError(
                f"Save folder appeared during profile-pair audit: {save_folder}"
            )


def _write_json_once(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_inputs: Mapping[Path, str],
    save_folders: Sequence[Path],
) -> None:
    """Stage deterministic bytes, rehash inputs, then atomically publish without overwrite."""
    path = _absolute_lexical_path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite immutable profile-pair receipt {path}")
    _reject_symlink_components(path, name="receipt output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        # These are deliberately the final operations before the no-replace publication link.
        _verify_inputs(expected_inputs)
        _verify_save_folders_absent(save_folders)
        _reject_symlink_components(path, name="receipt output")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite immutable profile-pair receipt {path}"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def build_profile_pair_receipt(
    *,
    recipe_path: str | Path,
    expected_recipe_sha256: str,
    control_profile_path: str | Path,
    expected_control_profile_sha256: str,
    treatment_profile_path: str | Path,
    expected_treatment_profile_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build and publish one immutable causal profile-pair audit receipt.

    :param recipe_path: Training recipe to load dynamically.
    :param expected_recipe_sha256: Exact raw SHA-256 of the recipe.
    :param control_profile_path: Reviewed frozen-vision control profile.
    :param expected_control_profile_sha256: Exact raw SHA-256 of the control profile.
    :param treatment_profile_path: Reviewed vision-unfrozen treatment profile.
    :param expected_treatment_profile_sha256: Exact raw SHA-256 of the treatment profile.
    :param output_path: New receipt path; an existing path is never replaced.

    :returns: The deterministic receipt that was published.
    """
    expected_recipe_sha256 = _require_sha256(expected_recipe_sha256, name="expected recipe SHA-256")
    expected_control_profile_sha256 = _require_sha256(
        expected_control_profile_sha256, name="expected control profile SHA-256"
    )
    expected_treatment_profile_sha256 = _require_sha256(
        expected_treatment_profile_sha256, name="expected treatment profile SHA-256"
    )
    output = _absolute_lexical_path(output_path)
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite immutable profile-pair receipt {output}")
    _reject_symlink_components(output, name="receipt output")

    recipe_path = _pinned_file(recipe_path, expected_recipe_sha256, name="recipe")
    control_profile_path = _pinned_file(
        control_profile_path,
        expected_control_profile_sha256,
        name="control profile",
    )
    treatment_profile_path = _pinned_file(
        treatment_profile_path,
        expected_treatment_profile_sha256,
        name="treatment profile",
    )
    if control_profile_path == treatment_profile_path:
        raise ProfilePairAuditError("Control and treatment profile paths must be distinct")

    repository_root = _repository_root(recipe_path)
    command_path = _recipe_command_path(recipe_path, repository_root)
    producer_path = Path(__file__).resolve()
    try:
        producer_root = producer_path.parents[3]
        producer_repository_path = producer_path.relative_to(producer_root).as_posix()
    except (IndexError, ValueError) as error:
        raise ProfilePairAuditError(
            "Profile-pair auditor lacks its expected src/scripts/eval repository layout"
        ) from error
    if producer_repository_path != PRODUCER_REPOSITORY_PATH:
        raise ProfilePairAuditError(
            f"Profile-pair auditor must run from {PRODUCER_REPOSITORY_PATH!r}"
        )
    producer_sha256 = _sha256_file(producer_path)
    recipe = _load_pinned_recipe(recipe_path, expected_recipe_sha256)
    try:
        with _recipe_main_module(recipe):
            recipe.prepare_cli_environment()
    except Exception as error:
        raise ProfilePairAuditError(
            f"Pinned recipe CLI environment preparation failed: {error}"
        ) from error

    control_profile, control_overrides = _load_reviewed_profile(recipe, control_profile_path)
    treatment_profile, treatment_overrides = _load_reviewed_profile(recipe, treatment_profile_path)
    profile_audit = _audit_profiles(
        control_profile,
        treatment_profile,
        repository_root=repository_root,
        control_path=control_profile_path,
        control_sha256=expected_control_profile_sha256,
        treatment_path=treatment_profile_path,
        treatment_sha256=expected_treatment_profile_sha256,
    )
    model_variant = profile_audit["model_variant"]
    policy = _model_variant_policy(model_variant)
    if (
        getattr(recipe, policy["workspace_recipe_constant"], None) != policy["workspace"]
        or getattr(recipe, "BEAKER_CLUSTER", None) != CANONICAL_CLUSTER
        or getattr(recipe, "BEAKER_BUDGET", None) != CANONICAL_BUDGET
    ):
        raise ProfilePairAuditError(
            f"Pinned recipe does not declare the canonical {policy['workspace']}/Holmes "
            "launch constants"
        )

    _clear_perception_provenance_cache(recipe)
    _, control_config = _build_profile_config(
        recipe,
        command_path=command_path,
        profile=control_profile,
        overrides=control_overrides,
    )
    _clear_perception_provenance_cache(recipe)
    _, treatment_config = _build_profile_config(
        recipe,
        command_path=command_path,
        profile=treatment_profile,
        overrides=treatment_overrides,
    )
    config_audit = _audit_configs(
        control_config,
        treatment_config,
        control_profile=control_profile,
        treatment_profile=treatment_profile,
        command_path=command_path,
        control_relative_path=profile_audit["control_relative_path"],
        control_sha256=expected_control_profile_sha256,
        treatment_relative_path=profile_audit["treatment_relative_path"],
        treatment_sha256=expected_treatment_profile_sha256,
        model_variant=model_variant,
    )

    allowlist_path = profile_audit["allowlist_path"]
    allowlist_sha256 = profile_audit["allowlist_sha256"]
    identity_config_paths = (
        _SSMAX_IDENTITY_CONFIG_PATHS
        if model_variant in SSMAX_MODEL_VARIANTS
        else _IDENTITY_CONFIG_PATHS
    )
    receipt = {
        "format": FORMAT,
        "version": policy["receipt_version"],
        "status": "passed",
        "recipe_execution_module": "__main__",
        "producer": {
            "path": str(producer_path),
            "repository_path": producer_repository_path,
            "sha256": producer_sha256,
        },
        "recipe": {
            "path": str(recipe_path),
            "command_path": command_path,
            "sha256": expected_recipe_sha256,
        },
        "review_allowlist": {
            "path": str(allowlist_path),
            "repository_path": profile_audit["allowlist_relative_path"],
            "sha256": allowlist_sha256,
        },
        "profiles": {
            CONTROL_ARM: {
                "name": profile_audit["control_name"],
                "path": str(control_profile_path),
                "repository_path": profile_audit["control_relative_path"],
                "sha256": expected_control_profile_sha256,
            },
            TREATMENT_ARM: {
                "name": profile_audit["treatment_name"],
                "path": str(treatment_profile_path),
                "repository_path": profile_audit["treatment_relative_path"],
                "sha256": expected_treatment_profile_sha256,
            },
        },
        "launch_contract": dict(policy["launch"]),
        "comparison": {
            "allowed_identity_config_paths": list(identity_config_paths),
            "allowed_arm_config_paths": list(_ARM_CONFIG_PATHS),
            "arm_config_sha256": config_audit["arm_config_sha256"],
            "shared_config_sha256": config_audit["shared_config_sha256"],
            "trainable_contract_sha256": config_audit["trainable_contract_sha256"],
        },
        "data": config_audit["data"],
        "git": config_audit["git_and_parent"]["git"],
        "initialization": config_audit["git_and_parent"]["initialization"],
        "perception_contract": config_audit["perception_contract"],
        "save_folders": {
            "status": "verified_absent_and_distinct",
            CONTROL_ARM: config_audit["save_folders"][CONTROL_ARM],
            TREATMENT_ARM: config_audit["save_folders"][TREATMENT_ARM],
        },
    }
    if model_variant in SSMAX_MODEL_VARIANTS:
        receipt["model_variant"] = model_variant
    expected_inputs = {
        producer_path: producer_sha256,
        recipe_path: expected_recipe_sha256,
        control_profile_path: expected_control_profile_sha256,
        treatment_profile_path: expected_treatment_profile_sha256,
        allowlist_path: allowlist_sha256,
    }
    save_folders = [
        Path(config_audit["save_folders"][CONTROL_ARM]),
        Path(config_audit["save_folders"][TREATMENT_ARM]),
    ]
    for save_folder in save_folders:
        try:
            output.relative_to(save_folder)
        except ValueError:
            pass
        else:
            raise ProfilePairAuditError("Receipt output may not be inside either run save folder")
    _write_json_once(
        output,
        receipt,
        expected_inputs=expected_inputs,
        save_folders=save_folders,
    )
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--control-profile", type=Path, required=True)
    parser.add_argument("--expected-control-profile-sha256", required=True)
    parser.add_argument("--treatment-profile", type=Path, required=True)
    parser.add_argument("--expected-treatment-profile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the profile-pair audit and print its immutable receipt identity."""
    args = _parse_args(argv)
    receipt = build_profile_pair_receipt(
        recipe_path=args.recipe,
        expected_recipe_sha256=args.expected_recipe_sha256,
        control_profile_path=args.control_profile,
        expected_control_profile_sha256=args.expected_control_profile_sha256,
        treatment_profile_path=args.treatment_profile,
        expected_treatment_profile_sha256=args.expected_treatment_profile_sha256,
        output_path=args.output,
    )
    output = _absolute_lexical_path(args.output)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": _sha256_file(output),
                "status": receipt["status"],
                "shared_config_sha256": receipt["comparison"]["shared_config_sha256"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
