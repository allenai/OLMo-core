"""Synthetic tests for the perception control/treatment profile-pair auditor."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_perception_profile_pair.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vision_alignment_perception_profile_pair_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_perception_data_contract_sha256(recipe) -> str:
    data = recipe.VisionAlignmentDataConfig(
        sequence_length=2560,
        mixture=recipe.VisionAlignmentMixtureConfig(phase="perception"),
    )
    data.pixmo_cap_path = (
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
        "artifacts/pixmo-cap-content-disjoint-v1/dataset"
    )
    data.perception_provenance_path = (
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
        "artifacts/perception-provenance-v2/vision-alignment-perception-provenance.json"
    )
    data.perception_provenance_sha256 = (
        "73cb3920676db5e16d789f7257800dcb44b2553b6463cff81beb740213d921e2"
    )
    data.source_audit_path = (
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
        "artifacts/perception-source-audit-v2.json"
    )
    data.source_audit_fingerprint = (
        "2e9bf63765c674b6a1161cd12313580e29dac028f990bfd9d304a00d844d5d3b"
    )
    data.mixture.mean_loss_weight = {
        "audited_alignment": 18.58728085551411,
        "cosyn_point": 20.31768759561237,
        "ocr_document": 4.165374373900704,
        "pixmo_caption": 29.374187365814578,
        "pixmo_points_basic": 33.16318166248675,
        "pixmo_points_high_frequency": 27.629776440531714,
        "pixmo_transcript": 29.7867612372429,
        "scalar_count": 3.464101552963257,
    }
    evaluation = recipe.VisionAlignmentEvalConfig(
        interval=500,
        examples_per_source=512,
        rank_batch_instances=4,
        seed=6198,
        eval_on_startup=True,
        eval_on_finish=True,
    )
    collator = recipe.MultimodalCollatorConfig(
        pad_token_id=100277,
        label_ignore_index=-100,
        pad_sequence_length=2560,
    )
    return recipe._canonical_sha256(
        {
            "phase": "perception",
            "formatter_version": recipe.FORMATTER_VERSION,
            "data": data.as_config_dict(),
            "evaluation": evaluation.as_config_dict(),
            "collator": collator.as_config_dict(),
            "global_batch_size": 327680,
            "data_seed": 95818,
        }
    )


def _load_real_recipe_with_legacy_generated_module(path: Path):
    raw = path.read_bytes()
    recipe_sha256 = hashlib.sha256(raw).hexdigest()
    module_name = f"_vision_alignment_profile_pair_recipe_{recipe_sha256[:20]}"
    recipe = types.ModuleType(module_name)
    recipe.__file__ = str(path)
    recipe.__loader__ = importlib.machinery.SourceFileLoader(module_name, str(path))
    recipe.__package__ = ""
    recipe.__spec__ = importlib.machinery.ModuleSpec(
        module_name, recipe.__loader__, origin=str(path)
    )
    sys.modules[module_name] = recipe
    try:
        exec(compile(raw, str(path), "exec"), recipe.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    return recipe


_FAKE_RECIPE = r"""
import copy
import hashlib
import json
from pathlib import Path

BEAKER_WORKSPACE = "ai2/molmofication"
SSMAX_BEAKER_WORKSPACE = "ai2/scaling-ladders"
BEAKER_CLUSTER = "ai2/holmes"
BEAKER_BUDGET = "ai2/oe-other"
PERCEPTION_PROFILE_ALLOWLIST = "configs/vision_moe/vision_alignment/perception/approved_profiles.json"
CASE_MODEL_VARIANT = __MODEL_VARIANT__
DRIFT_CONFIG = __DRIFT_CONFIG__
WRONG_WORKSPACE = __WRONG_WORKSPACE__
TYPE_ALIAS_DRIFT = __TYPE_ALIAS_DRIFT__
CONTRACT_TYPE_ALIAS_DRIFT = __CONTRACT_TYPE_ALIAS_DRIFT__
WRONG_FREEZE_LIST = __WRONG_FREEZE_LIST__
WRONG_VISION_LR = __WRONG_VISION_LR__
WRONG_GIT_BRANCH = __WRONG_GIT_BRANCH__
CLI_PREPARED = False
_PERCEPTION_PROVENANCE_RUNTIME_CACHE = {"stale": True}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_cli_environment():
    global CLI_PREPARED
    CLI_PREPARED = True


class FakeConfig:
    def __init__(self, value):
        self.value = value
        self.applied = False
        self.validated = False

    def as_config_dict(self):
        if not self.validated:
            raise RuntimeError("config was not validated after profile application")
        return copy.deepcopy(self.value)


def _module_sensitive_contract_sha256():
    payload = {
        "data": {
            "_CLASS_": f"{FakeConfig.__module__}.{FakeConfig.__name__}",
        }
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_profile(overrides):
    if not CLI_PREPARED:
        raise RuntimeError("CLI environment was not prepared before profile loading")
    assert len(overrides) == 1 and overrides[0].startswith("--profile=")
    path = Path(overrides[0].split("=", 1)[1]).resolve()
    root = Path(__file__).resolve().parents[3]
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    profile = json.loads(raw)
    allowlist_path = root / PERCEPTION_PROFILE_ALLOWLIST
    allowlist = json.loads(allowlist_path.read_text())
    profile_sha256 = hashlib.sha256(raw).hexdigest()
    if allowlist["profiles"].get(relative) != profile_sha256:
        raise ValueError("profile is not reviewed")
    profile["__reviewed_path__"] = relative
    profile["__reviewed_sha256__"] = profile_sha256
    profile["__reviewed_allowlist_path__"] = PERCEPTION_PROFILE_ALLOWLIST
    profile["__reviewed_allowlist_sha256__"] = _sha256(allowlist_path)
    return profile, [f"--phase={profile['phase']}", *profile["overrides"]]


def build_config(
    script,
    run_name,
    overrides,
    *,
    runtime,
    reviewed_profile_path,
    reviewed_profile_sha256,
    reviewed_profile_allowlist_path,
    reviewed_profile_allowlist_sha256,
):
    assert runtime is False
    if not CLI_PREPARED:
        raise RuntimeError("CLI environment was not prepared before config construction")
    if _PERCEPTION_PROVENANCE_RUNTIME_CACHE:
        raise RuntimeError("perception provenance cache was not independently cleared")
    selectors = [
        item.split("=", 1)[1]
        for item in overrides
        if item.startswith("--perception_trainability_arm=")
    ]
    assert len(selectors) == 1
    arm = selectors[0]
    control = arm == "frozen_vision_control"
    variant_selectors = [
        item.split("=", 1)[1]
        for item in overrides
        if item.startswith("--model_variant=")
    ]
    model_variant = variant_selectors[0] if variant_selectors else "s002"
    assert model_variant == CASE_MODEL_VARIANT
    is_ssmax = model_variant != "s002"
    _PERCEPTION_PROVENANCE_RUNTIME_CACHE["built_arm"] = arm
    root = Path(__file__).resolve().parents[3]
    command = [script, "train", run_name, f"--profile={reviewed_profile_path}"]
    data_seed = 999 if DRIFT_CONFIG and control else 95818
    expected_workspace = SSMAX_BEAKER_WORKSPACE if is_ssmax else BEAKER_WORKSPACE
    workspace = "ai2/OLMo-core" if WRONG_WORKSPACE else expected_workspace
    treatment_freeze = ["lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"]
    if WRONG_FREEZE_LIST:
        treatment_freeze.append("lm.extra.*")
    value = {
        "phase": "perception",
        "perception_trainability_arm": arm,
        "required_run_name": run_name,
        "expected_launch_command": command,
        "reviewed_profile_path": reviewed_profile_path,
        "reviewed_profile_sha256": reviewed_profile_sha256,
        "reviewed_profile_allowlist_path": reviewed_profile_allowlist_path,
        "reviewed_profile_allowlist_sha256": reviewed_profile_allowlist_sha256,
        "data_seed": data_seed,
        "launch": {
            "name": f"{run_name}-1234abcd",
            "cmd": command,
            "workspace": workspace,
            "clusters": [BEAKER_CLUSTER],
            "budget": BEAKER_BUDGET,
            "num_nodes": 2,
            "num_gpus": 8,
            "priority": "normal",
            "git": {
                "branch": (
                    "main"
                    if WRONG_GIT_BRANCH
                    else (
                        "rustin/vision-ssmax-molmofication" if is_ssmax else "vision-moe"
                    )
                ),
                "ref": "1" * 40,
            },
        },
        "trainer": {
            "save_folder": str(root / "run-output" / run_name),
            "save_overwrite": 1 if TYPE_ALIAS_DRIFT and not control else True,
            "max_duration": {"unit": "steps", "value": 4000},
            "callbacks": {
                "wandb": {"name": run_name, "project": "synthetic"},
                "checkpointer": {
                    "ephemeral_save_interval": 400,
                    "max_checkpoints": 6,
                    "save_async": False,
                    "pre_train_checkpoint": True,
                    **(
                        {"fixed_steps": [500, 1000, 2000, 3000, 4000]}
                        if is_ssmax
                        else {"save_interval": 1000}
                    ),
                },
                **(
                    {
                        "ssmax_health_ledger": {
                            "model_variant": model_variant,
                            "phase": "perception",
                            "run_name": run_name,
                        }
                    }
                    if is_ssmax
                    else {}
                ),
            },
        },
        "train_module": {
            "freeze_params": (["vision.*"] if control else []) + treatment_freeze,
            "optim": {
                "group_overrides": [
                    {"params": ["*lm.embeddings.weight"], "opts": {"lr": 5e-5}},
                    {"params": ["*connector.*"], "opts": {"lr": 5e-5}},
                    {
                        "params": ["*vision.*"],
                        "opts": {
                            "lr": 0.0 if control else (4e-6 if WRONG_VISION_LR else 3e-6),
                            "weight_decay": 0.0,
                        },
                    },
                ]
            },
            "rank_microbatch_size": 10240,
            "source_loss_mass_targets": {"source_a": 1.0},
            **(
                {
                    "_CLASS_": (
                        "olmo_core.train.train_module.transformer.multimodal_train_module."
                        "MultimodalTransformerTrainModuleConfig"
                    ),
                    "dp_config": {
                        "_CLASS_": (
                            "olmo_core.train.train_module.transformer.config."
                            "TransformerDataParallelConfig"
                        ),
                        "name": "hsdp",
                        "param_dtype": "bfloat16",
                        "reduce_dtype": "float32",
                    },
                }
                if is_ssmax
                else {"ep_config": {"degree": 8}}
            ),
        },
        "vision_alignment": {
            "lineage_id": run_name,
            "data_contract_sha256": _module_sensitive_contract_sha256(),
            "trainable_contract_sha256": ("d" if control else "e") * 64,
            "parent_checkpoint": "/synthetic/bridge/step500",
            "parent_config_sha256": "9" * 64,
            "parent_gate_sha256": "f" * 64,
        },
        "data": {
            "sequence_length": 2560,
            "perception_provenance_path": "/synthetic/perception-provenance.json",
            "perception_provenance_sha256": "a" * 64,
            "source_audit_path": "/synthetic/perception-source-audit.json",
            "source_audit_fingerprint": "b" * 64,
            "mixture": {"phase": "perception", "targets": {"source_a": 1.0}},
        },
        "evaluation": {
            "interval": 500,
            "examples_per_source": 512,
            "rank_batch_instances": 4,
            "seed": 6198,
            "eval_on_startup": True,
            "eval_on_finish": 1 if CONTRACT_TYPE_ALIAS_DRIFT else True,
        },
        "initialization": {
            "mode": "checkpoint",
            "checkpoint": "/synthetic/bridge/step500",
            "expected_parent_phase": "bridge",
            "parent_config_sha256": "9" * 64,
            "parent_gate_path": "/synthetic/bridge/parent-gate-v2.json",
            "parent_gate_sha256": "f" * 64,
        },
        "global_batch_size": 327680,
        "init_seed": 6198,
        "checkpoint_load_threads": 8,
        **({} if is_ssmax else {"router_lb_loss_weight": 0.015}),
    }
    if is_ssmax:
        value["model_variant"] = model_variant
        value["vision_alignment"]["model_variant"] = model_variant
    return FakeConfig(value)


def _apply_profile_launch(config, profile, *, run_name):
    assert profile["name"] == run_name
    config.value["launch"]["description"] = profile.get("description")
    config.value["launch"]["priority"] = profile["launch"]["priority"]
    config.value["launch"]["min_runtime"] = profile["launch"]["min_runtime"]
    config.applied = True
    return config


def _validate_phase_contract(config, run_name, *, runtime):
    assert runtime is False
    if not config.applied:
        raise RuntimeError("profile launch was not applied")
    if config.value["required_run_name"] != run_name:
        raise RuntimeError("run identity mismatch")
    config.validated = True


if __name__ == "__main__":
    print(json.dumps({"data_contract_sha256": _module_sensitive_contract_sha256()}))
"""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _make_case(
    tmp_path: Path,
    *,
    model_variant: str = "s002",
    config_drift: bool = False,
    wrong_workspace: bool = False,
    profile_drift: bool = False,
    type_alias_drift: bool = False,
    profile_type_alias_drift: bool = False,
    contract_type_alias_drift: bool = False,
    wrong_freeze_list: bool = False,
    wrong_vision_lr: bool = False,
    wrong_git_branch: bool = False,
) -> dict[str, Any]:
    root = tmp_path / "repo"
    recipe = root / "src" / "scripts" / "train" / "Vision-Alignment.py"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        _FAKE_RECIPE.replace("__MODEL_VARIANT__", repr(model_variant))
        .replace("__DRIFT_CONFIG__", repr(config_drift))
        .replace("__WRONG_WORKSPACE__", repr(wrong_workspace))
        .replace("__TYPE_ALIAS_DRIFT__", repr(type_alias_drift))
        .replace("__CONTRACT_TYPE_ALIAS_DRIFT__", repr(contract_type_alias_drift))
        .replace("__WRONG_FREEZE_LIST__", repr(wrong_freeze_list))
        .replace("__WRONG_VISION_LR__", repr(wrong_vision_lr))
        .replace("__WRONG_GIT_BRANCH__", repr(wrong_git_branch))
    )
    profile_root = root / "configs" / "vision_moe" / "vision_alignment" / "perception"
    launch = {
        "num_nodes": 2,
        "num_gpus": 8,
        "workspace": (
            "ai2/scaling-ladders" if model_variant.startswith("ssmax_") else "ai2/molmofication"
        ),
        "cluster": "ai2/holmes",
        "budget": "ai2/oe-other",
        "priority": "urgent",
        "min_runtime": "8h",
    }
    common_overrides = [
        *([f"--model_variant={model_variant}"] if model_variant.startswith("ssmax_") else []),
        "--data.perception_provenance_path=/synthetic/perception-provenance.json",
        f"--data.perception_provenance_sha256={'a' * 64}",
        "--data.source_audit_path=/synthetic/perception-source-audit.json",
        f"--data.source_audit_fingerprint={'b' * 64}",
        "--initialization.checkpoint=/synthetic/bridge/step500",
    ]
    control = profile_root / "control.yaml"
    treatment = profile_root / "treatment.yaml"
    if model_variant.startswith("ssmax_"):
        run_prefix = f"vision-{model_variant.replace('_', '-')}-1p4b-cx8-perception"
        control_name = f"{run_prefix}-frozen-vision-control-v1"
        treatment_name = f"{run_prefix}-treatment-v1"
    else:
        control_name = "perception-frozen-control"
        treatment_name = "perception-treatment"
    _write_json(
        control,
        {
            "version": True if profile_type_alias_drift else 1,
            "name": control_name,
            "description": "Frozen vision causal control",
            "phase": "perception",
            "launch": launch,
            "overrides": [
                *common_overrides,
                "--perception_trainability_arm=frozen_vision_control",
            ],
        },
    )
    _write_json(
        treatment,
        {
            "version": 1,
            "name": treatment_name,
            "description": "Vision-unfrozen treatment",
            "phase": "perception",
            "launch": launch,
            "overrides": [
                *common_overrides,
                *(["--data_seed=123"] if profile_drift else []),
                "--perception_trainability_arm=treatment",
            ],
        },
    )
    control_relative = control.relative_to(root).as_posix()
    treatment_relative = treatment.relative_to(root).as_posix()
    allowlist = profile_root / "approved_profiles.json"
    _write_json(
        allowlist,
        {
            "format": "vision_alignment_perception_profile_allowlist",
            "version": 1,
            "profiles": {
                control_relative: _sha256(control),
                treatment_relative: _sha256(treatment),
            },
        },
    )
    return {
        "root": root,
        "recipe": recipe,
        "recipe_sha256": _sha256(recipe),
        "control": control,
        "control_sha256": _sha256(control),
        "treatment": treatment,
        "treatment_sha256": _sha256(treatment),
        "allowlist": allowlist,
        "output": root / "audits" / "profile-pair.json",
    }


def _kwargs(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipe_path": case["recipe"],
        "expected_recipe_sha256": case["recipe_sha256"],
        "control_profile_path": case["control"],
        "expected_control_profile_sha256": case["control_sha256"],
        "treatment_profile_path": case["treatment"],
        "expected_treatment_profile_sha256": case["treatment_sha256"],
        "output_path": case["output"],
    }


def test_builds_deterministic_immutable_pair_receipt(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path)

    receipt = module.build_profile_pair_receipt(**_kwargs(case))
    first_raw = case["output"].read_bytes()
    assert json.loads(first_raw) == receipt
    assert receipt["status"] == "passed"
    assert receipt["version"] == 2
    assert (
        "/trainer/callbacks/ssmax_health_ledger/run_name"
        not in receipt["comparison"]["allowed_identity_config_paths"]
    )
    assert receipt["recipe_execution_module"] == "__main__"
    producer = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_perception_profile_pair.py"
    )
    assert receipt["producer"] == {
        "path": str(producer),
        "repository_path": "src/scripts/eval/vision_alignment_perception_profile_pair.py",
        "sha256": _sha256(producer),
    }
    assert receipt["launch_contract"] == {
        "budget": "ai2/oe-other",
        "cluster": "ai2/holmes",
        "min_runtime": "8h",
        "num_gpus": 8,
        "num_nodes": 2,
        "priority": "urgent",
        "workspace": "ai2/molmofication",
    }
    direct = subprocess.run(
        [sys.executable, str(case["recipe"])],
        check=True,
        capture_output=True,
        text=True,
    )
    direct_contract_sha256 = json.loads(direct.stdout)["data_contract_sha256"]
    assert receipt["data"]["data_contract_sha256"] == direct_contract_sha256
    assert receipt["data"]["perception_provenance_sha256"] == "a" * 64
    assert receipt["save_folders"]["status"] == "verified_absent_and_distinct"
    assert receipt["comparison"]["trainable_contract_sha256"] == {
        "frozen_vision_control": "d" * 64,
        "treatment": "e" * 64,
    }
    assert receipt["git"] == {"branch": "vision-moe", "ref": "1" * 40}
    assert receipt["initialization"] == {
        "checkpoint": "/synthetic/bridge/step500",
        "parent_config_sha256": "9" * 64,
        "parent_gate_path": "/synthetic/bridge/parent-gate-v2.json",
        "parent_gate_sha256": "f" * 64,
    }
    assert receipt["perception_contract"] == module._PERCEPTION_CONSTANTS

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.build_profile_pair_receipt(**_kwargs(case))

    case["output"].unlink()
    second = module.build_profile_pair_receipt(**_kwargs(case))
    assert second == receipt
    assert case["output"].read_bytes() == first_raw


@pytest.mark.parametrize("model_variant", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_builds_strict_ssmax_dense_hsdp_pair_receipt(tmp_path: Path, model_variant: str) -> None:
    module = _load_module()
    case = _make_case(tmp_path, model_variant=model_variant)

    receipt = module.build_profile_pair_receipt(**_kwargs(case))

    assert receipt["version"] == 3
    assert receipt["model_variant"] == model_variant
    assert receipt["launch_contract"]["workspace"] == "ai2/scaling-ladders"
    assert receipt["git"]["branch"] == "rustin/vision-ssmax-molmofication"
    assert receipt["perception_contract"] == module._SSMAX_PERCEPTION_CONSTANTS
    assert receipt["perception_contract"]["checkpointer"]["fixed_steps"] == [
        500,
        1000,
        2000,
        3000,
        4000,
    ]
    assert receipt["perception_contract"]["router_lb_loss_weight"] is None
    assert receipt["perception_contract"]["parallelism"]["data_parallel"] == {
        "class": module._SSMAX_DP_CONFIG_CLASS,
        "name": "hsdp",
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
    }
    assert (
        "/trainer/callbacks/ssmax_health_ledger/run_name"
        in receipt["comparison"]["allowed_identity_config_paths"]
    )


@pytest.mark.parametrize("drift", ["missing", "model_variant", "phase", "run_name"])
def test_rejects_ssmax_health_ledger_identity_drift(
    monkeypatch, tmp_path: Path, drift: str
) -> None:
    module = _load_module()
    case = _make_case(tmp_path, model_variant="ssmax_head_qknorm")
    original = module._build_profile_config

    def drift_treatment(*args, **kwargs):
        config, config_dict = original(*args, **kwargs)
        if config_dict["perception_trainability_arm"] == "treatment":
            callbacks = config_dict["trainer"]["callbacks"]
            if drift == "missing":
                callbacks.pop("ssmax_health_ledger")
            else:
                callbacks["ssmax_health_ledger"][drift] = "wrong"
        return config, config_dict

    monkeypatch.setattr(module, "_build_profile_config", drift_treatment)
    with pytest.raises(module.ProfilePairAuditError, match="SSMax health-ledger"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("fixed_steps", "checkpointer fixed_steps differs"),
        ("router", "router_lb_loss_weight must be omitted"),
        ("expert_parallel", "ep_config must be omitted"),
    ],
)
def test_rejects_ssmax_dense_contract_drift(
    monkeypatch, tmp_path: Path, drift: str, message: str
) -> None:
    module = _load_module()
    case = _make_case(tmp_path, model_variant="ssmax_head_qknorm")
    original = module._build_profile_config

    def drift_both_arms(*args, **kwargs):
        config, config_dict = original(*args, **kwargs)
        if drift == "fixed_steps":
            config_dict["trainer"]["callbacks"]["checkpointer"]["fixed_steps"] = [1000, 2000]
        elif drift == "router":
            config_dict["router_lb_loss_weight"] = 0.015
        else:
            config_dict["train_module"]["ep_config"] = {"degree": 8}
        return config, config_dict

    monkeypatch.setattr(module, "_build_profile_config", drift_both_arms)
    with pytest.raises(module.ProfilePairAuditError, match=message):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_real_recipe_data_contract_matches_direct_main_execution() -> None:
    module = _load_module()
    recipe_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "train" / "Vision-Alignment.py"
    ).resolve()
    recipe_sha256 = _sha256(recipe_path)

    recipe = module._load_pinned_recipe(recipe_path, recipe_sha256)
    assert recipe.__name__ == "__main__"
    assert recipe.__package__ is None
    assert recipe.__spec__ is None
    with module._recipe_main_module(recipe):
        direct_contract_sha256 = _production_perception_data_contract_sha256(recipe)
    assert (
        direct_contract_sha256 == "1116f73987da8c94fb8158a2b4e38629fc18cd3227d2c88d5cadeafeadbfd916"
    )

    legacy_recipe = _load_real_recipe_with_legacy_generated_module(recipe_path)
    legacy_contract_sha256 = _production_perception_data_contract_sha256(legacy_recipe)
    assert legacy_contract_sha256 != direct_contract_sha256


def test_rejects_recipe_that_differs_from_caller_pin(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path)
    kwargs = _kwargs(case)
    kwargs["expected_recipe_sha256"] = "0" * 64

    with pytest.raises(module.ProfilePairAuditError, match="recipe SHA-256 differs"):
        module.build_profile_pair_receipt(**kwargs)
    assert not case["output"].exists()


def test_rejects_reviewed_profile_difference_outside_arm(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, profile_drift=True)

    with pytest.raises(module.ProfilePairAuditError, match="outside their exact arm selector"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_cross_variant_ssmax_profile_pair(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, model_variant="ssmax_head_qknorm")
    treatment = json.loads(case["treatment"].read_text())
    treatment["overrides"] = [
        value.replace("ssmax_head_qknorm", "ssmax_no_qknorm") for value in treatment["overrides"]
    ]
    _write_json(case["treatment"], treatment)
    case["treatment_sha256"] = _sha256(case["treatment"])
    allowlist = json.loads(case["allowlist"].read_text())
    relative = case["treatment"].relative_to(case["root"]).as_posix()
    allowlist["profiles"][relative] = case["treatment_sha256"]
    _write_json(case["allowlist"], allowlist)

    with pytest.raises(module.ProfilePairAuditError, match="different model variants"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_full_config_difference_outside_identity_and_arm(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, config_drift=True)

    with pytest.raises(module.ProfilePairAuditError, match="differ outside"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_bool_int_alias_outside_identity_and_arm(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, type_alias_drift=True)

    with pytest.raises(module.ProfilePairAuditError, match="differ outside"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_bool_int_alias_in_reviewed_public_profile(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, profile_type_alias_drift=True)

    with pytest.raises(module.ProfilePairAuditError, match="profile documents differ"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_bool_int_alias_in_perception_constants(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, contract_type_alias_drift=True)

    with pytest.raises(module.ProfilePairAuditError, match="evaluation eval_on_finish differs"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_noncanonical_exact_freeze_lists(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, wrong_freeze_list=True)

    with pytest.raises(module.ProfilePairAuditError, match="treatment freeze list differs"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_noncanonical_treatment_vision_lr(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, wrong_vision_lr=True)

    with pytest.raises(module.ProfilePairAuditError, match="treatment vision LR differs"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_noncanonical_common_git_branch(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, wrong_git_branch=True)

    with pytest.raises(module.ProfilePairAuditError, match="git branch differs"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_noncanonical_workspace_at_final_config_boundary(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path, wrong_workspace=True)

    with pytest.raises(module.ProfilePairAuditError, match="exact molmofication/Holmes"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_preexisting_save_folder(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path)
    (case["root"] / "run-output" / "perception-frozen-control").mkdir(parents=True)

    with pytest.raises(module.ProfilePairAuditError, match="save folder already exists"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rejects_dangling_save_folder_symlink(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path)
    run_output = case["root"] / "run-output"
    run_output.mkdir()
    dangling = run_output / "perception-frozen-control"
    dangling.symlink_to(run_output / "absent-target", target_is_directory=True)

    with pytest.raises(module.ProfilePairAuditError, match="may not contain symlinks"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert dangling.is_symlink()
    assert not case["output"].exists()


def test_refuses_dangling_output_symlink_without_redirecting(tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path)
    case["output"].parent.mkdir(parents=True)
    redirected = case["output"].with_name("redirected.json")
    case["output"].symlink_to(redirected)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert case["output"].is_symlink()
    assert not redirected.exists()


def test_rehashes_every_input_immediately_before_publish(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    case = _make_case(tmp_path)
    original = module._build_profile_config
    calls = 0

    def mutate_after_build(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            case["control"].write_bytes(case["control"].read_bytes() + b" ")
        return result

    monkeypatch.setattr(module, "_build_profile_config", mutate_after_build)
    with pytest.raises(module.ProfilePairAuditError, match="Input changed during audit"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert not case["output"].exists()


def test_rehashes_pair_auditor_producer_immediately_before_publish(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    case = _make_case(tmp_path)
    producer = Path(module.__file__).resolve()
    original = module._sha256_file
    producer_hash_calls = 0

    def drift_producer_hash(path):
        nonlocal producer_hash_calls
        actual = original(path)
        if path == producer:
            producer_hash_calls += 1
            if producer_hash_calls > 1:
                return "0" * 64
        return actual

    monkeypatch.setattr(module, "_sha256_file", drift_producer_hash)
    with pytest.raises(module.ProfilePairAuditError, match="Input changed during audit"):
        module.build_profile_pair_receipt(**_kwargs(case))
    assert producer_hash_calls == 2
    assert not case["output"].exists()
