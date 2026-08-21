"""Focused tests for the distributed production perception runtime preflight."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_perception_runtime_preflight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vision_alignment_perception_runtime_preflight_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_packets(module, git_ref: str = "a" * 40, model_variant: str = "s002"):
    policy = module._model_variant_policy(model_variant)
    packets = []
    for rank in range(module.WORLD_SIZE):
        replica_rank = rank // module.LOCAL_WORLD_SIZE
        local_rank = rank % module.LOCAL_WORLD_SIZE
        node_id = f"node-{replica_rank}"
        hostname = f"holmes-test-{replica_rank}.example.ai2.in"
        packets.append(
            {
                "rank": rank,
                "env_rank": rank,
                "local_rank": local_rank,
                "world_size": module.WORLD_SIZE,
                "local_world_size": module.LOCAL_WORLD_SIZE,
                "num_nodes": module.NUM_NODES,
                "replica_count": module.NUM_NODES,
                "replica_rank": replica_rank,
                "assigned_gpu_count": module.LOCAL_WORLD_SIZE,
                "cuda_device_count": module.LOCAL_WORLD_SIZE,
                "cuda_device": local_rank,
                "workspace_id": policy["workspace_id"],
                "experiment_id": "experiment-0",
                "workload_id": "experiment-0",
                "task_id": f"task-{replica_rank}",
                "job_id": f"job-{replica_rank}",
                "job_kind": "batch",
                "node_id": node_id,
                "hostname": hostname,
                "leader_node_id": "node-0",
                "leader_hostname": "holmes-test-0.example.ai2.in",
                "git_branch": policy["git_branch"],
                "git_ref": git_ref,
                "checkout_ref": git_ref,
                "tracked_checkout_dirty": False,
                "credential_env_names": [],
            }
        )
    return packets


def test_validate_rank_metadata_accepts_exact_two_by_eight_holmes_topology() -> None:
    module = _load_module()
    summary = module._validate_rank_metadata(_valid_packets(module), expected_git_ref="a" * 40)
    assert summary == {
        "world_size": 16,
        "nodes": 2,
        "gpus_per_node": 8,
        "workspace": "ai2/molmofication",
        "workspace_id": "01KSTRJHG4A32N7GDM82KY8J3E",
        "cluster": "ai2/holmes",
        "experiment_id": "experiment-0",
        "node_hostnames": [
            "holmes-test-0.example.ai2.in",
            "holmes-test-1.example.ai2.in",
        ],
    }


@pytest.mark.parametrize("model_variant", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_validate_rank_metadata_accepts_scaling_ladders_ssmax_topology(
    model_variant: str,
) -> None:
    module = _load_module()
    summary = module._validate_rank_metadata(
        _valid_packets(module, model_variant=model_variant),
        expected_git_ref="a" * 40,
        model_variant=model_variant,
    )
    assert summary["workspace"] == "ai2/scaling-ladders"
    assert summary["workspace_id"] == "01KSTRR20XQE9V505A61SW3EBS"
    assert summary["world_size"] == 16
    assert summary["nodes"] == 2


def test_validate_rank_metadata_propagates_exact_rank_error() -> None:
    module = _load_module()
    packets = _valid_packets(module)
    packets[7] = {"rank": 7, "error": "ValueError: missing BEAKER_NODE_ID"}
    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match=r"rank 7: ValueError: missing BEAKER_NODE_ID",
    ):
        module._validate_rank_metadata(packets, expected_git_ref="a" * 40)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda packet: packet.update(workspace_id="wrong"), "workspace_id differs"),
        (lambda packet: packet.update(assigned_gpu_count=7), "assigned_gpu_count differs"),
        (lambda packet: packet.update(hostname="jupiter-test.example.ai2.in"), "Holmes metadata"),
        (lambda packet: packet.update(checkout_ref="b" * 40), "checkout_ref differs"),
        (lambda packet: packet.update(tracked_checkout_dirty=True), "tracked_checkout_dirty"),
        (
            lambda packet: packet.update(credential_env_names=["AWS_SESSION_TOKEN"]),
            "credential_env_names differs",
        ),
    ],
)
def test_validate_rank_metadata_rejects_authority_or_checkout_drift(mutation, message) -> None:
    module = _load_module()
    packets = _valid_packets(module)
    mutation(packets[9])
    with pytest.raises(module.PerceptionRuntimePreflightError, match=message):
        module._validate_rank_metadata(packets, expected_git_ref="a" * 40)


def test_validate_rank_metadata_requires_two_distinct_eight_rank_nodes() -> None:
    module = _load_module()
    packets = _valid_packets(module)
    packets[8]["node_id"] = "node-0"
    packets[8]["hostname"] = "holmes-test-0.example.ai2.in"
    with pytest.raises(module.PerceptionRuntimePreflightError, match="contains 9 ranks"):
        module._validate_rank_metadata(packets, expected_git_ref="a" * 40)


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--expected-recipe-sha256", "A" * 64, "64 lowercase"),
        ("--expected-profile-sha256", "a" * 63, "64 lowercase"),
        ("--expected-profile-pair-receipt-sha256", "d" * 65, "64 lowercase"),
        ("--expected-git-ref", "a" * 64, "40 lowercase"),
    ],
)
def test_cli_rejects_malformed_identity_pins(flag, value, message, capsys) -> None:
    module = _load_module()
    argv = [
        "--recipe=recipe.py",
        f"--expected-recipe-sha256={'a' * 64}",
        "--profile=profile.yaml",
        f"--expected-profile-sha256={'b' * 64}",
        "--profile-pair-receipt=pair.json",
        f"--expected-profile-pair-receipt-sha256={'d' * 64}",
        f"--expected-git-ref={'c' * 40}",
    ]
    prefix = f"{flag}="
    argv = [f"{flag}={value}" if item.startswith(prefix) else item for item in argv]
    with pytest.raises(SystemExit, match="2"):
        module._parse_args(argv)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("model_variant", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_cli_accepts_explicit_ssmax_model_variant(model_variant: str) -> None:
    module = _load_module()
    args = module._parse_args(
        [
            f"--model-variant={model_variant}",
            "--recipe=recipe.py",
            f"--expected-recipe-sha256={'a' * 64}",
            "--profile=profile.yaml",
            f"--expected-profile-sha256={'b' * 64}",
            "--profile-pair-receipt=pair.json",
            f"--expected-profile-pair-receipt-sha256={'d' * 64}",
            f"--expected-git-ref={'c' * 40}",
        ]
    )
    assert args.model_variant == model_variant
    assert args.protocol_version == "v1"


@pytest.mark.parametrize("model_variant", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_cli_accepts_explicit_ssmax_v2_protocol(model_variant: str) -> None:
    module = _load_module()
    args = module._parse_args(
        [
            f"--model-variant={model_variant}",
            "--protocol-version=v2",
            "--recipe=recipe.py",
            f"--expected-recipe-sha256={'a' * 64}",
            "--profile=profile.yaml",
            f"--expected-profile-sha256={'b' * 64}",
            "--profile-pair-receipt=pair.json",
            f"--expected-profile-pair-receipt-sha256={'d' * 64}",
            f"--expected-git-ref={'c' * 40}",
        ]
    )
    assert args.model_variant == model_variant
    assert args.protocol_version == "v2"


def test_v2_protocol_rejects_legacy_s002_lineage() -> None:
    module = _load_module()
    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match="supported only for SSMax",
    ):
        module._model_variant_policy("s002", "v2")


def test_pinned_file_rejects_changed_bytes(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "profile.yaml"
    path.write_bytes(b"reviewed bytes\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert module._pinned_file(path, expected, name="profile") == path.resolve()
    path.write_bytes(b"changed bytes\n")
    with pytest.raises(module.PerceptionRuntimePreflightError, match="SHA-256 differs"):
        module._pinned_file(path, expected, name="profile")


def test_runtime_source_identity_requires_exact_gantry_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    repository_root = tmp_path / "gantry-runtime"
    source_root = repository_root / "src"
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
    monkeypatch.setattr(module.olmo_core, "__file__", str(expected["olmo_core"]))
    monkeypatch.setattr(
        module.vision_alignment_perception_provenance,
        "__file__",
        str(expected["perception_provenance"]),
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{source_root}{os.pathsep}{repository_root}",
    )

    assert module._runtime_source_identity(repository_root) == {
        name: path.relative_to(repository_root).as_posix() for name, path in expected.items()
    }


@pytest.mark.parametrize(
    "pythonpath",
    [
        "{source}",
        "{root}{sep}{source}",
        "{source}{sep}{root}{sep}/unreviewed",
    ],
)
def test_runtime_source_identity_rejects_pythonpath_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pythonpath: str,
) -> None:
    module = _load_module()
    repository_root = tmp_path / "gantry-runtime"
    source_root = repository_root / "src"
    monkeypatch.setenv(
        "PYTHONPATH",
        pythonpath.format(source=source_root, root=repository_root, sep=os.pathsep),
    )

    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match="exact Gantry source/check-out identity",
    ):
        module._runtime_source_identity(repository_root)


def test_pinned_recipe_uses_direct_main_identity_without_running_cli(tmp_path: Path) -> None:
    module = _load_module()
    recipe = tmp_path / "recipe.py"
    marker = tmp_path / "cli-ran"
    recipe.write_text(
        f"""import hashlib

class ModuleSensitiveConfig:
    pass

_PERCEPTION_PROVENANCE_RUNTIME_CACHE = {{}}

def contract_sha256():
    value = f"{{ModuleSensitiveConfig.__module__}}.{{ModuleSensitiveConfig.__name__}}"
    return hashlib.sha256(value.encode()).hexdigest()

def _load_profile(*args, **kwargs): pass
def build_config(*args, **kwargs): pass
def _apply_profile_launch(*args, **kwargs): pass
def _validate_phase_contract(*args, **kwargs): pass
def _perception_provenance(*args, **kwargs): pass

if __name__ == "__main__":
    print(contract_sha256())
    open({str(marker)!r}, "w").close()
"""
    )
    expected_sha256 = hashlib.sha256(recipe.read_bytes()).hexdigest()
    direct = subprocess.run(
        [sys.executable, str(recipe)], check=True, capture_output=True, text=True
    ).stdout.strip()
    marker.unlink()

    loaded = module._load_pinned_recipe(recipe, expected_sha256)

    assert loaded.__name__ == "__main__"
    assert loaded.__package__ is None
    assert loaded.__spec__ is None
    assert loaded.ModuleSensitiveConfig.__module__ == "__main__"
    assert loaded.contract_sha256() == direct
    assert not marker.exists()


def _profile_pair_case(
    tmp_path: Path,
    module,
    model_variant: str = "s002",
    protocol_version: str = "v1",
):
    policy = module._model_variant_policy(model_variant, protocol_version)
    recipe = tmp_path / module.RECIPE_REPOSITORY_PATH
    recipe.parent.mkdir(parents=True)
    recipe.write_text("# pinned recipe\n")
    profile = (
        tmp_path
        / "configs"
        / "vision_moe"
        / "vision_alignment"
        / "perception"
        / f"treatment_{protocol_version}.yaml"
    )
    profile.parent.mkdir(parents=True)
    profile.write_text(f"name: {policy['profile_names']['treatment']}\n")
    control = profile.with_name(f"frozen_vision_control_{protocol_version}.yaml")
    control.write_text(f"name: {policy['profile_names']['frozen_vision_control']}\n")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sha_fields = {
        "config_sha256": "1" * 64,
        "data_contract_sha256": "b" * 64,
        "evaluation_config_sha256": "2" * 64,
        "perception_provenance_sha256": "c" * 64,
        "source_audit_fingerprint": "3" * 64,
    }
    receipt = {
        "format": module.PROFILE_PAIR_FORMAT,
        "version": policy["profile_pair_version"],
        "status": "passed",
        "recipe_execution_module": "__main__",
        "producer": {
            "path": str(tmp_path / "producer.py"),
            "repository_path": "src/scripts/eval/producer.py",
            "sha256": "4" * 64,
        },
        "recipe": {
            "path": str(recipe),
            "command_path": module.RECIPE_REPOSITORY_PATH,
            "sha256": sha256(recipe),
        },
        "review_allowlist": {
            "path": str(tmp_path / "allowlist.json"),
            "repository_path": "configs/allowlist.json",
            "sha256": "5" * 64,
        },
        "git": {"branch": policy["git_branch"], "ref": "a" * 40},
        "launch_contract": {
            "workspace": policy["workspace"],
            "cluster": module.CANONICAL_CLUSTER,
            "budget": module.CANONICAL_BUDGET,
            "num_nodes": module.NUM_NODES,
            "num_gpus": module.LOCAL_WORLD_SIZE,
            "priority": "urgent",
            "min_runtime": "8h",
        },
        "profiles": {
            "frozen_vision_control": {
                "name": policy["profile_names"]["frozen_vision_control"],
                "path": str(control),
                "repository_path": control.relative_to(tmp_path).as_posix(),
                "sha256": sha256(control),
            },
            "treatment": {
                "name": policy["profile_names"]["treatment"],
                "path": str(profile),
                "repository_path": profile.relative_to(tmp_path).as_posix(),
                "sha256": sha256(profile),
            },
        },
        "comparison": {
            "allowed_identity_config_paths": list(
                module._SSMAX_ALLOWED_IDENTITY_CONFIG_PATHS
                if model_variant in module.SSMAX_MODEL_VARIANTS
                else module._ALLOWED_IDENTITY_CONFIG_PATHS
            ),
            "allowed_arm_config_paths": list(module._ALLOWED_ARM_CONFIG_PATHS),
            "arm_config_sha256": {
                "frozen_vision_control": "6" * 64,
                "treatment": "7" * 64,
            },
            "shared_config_sha256": "8" * 64,
            "trainable_contract_sha256": {
                "frozen_vision_control": "9" * 64,
                "treatment": "a" * 64,
            },
        },
        "data": sha_fields,
        "initialization": {
            "checkpoint": str(tmp_path / "parent"),
            "parent_config_sha256": "d" * 64,
            "parent_gate_path": str(tmp_path / "gate.json"),
            "parent_gate_sha256": "e" * 64,
        },
        "perception_contract": copy.deepcopy(policy["perception_contract"]),
        "save_folders": {
            "status": "verified_absent_and_distinct",
            "frozen_vision_control": str(tmp_path / "control-output"),
            "treatment": str(tmp_path / "treatment-output"),
        },
    }
    if model_variant in module.SSMAX_MODEL_VARIANTS:
        receipt["model_variant"] = model_variant
    if policy["optimizer_guard_contract"] is not None:
        receipt["optimizer_guard_contract"] = copy.deepcopy(policy["optimizer_guard_contract"])
    return {
        "root": tmp_path,
        "recipe_sha256": sha256(recipe),
        "control": control,
        "profile": profile,
        "profile_sha256": sha256(profile),
        "receipt": receipt,
        "model_variant": model_variant,
        "protocol_version": protocol_version,
        "receipt_path": tmp_path / "artifacts" / policy["profile_pair_name"],
    }


def _write_receipt(case) -> str:
    case["receipt_path"].parent.mkdir(exist_ok=True)
    case["receipt_path"].write_text(
        json.dumps(case["receipt"], sort_keys=True, separators=(",", ":")) + "\n"
    )
    return hashlib.sha256(case["receipt_path"].read_bytes()).hexdigest()


def _load_receipt_case(module, case, receipt_sha256):
    return module._load_profile_pair_receipt(
        case["receipt_path"],
        receipt_sha256,
        repository_root=case["root"],
        recipe_sha256=case["recipe_sha256"],
        profile_path=case["profile"],
        profile_sha256=case["profile_sha256"],
        git_ref="a" * 40,
        model_variant=case["model_variant"],
        protocol_version=case["protocol_version"],
    )


def test_profile_pair_receipt_binds_runtime_recipe_profile_git_and_data(tmp_path: Path) -> None:
    module = _load_module()
    case = _profile_pair_case(tmp_path, module)
    receipt_sha256 = _write_receipt(case)

    assert _load_receipt_case(module, case, receipt_sha256) == {
        "path": str(case["receipt_path"].resolve()),
        "sha256": receipt_sha256,
        "version": 2,
        "model_variant": "s002",
        "arm": "treatment",
        "profile_name": "vision-alignment-perception-treatment-v1",
        "data_contract_sha256": "b" * 64,
        "perception_provenance_sha256": "c" * 64,
        "control_save_folder": str(tmp_path / "control-output"),
        "treatment_save_folder": str(tmp_path / "treatment-output"),
    }


def test_profile_pair_receipt_reports_the_selected_control_arm(tmp_path: Path) -> None:
    module = _load_module()
    case = _profile_pair_case(tmp_path, module, model_variant="ssmax_head_qknorm")
    case["profile"] = case["control"]
    case["profile_sha256"] = hashlib.sha256(case["control"].read_bytes()).hexdigest()
    receipt_sha256 = _write_receipt(case)

    summary = _load_receipt_case(module, case, receipt_sha256)

    assert summary["arm"] == "frozen_vision_control"
    assert (
        summary["profile_name"]
        == module.SSMAX_PROFILE_NAMES["ssmax_head_qknorm"]["frozen_vision_control"]
    )


@pytest.mark.parametrize("model_variant", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_ssmax_v3_profile_pair_receipt_binds_exact_lineage(
    tmp_path: Path, model_variant: str
) -> None:
    module = _load_module()
    case = _profile_pair_case(tmp_path, module, model_variant=model_variant)
    receipt_sha256 = _write_receipt(case)

    summary = _load_receipt_case(module, case, receipt_sha256)

    assert summary["version"] == 3
    assert summary["model_variant"] == model_variant
    assert summary["profile_name"] == module.SSMAX_PROFILE_NAMES[model_variant]["treatment"]
    assert summary["arm"] == "treatment"
    assert (
        "/trainer/callbacks/ssmax_health_ledger/run_name"
        in case["receipt"]["comparison"]["allowed_identity_config_paths"]
    )


def test_ssmax_v3_profile_pair_receipt_rejects_legacy_identity_paths(tmp_path: Path) -> None:
    module = _load_module()
    case = _profile_pair_case(tmp_path, module, model_variant="ssmax_head_qknorm")
    case["receipt"]["comparison"]["allowed_identity_config_paths"] = list(
        module._ALLOWED_IDENTITY_CONFIG_PATHS
    )
    receipt_sha256 = _write_receipt(case)

    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match="exact identity and causal-arm difference",
    ):
        _load_receipt_case(module, case, receipt_sha256)


@pytest.mark.parametrize("model_variant", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_ssmax_v4_profile_pair_receipt_binds_v2_guard_policy(
    tmp_path: Path, model_variant: str
) -> None:
    module = _load_module()
    case = _profile_pair_case(
        tmp_path,
        module,
        model_variant=model_variant,
        protocol_version="v2",
    )
    receipt_sha256 = _write_receipt(case)

    summary = _load_receipt_case(module, case, receipt_sha256)

    assert summary["version"] == 4
    assert summary["protocol_version"] == "v2"
    assert summary["model_variant"] == model_variant
    assert summary["profile_name"] == module.SSMAX_V2_PROFILE_NAMES[model_variant]["treatment"]
    assert summary["optimizer_guard_contract"] == module._SSMAX_V2_OPTIMIZER_GUARD_CONTRACT


def test_ssmax_v4_profile_pair_receipt_rejects_guard_policy_drift(tmp_path: Path) -> None:
    module = _load_module()
    case = _profile_pair_case(
        tmp_path,
        module,
        model_variant="ssmax_head_qknorm",
        protocol_version="v2",
    )
    case["receipt"]["optimizer_guard_contract"]["eligibility"]["maximum_optimizer_guard_skips"] = 9
    receipt_sha256 = _write_receipt(case)

    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match="optimizer-guard contract differs",
    ):
        _load_receipt_case(module, case, receipt_sha256)


def test_ssmax_v2_protocol_rejects_v1_receipt_name(tmp_path: Path) -> None:
    module = _load_module()
    case = _profile_pair_case(
        tmp_path,
        module,
        model_variant="ssmax_head_qknorm",
        protocol_version="v2",
    )
    case["receipt_path"] = case["receipt_path"].with_name(
        module.SSMAX_PROFILE_PAIR_NAMES["ssmax_head_qknorm"]
    )
    receipt_sha256 = _write_receipt(case)

    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match="must be artifacts/ssmax-head-qknorm-perception-profile-pair-v4.json",
    ):
        _load_receipt_case(module, case, receipt_sha256)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda receipt: receipt.update(version=1), "identity, version, or passed"),
        (lambda receipt: receipt.update(status="failed"), "identity, version, or passed"),
        (lambda receipt: receipt["git"].update(ref="d" * 40), "git identity"),
        (lambda receipt: receipt["recipe"].update(sha256="e" * 64), "recipe differs"),
        (
            lambda receipt: receipt["profiles"]["treatment"].update(sha256="f" * 64),
            "not exactly one arm",
        ),
        (
            lambda receipt: receipt["data"].update(data_contract_sha256="invalid"),
            "data_contract_sha256",
        ),
        (
            lambda receipt: receipt["comparison"].update(allowed_arm_config_paths=[]),
            "exact identity and causal-arm difference",
        ),
        (
            lambda receipt: receipt["perception_contract"]["checkpointer"].update(
                fixed_steps=[500]
            ),
            "perception contract differs",
        ),
    ],
)
def test_profile_pair_receipt_rejects_contract_drift(tmp_path: Path, mutate, message) -> None:
    module = _load_module()
    case = _profile_pair_case(tmp_path, module)
    mutate(case["receipt"])
    receipt_sha256 = _write_receipt(case)

    with pytest.raises(module.PerceptionRuntimePreflightError, match=message):
        _load_receipt_case(module, case, receipt_sha256)


def _ssmax_runtime_config(module, model_variant: str):
    raw = {
        "model_variant": model_variant,
        "vision_alignment": {"model_variant": model_variant},
        "trainer": {
            "max_duration": {"unit": "steps", "value": 4000},
            "callbacks": {
                "checkpointer": {
                    "ephemeral_save_interval": 400,
                    "fixed_steps": [500, 1000, 2000, 3000, 4000],
                    "max_checkpoints": 6,
                    "save_async": False,
                    "pre_train_checkpoint": True,
                }
            },
        },
        "evaluation": {
            "interval": 500,
            "examples_per_source": 512,
            "rank_batch_instances": 4,
            "seed": 6198,
            "eval_on_startup": True,
            "eval_on_finish": True,
        },
        "data": {"sequence_length": 2560},
        "global_batch_size": 327680,
        "data_seed": 95818,
        "init_seed": 6198,
        "checkpoint_load_threads": 8,
        "train_module": {
            "_CLASS_": module._SSMAX_TRAIN_MODULE_CLASS,
            "rank_microbatch_size": 10240,
            "max_grad_norm": 1.0,
            "optim": {
                "type": "skip_step_adamw",
                "rolling_interval_length": 128,
                "sigma_factor": 12,
            },
            "dp_config": {
                "_CLASS_": module._SSMAX_DP_CONFIG_CLASS,
                "name": "hsdp",
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
            },
        },
    }
    return raw, SimpleNamespace(as_config_dict=lambda: copy.deepcopy(raw))


@pytest.mark.parametrize("model_variant", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_runtime_perception_contract_accepts_exact_ssmax_dense_hsdp(
    model_variant: str,
) -> None:
    module = _load_module()
    _, config = _ssmax_runtime_config(module, model_variant)
    module._validate_runtime_perception_contract(config, model_variant=model_variant)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["trainer"]["callbacks"]["checkpointer"].update(
                fixed_steps=[1000, 2000, 3000, 4000]
            ),
            "contract differs",
        ),
        (lambda raw: raw.update(router_lb_loss_weight=0.015), "must be omitted"),
        (lambda raw: raw["train_module"].update(ep_config={"degree": 8}), "must be omitted"),
        (
            lambda raw: raw["train_module"]["dp_config"].update(name="fsdp"),
            "contract differs",
        ),
    ],
)
def test_runtime_perception_contract_rejects_ssmax_drift(mutate, message) -> None:
    module = _load_module()
    raw, config = _ssmax_runtime_config(module, "ssmax_head_qknorm")
    mutate(raw)
    with pytest.raises(module.PerceptionRuntimePreflightError, match=message):
        module._validate_runtime_perception_contract(config, model_variant="ssmax_head_qknorm")


def test_runtime_optimizer_guard_accepts_exact_v2_training_config() -> None:
    module = _load_module()
    _, config = _ssmax_runtime_config(module, "ssmax_head_qknorm")

    module._validate_runtime_optimizer_guard_contract(
        config,
        expected_contract=module._SSMAX_V2_OPTIMIZER_GUARD_CONTRACT,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["train_module"]["optim"].update(type="adamw"),
        lambda raw: raw["train_module"]["optim"].update(rolling_interval_length=64),
        lambda raw: raw["train_module"]["optim"].update(sigma_factor=18),
        lambda raw: raw["train_module"].update(max_grad_norm=2.0),
    ],
)
def test_runtime_optimizer_guard_rejects_v2_training_drift(mutate) -> None:
    module = _load_module()
    raw, config = _ssmax_runtime_config(module, "ssmax_head_qknorm")
    mutate(raw)

    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match="Runtime optimizer guard differs",
    ):
        module._validate_runtime_optimizer_guard_contract(
            config,
            expected_contract=module._SSMAX_V2_OPTIMIZER_GUARD_CONTRACT,
        )


def test_credential_env_names_reports_names_only() -> None:
    module = _load_module()
    environment = {
        "AWS_ACCESS_KEY_ID": "secret-value-must-not-be-returned",
        "RUSTINS_AWS_SESSION_TOKEN": "another-secret-value",
        "S3_PROFILE": "S3",
        "WEKA_PROFILE": "WEKA",
    }
    assert module._credential_env_names(environment) == [
        "AWS_ACCESS_KEY_ID",
        "RUSTINS_AWS_SESSION_TOKEN",
    ]
