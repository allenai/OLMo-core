from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader

from olmo_core.eval import vision_alignment_ssmax_bridge as bridge
from olmo_core.train.callbacks import SSMaxHealthLedgerCallback

_TIMESTAMP = "2026-08-20T00:00:00+00:00"
_HASH = "a" * 64
_PAIRING_HASHES = {
    "pixmo_caption": "b" * 64,
    "pixmo_transcript": "c" * 64,
}
_LOADED_MODEL_KEYS = ["lm.a", "lm.b"]
_TRAINER_STATE_SHA = "9" * 64
_TRAINER_STATE_SIZE = 123
_TEST_EVALUATION_CONTRACT = {
    "sources": list(bridge.SOURCES),
    "steps": list(bridge.REQUIRED_STEPS),
    "examples_per_source": 2,
    "pairing_seed": 6198,
    "bootstrap_seed": 17,
    "bootstrap_samples": 128,
    "rank_batch_instances": 1,
    "windows": list(bridge.WINDOWS),
}
_TEST_TOPOLOGY_CONTRACT = {
    "world_size": 1,
    "num_nodes": 1,
    "gpus_per_node": 1,
    "data_parallel": "hsdp",
}


@pytest.fixture(autouse=True)
def _use_small_synthetic_protocol_contracts(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep synthetic bootstrap tests small while separately testing production constants."""

    if request.node.name == "test_checked_in_specs_form_one_controlled_pair_and_reject_arm_drift":
        return
    monkeypatch.setattr(bridge, "BRIDGE_EVALUATION_CONTRACT", _TEST_EVALUATION_CONTRACT)
    monkeypatch.setattr(bridge, "BRIDGE_TOPOLOGY_CONTRACT", _TEST_TOPOLOGY_CONTRACT)


def _trainer_state_inventory() -> list[dict[str, Any]]:
    return [
        {
            "rank": 0,
            "path": "train/rank0.pt",
            "size": _TRAINER_STATE_SIZE,
            "sha256": _TRAINER_STATE_SHA,
        }
    ]


def _checkpoint_reference(tmp_path: Path, arm: str, step: int) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "path": str(tmp_path / arm / f"step{step}"),
        "global_step": step,
        "config_sha256": "d" * 64,
        "marker_sha256": "e" * 64,
        "dcp_metadata_sha256": "f" * 64,
        "state_file_count": 2,
        "state_file_inventory_sha256": "1" * 64,
        "trainer_state_count": 1,
        "trainer_state_inventory_sha256": bridge.canonical_sha256(_trainer_state_inventory()),
    }
    reference["identity_sha256"] = bridge.canonical_sha256(reference)
    return reference


def _manifest(tmp_path: Path, arm: str = "ssmax_head_qknorm") -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": bridge.MANIFEST_FORMAT,
        "version": bridge.SCHEMA_VERSION,
        "created_at": _TIMESTAMP,
        "pair_id": "controlled-ssmax-pair-v1",
        "arm": arm,
        "model_variant": arm,
        "run_name": f"{arm}-bridge-v1",
        "parent": {
            "checkpoint": str(tmp_path / f"{arm}-parent"),
            "config_sha256": _HASH,
            "data_paths_sha256": _HASH,
            "marker_sha256": _HASH,
            "dcp_metadata_sha256": _HASH,
            "trainer_state_sha256": _HASH,
            "model_keyset_sha256": bridge.canonical_sha256(["a", "b"]),
            "model_inventory_sha256": _HASH,
            "checkpoint_identity_sha256": _HASH,
            "state_file_count": 3,
            "state_file_inventory_sha256": _HASH,
            "trainer_state_count": 1,
            "trainer_state_inventory_sha256": _HASH,
            "source_commit": "1" * 40,
            "olmo_core_commit": "2" * 40,
            "parameter_count": 6,
            "tensor_count": 2,
        },
        "parent_load_receipt": {"path": "/evidence/parent-load.json", "sha256": _HASH},
        "git": {
            "repo": "allenai/OLMo-core",
            "repo_url": "https://github.com/allenai/OLMo-core",
            "ref": "3" * 40,
        },
        "manifest_spec": {
            "repo_relative_path": bridge.MANIFEST_SPEC_RELATIVE_PATHS[arm],
            "sha256": "6" * 64,
            "git_ref": "3" * 40,
        },
        "producers": {
            producer: {
                "repo_relative_path": relative,
                "sha256": str(index + 4) * 64,
                "git_ref": "3" * 40,
            }
            for index, (producer, relative) in enumerate(bridge.PRODUCER_RELATIVE_PATHS.items())
        },
        "training_profile": {"path": "/profiles/bridge.yaml", "sha256": _HASH},
        "recipe": {"path": "/recipes/Vision-Alignment.py", "sha256": _HASH},
        "validation": {"path": "/data/validation.json", "sha256": _HASH},
        "attention_probe": {"path": "/data/attention.json", "sha256": _HASH},
        "pairings": {
            source: {"path": f"/data/{source}.json", "sha256": _PAIRING_HASHES[source]}
            for source in bridge.SOURCES
        },
        "evaluation": copy.deepcopy(_TEST_EVALUATION_CONTRACT),
        "topology": copy.deepcopy(_TEST_TOPOLOGY_CONTRACT),
        "policy": {
            "positive_gap_ci_steps": [250, 300, 400, 500],
            "step0_gap_role": "descriptive_baseline_only",
            "retention_reference_step": 250,
            "retention_candidate_step": 500,
            "retention_windows": ["first_8", "first_32"],
            "minimum_gap_retention": 0.8,
            "correct_ce_reference_step": 250,
            "correct_ce_candidate_step": 500,
            "correct_ce_max_relative_increase": 0.02,
            "require_step0_to_final_correct_ce_improvement": True,
            "loss_mass_share_tolerance": 0.02,
            "maximum_data_errors": 0,
        },
        "checkpoints": {
            str(step): _checkpoint_reference(tmp_path, arm, step) for step in bridge.REQUIRED_STEPS
        },
    }
    manifest["content_sha256"] = bridge.canonical_sha256(manifest)
    return manifest


def _saved_config(
    spec: Mapping[str, Any],
    *,
    checkpoint_root: Path,
    profile_path: str,
    profile_sha256: str,
    validation_path: Path,
    validation_sha256: str,
    git_ref: str = "3" * 40,
) -> dict[str, Any]:
    parent = spec["parent"]
    sequence_length = 2560
    return {
        "model_variant": spec["model_variant"],
        "phase": "bridge",
        "required_run_name": spec["run_name"],
        "expected_launch_command": [
            "src/scripts/train/Vision-Alignment.py",
            "train",
            spec["run_name"],
            f"--profile={profile_path}",
        ],
        "reviewed_profile_path": profile_path,
        "reviewed_profile_sha256": profile_sha256,
        "artifacts": {
            "base_checkpoint": parent["checkpoint"],
            "base_config_sha256": parent["config_sha256"],
            "base_data_paths_sha256": parent["data_paths_sha256"],
            "base_checkpoint_marker_sha256": parent["marker_sha256"],
            "base_checkpoint_metadata_sha256": parent["dcp_metadata_sha256"],
            "base_trainer_state_sha256": parent["trainer_state_sha256"],
            "base_model_keyset_sha256": parent["model_keyset_sha256"],
            "base_model_inventory_sha256": parent["model_inventory_sha256"],
            "base_checkpoint_identity_sha256": parent["checkpoint_identity_sha256"],
            "base_checkpoint_state_file_count": parent["state_file_count"],
            "base_checkpoint_state_file_inventory_sha256": parent["state_file_inventory_sha256"],
            "base_checkpoint_trainer_state_count": parent["trainer_state_count"],
            "base_checkpoint_trainer_state_inventory_sha256": parent[
                "trainer_state_inventory_sha256"
            ],
            "source_commit": parent["source_commit"],
            "source_olmo_core_commit": parent["olmo_core_commit"],
            "expected_lm_parameter_count": parent["parameter_count"],
            "expected_lm_tensor_count": parent["tensor_count"],
        },
        "vision_alignment": {
            "model_variant": spec["model_variant"],
            "phase": "bridge",
            "lineage_id": spec["run_name"],
        },
        "data": {
            "sequence_length": sequence_length,
            "pack_sequences": False,
            "allow_unpinned_synthetic_smoke": False,
            "mixture": {"phase": "bridge"},
        },
        "evaluation": {
            "interval": 100,
            "examples_per_source": spec["evaluation"]["examples_per_source"],
            "rank_batch_instances": spec["evaluation"]["rank_batch_instances"],
            "seed": spec["evaluation"]["pairing_seed"],
            "eval_on_startup": True,
            "eval_on_finish": True,
            "validation_manifest_path": str(validation_path),
            "validation_manifest_sha256": validation_sha256,
        },
        "launch": {
            "cmd": [
                "src/scripts/train/Vision-Alignment.py",
                "train",
                spec["run_name"],
                f"--profile={profile_path}",
            ],
            "num_nodes": spec["topology"]["num_nodes"],
            "num_gpus": spec["topology"]["gpus_per_node"],
            "workspace": "ai2/scaling-ladders",
            "budget": "ai2/oe-other",
            "clusters": ["ai2/holmes"],
            "priority": "urgent",
            "min_runtime": "8h",
            "shared_filesystem": True,
            "git": {
                "repo": "allenai/OLMo-core",
                "repo_url": "https://github.com/allenai/OLMo-core",
                "ref": git_ref,
                "branch": "rustin/vision-ssmax-molmofication",
            },
        },
        "train_module": {
            "dp_config": {
                "name": "hsdp",
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
                "wrapping_strategy": "blocks",
                "reduce_grads_in_fp32": True,
                "accumulate_grads_in_fp32": True,
            },
            "rank_microbatch_size": spec["evaluation"]["rank_batch_instances"] * sequence_length,
            "new_component_init_seed": spec["evaluation"]["pairing_seed"],
            "source_loss_mass_targets": {"pixmo_caption": 0.7, "pixmo_transcript": 0.3},
        },
        "global_batch_size": bridge.BRIDGE_GLOBAL_BATCH_INSTANCES * sequence_length,
        "trainer": {
            "save_folder": str(checkpoint_root),
            "max_duration": {"value": 500, "unit": "steps"},
            "no_checkpoints": False,
            "callbacks": {
                "checkpointer": {
                    "ephemeral_save_interval": 50,
                    "pre_train_checkpoint": True,
                    "fixed_steps": list(bridge.REQUIRED_STEPS[1:]),
                    "max_checkpoints": len(bridge.REQUIRED_STEPS),
                    "enabled": True,
                },
                "ssmax_health_ledger": {
                    "model_variant": spec["model_variant"],
                    "phase": "bridge",
                    "run_name": spec["run_name"],
                    "enabled": True,
                },
            },
        },
    }


def _parent_load_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent = manifest["parent"]
    payload: dict[str, Any] = {
        "format": "vision_alignment_ssmax_parent_load_receipt",
        "version": 1,
        "model_variant": manifest["model_variant"],
        "parent_checkpoint": parent["checkpoint"],
        "parent_config_sha256": parent["config_sha256"],
        "parent_data_paths_sha256": parent["data_paths_sha256"],
        "parent_checkpoint_marker_sha256": parent["marker_sha256"],
        "parent_dcp_metadata_sha256": parent["dcp_metadata_sha256"],
        "parent_trainer_state_sha256": parent["trainer_state_sha256"],
        "parent_source_commit": parent["source_commit"],
        "parent_olmo_core_commit": parent["olmo_core_commit"],
        "parent_model_keyset_sha256": parent["model_keyset_sha256"],
        "parent_model_inventory_sha256": parent["model_inventory_sha256"],
        "parent_checkpoint_identity_sha256": parent["checkpoint_identity_sha256"],
        "parent_state_file_count": parent["state_file_count"],
        "parent_state_file_inventory_sha256": parent["state_file_inventory_sha256"],
        "parent_trainer_state_count": parent["trainer_state_count"],
        "parent_trainer_state_inventory_sha256": parent["trainer_state_inventory_sha256"],
        "checkpoint_dir": parent["checkpoint"],
        "loaded_parameter_numel": parent["parameter_count"],
        "loaded_model_tensor_count": parent["tensor_count"],
        "loaded_parameter_count": parent["tensor_count"],
        "loaded_tensor_dtype_counts": {"torch.float32": parent["tensor_count"]},
        "loaded_tensor_layout_counts": {"torch.strided": parent["tensor_count"]},
        "loaded_model_keys": _LOADED_MODEL_KEYS,
        "loaded_parameter_keys": _LOADED_MODEL_KEYS,
        "missing_initialized_tensor_count": 3,
        "missing_initialized_parameter_count": 3,
        "missing_initialized_model_keys": ["connector.weight", "vision.weight", "lm.image_rows"],
    }
    payload["fingerprint"] = bridge.canonical_sha256(payload)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return bridge.artifact_reference(path)


def _receipt_manifest_reference(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": "/evidence/bridge-manifest.json",
        "sha256": _HASH,
        "content_sha256": str(manifest["content_sha256"]),
    }


def _component_state(*, connector_sha: str = "4" * 64) -> dict[str, Any]:
    state: dict[str, Any] = {
        "protocol": "same-topology-rank-shard-sha256-v1",
        "vision": {"inventory_sha256": "3" * 64, "tensor_count": 1},
        "connector": {"inventory_sha256": connector_sha, "tensor_count": 1},
        "image_embedding_rows": {"sha256": "5" * 64, "shape": [6, 4]},
    }
    state["sha256"] = bridge.canonical_sha256(state)
    return state


def _records(step: int, *, gap_delta: float = 0.0, bad_final: bool = False) -> list[dict[str, Any]]:
    correct_by_step = {
        0: 2.0,
        100: 1.9,
        200: 1.8,
        250: 1.6,
        300: 1.5,
        400: 1.4,
        500: 1.2,
    }
    gap_by_step = {0: 0.0, 100: 0.2, 200: 0.3, 250: 0.5, 300: 0.55, 400: 0.6, 500: 0.5}
    correct = 1.8 if bad_final and step == 500 else correct_by_step[step]
    gap = 0.1 if bad_final and step == 500 else gap_by_step[step]
    output = []
    for index in range(2):
        row_gap = (-0.1 if index == 0 else 0.1) + gap_delta if step == 0 else gap + gap_delta
        correct_windows = {window: correct + index * 0.01 for window in bridge.WINDOWS}
        gap_windows = {window: row_gap for window in bridge.WINDOWS}
        output.append(
            {
                "pairing_position": index,
                "recipient_index": index,
                "donor_index": 1 - index,
                "response_tokens": 8 + index,
                "correct_ce": correct_windows,
                "wrong_ce": {
                    window: correct_windows[window] + gap_windows[window]
                    for window in bridge.WINDOWS
                },
                "ce_gap_wrong_minus_correct": gap_windows,
            }
        )
    return output


def _matched_receipt(
    manifest: Mapping[str, Any],
    step: int,
    *,
    component_state: Mapping[str, Any] | None = None,
    gap_delta: float = 0.0,
    bad_final: bool = False,
) -> dict[str, Any]:
    results = {}
    for source_index, source in enumerate(bridge.SOURCES):
        records = _records(step, gap_delta=gap_delta, bad_final=bad_final)
        results[source] = {
            "pairing_sha256": manifest["pairings"][source]["sha256"],
            "metrics": bridge.aggregate_matched_records(
                records,
                bootstrap_seed=manifest["evaluation"]["bootstrap_seed"] + source_index * 1_000_000,
                bootstrap_samples=manifest["evaluation"]["bootstrap_samples"],
            ),
            "per_example": records,
        }
    attention_diagnostics: dict[str, Any] = {
        "format": "ssmax_attention_diagnostics",
        "version": 1,
        "checkpoint": manifest["checkpoints"][str(step)],
        "protocol": {"manifest_sha256": manifest["attention_probe"]["sha256"]},
    }
    attention_diagnostics["report_sha256"] = hashlib.sha256(
        (
            json.dumps(
                attention_diagnostics,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "format": bridge.MATCHED_STATE_RECEIPT_FORMAT,
        "version": bridge.SCHEMA_VERSION,
        "status": "passed",
        "created_at": _TIMESTAMP,
        "manifest": _receipt_manifest_reference(manifest),
        "pair_id": manifest["pair_id"],
        "arm": manifest["arm"],
        "model_variant": manifest["model_variant"],
        "step": step,
        "checkpoint": manifest["checkpoints"][str(step)],
        "step0_checkpoint": manifest["checkpoints"]["0"],
        "pairings": manifest["pairings"],
        "strict_generic_dcp_load": {
            "complete": True,
            "strict": True,
            "load_completed": True,
        },
        "step0_strict_generic_dcp_load": {
            "complete": True,
            "strict": True,
            "load_completed": True,
        },
        "frozen_state": {"complete": True, "mismatch_count": 0},
        "component_state": dict(component_state or _component_state()),
        "validation": manifest["validation"],
        "protocol": {"name": "fixture"},
        "attention_diagnostics": attention_diagnostics,
        "results": results,
        "evaluator": manifest["producers"][bridge.MATCHED_STATE_PRODUCER],
    }
    payload["content_sha256"] = bridge.canonical_sha256(payload)
    return payload


def _health_receipt(manifest: Mapping[str, Any], step: int) -> dict[str, Any]:
    cursor_sha = f"{step:064x}"
    callback = SSMaxHealthLedgerCallback(
        model_variant=str(manifest["model_variant"]),
        phase="bridge",
        run_name=str(manifest["run_name"]),
    )
    callback.trainer = SimpleNamespace(
        global_step=step,
        data_loader=SimpleNamespace(state_dict=lambda: {"total_data_errors": 0}),
    )
    for global_step in range(1, step + 1):
        callback.log_metrics(
            global_step,
            {
                "train/CE loss": 2.0,
                "optim/total grad norm": 1.0,
                "optim/step skipped": 0.0,
            },
        )
    ledger = callback.state_dict()
    payload: dict[str, Any] = {
        "format": bridge.HEALTH_RECEIPT_FORMAT,
        "version": bridge.SCHEMA_VERSION,
        "status": "passed",
        "created_at": _TIMESTAMP,
        "manifest": _receipt_manifest_reference(manifest),
        "pair_id": manifest["pair_id"],
        "arm": manifest["arm"],
        "model_variant": manifest["model_variant"],
        "step": step,
        "checkpoint": manifest["checkpoints"][str(step)],
        "protocol": {"name": "fixture", "end_step": step},
        "loader": {
            "data_contract_sha256": "7" * 64,
            "dataset_fingerprints_sha256": "8" * 64,
            "initial_state_sha256": "a" * 64,
            "batches_replayed": step,
            "rank_states_global_step": step,
            "rank_states_batches_processed": step,
            "checkpoint_final_state_sha256": cursor_sha,
            "replayed_final_state_sha256": cursor_sha,
            "rank_state_inventory_sha256": bridge.canonical_sha256(_trainer_state_inventory()),
            "rank_state_count": 1,
            "dp_world_size": 1,
            "total_data_errors": 0,
        },
        "sources": {
            source: {"active_loss_mass_share": 0.5, "target_loss_mass": 0.5}
            for source in bridge.SOURCES
        },
        "health_ledger": {
            "rank_ledgers": [ledger],
            "event_chain_sha256": ledger["event_chain_sha256"],
            "counters": {
                "data_errors": 0,
                "optimizer_guard_skips": 0,
                "nonfinite_losses": 0,
                "nonfinite_gradients": 0,
            },
        },
        "summary": {"within_tolerance": True},
        "evidence": {
            "recipe": manifest["recipe"],
            "producer": manifest["producers"][bridge.HEALTH_PRODUCER],
            "rank_state_inventory": _trainer_state_inventory(),
        },
    }
    payload["content_sha256"] = bridge.canonical_sha256(payload)
    return payload


def _write_trajectory_receipts(
    tmp_path: Path,
    manifest: Mapping[str, Any],
    *,
    label: str,
    gap_delta: float = 0.0,
    bad_final: bool = False,
    component_state: Mapping[str, Any] | None = None,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    matched = {}
    health = {}
    for step in bridge.REQUIRED_STEPS:
        matched[step] = _write_json(
            tmp_path / label / f"matched-step{step}.json",
            _matched_receipt(
                manifest,
                step,
                component_state=component_state,
                gap_delta=gap_delta if step == 500 else 0.0,
                bad_final=bad_final,
            ),
        )
        health[step] = _write_json(
            tmp_path / label / f"health-step{step}.json",
            _health_receipt(manifest, step),
        )
    return matched, health


def _write_promotion_report(
    tmp_path: Path,
    *,
    label: str,
    manifest_path: Path,
    matched: Mapping[int, Mapping[str, str]],
    health: Mapping[int, Mapping[str, str]],
) -> dict[str, str]:
    report = bridge.build_promotion_report(
        manifest_path=manifest_path,
        matched_receipts=matched,
        health_receipts=health,
        created_at=_TIMESTAMP,
        verify_live_manifest=False,
    )
    assert report["status"] == "passed"
    assert report["deviations"] == []
    return _write_json(tmp_path / label / "promotion.json", report)


def test_checked_in_specs_form_one_controlled_pair_and_reject_arm_drift(tmp_path: Path) -> None:
    config_root = (
        Path(__file__).resolve().parents[3] / "configs" / "vision_moe" / "vision_alignment" / "eval"
    )
    head = bridge.load_manifest_spec(config_root / "ssmax_head_qknorm_bridge_manifest_v1.json")
    no_qk = bridge.load_manifest_spec(config_root / "ssmax_no_qknorm_bridge_manifest_v1.json")

    assert head["pair_id"] == no_qk["pair_id"]
    assert head["evaluation"] == no_qk["evaluation"]
    assert head["topology"] == no_qk["topology"]
    assert head["pairing_paths"] == no_qk["pairing_paths"]
    assert {head["model_variant"], no_qk["model_variant"]} == set(bridge.MODEL_VARIANTS)
    assert head["evaluation"] == bridge.BRIDGE_EVALUATION_CONTRACT
    assert head["topology"] == bridge.BRIDGE_TOPOLOGY_CONTRACT
    assert head["policy"] == bridge.BRIDGE_POLICY_CONTRACT
    assert (
        bridge.BRIDGE_GLOBAL_BATCH_INSTANCES
        // (head["topology"]["world_size"] * head["evaluation"]["rank_batch_instances"])
        == 2
    )

    invalid = copy.deepcopy(head)
    invalid["arm"] = "ssmax_no_qknorm"
    invalid_path = tmp_path / "invalid-spec.json"
    invalid_path.write_text(json.dumps(invalid))
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="arm/model_variant"):
        bridge.load_manifest_spec(invalid_path)

    for name, mutation, match in (
        (
            "weak-bootstrap",
            lambda value: value["evaluation"].__setitem__("bootstrap_samples", 1),
            "evaluation differs from the locked contract",
        ),
        (
            "empty-positive-steps",
            lambda value: value["policy"].__setitem__("positive_gap_ci_steps", []),
            "policy differs from the locked contract",
        ),
        (
            "loose-data-errors",
            lambda value: value["policy"].__setitem__("maximum_data_errors", 10_000),
            "policy differs from the locked contract",
        ),
    ):
        weakened = copy.deepcopy(head)
        mutation(weakened)
        weakened_path = tmp_path / f"{name}.json"
        weakened_path.write_text(json.dumps(weakened))
        with pytest.raises(bridge.SSMaxBridgeEvidenceError, match=match):
            bridge.load_manifest_spec(weakened_path)


def test_finalized_manifest_validation_binds_content_and_checkpoint_inventory(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    assert bridge.validate_manifest(manifest, verify_live=False) == manifest

    changed = copy.deepcopy(manifest)
    changed["run_name"] = "silently-changed"
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="content SHA-256"):
        bridge.validate_manifest(changed, verify_live=False)

    changed = copy.deepcopy(manifest)
    changed["checkpoints"]["250"]["state_file_count"] += 1
    changed["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="identity is malformed"):
        bridge.validate_manifest(changed, verify_live=False)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda config: config["data"].__setitem__("pack_sequences", True), "pack_sequences"),
        (
            lambda config: config["trainer"]["callbacks"]["checkpointer"].__setitem__(
                "fixed_steps", [100, 500]
            ),
            "fixed_steps",
        ),
        (
            lambda config: config.__setitem__(
                "global_batch_size",
                config["train_module"]["rank_microbatch_size"],
            ),
            "global_batch_size",
        ),
        (
            lambda config: config["artifacts"].__setitem__("base_model_inventory_sha256", "9" * 64),
            "base_model_inventory_sha256",
        ),
        (
            lambda config: config.__setitem__("reviewed_profile_sha256", "9" * 64),
            "reviewed_profile_sha256",
        ),
        (
            lambda config: config["expected_launch_command"].append(
                "--trainer.metrics_collect_interval=7"
            ),
            "expected_launch_command",
        ),
        (
            lambda config: config["launch"]["cmd"].append("--trainer.metrics_collect_interval=7"),
            "launch.cmd",
        ),
        (
            lambda config: config["launch"]["git"].__setitem__("branch", None),
            "launch.git.branch",
        ),
    ],
)
def test_saved_bridge_config_is_bound_to_spec_and_accumulated_global_batch(
    tmp_path: Path, mutation, match: str
) -> None:
    manifest = _manifest(tmp_path)
    repository = tmp_path / "repo"
    recipe = repository / "src" / "scripts" / "train" / "Vision-Alignment.py"
    profile = repository / "configs" / "vision_moe" / "vision_alignment" / "bridge" / "profile.yaml"
    recipe.parent.mkdir(parents=True)
    profile.parent.mkdir(parents=True)
    recipe.write_text("# recipe\n")
    profile.write_text("# profile\n")
    validation = tmp_path / "validation.json"
    validation.write_text("{}\n")
    profile_sha = bridge.sha256_file(profile)
    validation_sha = bridge.sha256_file(validation)
    config = _saved_config(
        manifest,
        checkpoint_root=tmp_path / "checkpoints",
        profile_path=profile.relative_to(repository).as_posix(),
        profile_sha256=profile_sha,
        validation_path=validation,
        validation_sha256=validation_sha,
    )

    git = bridge._validate_saved_bridge_config(
        manifest,
        config,
        checkpoint_root=tmp_path / "checkpoints",
        profile_path=profile,
        profile_sha256=profile_sha,
        recipe_path=recipe,
        validation_path=validation,
        validation_sha256=validation_sha,
    )
    assert git["ref"] == "3" * 40
    assert config["global_batch_size"] == 128 * 2560
    assert (
        bridge.BRIDGE_GLOBAL_BATCH_INSTANCES
        % (manifest["topology"]["world_size"] * manifest["evaluation"]["rank_batch_instances"])
        == 0
    )

    changed = copy.deepcopy(config)
    mutation(changed)
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match=match):
        bridge._validate_saved_bridge_config(
            manifest,
            changed,
            checkpoint_root=tmp_path / "checkpoints",
            profile_path=profile,
            profile_sha256=profile_sha,
            recipe_path=recipe,
            validation_path=validation,
            validation_sha256=validation_sha,
        )


@pytest.mark.parametrize("modified", ["recipe", "profile"])
def test_git_provenance_rejects_modified_recipe_or_profile(tmp_path: Path, modified: str) -> None:
    repository = tmp_path / "repo"
    recipe = repository / "src" / "scripts" / "train" / "Vision-Alignment.py"
    profile = repository / "configs" / "vision" / "bridge.yaml"
    recipe.parent.mkdir(parents=True)
    profile.parent.mkdir(parents=True)
    recipe.write_text("# committed recipe\n")
    profile.write_text("# committed profile\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "tests@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "SSMax tests"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    ref = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    git = {
        "repo": "allenai/OLMo-core",
        "repo_url": "https://github.com/allenai/OLMo-core",
        "ref": ref,
    }
    bridge._validate_saved_git_checkout(git, recipe_path=recipe, profile_path=profile)

    path = recipe if modified == "recipe" else profile
    path.write_text(path.read_text() + "# modified\n")
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="not clean"):
        bridge._validate_saved_git_checkout(git, recipe_path=recipe, profile_path=profile)


def test_manifest_producer_source_rejects_wrong_ref_and_source_mutation(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    sources = {}
    for producer, relative in bridge.PRODUCER_RELATIVE_PATHS.items():
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# committed {producer} producer\n")
        sources[producer] = source
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "tests@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "SSMax tests"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    ref = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = _manifest(tmp_path)
    manifest["git"]["ref"] = ref
    manifest["producers"] = {
        producer: {
            "repo_relative_path": relative,
            "sha256": bridge.sha256_file(sources[producer]),
            "git_ref": ref,
        }
        for producer, relative in bridge.PRODUCER_RELATIVE_PATHS.items()
    }

    assert (
        bridge.validate_manifest_producer_source(
            manifest,
            producer=bridge.MATCHED_STATE_PRODUCER,
            source_path=sources[bridge.MATCHED_STATE_PRODUCER],
        )
        == manifest["producers"][bridge.MATCHED_STATE_PRODUCER]
    )

    wrong_ref = copy.deepcopy(manifest)
    wrong_ref["git"]["ref"] = "0" * 40
    for reference in wrong_ref["producers"].values():
        reference["git_ref"] = "0" * 40
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="HEAD differs"):
        bridge.validate_manifest_producer_source(
            wrong_ref,
            producer=bridge.MATCHED_STATE_PRODUCER,
            source_path=sources[bridge.MATCHED_STATE_PRODUCER],
        )

    matched_relative = bridge.PRODUCER_RELATIVE_PATHS[bridge.MATCHED_STATE_PRODUCER]
    subprocess.run(
        ["git", "-C", str(repository), "update-index", "--assume-unchanged", matched_relative],
        check=True,
    )
    sources[bridge.MATCHED_STATE_PRODUCER].write_text("# locally mutated producer\n")
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="source identity"):
        bridge.validate_manifest_producer_source(
            manifest,
            producer=bridge.MATCHED_STATE_PRODUCER,
            source_path=sources[bridge.MATCHED_STATE_PRODUCER],
        )


def _write_native_checkpoint(
    root: Path,
    *,
    step: int,
    ephemeral: bool = False,
    marker_step: int | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config or {}, sort_keys=True) + "\n")
    marker = {"ephemeral": ephemeral, "version": "test"}
    if marker_step is not None:
        marker["global_step"] = marker_step
    (root / ".metadata.json").write_text(json.dumps(marker))
    dcp.save({"model": {"weight": torch.ones(2)}}, checkpoint_id=root / "model_and_optim")
    (root / "train").mkdir()
    torch.save({"global_step": step, "world_size": 1}, root / "train" / "rank0.pt")


def test_checkpoint_identity_requires_permanent_complete_step_agreement(tmp_path: Path) -> None:
    permanent = tmp_path / "permanent" / "step7"
    _write_native_checkpoint(permanent, step=7)
    assert bridge.checkpoint_identity(permanent, workers=1)["global_step"] == 7

    ephemeral = tmp_path / "ephemeral" / "step7"
    _write_native_checkpoint(ephemeral, step=7, ephemeral=True)
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="completed permanent"):
        bridge.checkpoint_identity(ephemeral, workers=1)

    wrong_step = tmp_path / "wrong-step" / "step7"
    _write_native_checkpoint(wrong_step, step=7, marker_step=8)
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="marker global_step"):
        bridge.checkpoint_identity(wrong_step, workers=1)


def test_parent_load_receipt_binds_exact_parent_pins_and_inventory(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    receipt = _parent_load_receipt(manifest)
    assert (
        bridge._validate_parent_load_receipt_payload(
            receipt,
            parent=manifest["parent"],
            model_variant=manifest["model_variant"],
        )
        == receipt
    )

    changed = copy.deepcopy(receipt)
    changed["parent_model_inventory_sha256"] = "9" * 64
    changed["fingerprint"] = bridge.canonical_sha256(
        {key: value for key, value in changed.items() if key != "fingerprint"}
    )
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="model_inventory_sha256"):
        bridge._validate_parent_load_receipt_payload(
            changed,
            parent=manifest["parent"],
            model_variant=manifest["model_variant"],
        )


def test_build_manifest_end_to_end_binds_saved_config_git_and_parent_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    recipe = repository / "src" / "scripts" / "train" / "Vision-Alignment.py"
    profile = repository / "configs" / "vision" / "bridge.yaml"
    recipe.parent.mkdir(parents=True)
    profile.parent.mkdir(parents=True)
    recipe.write_text("# exact recipe\n")
    profile.write_text("# exact profile\n")
    for producer, relative in bridge.PRODUCER_RELATIVE_PATHS.items():
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# exact {producer} producer\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "tests@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "SSMax tests"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    git_ref = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()

    checkpoint_root = tmp_path / "checkpoints"
    validation = tmp_path / "validation.json"
    attention_probe = tmp_path / "attention-probe.json"
    validation.write_text("{}\n")
    attention_probe.write_text("{}\n")
    base = _manifest(tmp_path)
    config = _saved_config(
        base,
        checkpoint_root=checkpoint_root,
        profile_path=profile.relative_to(repository).as_posix(),
        profile_sha256=bridge.sha256_file(profile),
        validation_path=validation,
        validation_sha256=bridge.sha256_file(validation),
        git_ref=git_ref,
    )
    for step in bridge.REQUIRED_STEPS:
        _write_native_checkpoint(checkpoint_root / f"step{step}", step=step, config=config)
    _write_json(
        checkpoint_root / "bridge-parent-load-receipt.json",
        _parent_load_receipt(base),
    )
    pairing_references = {
        source: _write_json(tmp_path / f"{source}-pairing.json", {}) for source in bridge.SOURCES
    }
    pairing_sha = pairing_references[bridge.SOURCES[0]]["sha256"]
    assert {reference["sha256"] for reference in pairing_references.values()} == {pairing_sha}
    monkeypatch.setattr(
        bridge, "validate_matched_wrong_image_pairing", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        bridge,
        "matched_wrong_image_pairing_sha256",
        lambda _payload: pairing_sha,
    )
    spec = {
        "format": bridge.MANIFEST_SPEC_FORMAT,
        "version": bridge.SCHEMA_VERSION,
        "pair_id": base["pair_id"],
        "arm": base["arm"],
        "model_variant": base["model_variant"],
        "run_name": base["run_name"],
        "checkpoint_root": str(checkpoint_root),
        "parent": base["parent"],
        "training_profile": str(profile),
        "recipe": str(recipe),
        "validation": str(validation),
        "attention_probe": str(attention_probe),
        "pairing_paths": {
            source: reference["path"] for source, reference in pairing_references.items()
        },
        "evaluation": base["evaluation"],
        "topology": base["topology"],
        "policy": base["policy"],
    }
    spec_path = repository / bridge.MANIFEST_SPEC_RELATIVE_PATHS[base["model_variant"]]
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "bind manifest spec"], check=True
    )
    git_ref = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    config["launch"]["git"]["ref"] = git_ref
    for step in bridge.REQUIRED_STEPS:
        (checkpoint_root / f"step{step}" / "config.json").write_text(
            json.dumps(config, sort_keys=True) + "\n"
        )

    finalized = bridge.build_manifest(
        spec,
        spec_path=spec_path,
        pairing_references=pairing_references,
        created_at=_TIMESTAMP,
        hash_workers=1,
    )

    assert finalized["git"]["ref"] == git_ref
    assert finalized["manifest_spec"] == {
        "repo_relative_path": bridge.MANIFEST_SPEC_RELATIVE_PATHS[base["model_variant"]],
        "sha256": bridge.sha256_file(spec_path),
        "git_ref": git_ref,
    }
    assert finalized["producers"] == {
        producer: {
            "repo_relative_path": relative,
            "sha256": bridge.sha256_file(repository / relative),
            "git_ref": git_ref,
        }
        for producer, relative in bridge.PRODUCER_RELATIVE_PATHS.items()
    }
    assert finalized["parent_load_receipt"] == bridge.artifact_reference(
        checkpoint_root / "bridge-parent-load-receipt.json"
    )
    assert finalized["checkpoints"]["500"]["config_sha256"] == bridge.sha256_file(
        checkpoint_root / "step500" / "config.json"
    )


class _ToyState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.register_buffer("counts", torch.tensor([2, 4], dtype=torch.int64))


def test_generic_dcp_inventory_requires_exact_key_shape_dtype_and_partition(tmp_path: Path) -> None:
    model = _ToyState()
    checkpoint = tmp_path / "model_and_optim"
    dcp.save(
        {"model": model.state_dict(), "optim": {"step": torch.tensor(7)}},
        checkpoint_id=checkpoint,
    )
    metadata = FileSystemReader(checkpoint).read_metadata()

    inventory = bridge.verify_generic_dcp_load_inventory(
        metadata=metadata,
        state_dict_to_load={"model": model.state_dict()},
        parameter_names=("weight",),
        buffer_names=("counts",),
    )
    assert inventory.checkpoint_key_count == 3
    assert inventory.model_tensor_count == 2
    assert inventory.model_parameter_tensor_count == 1
    assert inventory.model_buffer_tensor_count == 1
    assert inventory.as_dict()["complete"] is True

    wrong_shape = {"model": {**model.state_dict(), "weight": torch.zeros(3, 2)}}
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="metadata differs for model.weight"):
        bridge.verify_generic_dcp_load_inventory(
            metadata=metadata,
            state_dict_to_load=wrong_shape,
            parameter_names=("weight",),
            buffer_names=("counts",),
        )
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="does not exactly partition"):
        bridge.verify_generic_dcp_load_inventory(
            metadata=metadata,
            state_dict_to_load={"model": model.state_dict()},
            parameter_names=("weight",),
            buffer_names=(),
        )


def test_matched_aggregation_is_deterministic_and_uses_wrong_minus_correct() -> None:
    records = _records(0)
    first = bridge.aggregate_matched_records(records, bootstrap_seed=91, bootstrap_samples=256)
    second = bridge.aggregate_matched_records(records, bootstrap_seed=91, bootstrap_samples=256)

    assert first == second
    for window in bridge.WINDOWS:
        assert first[window]["gap_wrong_minus_correct_mean"] == pytest.approx(0.0)
        assert first[window]["win_rate"] == 0.5
        assert first[window]["tie_rate"] == 0.0
        assert first[window]["mean_gap_bootstrap_ci"]["confidence"] == 0.95
        assert first[window]["mean_gap_bootstrap_ci"]["low"] == pytest.approx(-0.1)
        assert first[window]["mean_gap_bootstrap_ci"]["high"] == pytest.approx(0.1)

    invalid = copy.deepcopy(records)
    invalid[0]["wrong_ce"]["all"] = float("nan")
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="finite vector"):
        bridge.aggregate_matched_records(invalid, bootstrap_seed=91, bootstrap_samples=32)


def test_promotion_gates_pass_and_reject_regressed_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    candidate = manifest["checkpoints"]["500"]
    candidate_config = {
        "vision_alignment": {
            "phase": "bridge",
            "model_variant": manifest["model_variant"],
            "recipe_version": 1,
            "formatter_version": "vision-alignment-document-v1",
            "data_contract_sha256": "6" * 64,
            "trainable_contract_sha256": "7" * 64,
        }
    }
    candidate_config_ref = _write_json(Path(candidate["path"]) / "config.json", candidate_config)
    candidate["config_sha256"] = candidate_config_ref["sha256"]
    candidate.pop("identity_sha256")
    candidate["identity_sha256"] = bridge.canonical_sha256(candidate)
    manifest.pop("content_sha256")
    manifest["content_sha256"] = bridge.canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(bridge, "load_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        bridge,
        "compare_ssmax_attention_reports",
        lambda _reference, _candidate: {"flag_count": 0, "flags": []},
    )

    matched, health = _write_trajectory_receipts(tmp_path, manifest, label="passing")
    report = bridge.build_promotion_report(
        manifest_path=manifest_path,
        matched_receipts=matched,
        health_receipts=health,
        created_at=_TIMESTAMP,
    )
    assert report["status"] == "passed"
    assert report["deviations"] == []
    assert all(
        report["trajectory"][source][window]["step0_to_final_correct_ce_improvement"]["mean"] > 0
        for source in bridge.SOURCES
        for window in bridge.WINDOWS
    )
    report_reference = _write_json(tmp_path / "promotion-report.json", report)
    summary = bridge.validate_promotion_report_reference(
        report_reference,
        expected_checkpoint=Path(manifest["checkpoints"]["500"]["path"]),
        expected_checkpoint_config_sha256=manifest["checkpoints"]["500"]["config_sha256"],
        expected_model_variant=manifest["model_variant"],
        verify_live_checkpoint=False,
    )
    assert summary["candidate"] == manifest["checkpoints"]["500"]
    forged_report = copy.deepcopy(report)
    forged_report["trajectory"]["pixmo_caption"]["all"]["step0_to_final_correct_ce_improvement"][
        "mean"
    ] = 999.0
    forged_report["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in forged_report.items() if key != "content_sha256"}
    )
    forged_reference = _write_json(tmp_path / "forged-promotion-report.json", forged_report)
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="rebuilt from its pinned receipts"):
        bridge.validate_promotion_report_reference(
            forged_reference,
            expected_checkpoint=Path(manifest["checkpoints"]["500"]["path"]),
            expected_checkpoint_config_sha256=manifest["checkpoints"]["500"]["config_sha256"],
            expected_model_variant=manifest["model_variant"],
            verify_live_checkpoint=False,
        )
    gate = bridge.build_parent_gate(
        promotion_report_path=Path(report_reference["path"]),
        expected_promotion_report_sha256=report_reference["sha256"],
        approved_by="reviewer@example.com",
        approved_at=_TIMESTAMP,
        verify_live_checkpoint=False,
    )
    assert gate["waivers"] == []
    assert gate["data_contract_sha256"] == "6" * 64
    gate_summary = bridge.validate_ssmax_bridge_parent_gate(
        gate,
        expected_checkpoint=Path(manifest["checkpoints"]["500"]["path"]),
        expected_checkpoint_config_sha256=manifest["checkpoints"]["500"]["config_sha256"],
        expected_model_variant=manifest["model_variant"],
        expected_data_contract_sha256="6" * 64,
        expected_trainable_contract_sha256="7" * 64,
        verify_live_checkpoint=False,
    )
    assert gate_summary["candidate"]["identity_sha256"] == gate["checkpoint_identity_sha256"]
    changed_gate = {**gate, "waivers": [{"id": "not-allowed"}]}
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="does not permit waivers"):
        bridge.validate_ssmax_bridge_parent_gate(
            changed_gate,
            expected_checkpoint=Path(manifest["checkpoints"]["500"]["path"]),
            expected_checkpoint_config_sha256=manifest["checkpoints"]["500"]["config_sha256"],
            expected_model_variant=manifest["model_variant"],
            expected_data_contract_sha256="6" * 64,
            expected_trainable_contract_sha256="7" * 64,
            verify_live_checkpoint=False,
        )

    failed_health = dict(health)
    failed_payload = _health_receipt(manifest, 100)
    failed_payload["status"] = "failed"
    failed_payload["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in failed_payload.items() if key != "content_sha256"}
    )
    failed_health[100] = _write_json(
        tmp_path / "failed-health" / "health-step100.json", failed_payload
    )
    report = bridge.build_promotion_report(
        manifest_path=manifest_path,
        matched_receipts=matched,
        health_receipts=failed_health,
        created_at=_TIMESTAMP,
    )
    assert report["status"] == "rejected"
    assert {"kind": "health_receipt_status", "step": 100} in report["deviations"]

    regressed, _ = _write_trajectory_receipts(tmp_path, manifest, label="regressed", bad_final=True)
    report = bridge.build_promotion_report(
        manifest_path=manifest_path,
        matched_receipts=regressed,
        health_receipts=health,
        created_at=_TIMESTAMP,
    )
    assert report["status"] == "rejected"
    kinds = {deviation["kind"] for deviation in report["deviations"]}
    assert {"gap_retention", "correct_ce_regression"} <= kinds


def test_promotion_rejects_receipt_without_completed_strict_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(bridge, "load_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        bridge,
        "compare_ssmax_attention_reports",
        lambda _reference, _candidate: {"flag_count": 0, "flags": []},
    )
    matched, health = _write_trajectory_receipts(tmp_path, manifest, label="incomplete-load")
    payload_path = Path(matched[100]["path"])
    payload = bridge.load_json(payload_path)
    payload["strict_generic_dcp_load"]["load_completed"] = False
    payload["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    matched[100] = _write_json(payload_path, payload)

    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="completed strict generic DCP load"):
        bridge.build_promotion_report(
            manifest_path=manifest_path,
            matched_receipts=matched,
            health_receipts=health,
            created_at=_TIMESTAMP,
        )


@pytest.mark.parametrize(
    ("receipt_kind", "mutation", "match"),
    [
        (
            "matched",
            lambda payload: payload["evaluator"].__setitem__("sha256", "8" * 64),
            "evaluator source identity differs",
        ),
        (
            "health",
            lambda payload: payload["evidence"]["producer"].__setitem__(
                "repo_relative_path", "src/scripts/eval/not-the-health-producer.py"
            ),
            "producer source identity differs",
        ),
        (
            "matched",
            lambda payload: payload.__setitem__("unexpected", True),
            "receipt fields differ",
        ),
    ],
)
def test_promotion_requires_exact_receipt_schema_and_producer_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_kind: str,
    mutation,
    match: str,
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(bridge, "load_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        bridge,
        "compare_ssmax_attention_reports",
        lambda _reference, _candidate: {"flag_count": 0, "flags": []},
    )
    matched, health = _write_trajectory_receipts(tmp_path, manifest, label="producer-identity")
    references = matched if receipt_kind == "matched" else health
    receipt_path = Path(references[100]["path"])
    payload = bridge.load_json(receipt_path)
    mutation(payload)
    payload["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    references[100] = _write_json(receipt_path, payload)

    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match=match):
        bridge.build_promotion_report(
            manifest_path=manifest_path,
            matched_receipts=matched,
            health_receipts=health,
            created_at=_TIMESTAMP,
        )


def test_promotion_rejects_tampered_attention_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(bridge, "load_manifest", lambda *_args, **_kwargs: manifest)
    matched, health = _write_trajectory_receipts(tmp_path, manifest, label="attention-drift")
    payload_path = Path(matched[100]["path"])
    payload = bridge.load_json(payload_path)
    payload["attention_diagnostics"]["tampered"] = True
    payload["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    matched[100] = _write_json(payload_path, payload)

    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="report SHA mismatch"):
        bridge.build_promotion_report(
            manifest_path=manifest_path,
            matched_receipts=matched,
            health_receipts=health,
            created_at=_TIMESTAMP,
        )


def test_bridge_promotion_binds_checkpoint_native_health_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(bridge, "load_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        bridge,
        "compare_ssmax_attention_reports",
        lambda _reference, _candidate: {"flag_count": 0, "flags": []},
    )
    matched, health = _write_trajectory_receipts(tmp_path, manifest, label="health-bytes")

    changed = _health_receipt(manifest, 100)
    changed["evidence"]["rank_state_inventory"][0]["sha256"] = "8" * 64
    changed["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    health[100] = _write_json(tmp_path / "tampered-health.json", changed)
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="trainer-state bytes differ"):
        bridge.build_promotion_report(
            manifest_path=manifest_path,
            matched_receipts=matched,
            health_receipts=health,
            created_at=_TIMESTAMP,
        )


def test_controlled_pair_requires_bit_identical_step0_new_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _manifest(tmp_path, "ssmax_head_qknorm")
    right = _manifest(tmp_path, "ssmax_no_qknorm")
    left_path = tmp_path / "left-manifest.json"
    right_path = tmp_path / "right-manifest.json"
    _write_json(left_path, left)
    _write_json(right_path, right)
    manifests = {left_path.resolve(): left, right_path.resolve(): right}
    monkeypatch.setattr(
        bridge,
        "load_manifest",
        lambda path, **_kwargs: manifests[Path(path).resolve()],
    )
    monkeypatch.setattr(bridge, "validate_ssmax_attention_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "compare_ssmax_attention_reports",
        lambda left_report, right_report: {
            "left_checkpoint": left_report["checkpoint"]["global_step"],
            "right_checkpoint": right_report["checkpoint"]["global_step"],
        },
    )
    components = _component_state()
    left_matched, left_health = _write_trajectory_receipts(
        tmp_path,
        left,
        label="left",
        gap_delta=0.2,
        component_state=components,
    )
    right_matched, right_health = _write_trajectory_receipts(
        tmp_path,
        right,
        label="right",
        component_state=components,
    )

    left_promotion = _write_promotion_report(
        tmp_path,
        label="left",
        manifest_path=left_path,
        matched=left_matched,
        health=left_health,
    )
    right_promotion = _write_promotion_report(
        tmp_path,
        label="right",
        manifest_path=right_path,
        matched=right_matched,
        health=right_health,
    )
    comparison = bridge.build_pair_comparison(
        left_promotion_report=left_promotion,
        right_promotion_report=right_promotion,
        created_at=_TIMESTAMP,
        verify_live_checkpoint=False,
    )
    assert comparison["step0_new_component_attestation"]["vision_bit_identical"] is True
    assert comparison["step0_new_component_attestation"]["connector_bit_identical"] is True
    assert comparison["step500_gap_dominant_arm"] == "ssmax_head_qknorm"
    assert comparison["step500_absolute_gap_dominant_arm"] == "ssmax_head_qknorm"
    assert comparison["step500_adaptation_gap_dominant_arm"] == "ssmax_head_qknorm"
    adaptation = comparison["comparison"]["500"]["pixmo_caption"]["all"]["adaptation_from_step0"]
    assert adaptation["left_gap_change"]["mean"] == pytest.approx(0.7)
    assert adaptation["right_gap_change"]["mean"] == pytest.approx(0.5)
    assert adaptation["gap_change_did_left_minus_right"]["mean"] == pytest.approx(0.2)
    assert adaptation["correct_ce_improvement_did_left_minus_right"]["mean"] == pytest.approx(0.0)
    assert comparison["comparison"]["0"]["pixmo_caption"]["all"]["adaptation_from_step0"][
        "gap_change_did_left_minus_right"
    ]["mean"] == pytest.approx(0.0)
    assert set(comparison["attention_comparison"]) == {str(step) for step in bridge.REQUIRED_STEPS}
    assert comparison["attention_comparison"]["500"]["left_minus_right"] == {
        "left_checkpoint": 500,
        "right_checkpoint": 500,
    }

    changed_components = _component_state(connector_sha="9" * 64)
    bad_right, bad_right_health = _write_trajectory_receipts(
        tmp_path,
        right,
        label="right-changed-component",
        component_state=changed_components,
    )
    with pytest.raises(
        bridge.SSMaxBridgeEvidenceError,
        match="step0 new components are not bit-identical: .*connector",
    ):
        bad_right_promotion = _write_promotion_report(
            tmp_path,
            label="right-changed-component-promotion",
            manifest_path=right_path,
            matched=bad_right,
            health=bad_right_health,
        )
        bridge.build_pair_comparison(
            left_promotion_report=left_promotion,
            right_promotion_report=bad_right_promotion,
            created_at=_TIMESTAMP,
            verify_live_checkpoint=False,
        )

    drifted_health = dict(right_health)
    drifted_payload = bridge.load_json(Path(right_health[100]["path"]))
    drifted_payload["loader"]["dataset_fingerprints_sha256"] = "9" * 64
    drifted_payload["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in drifted_payload.items() if key != "content_sha256"}
    )
    drifted_health[100] = _write_json(
        tmp_path / "right-health-drift" / "step100.json", drifted_payload
    )
    drifted_promotion = _write_promotion_report(
        tmp_path,
        label="right-health-drift-promotion",
        manifest_path=right_path,
        matched=right_matched,
        health=drifted_health,
    )
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="health/data trajectories differ"):
        bridge.build_pair_comparison(
            left_promotion_report=left_promotion,
            right_promotion_report=drifted_promotion,
            created_at=_TIMESTAMP,
            verify_live_checkpoint=False,
        )

    failed_health = dict(right_health)
    failed_payload = bridge.load_json(Path(right_health[100]["path"]))
    failed_payload["status"] = "failed"
    failed_payload["content_sha256"] = bridge.canonical_sha256(
        {key: value for key, value in failed_payload.items() if key != "content_sha256"}
    )
    failed_health[100] = _write_json(
        tmp_path / "right-health-failed" / "step100.json", failed_payload
    )
    rejected_report = bridge.build_promotion_report(
        manifest_path=right_path,
        matched_receipts=right_matched,
        health_receipts=failed_health,
        created_at=_TIMESTAMP,
        verify_live_manifest=False,
    )
    assert rejected_report["status"] == "rejected"
    rejected_promotion = _write_json(tmp_path / "right-rejected-promotion.json", rejected_report)
    with pytest.raises(bridge.SSMaxBridgeEvidenceError, match="not eligible"):
        bridge.build_pair_comparison(
            left_promotion_report=left_promotion,
            right_promotion_report=rejected_promotion,
            created_at=_TIMESTAMP,
            verify_live_checkpoint=False,
        )


def test_controlled_pair_does_not_confuse_starting_advantage_with_adaptation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _manifest(tmp_path, "ssmax_head_qknorm")
    right = _manifest(tmp_path, "ssmax_no_qknorm")
    left_path = tmp_path / "left-manifest.json"
    right_path = tmp_path / "right-manifest.json"
    _write_json(left_path, left)
    _write_json(right_path, right)
    manifests = {left_path.resolve(): left, right_path.resolve(): right}
    monkeypatch.setattr(
        bridge,
        "load_manifest",
        lambda path, **_kwargs: manifests[Path(path).resolve()],
    )
    monkeypatch.setattr(bridge, "validate_ssmax_attention_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "compare_ssmax_attention_reports",
        lambda *_args, **_kwargs: {"flag_count": 0, "flags": []},
    )
    components = _component_state()
    left_matched = {
        step: _write_json(
            tmp_path / "left-starting-advantage" / f"step{step}.json",
            _matched_receipt(
                left,
                step,
                component_state=components,
                gap_delta=0.2,
            ),
        )
        for step in bridge.REQUIRED_STEPS
    }
    _, left_health = _write_trajectory_receipts(
        tmp_path,
        left,
        label="left-starting-health",
        component_state=components,
    )
    right_matched, right_health = _write_trajectory_receipts(
        tmp_path,
        right,
        label="right-no-starting-advantage",
        component_state=components,
    )
    left_promotion = _write_promotion_report(
        tmp_path,
        label="left-starting-promotion",
        manifest_path=left_path,
        matched=left_matched,
        health=left_health,
    )
    right_promotion = _write_promotion_report(
        tmp_path,
        label="right-starting-promotion",
        manifest_path=right_path,
        matched=right_matched,
        health=right_health,
    )

    comparison = bridge.build_pair_comparison(
        left_promotion_report=left_promotion,
        right_promotion_report=right_promotion,
        created_at=_TIMESTAMP,
        verify_live_checkpoint=False,
    )

    assert comparison["step500_absolute_gap_dominant_arm"] == "ssmax_head_qknorm"
    assert comparison["step500_adaptation_gap_dominant_arm"] is None
    did = comparison["comparison"]["500"]["pixmo_caption"]["all"]["adaptation_from_step0"][
        "gap_change_did_left_minus_right"
    ]
    assert did["mean"] == pytest.approx(0.0)
