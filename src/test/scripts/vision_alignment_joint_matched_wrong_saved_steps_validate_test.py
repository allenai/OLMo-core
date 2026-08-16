from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_joint_matched_wrong_saved_steps_validate.py"
    )
    name = "_vision_alignment_joint_matched_wrong_saved_steps_validate_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _endpoint(step: int) -> dict[str, Any]:
    return {
        "contract": module.ENDPOINT_CONTRACT,
        "admissible_steps": list(module.STEPS),
        "step": step,
        "storage_class": module.EXPECTED_STORAGE_CLASS[step],
        "nearest_step_substitution": False,
    }


def _state_inventory(step: int) -> list[dict[str, Any]]:
    records = [
        {
            "path": "model_and_optim/.metadata",
            "size": 4_208_572,
            "sha256": module.EXPECTED_DCP_METADATA_SHA256[step],
        }
    ]
    shard_paths = sorted(module._expected_state_paths() - {"model_and_optim/.metadata"})
    remaining = module.EXPECTED_MODEL_AND_OPTIM_BYTES - 4_208_572
    for position, path in enumerate(shard_paths):
        size = 1 if position < len(shard_paths) - 1 else remaining - len(shard_paths) + 1
        records.append({"path": path, "size": size, "sha256": f"{position + 1:064x}"})
    return records


def _trainer_inventory() -> list[dict[str, Any]]:
    return [
        {
            "path": f"train/rank{rank}.pt",
            "size": 17_621 if rank < 10 else 17_629,
            "sha256": f"{rank + 1:064x}",
        }
        for rank in range(16)
    ]


def _checkpoint_identity(step: int, root: Path) -> dict[str, Any]:
    state_inventory = _state_inventory(step)
    trainer_inventory = _trainer_inventory()
    identity = {
        "root": str(root),
        "state_dir": str(root / "model_and_optim"),
        "config_sha256": module._v1.EXPECTED_CONFIG_SHA256,
        "checkpoint_marker_sha256": module.EXPECTED_MARKER_SHA256[step],
        "dcp_metadata_sha256": module.EXPECTED_DCP_METADATA_SHA256[step],
        "state_file_hash_algorithm": "sha256",
        "state_file_inventory_sha256": module._v1._canonical_sha256(state_inventory),
        "state_file_inventory": state_inventory,
        "model_and_optim_identity_sha256": "",
        "checkpoint_step": step,
        "permanent": not module.EXPECTED_MARKER[step]["ephemeral"],
        "checkpoint_marker": dict(module.EXPECTED_MARKER[step]),
        "trainer_state_rank_count": 16,
        "trainer_state_file_inventory": trainer_inventory,
        "trainer_state_file_inventory_sha256": module._v1._canonical_sha256(trainer_inventory),
        "trainer_state_summary": {
            "global_step": step,
            "global_train_tokens_seen": step * 1_048_576,
            "max_steps": 16_000,
            "world_size": 16,
            "batches_processed": step,
            "consecutive_data_errors": 0,
            "wandb_run_id": module.EXPECTED_WANDB_RUN_ID,
            "wandb_name": "vision-alignment-joint-v1",
        },
        "trainer_state_total_data_errors_by_rank": list(module.EXPECTED_TRAINER_ERRORS),
        "trainer_state_total_data_errors_sum": sum(module.EXPECTED_TRAINER_ERRORS),
        "identity_sha256": "",
    }
    model_fields = (
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
    )
    identity["model_and_optim_identity_sha256"] = module._v1._canonical_sha256(
        {field: identity[field] for field in model_fields}
    )
    identity["identity_sha256"] = module._v1._canonical_sha256(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    return identity


def _policy() -> dict[str, Any]:
    return {
        "output_overwrite_enabled": False,
        "pairing_manifest_requires_sha256_pin": True,
        "all_pairings_rehashed": True,
        "all_pairings_deterministically_rebuilt": True,
        "checkpoint_private_snapshot": (
            "full model_and_optim same-FD verified byte copy; load only; delete after load"
        ),
        "checkpoint_post_identity_rehashed": True,
        "native_sources_full_hash_pre_and_post": True,
        "descriptive_only": True,
        "promotion_eligible": False,
        "checkpoint_source_marker_preserved": True,
        "retained_ephemeral_not_promoted_to_permanent": True,
        "nearest_step_substitution_allowed": False,
    }


@pytest.mark.parametrize("step", module.STEPS)
def test_endpoint_schema_and_storage_class_are_exact(step):
    assert module._validate_endpoint(_endpoint(step), step=step, name="endpoint") == _endpoint(step)
    changed = _endpoint(step)
    changed["nearest_step_substitution"] = True
    with pytest.raises(ValueError, match="saved-endpoint contract"):
        module._validate_endpoint(changed, step=step, name="endpoint")


@pytest.mark.parametrize("step", module.STEPS)
def test_checkpoint_identity_freezes_inventory_marker_and_error_panel(tmp_path, monkeypatch, step):
    monkeypatch.setattr(module, "EXPECTED_CHECKPOINT_BASE", tmp_path)
    root = tmp_path / f"step{step}"
    (root / "model_and_optim").mkdir(parents=True)
    (root / "train").mkdir()
    (root / "config.json").write_bytes(b"x" * module.EXPECTED_CONFIG_BYTES)
    (root / ".metadata.json").write_text(json.dumps(module.EXPECTED_MARKER[step]))
    monkeypatch.setattr(
        module._v1,
        "_validate_inventory",
        lambda value, **_kwargs: list(value),
    )
    config = {
        "path": str(root / "config.json"),
        "sha256": module._v1.EXPECTED_CONFIG_SHA256,
        "step": step,
    }
    identity = _checkpoint_identity(step, root)
    module._validate_checkpoint_identity(
        identity,
        config=config,
        endpoint=_endpoint(step),
        name="checkpoint",
        verify_live_files=False,
    )

    unexpected = root / "undeclared"
    unexpected.write_text("not part of the endpoint")
    with pytest.raises(ValueError, match="root entries"):
        module._validate_checkpoint_identity(
            identity,
            config=config,
            endpoint=_endpoint(step),
            name="checkpoint",
            verify_live_files=False,
        )
    unexpected.unlink()

    changed = copy.deepcopy(identity)
    changed["trainer_state_total_data_errors_by_rank"][0] = 1
    changed["trainer_state_total_data_errors_sum"] = 2
    changed["identity_sha256"] = module._v1._canonical_sha256(
        {key: value for key, value in changed.items() if key != "identity_sha256"}
    )
    with pytest.raises(ValueError, match="data-error panel"):
        module._validate_checkpoint_identity(
            changed,
            config=config,
            endpoint=_endpoint(step),
            name="checkpoint",
            verify_live_files=False,
        )


def test_retained_ephemeral_cannot_be_relabelled_permanent(tmp_path, monkeypatch):
    step = 14400
    monkeypatch.setattr(module, "EXPECTED_CHECKPOINT_BASE", tmp_path)
    root = tmp_path / f"step{step}"
    (root / "model_and_optim").mkdir(parents=True)
    (root / "train").mkdir()
    (root / "config.json").write_bytes(b"x" * module.EXPECTED_CONFIG_BYTES)
    (root / ".metadata.json").write_text(json.dumps(module.EXPECTED_MARKER[step]))
    monkeypatch.setattr(module._v1, "_validate_inventory", lambda value, **_kwargs: list(value))
    config = {
        "path": str(root / "config.json"),
        "sha256": module._v1.EXPECTED_CONFIG_SHA256,
        "step": step,
    }
    identity = _checkpoint_identity(step, root)
    identity["permanent"] = True
    identity["identity_sha256"] = module._v1._canonical_sha256(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    with pytest.raises(ValueError, match="storage class"):
        module._validate_checkpoint_identity(
            identity,
            config=config,
            endpoint=_endpoint(step),
            name="checkpoint",
            verify_live_files=False,
        )


def test_saved_protocol_normalizes_through_shared_v1_contract(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        module._v1,
        "_validate_protocol",
        lambda value, *, name: captured.update({"value": value, "name": name}) or value,
    )
    protocol = {field: None for field in module.PROTOCOL_FIELDS}
    protocol.update(
        {
            "name": module.PROTOCOL_NAME,
            "primary_statistic": module.PRIMARY_STATISTIC,
            "endpoint_set_version": module.ENDPOINT_SET_VERSION,
            "admissible_steps": list(module.STEPS),
            "evaluated_step": 12000,
            "nearest_step_substitution": False,
        }
    )
    module._validate_protocol(protocol, step=12000, name="protocol")
    assert captured["value"]["name"] == module._v1.EVALUATOR_PROTOCOL_NAME
    assert captured["value"]["primary_statistic"].endswith("step4000 to step8000")
    assert set(captured["value"]) == set(module.V1_PROTOCOL_FIELDS)
    changed = dict(protocol, evaluated_step=14400)
    with pytest.raises(ValueError, match="saved-endpoint protocol"):
        module._validate_protocol(changed, step=12000, name="protocol")


def test_artifact_policy_preserves_source_marker_and_no_overwrite():
    module._validate_artifact_policy(_policy(), name="policy")
    changed = dict(_policy(), retained_ephemeral_not_promoted_to_permanent=False)
    with pytest.raises(ValueError, match="immutability policy"):
        module._validate_artifact_policy(changed, name="policy")


def test_producer_exactly_binds_validator_evaluator_and_shared_helpers():
    pairing_source = module.inspect.getsourcefile(module._v1.validate_matched_wrong_image_pairing)
    assert pairing_source is not None
    live = {
        "": Path(module.__file__).with_name("vision_alignment_joint_matched_wrong_saved_steps.py"),
        "validator_": Path(module.__file__),
        "scoring_engine_": Path(module.__file__).with_name(
            "vision_alignment_joint_matched_wrong.py"
        ),
        "perception_helper_": Path(module.__file__).with_name(
            "vision_alignment_perception_matched_wrong.py"
        ),
        "bridge_helper_": Path(module.__file__).with_name("vision_alignment_matched_wrong.py"),
        "pairing_implementation_": Path(pairing_source),
        "training_contract_": Path(module.__file__).resolve().parents[1]
        / "train"
        / "Vision-Alignment.py",
    }
    assert all(path.is_file() for path in live.values())
    producer = {
        key: value
        for prefix, path in live.items()
        for key, value in (
            (f"{prefix}path", str(path)),
            (f"{prefix}sha256", module._v1._sha256_file(path)),
        )
    }
    module._validate_producer(producer, name="producer")
    with pytest.raises(ValueError, match="live reviewed implementation"):
        module._validate_producer({**producer, "validator_sha256": "0" * 64}, name="producer")


def test_public_hook_rederives_rows_metrics_pairings_and_native(monkeypatch):
    step = 12000
    native_identity = {
        "manifest_order_sha256": "a" * 64,
        "row_provenance_sha256": "b" * 64,
    }
    protocol = {
        "descriptive_only": True,
        "promotion_eligible": False,
        "native_identity": native_identity,
        "native_row_provenance_sha256": "b" * 64,
    }
    receipt = {field: {} for field in module.RECEIPT_FIELDS - {"format", "version", "status"}}
    receipt.update(
        {
            "format": module.FORMAT,
            "version": module.VERSION,
            "status": module.STATUS,
            "created_at": "2026-08-16T00:00:00+00:00",
            "endpoint": _endpoint(step),
            "checkpoint_config": {"step": step},
            "protocol": protocol,
            "visual_results": {source: {} for source in module._v1.SOURCE_NAMES},
            "blank_results": {source: {} for source in module._v1.BLANK_SOURCE_NAMES},
        }
    )
    receipt["content_sha256"] = module._v1._canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    raw_sha = "f" * 64
    monkeypatch.setattr(module._v1, "_direct_existing_path", lambda path, **_kwargs: Path(path))
    monkeypatch.setattr(
        module._v1,
        "_load_json_bytes",
        lambda _path, **_kwargs: (receipt, raw_sha),
    )
    monkeypatch.setattr(
        module._v1,
        "_validate_checkpoint_config",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(module, "_validate_checkpoint_identity", lambda value, **_kwargs: value)
    monkeypatch.setattr(module._v1, "_validate_load_coverage", lambda value, **_kwargs: value)
    monkeypatch.setattr(module, "_validate_producer", lambda value, **_kwargs: value)
    monkeypatch.setattr(module._v1, "_validate_git", lambda value, **_kwargs: value)
    monkeypatch.setattr(module._v1, "_validate_projection", lambda value, **_kwargs: value)
    monkeypatch.setattr(module._v1, "_validate_source_audit", lambda value, **_kwargs: value)
    monkeypatch.setattr(module._v1, "_validate_registry_domains", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module._v1, "_validate_tokenizer", lambda value, **_kwargs: value)
    monkeypatch.setattr(module, "_validate_protocol", lambda value, **_kwargs: value)
    monkeypatch.setattr(module, "_validate_artifact_policy", lambda value, **_kwargs: value)
    pairings = {source: {"coverage": {}} for source in module._v1.SOURCE_NAMES}
    monkeypatch.setattr(
        module._v1,
        "_validate_pairing_manifest",
        lambda *_args, **_kwargs: (
            {
                "path": str(module.EXPECTED_PAIRING_MANIFEST_PATH),
                "sha256": module.EXPECTED_PAIRING_MANIFEST_SHA256,
                "content_sha256": module.EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256,
            },
            {},
            pairings,
        ),
    )
    calls = {"visual": 0, "blank": 0, "native": 0}

    def visual(*_args, **_kwargs):
        calls["visual"] += 1
        return []

    def blank(*_args, **_kwargs):
        calls["blank"] += 1
        return []

    def native(*_args, **_kwargs):
        calls["native"] += 1
        return [], "a" * 64, "b" * 64, module._v1._canonical_sha256(native_identity)

    monkeypatch.setattr(module._v1, "_validate_visual_rows", visual)
    monkeypatch.setattr(module._v1, "_validate_blank_rows", blank)
    monkeypatch.setattr(module._v1, "_validate_native_rows", native)
    monkeypatch.setattr(module._v1, "_validate_native_identity", lambda value, **_kwargs: value)
    monkeypatch.setattr(module._v1, "matched_wrong_image_pairing_sha256", lambda _pairing: "c" * 64)
    module.validate_evaluator_receipt("receipt.json", raw_sha, step)
    assert calls == {"visual": 8, "blank": 2, "native": 1}

    monkeypatch.setattr(
        module._v1,
        "_validate_pairing_manifest",
        lambda *_args, **_kwargs: (
            {
                "path": str(module.EXPECTED_PAIRING_MANIFEST_PATH),
                "sha256": "0" * 64,
                "content_sha256": module.EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256,
            },
            {},
            pairings,
        ),
    )
    with pytest.raises(ValueError, match="exact shared V1 artifact"):
        module.validate_evaluator_receipt("receipt.json", raw_sha, step)


def test_public_hook_rejects_unsupported_endpoint_before_io():
    with pytest.raises(ValueError, match="must be one of"):
        module.validate_evaluator_receipt("receipt.json", "0" * 64, 14000)
