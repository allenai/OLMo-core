import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint.metadata import TensorProperties


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_external_academic_phase_boundaries.py"
    )
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_external_academic_phase_boundaries_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def boundaries():
    return _load_module()


@pytest.fixture(scope="module")
def live_payloads(boundaries):
    output = {}
    for key, spec in boundaries.BOUNDARIES.items():
        raw_config = boundaries.academic._load_json_strict(spec["checkpoint"] / "config.json")
        output[key] = {
            "checkpoint": boundaries._checkpoint_identity(spec["checkpoint"]),
            "phase_boundary": boundaries._phase_boundary_payload(raw_config, spec),
            "provenance": boundaries._provenance_payload(spec),
        }
    return output


def test_exact_two_checkpoint_admission_and_distinct_policy(boundaries):
    assert set(boundaries.BOUNDARIES) == {"bridge_step500", "perception_step4000"}
    assert boundaries._boundary_spec(boundaries.BRIDGE_CHECKPOINT)["role"] == (
        "bridge_pre_vision_tower_unfreeze"
    )
    assert boundaries._boundary_spec(boundaries.PERCEPTION_CHECKPOINT)["role"] == (
        "perception_treatment_pre_joint_language_model_unfreeze"
    )
    with pytest.raises(ValueError, match="admits only canonical bridge"):
        boundaries._boundary_spec(boundaries.BRIDGE_CHECKPOINT.parent / "step400")
    assert boundaries.RECEIPT_FORMAT == (
        "vision_alignment_external_academic_phase_boundary_receipt"
    )
    assert boundaries._artifact_policy() == {
        "descriptive_only": True,
        "promotion_eligible": False,
        "phase_boundary_comparison_evidence": True,
        "checkpoint_selection_evidence": False,
        "causal_phase_effect_evidence": False,
    }


def test_wrapper_preserves_frozen_evaluator_and_full_identity_scope(boundaries):
    implementation = boundaries._implementation_identity()
    assert implementation["frozen_evaluator"]["files"]["evaluator"]["sha256"] == (
        boundaries.EXPECTED_FROZEN_EVALUATOR_SHA256
    )
    scope = boundaries._checkpoint_identity_scope()
    assert scope["distcp_shards"] == (
        "full-file SHA-256 from approved evidence, verified pre and post"
    )
    assert scope["trainer_states"] == (
        "full-file SHA-256 for every rank plus safe progress projection"
    )
    assert implementation["wrapper_files"]["wrapper"]["sha256"] == (
        "d3d2be012b8d5fc9df7252337b6ca61a44b0822a516593f0a4f02453c38cf4e5"
    )


def test_manifest_builder_git_is_distinct_from_launch_git(boundaries):
    manifest = {
        "content_sha256": boundaries.EXPECTED_MANIFEST_CONTENT_SHA256,
        "git": boundaries.EXPECTED_MANIFEST_GIT,
        "selection": {"partial": True, "panel_status": "confirmatory"},
    }
    identity = {
        "path": str(boundaries.EXPECTED_MANIFEST.resolve()),
        "bytes": 1_672_252,
        "sha256": boundaries.EXPECTED_MANIFEST_SHA256,
    }
    reference = boundaries._manifest_reference(manifest, identity)
    assert reference["builder_git"] == boundaries.EXPECTED_MANIFEST_GIT
    assert "launch_git" not in reference


def test_real_checkpoint_trainer_dcp_and_provenance_identities(boundaries, live_payloads):
    bridge = live_payloads["bridge_step500"]
    perception = live_payloads["perception_step4000"]
    assert bridge["checkpoint"]["trainer_state_summary"]["global_step"] == 500
    assert perception["checkpoint"]["trainer_state_summary"]["global_step"] == 4_000
    assert bridge["checkpoint"]["dcp_key_projection"]["model_tensor_key_count"] == 818
    assert perception["checkpoint"]["dcp_key_projection"]["model_tensor_key_count"] == 818
    assert bridge["checkpoint"]["full_dcp_identity"]["identity_sha256"] == (
        "671c3b0034ee73f0ed74a99e24a9970673db1ec2a9b9c14d8f0facadb6b54e9e"
    )
    assert perception["checkpoint"]["full_dcp_identity"]["identity_sha256"] == (
        "10b81d98490a0ba5e9e209422db235b64b43c16187091ada5674b8079c51848f"
    )
    assert bridge["provenance"]["approval_gate"]["sha256"] == (
        "e6dea8f8f1fd52c2b008e5460854169a893a814bd19da77b1567330116282b6a"
    )
    assert perception["provenance"]["approval_gate"]["sha256"] == (
        "6f110f00becd2f6360fcb0dd8f85fd78e4bcba787087ef44f3159c5f8d486316"
    )


def test_real_phase_transition_freeze_semantics(boundaries, live_payloads):
    bridge = live_payloads["bridge_step500"]["phase_boundary"]
    assert bridge["completed_phase"]["freeze_params"] == [
        "vision.*",
        "lm.embedding_norm.*",
        "lm.blocks.*",
        "lm.lm_head.*",
    ]
    assert bridge["next_phase"]["phase"] == "perception"
    assert bridge["next_phase"]["loads_exact_boundary_checkpoint"] is True
    assert bridge["freeze_transition"]["newly_trainable"] == ["vision.*"]

    perception = live_payloads["perception_step4000"]["phase_boundary"]
    assert perception["completed_phase"]["trainability_arm"] == "treatment"
    assert perception["next_phase"]["phase"] == "joint"
    assert perception["next_phase"]["freeze_params"] == ["lm.lm_head.w_out.weight"]
    assert perception["freeze_transition"]["newly_trainable"] == [
        "lm.embedding_norm.*",
        "lm.blocks.*",
        "lm.lm_head.norm.weight",
    ]
    assert perception["freeze_transition"]["still_frozen"] == ["lm.lm_head.w_out.weight"]


def test_checkpoint_payload_tamper_cases(boundaries, live_payloads):
    original = live_payloads["bridge_step500"]["checkpoint"]
    checkpoint, spec = boundaries._validate_checkpoint_payload(copy.deepcopy(original))
    assert checkpoint["boundary_key"] == "bridge_step500"
    assert spec["checkpoint"] == boundaries.BRIDGE_CHECKPOINT

    config_tamper = copy.deepcopy(original)
    config_tamper["config"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Serialized phase-boundary checkpoint identity"):
        boundaries._validate_checkpoint_payload(config_tamper)

    trainer_tamper = copy.deepcopy(original)
    trainer_tamper["trainer_state_inventory"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="trainer-state identity"):
        boundaries._validate_checkpoint_payload(trainer_tamper)

    dcp_tamper = copy.deepcopy(original)
    dcp_tamper["dcp_key_projection"]["model_tensor_key_count"] -= 1
    with pytest.raises(ValueError, match="Serialized phase-boundary checkpoint identity"):
        boundaries._validate_checkpoint_payload(dcp_tamper)

    evidence_tamper = copy.deepcopy(original)
    evidence_tamper["full_dcp_identity"]["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Serialized phase-boundary checkpoint identity"):
        boundaries._validate_checkpoint_payload(evidence_tamper)


def test_full_dcp_evidence_inventory_tamper_is_rejected(boundaries, monkeypatch):
    spec = boundaries.BOUNDARIES["bridge_step500"]
    real_load = boundaries.academic._load_json_strict
    evidence_path = Path(spec["full_dcp_evidence_path"]).resolve()
    evidence = real_load(evidence_path)
    evidence["checkpoint"]["state_file_inventory"][0]["sha256"] = "0" * 64

    def load_tampered(path):
        if Path(path).resolve() == evidence_path:
            return evidence
        return real_load(path)

    monkeypatch.setattr(boundaries.academic, "_load_json_strict", load_tampered)
    with pytest.raises(ValueError, match="identity is not canonical"):
        boundaries._checkpoint_full_identity(spec, include_inventory=False)


def test_live_full_dcp_verifier_detects_content_tamper(boundaries, monkeypatch, tmp_path):
    root = tmp_path / "checkpoint"
    state_dir = root / "model_and_optim"
    state_dir.mkdir(parents=True)
    metadata = state_dir / ".metadata"
    shard = state_dir / "__0_0.distcp"
    metadata.write_bytes(b"metadata")
    shard.write_bytes(b"checkpoint shard")

    def row(path):
        raw = path.read_bytes()
        return {
            "path": path.relative_to(root).as_posix(),
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    identity = {
        "root": str(root),
        "state_dir": str(state_dir),
        "state_file_inventory": [row(metadata), row(shard)],
    }
    monkeypatch.setattr(
        boundaries,
        "_checkpoint_full_identity",
        lambda spec, include_inventory: identity,
    )
    boundaries._validate_live_full_dcp({}, hash_workers=2)
    shard.write_bytes(b"tampered shard")
    with pytest.raises(ValueError, match="content differs"):
        boundaries._validate_live_full_dcp({}, hash_workers=2)


def _valid_model_load(boundaries, spec):
    coverage = copy.deepcopy(spec["load_coverage"])
    return {
        "checkpoint_kind": "multimodal_stage1",
        "api": "MultimodalOLMoDDPTrainModule.load_state_dict_direct",
        "state_dir": str(spec["checkpoint"] / "model_and_optim"),
        "eval_only": True,
        "load_optimizer_state": False,
        "process_group": "WORLD",
        "world_size": boundaries.academic.EP_DEGREE,
        "ep_degree": boundaries.academic.EP_DEGREE,
        "expert_parallel_path": "sync_1d",
        "checkpoint_load_threads": 8,
        "coverage": coverage,
        "all_rank_completion": [
            {
                "rank": rank,
                "coverage_sha256": coverage["sha256"],
                "remaining_meta_parameter_count": 0,
            }
            for rank in range(boundaries.academic.EP_DEGREE)
        ],
    }


def test_perception_eval_freeze_topology_has_403_key_overlap(boundaries):
    spec = boundaries.BOUNDARIES["perception_step4000"]
    metadata = boundaries.FileSystemReader(spec["checkpoint"] / "model_and_optim").read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    stable_frozen_lm = {key for key in checkpoint_keys if key.startswith("frozen_model.lm.")}
    vision_optimizer_main = {
        key for key in checkpoint_keys if key.startswith("module.vision.") and key.endswith(".main")
    }
    all_optimizer_main = {key for key in checkpoint_keys if key.endswith(".main")}
    coverage = spec["load_coverage"]
    assert len(stable_frozen_lm) == 403
    assert len(vision_optimizer_main) == 403
    assert len(all_optimizer_main) == coverage["eval_state_key_count"] == 415
    assert (
        coverage["frozen_state_key_count"]
        == (len(stable_frozen_lm) + len(vision_optimizer_main))
        == 806
    )
    assert coverage["eval_state_key_count"] + coverage["frozen_state_key_count"] - coverage[
        "prepared_load_key_count"
    ] == len(vision_optimizer_main)
    assert coverage["prepared_load_key_count"] == coverage["model_parameter_count"] == 818
    assert coverage["sha256"] == (
        "6ec333ddba34dce5dd512448a1c235883ee9739c311ec505275144efe44e47c2"
    )


def test_native_coverage_models_eval_frozen_main_key_overlap(boundaries, monkeypatch):
    model_part = torch.nn.Module()
    for name in ("vision", "connector", "lm"):
        model_part.register_parameter(name, torch.nn.Parameter(torch.zeros(1)))
    parameters = dict(model_part.named_parameters())
    keys = {
        "vision": "module.vision.main",
        "connector": "module.connector.main",
        "lm": "frozen_model.lm",
    }
    tensor_metadata = boundaries.TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size([1]),
        chunks=[],
    )
    metadata = SimpleNamespace(state_dict_metadata={key: tensor_metadata for key in keys.values()})
    monkeypatch.setattr(
        boundaries,
        "FileSystemReader",
        lambda state_dir: SimpleNamespace(read_metadata=lambda: metadata),
    )

    class EvalFrozenVisionModule:
        model_parts = [model_part]

        @staticmethod
        def _get_model_state_dict_for_eval_load(metadata):
            return {
                keys["vision"]: parameters["vision"],
                keys["connector"]: parameters["connector"],
            }

        @staticmethod
        def _resolve_model_checkpoint_key(name, checkpoint_keys):
            return keys.get(name)

        @staticmethod
        def _frozen_checkpoint_model_param_state_dict_for_load(checkpoint_keys):
            # Stable frozen LM plus the optimizer-main vision key added because the current
            # academic eval module freezes vision.*.
            return {
                keys["vision"]: parameters["vision"],
                keys["lm"]: parameters["lm"],
            }

        @staticmethod
        def _frozen_checkpoint_param_state_dict_for_load(checkpoint_keys):
            return {
                keys["vision"]: parameters["vision"],
                keys["lm"]: parameters["lm"],
            }

        @staticmethod
        def _persistent_model_buffer_state_dict():
            return {}

    report = boundaries._native_checkpoint_load_coverage(EvalFrozenVisionModule(), Path("/unused"))
    assert report["eval_state_key_count"] == 2
    assert report["frozen_state_key_count"] == 2
    assert report["prepared_load_key_count"] == 3
    assert report["model_parameter_checkpoint_key_count"] == 3


def test_runtime_coverage_mismatch_reports_exact_fields(boundaries):
    spec = boundaries.BOUNDARIES["perception_step4000"]
    wrong = copy.deepcopy(spec["load_coverage"])
    wrong["frozen_state_key_count"] = 403
    with pytest.raises(RuntimeError) as error:
        boundaries._model_load_payload(
            wrong,
            SimpleNamespace(model_parts=[]),
            spec["checkpoint"] / "model_and_optim",
            spec,
            checkpoint_load_threads=8,
        )
    message = str(error.value)
    assert "frozen_state_key_count: expected=806, actual=403" in message
    assert "sha256: expected='6ec333dd" in message
    assert "actual='1ec73fb" in message


@pytest.mark.parametrize("boundary_key", ["bridge_step500", "perception_step4000"])
def test_model_load_requires_exact_full_all_rank_coverage(boundaries, boundary_key):
    spec = boundaries.BOUNDARIES[boundary_key]
    checkpoint = {"state_dir": str(spec["checkpoint"] / "model_and_optim")}
    value = _valid_model_load(boundaries, spec)
    assert boundaries._validate_model_load(value, checkpoint, spec) == value

    missing_rank = copy.deepcopy(value)
    missing_rank["all_rank_completion"].pop()
    with pytest.raises(ValueError, match="all-rank load completion"):
        boundaries._validate_model_load(missing_rank, checkpoint, spec)

    missing_parameter = copy.deepcopy(value)
    missing_parameter["coverage"]["model_parameter_count"] -= 1
    with pytest.raises(ValueError, match="model-load declaration"):
        boundaries._validate_model_load(missing_parameter, checkpoint, spec)


def test_sequence_cap_audit_rederives_every_control_input(boundaries, monkeypatch):
    tasks = {}
    for task in boundaries.academic.DEFAULT_TASKS:
        options = ("yes", "no") if task in ("ai2d", "a_okvqa_mc") else ()
        input_tokens = 1_246 if options else 1_235
        tasks[task] = {
            "examples": [
                {
                    "example_id": f"{task}-{index}",
                    "image_grid_signature": [14, 14, 14, 14],
                    "controls": {
                        control: {"input_tokens": input_tokens}
                        for control in boundaries.academic.CONTROLS
                    },
                    "_options": options,
                }
                for index in range(boundaries.academic.DEFAULT_EXAMPLES_PER_TASK)
            ]
        }
    monkeypatch.setattr(
        boundaries.academic,
        "_receipt_example_from_row",
        lambda task, row: SimpleNamespace(
            task=task,
            example_id=row["example_id"],
            question="Question?",
            options=row["_options"],
        ),
    )
    monkeypatch.setattr(boundaries.academic, "build_image_token_ids", lambda *grid: [100280])
    monkeypatch.setattr(boundaries.academic, "_build_mc_prompt", lambda *args: "mc prompt")
    monkeypatch.setattr(boundaries.academic, "_free_answer_prompt", lambda question: "prompt")
    monkeypatch.setattr(
        boundaries.academic,
        "document_prompt_ids",
        lambda tokenizer, prompt, image_ids: [1] * (1_246 if "mc" in prompt else 1_235),
    )
    boundaries._validate_sequence_cap(tasks, SimpleNamespace())

    tasks["vqav2"]["examples"][0]["controls"]["blank"]["input_tokens"] = 1_234
    with pytest.raises(ValueError, match="input-token count was not rederived"):
        boundaries._validate_sequence_cap(tasks, SimpleNamespace())


def test_receipt_validator_delegates_frozen_row_and_metric_rederivation(
    boundaries, monkeypatch, tmp_path
):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = {"selection": {}}
    manifest_reference = {"path": str(manifest_path)}
    checkpoint = {"config": {"path": str(config_path)}, "state_dir": "/state"}
    spec = boundaries.BOUNDARIES["bridge_step500"]
    implementation = {"frozen": True}
    phase_boundary = {"role": "bridge"}
    provenance = {"approved": True}
    tokenizer = SimpleNamespace(eos_token_id=100257, pad_token_id=100277)
    token_ids = SimpleNamespace(image_token_ids={100278}, as_config_dict=lambda: {})
    tokenizer_payload = {"tokenizer": "exact"}
    protocol = {"panel": "frozen"}

    monkeypatch.setattr(boundaries, "_implementation_identity", lambda: implementation)
    monkeypatch.setattr(
        boundaries, "_manifest_reference", lambda manifest, identity: manifest_reference
    )
    monkeypatch.setattr(
        boundaries.academic,
        "_file_identity",
        lambda path: {"path": str(path), "bytes": 2, "sha256": "a" * 64},
    )
    monkeypatch.setattr(
        boundaries, "_validate_checkpoint_payload", lambda value: (checkpoint, spec)
    )
    monkeypatch.setattr(
        boundaries, "_phase_boundary_payload", lambda raw, boundary_spec: phase_boundary
    )
    monkeypatch.setattr(boundaries, "_provenance_payload", lambda boundary_spec: provenance)
    monkeypatch.setattr(boundaries, "_validate_model_load", lambda *args: args[0])
    monkeypatch.setattr(
        boundaries, "_tokenizer_payload", lambda tokenizer, token_ids: tokenizer_payload
    )
    monkeypatch.setattr(boundaries.academic, "_protocol_payload", lambda manifest: protocol)
    monkeypatch.setattr(boundaries, "_validate_sequence_cap", lambda tasks, tokenizer: None)

    payload = {
        "schema_version": boundaries.SCHEMA_VERSION,
        "format": boundaries.RECEIPT_FORMAT,
        "protocol_name": boundaries.PROTOCOL_NAME,
        "created_at": "2026-08-17T00:00:00+00:00",
        "launch_git": {"revision": "1" * 40, "dirty": False},
        "implementation": implementation,
        "manifest": manifest_reference,
        "checkpoint": checkpoint,
        "phase_boundary": phase_boundary,
        "provenance": provenance,
        "model_load": {"loaded": True},
        "artifact_policy": boundaries._artifact_policy(),
        "interpretation_limits": boundaries._interpretation_limits(spec),
        "tokenizer": tokenizer_payload,
        "protocol": protocol,
        "tasks": {"tampered": True},
    }
    receipt = boundaries.academic._attach_content_sha256(payload)

    content_tamper = copy.deepcopy(receipt)
    content_tamper["artifact_policy"]["promotion_eligible"] = True
    with pytest.raises(ValueError, match="content SHA-256 differs"):
        boundaries._validate_receipt_payload(
            content_tamper,
            manifest=manifest,
            loaded=None,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )

    def reject_rows(*args, **kwargs):
        raise ValueError("decoded generated tokens differ")

    monkeypatch.setattr(boundaries.academic, "_validate_receipt_tasks", reject_rows)
    with pytest.raises(ValueError, match="decoded generated tokens differ"):
        boundaries._validate_receipt_payload(
            receipt,
            manifest=manifest,
            loaded=None,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )


def test_public_validator_rejects_bad_external_raw_sha_before_dependencies(boundaries, tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"not": "a receipt"}), encoding="utf-8")
    with pytest.raises(ValueError, match="raw SHA-256 differs"):
        boundaries.validate_phase_boundary_receipt(receipt, "0" * 64)
    with pytest.raises(ValueError, match="must be lowercase hex"):
        boundaries.validate_phase_boundary_receipt(receipt, "invalid")


def test_publication_is_write_once(boundaries, tmp_path):
    output = tmp_path / "receipt.json"
    boundaries.academic._write_json_no_overwrite(output, {"first": True})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        boundaries.academic._write_json_no_overwrite(output, {"second": True})


def test_cli_exposes_evaluate_and_strict_public_validation(boundaries):
    parser = boundaries._parser()
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--manifest",
            str(boundaries.EXPECTED_MANIFEST),
            "--checkpoint",
            str(boundaries.BRIDGE_CHECKPOINT),
            "--output",
            "/tmp/new-receipt.json",
        ]
    )
    assert evaluate.checkpoint_hash_workers == 16
    assert evaluate.checkpoint_load_threads == 8
    validate = parser.parse_args(
        ["validate-receipt", "--receipt", "/tmp/r.json", "--expected-sha256", "1" * 64]
    )
    assert validate.checkpoint_hash_workers == 16
