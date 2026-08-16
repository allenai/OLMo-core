import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_external_academic_legacy_stage1.py"
    )
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_external_academic_legacy_stage1_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def legacy():
    return _load_module()


def test_wrapper_preserves_frozen_evaluator_and_discloses_identity_scope(legacy):
    implementation = legacy._implementation_identity()
    assert implementation["frozen_evaluator"]["files"]["evaluator"]["sha256"] == (
        "29a2c0fa37993e211ef634914aa740119178be3770cc1c178688ca9986e441f0"
    )
    scope = legacy._checkpoint_identity_scope()
    assert "not hashed" in scope["distcp_shards"]
    assert scope["dcp_metadata"] == "full-file SHA-256"
    assert scope["trainer_states"] == "full-file SHA-256 for every rank"


def test_manifest_builder_git_is_separate_from_launch_git(legacy):
    manifest = {
        "content_sha256": legacy.EXPECTED_MANIFEST_CONTENT_SHA256,
        "git": legacy.EXPECTED_MANIFEST_GIT,
        "selection": {"partial": True, "panel_status": "confirmatory"},
    }
    identity = {
        "path": str(legacy.EXPECTED_MANIFEST.resolve()),
        "bytes": 1_672_252,
        "sha256": legacy.EXPECTED_MANIFEST_SHA256,
    }
    reference = legacy._manifest_reference(manifest, identity)
    assert reference["builder_git"] == legacy.EXPECTED_MANIFEST_GIT
    assert "launch_git" not in reference


def test_tokenizer_payload_distinguishes_eval_pin_from_legacy_training(legacy):
    tokenizer = SimpleNamespace(eos_token_id=100257, pad_token_id=100277)
    token_ids = SimpleNamespace(
        as_config_dict=lambda: {
            "im_start_id": 100278,
            "_CLASS_": "olmo_core.nn.vision.molmo2_tokens.Molmo2TokenIds",
        }
    )
    payload = legacy._tokenizer_payload(tokenizer, token_ids)
    assert payload["revision"] == legacy.TOKENIZER_REVISION
    assert payload["fingerprint"] == legacy.TOKENIZER_FINGERPRINT
    assert payload["historical_training_revision_was_pinned"] is False
    assert (
        "pins only the identifier"
        in legacy._interpretation_limits()["training_tokenizer_provenance"]
    )


def _valid_model_load(legacy):
    coverage = {
        "resolved_model_parameter_count": legacy.EXPECTED_MODEL_KEY_COUNT,
        "resolved_checkpoint_keys_sha256": legacy.EXPECTED_MODEL_KEYS_SHA256,
        "remaining_meta_parameter_count": 0,
    }
    return {
        "checkpoint_kind": "multimodal_stage1",
        "api": "MultimodalOLMoDDPTrainModule.load_state_dict_direct",
        "state_dir": "/exact/model_and_optim",
        "eval_only": True,
        "load_optimizer_state": False,
        "process_group": "WORLD",
        "world_size": 8,
        "ep_degree": 8,
        "expert_parallel_path": "sync_1d",
        "checkpoint_load_threads": 8,
        "coverage": coverage,
        "all_rank_completion": [
            {"rank": rank, **coverage} for rank in range(legacy.academic.EP_DEGREE)
        ],
    }


def test_model_load_requires_full_ep8_rank_and_key_coverage(legacy):
    checkpoint = {"state_dir": "/exact/model_and_optim"}
    value = _valid_model_load(legacy)
    assert legacy._validate_model_load(value, checkpoint) == value

    missing_rank = copy.deepcopy(value)
    missing_rank["all_rank_completion"].pop()
    with pytest.raises(ValueError, match="all-rank load completion"):
        legacy._validate_model_load(missing_rank, checkpoint)

    missing_model_key = copy.deepcopy(value)
    missing_model_key["coverage"]["resolved_model_parameter_count"] -= 1
    with pytest.raises(ValueError, match="model-load declaration"):
        legacy._validate_model_load(missing_model_key, checkpoint)


def test_sequence_cap_audit_rederives_every_control_input(legacy, monkeypatch):
    tasks = {}
    for task in legacy.academic.DEFAULT_TASKS:
        options = ("yes", "no") if task in ("ai2d", "a_okvqa_mc") else ()
        input_tokens = 1_246 if options else 1_235
        tasks[task] = {
            "examples": [
                {
                    "example_id": f"{task}-{index}",
                    "image_grid_signature": [14, 14, 14, 14],
                    "controls": {
                        control: {"input_tokens": input_tokens}
                        for control in legacy.academic.CONTROLS
                    },
                    "_options": options,
                }
                for index in range(legacy.academic.DEFAULT_EXAMPLES_PER_TASK)
            ]
        }
    monkeypatch.setattr(
        legacy.academic,
        "_receipt_example_from_row",
        lambda task, row: SimpleNamespace(
            task=task,
            example_id=row["example_id"],
            question="Question?",
            options=row["_options"],
        ),
    )
    monkeypatch.setattr(legacy.academic, "build_image_token_ids", lambda *grid: [100280])
    monkeypatch.setattr(legacy.academic, "_build_mc_prompt", lambda question, options: "mc prompt")
    monkeypatch.setattr(legacy.academic, "_free_answer_prompt", lambda question: "prompt")
    monkeypatch.setattr(
        legacy.academic,
        "document_prompt_ids",
        lambda tokenizer, prompt, image_ids: [1] * (1_246 if "mc" in prompt else 1_235),
    )
    legacy._validate_legacy_sequence_cap(tasks, SimpleNamespace())

    tasks["vqav2"]["examples"][0]["controls"]["blank"]["input_tokens"] = 1_234
    with pytest.raises(ValueError, match="input-token count was not rederived"):
        legacy._validate_legacy_sequence_cap(tasks, SimpleNamespace())


def test_receipt_validator_delegates_row_and_metric_rederivation(legacy, monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = {"selection": {}}
    manifest_reference = {"path": str(manifest_path)}
    checkpoint = {"config": {"path": str(config_path)}, "state_dir": "/state"}
    implementation = {"frozen": True}
    lineage = {"run": "legacy"}
    tokenizer = SimpleNamespace(eos_token_id=100257, pad_token_id=100277)
    token_ids = SimpleNamespace(image_token_ids={100278}, as_config_dict=lambda: {})
    tokenizer_payload = {"tokenizer": "exact"}
    protocol = {"panel": "frozen"}

    monkeypatch.setattr(legacy, "_implementation_identity", lambda: implementation)
    monkeypatch.setattr(
        legacy, "_manifest_reference", lambda manifest, identity: manifest_reference
    )
    monkeypatch.setattr(
        legacy.academic,
        "_file_identity",
        lambda path: {"path": str(path), "bytes": 2, "sha256": "a" * 64},
    )
    monkeypatch.setattr(legacy, "_validate_checkpoint_payload", lambda value: checkpoint)
    monkeypatch.setattr(legacy, "_legacy_lineage", lambda raw: lineage)
    monkeypatch.setattr(legacy, "_validate_model_load", lambda value, checkpoint: value)
    monkeypatch.setattr(
        legacy, "_tokenizer_payload", lambda tokenizer, token_ids: tokenizer_payload
    )
    monkeypatch.setattr(legacy.academic, "_protocol_payload", lambda manifest: protocol)

    payload = {
        "schema_version": legacy.SCHEMA_VERSION,
        "format": legacy.RECEIPT_FORMAT,
        "protocol_name": legacy.PROTOCOL_NAME,
        "created_at": "2026-08-16T00:00:00+00:00",
        "launch_git": {"revision": "1" * 40, "dirty": False},
        "implementation": implementation,
        "manifest": manifest_reference,
        "checkpoint": checkpoint,
        "legacy_stage1_lineage": lineage,
        "model_load": {"loaded": True},
        "artifact_policy": {
            "descriptive_only": True,
            "promotion_eligible": False,
            "historical_reference_comparison_evidence": True,
        },
        "interpretation_limits": legacy._interpretation_limits(),
        "tokenizer": tokenizer_payload,
        "protocol": protocol,
        "tasks": {"tampered": True},
    }
    receipt = legacy.academic._attach_content_sha256(payload)

    def reject_rows(*args, **kwargs):
        raise ValueError("decoded generated tokens differ")

    monkeypatch.setattr(legacy.academic, "_validate_receipt_tasks", reject_rows)
    with pytest.raises(ValueError, match="decoded generated tokens differ"):
        legacy._validate_receipt_payload(
            receipt,
            manifest=manifest,
            loaded=None,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )


@pytest.mark.skipif(
    not Path(
        "/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/"
        "s002-stage1-corrected-clean-32k-b300-20260807/step32000"
    ).exists(),
    reason="canonical legacy Stage-1 checkpoint is not mounted",
)
def test_live_checkpoint_config_trainer_and_dcp_identities(legacy):
    identity = legacy._checkpoint_identity(legacy.EXPECTED_CHECKPOINT)
    assert legacy._validate_checkpoint_payload(identity) == identity
    assert identity["root_file_count"] == 275
    assert identity["trainer_state_summary"]["global_step"] == 32_000

    tampered = copy.deepcopy(identity)
    tampered["trainer_state_summary"]["wandb_run_id"] = "wrong"
    with pytest.raises(ValueError, match="serialized checkpoint identity"):
        legacy._validate_checkpoint_payload(tampered)
