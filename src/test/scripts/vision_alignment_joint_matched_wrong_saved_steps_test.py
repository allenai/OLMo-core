"""Focused contracts for the V2 joint saved-endpoints evaluator."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def module():
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_joint_matched_wrong_saved_steps.py"
    )
    name = "_vision_alignment_joint_matched_wrong_saved_steps_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _args(module):
    return module._parser().parse_args(
        [
            f"--checkpoint={module.EXPECTED_CHECKPOINT_PARENT / 'step12000'}",
            f"--expected-config-sha256={module.EXPECTED_CONFIG_SHA256}",
            f"--pairing-dir={module.EXPECTED_PAIRING_DIR}",
            f"--pairing-manifest={module.EXPECTED_PAIRING_MANIFEST_PATH}",
            ("--expected-pairing-manifest-sha256=" f"{module.EXPECTED_PAIRING_MANIFEST_SHA256}"),
            "--output=/tmp/saved-step.json",
            "--work-dir=/tmp/saved-step-work",
        ]
    )


def _trainer_state(module, *, rank: int, step: int) -> dict[str, Any]:
    return {
        "global_step": step,
        "global_train_tokens_seen": step * 1_048_576,
        "global_train_petaflops": 1.0,
        "max_steps": 16000,
        "data_loader": {
            "batches_processed": step,
            "epoch": 1,
            "seed": 95818,
            "consecutive_data_errors": 0,
            "total_data_errors": module.ENDPOINTS[step].total_data_errors_by_rank[rank],
            "packing_state": {
                "dp_world_size": 16,
                "dp_rank": rank,
                "rank_instances": 8,
                "seq_len": 8192,
                "dataset_names": [
                    "audited_alignment",
                    "cosyn_point",
                    "count_numeric",
                    "native_text_replay",
                    "ocr_document",
                    "pixmo_caption",
                    "pixmo_points_basic",
                    "pixmo_points_high_frequency",
                    "pixmo_transcript",
                ],
            },
        },
        "epoch": 1,
        "world_size": 16,
        "rng": {},
        "callbacks": {
            "wandb": {
                "run_id": module.EXPECTED_WANDB_RUN_ID if rank == 0 else None,
                "step": step,
                "name": module.scoring.EXPECTED_LINEAGE,
                "project": "vision-alignment",
            }
        },
    }


def _protocol(module, step: int) -> dict[str, Any]:
    return module._protocol(
        step=step,
        examples=504,
        pairing_sha256={source: "a" * 64 for source in module.scoring.JOINT_VISUAL_SOURCE_NAMES},
        dp_world_size=8,
        checkpoint_config={
            "training_beaker_image": module.scoring.EXPECTED_BEAKER_IMAGE,
            "training_git_ref": "7e42a7e3064bd944806a5cf5d351ec4f6dc24e42",
            "sha256": module.EXPECTED_CONFIG_SHA256,
        },
        projection={"raw_sha256": module.scoring.EXPECTED_PROJECTION_SHA256},
        source_audit={"fingerprint": module.scoring.EXPECTED_SOURCE_AUDIT_FINGERPRINT},
        native_identity={
            "holdout_fingerprint": (
                "6418aa4e1c1652ff4a9c504a9eed883fd5d346bdbccbda3ceae2575da29a2766"
            ),
            "row_provenance_sha256": "b" * 64,
        },
    )


def _checkpoint(module, step: int) -> dict[str, Any]:
    spec = module.ENDPOINTS[step]
    return {
        "checkpoint_step": step,
        "permanent": not bool(spec.marker["ephemeral"]),
        "checkpoint_marker": dict(spec.marker),
        "checkpoint_marker_sha256": spec.marker_sha256,
        "trainer_state_total_data_errors_by_rank": list(spec.total_data_errors_by_rank),
        "trainer_state_total_data_errors_sum": sum(spec.total_data_errors_by_rank),
    }


def test_args_freeze_ep8_config_and_exact_shared_v1_pairing(module, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    args = _args(module)
    module._validate_args(args)

    args.expected_pairing_manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="exact shared V1"):
        module._validate_args(args)
    args = _args(module)
    args.pairing_manifest = "/tmp/pairing-manifest.json"
    with pytest.raises(ValueError, match="manifest path"):
        module._validate_args(args)
    args = _args(module)
    args.examples = 504
    with pytest.raises(ValueError, match="examples=512"):
        module._validate_args(args)
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="WORLD_SIZE=8"):
        module._validate_args(_args(module))


def test_only_exact_saved_steps_are_admissible_without_substitution(module, monkeypatch, tmp_path):
    checkpoint_parent = tmp_path / "vision-alignment-joint-v1"
    monkeypatch.setattr(module, "EXPECTED_CHECKPOINT_PARENT", checkpoint_parent)
    for step in module.ADMISSIBLE_STEPS:
        assert module._step_from_root(checkpoint_parent / f"step{step}") == step
    for absent_or_wrong_step in (4000, 8000, 10000, 14000, 15000):
        with pytest.raises(ValueError, match="nearest-step substitution is forbidden"):
            module._step_from_root(checkpoint_parent / f"step{absent_or_wrong_step}")
    with pytest.raises(ValueError, match="nearest-step substitution is forbidden"):
        module._step_from_root(checkpoint_parent / "latest")


def test_main_initializes_and_tears_down_exactly_one_runtime(module):
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.main)))
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"_initialize_runtime", "_teardown_runtime"}
    ]
    assert calls.count("_initialize_runtime") == 1
    assert calls.count("_teardown_runtime") == 1


def test_endpoint_storage_classes_markers_and_exact_byte_panels(module):
    expected_errors = (2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    for step in module.ADMISSIBLE_STEPS:
        spec = module.ENDPOINTS[step]
        assert spec.total_data_errors_by_rank == expected_errors
        assert sum(spec.total_data_errors_by_rank) == 3
        assert module.EXPECTED_MODEL_AND_OPTIM_FILES == 257
        assert module.EXPECTED_MODEL_AND_OPTIM_BYTES == 384_970_228_158
        assert module.EXPECTED_CHECKPOINT_REGULAR_FILES == 275
        endpoint = module._endpoint_identity(step)
        assert endpoint["step"] == step
        assert endpoint["nearest_step_substitution"] is False
        if step == 14400:
            assert spec.marker["ephemeral"] is True
            assert endpoint["storage_class"] == "retained_ephemeral"
            assert spec.total_checkpoint_bytes == 384_970_543_908
        else:
            assert spec.marker["ephemeral"] is False
            assert endpoint["storage_class"] == "scheduled_permanent"
            assert spec.total_checkpoint_bytes == 384_970_543_909


@pytest.mark.parametrize("step", [12000, 14400, 16000])
def test_trainer_contract_freezes_real_progress_errors_and_run_identity(module, step):
    leader = module._validate_trainer_state(
        _trainer_state(module, rank=0, step=step), rank=0, step=step
    )
    rank8 = module._validate_trainer_state(
        _trainer_state(module, rank=8, step=step), rank=8, step=step
    )
    assert leader["total_data_errors"] == 2
    assert rank8["total_data_errors"] == 1
    assert leader["wandb_run_id"] == "4gxnu6we"

    bad_error = _trainer_state(module, rank=0, step=step)
    bad_error["data_loader"]["total_data_errors"] = 1
    with pytest.raises(ValueError, match="differs"):
        module._validate_trainer_state(bad_error, rank=0, step=step)
    bad_run = _trainer_state(module, rank=0, step=step)
    bad_run["callbacks"]["wandb"]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="differs"):
        module._validate_trainer_state(bad_run, rank=0, step=step)


def test_v2_protocol_is_per_checkpoint_descriptive_and_step_bound(module):
    protocol = _protocol(module, 14400)
    assert protocol["name"] == module.PROTOCOL_NAME
    assert protocol["primary_statistic"] == module.PRIMARY_STATISTIC
    assert protocol["per_checkpoint_statistic"] == module.PER_CHECKPOINT_STATISTIC
    assert protocol["endpoint_set_version"] == module.ENDPOINT_SET_VERSION
    assert protocol["admissible_steps"] == [12000, 14400, 16000]
    assert protocol["evaluated_step"] == 14400
    assert protocol["nearest_step_substitution"] is False
    assert "step4000 to step8000" not in protocol["primary_statistic"]


def test_v2_receipt_schema_preserves_retained_ephemeral_marker(module):
    step = 14400
    protocol = _protocol(module, step)
    pairing_manifest = {
        "path": str(module.EXPECTED_PAIRING_MANIFEST_PATH),
        "sha256": module.EXPECTED_PAIRING_MANIFEST_SHA256,
        "content_sha256": module.EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256,
    }
    payload = module._receipt_payload(
        step=step,
        checkpoint=_checkpoint(module, step),
        checkpoint_config={"step": step, "sha256": module.EXPECTED_CONFIG_SHA256},
        load_coverage={"complete": True},
        projection={"raw_sha256": module.scoring.EXPECTED_PROJECTION_SHA256},
        source_audit={"fingerprint": module.scoring.EXPECTED_SOURCE_AUDIT_FINGERPRINT},
        tokenizer={"fingerprint": "a" * 64},
        pairing_manifest=pairing_manifest,
        protocol=protocol,
        visual_results={source: {} for source in module.scoring.JOINT_VISUAL_SOURCE_NAMES},
        blank_results={source: {} for source in module.scoring.BLANK_SOURCE_NAMES},
        native_result={"examples": 1000},
        producer={"path": "saved-evaluator"},
        git={"revision": "a" * 40},
    )
    assert set(payload) == {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "git",
        "artifact_policy",
        "endpoint",
        "checkpoint",
        "checkpoint_config",
        "load_coverage",
        "projection",
        "source_audit",
        "tokenizer",
        "pairing_manifest",
        "protocol",
        "visual_results",
        "blank_results",
        "native_result",
        "content_sha256",
    }
    assert payload["version"] == 2
    assert payload["endpoint"]["storage_class"] == "retained_ephemeral"
    assert payload["checkpoint"]["permanent"] is False
    assert payload["checkpoint"]["checkpoint_marker"]["ephemeral"] is True
    assert payload["artifact_policy"]["retained_ephemeral_not_promoted_to_permanent"] is True
    unsigned = dict(payload)
    digest = unsigned.pop("content_sha256")
    assert digest == module.scoring._canonical_sha256(unsigned)

    tampered = _checkpoint(module, step)
    tampered["permanent"] = True
    with pytest.raises(ValueError, match="saved-endpoint contract"):
        module._receipt_payload(
            step=step,
            checkpoint=tampered,
            checkpoint_config={"step": step, "sha256": module.EXPECTED_CONFIG_SHA256},
            load_coverage={},
            projection={},
            source_audit={},
            tokenizer={},
            pairing_manifest=pairing_manifest,
            protocol=protocol,
            visual_results={source: {} for source in module.scoring.JOINT_VISUAL_SOURCE_NAMES},
            blank_results={source: {} for source in module.scoring.BLANK_SOURCE_NAMES},
            native_result={},
            producer={},
            git={},
        )


def test_producer_binds_new_validator_and_frozen_v1_bytes(module):
    engine = Path(module.scoring.__file__).resolve()
    comparator = engine.with_name("vision_alignment_joint_matched_wrong_compare.py")
    assert (
        module.scoring._stable_file_sha256(engine, name="engine")
        == module.EXPECTED_SCORING_ENGINE_SHA256
    )
    assert (
        module.scoring._stable_file_sha256(comparator, name="comparator")
        == module.EXPECTED_V1_COMPARATOR_SHA256
    )
    producer = module._producer_identity()
    assert set(producer) == {
        "path",
        "sha256",
        "validator_path",
        "validator_sha256",
        "scoring_engine_path",
        "scoring_engine_sha256",
        "perception_helper_path",
        "perception_helper_sha256",
        "bridge_helper_path",
        "bridge_helper_sha256",
        "pairing_implementation_path",
        "pairing_implementation_sha256",
        "training_contract_path",
        "training_contract_sha256",
    }
    assert producer["scoring_engine_sha256"] == module.EXPECTED_SCORING_ENGINE_SHA256


def test_candidate_is_externally_validated_before_immutable_publication(
    module, monkeypatch, tmp_path
):
    producer = {"path": "saved-evaluator", "sha256": "a" * 64}
    git = {
        "revision": "b" * 40,
        "dirty": False,
        "status_sha256": "c" * 64,
        "tracked_diff_sha256": "d" * 64,
    }
    payload = {"producer": producer, "git": git, "value": 1}
    observed = {}

    class Validator:
        @staticmethod
        def validate_evaluator_receipt(path, expected_sha256, step, verify_live_checkpoint=False):
            raw = Path(path).read_bytes()
            assert module.scoring.hashlib.sha256(raw).hexdigest() == expected_sha256
            assert step == 12000
            assert verify_live_checkpoint is False
            observed["raw"] = raw

    monkeypatch.setattr(module, "_load_validator_contract", lambda: Validator)
    monkeypatch.setattr(module, "_producer_identity", lambda: producer)
    monkeypatch.setattr(module.scoring.bridge, "_git_identity", lambda: git)
    output = tmp_path / "results" / "step12000.json"
    digest = module._write_validated_receipt(
        output,
        payload,
        work_dir=tmp_path / "work",
        step=12000,
    )
    assert output.read_bytes() == observed["raw"]
    assert digest == module.scoring.hashlib.sha256(observed["raw"]).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._write_validated_receipt(
            output,
            payload,
            work_dir=tmp_path / "work",
            step=12000,
        )
