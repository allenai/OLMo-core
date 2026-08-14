"""Contracts for the separate vision-alignment continued-pretraining recipe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch.nn as nn

from olmo_core.data.multimodal.vision_alignment_sources import serialized_example_sha256
from olmo_core.nn.vision import Molmo2TokenIds


def _load_module():
    path = Path(__file__).parents[2] / "scripts" / "train" / "Vision-Alignment.py"
    spec = importlib.util.spec_from_file_location("vision_alignment_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_pixmo_builder():
    path = Path(__file__).parents[2] / "scripts" / "data" / "build_vision_alignment_pixmo_cap.py"
    spec = importlib.util.spec_from_file_location("vision_alignment_pixmo_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_canonical_pixmo_source_policy(monkeypatch, vision_alignment, source_path, splits):
    monkeypatch.setattr(
        vision_alignment,
        "_CANONICAL_PIXMO_SOURCE_DATASET",
        str(Path(source_path).resolve()),
    )
    monkeypatch.setattr(
        vision_alignment,
        "_CANONICAL_PIXMO_SOURCE_SPLITS",
        {
            split: (entry["dataset_fingerprint"], entry["examples"])
            for split, entry in splits.items()
        },
    )


def _canonical_policy_config(vision_alignment, phase):
    phase = vision_alignment.VisionAlignmentPhase(phase)
    mixture = vision_alignment.VisionAlignmentMixtureConfig(phase=phase.value)
    mixture.mean_loss_weight = {source: 1.0 for source in mixture.resolved_targets()}
    data = vision_alignment.VisionAlignmentDataConfig(mixture=mixture)
    return SimpleNamespace(phase=phase, data=data)


def _joint_audit_fixture(tmp_path, vision_alignment):
    projection_path = (tmp_path / "vision-alignment-joint-visual-projection.json").resolve()
    projection_path.write_bytes(b"joint projection\n")
    parent_provenance_path = (tmp_path / "vision-alignment-perception-provenance.json").resolve()
    parent_provenance_path.write_bytes(b"perception provenance\n")
    native_path = (tmp_path / "native-train.json").resolve()
    native_path.write_bytes(b"native train manifest\n")
    receipt_path = (tmp_path / "native-verification.json").resolve()
    receipt_path.write_bytes(b"native verification receipt\n")
    catalog_path = (tmp_path / "vision-alignment-joint-source-catalog.json").resolve()
    catalog_path.write_bytes(b"joint source catalog\n")

    source_spec = SimpleNamespace(
        as_canonical_dict=lambda: {
            "phase": "joint",
            "sequence_length": vision_alignment._JOINT_SEQUENCE_LENGTH,
            "parent_perception_preprocessing_sha256": "a" * 64,
        }
    )
    projection = SimpleNamespace(
        path=projection_path,
        raw_sha256=vision_alignment._sha256_file(projection_path),
        content_sha256="b" * 64,
        source_spec=source_spec,
        source_spec_sha256="c" * 64,
        parent_provenance=SimpleNamespace(
            path=parent_provenance_path,
            raw_sha256=vision_alignment._sha256_file(parent_provenance_path),
        ),
    )
    mixture = vision_alignment.VisionAlignmentMixtureConfig(phase="joint")
    mixture.mean_loss_weight = {source_name: 1.0 for source_name in mixture.resolved_targets()}
    data = vision_alignment.VisionAlignmentDataConfig(mixture=mixture)
    data.sequence_length = vision_alignment._JOINT_SEQUENCE_LENGTH
    data.native_text_replay_fingerprint = "d" * 64
    data.native_text_replay = SimpleNamespace(
        manifest_path=str(native_path),
        expected_fingerprint=data.native_text_replay_fingerprint,
        verification_receipt_path=str(receipt_path),
        expected_verification_receipt_sha256=vision_alignment._sha256_file(receipt_path),
    )
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.joint,
        data=data,
    )

    inputs = {}
    summaries = {}
    input_descriptors = []
    for source_name in vision_alignment._JOINT_SOURCE_NAMES:
        kind = "native_text_replay" if source_name == "native_text_replay" else "visual"
        epochs = (
            vision_alignment._JOINT_NATIVE_PROBE_EPOCHS
            if kind == "native_text_replay"
            else vision_alignment._JOINT_VISUAL_PROBE_EPOCHS
        )
        unique_indices = (
            vision_alignment._JOINT_NATIVE_PROBE_INDICES
            if kind == "native_text_replay"
            else vision_alignment._JOINT_VISUAL_PROBE_INDICES
        )
        dataset_size = 2048
        dataset_fingerprint = hashlib.sha256(source_name.encode()).hexdigest()
        probe_indices = list(
            vision_alignment.select_deterministic_probe_indices(
                dataset_size,
                unique_indices,
                seed=vision_alignment._JOINT_PROBE_SEED,
                dataset_fingerprint=dataset_fingerprint,
            )
        )
        rows = unique_indices * len(epochs)
        row_hashes = [
            hashlib.sha256(f"{source_name}-{index}".encode()).hexdigest() for index in range(rows)
        ]
        source_path = tmp_path / f"{source_name}.jsonl"
        source_path.write_bytes(f"{source_name} pinned rows\n".encode())
        image_sha = (
            vision_alignment._canonical_sha256([])
            if kind == "native_text_replay"
            else hashlib.sha256(f"{source_name}-images".encode()).hexdigest()
        )
        maximum_length = (
            vision_alignment._JOINT_SEQUENCE_LENGTH if kind == "native_text_replay" else 64
        )
        source = {
            "name": source_name,
            "kind": kind,
            "format": "jsonl",
            "path": source_path.name,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_size": dataset_size,
            "sha256": vision_alignment._sha256_file(source_path),
            "probe_epochs": list(epochs),
            "probe_indices": probe_indices,
            "probe_indices_sha256": vision_alignment._canonical_sha256(probe_indices),
            "serialized_row_hashes_sha256": vision_alignment._canonical_sha256(row_hashes),
            "probe_image_content_sha256": image_sha,
            "max_observed_sequence_length": maximum_length,
            "truncated_rows": 0,
            "serialized_row_hashes": row_hashes,
        }
        inputs[source_name] = source
        token_length = maximum_length
        summaries[source_name] = {
            "examples": {"seen": rows, "valid": rows, "errors": 0},
            "raw_input_tokens": {
                "total": rows * token_length,
                "mean": float(token_length),
                "min": token_length,
                "max": token_length,
            },
            "positive_supervised_tokens": {
                "total": rows,
                "mean": 1.0,
                "min": 1,
                "max": 1,
            },
            "summed_loss_weight": {
                "total": float(rows),
                "mean": 1.0,
                "min": 1.0,
                "max": 1.0,
            },
            "mean_sum_loss_masks": 1.0,
            "image_crops": {
                "total": 0,
                "mean": 0.0,
                "min": 0,
                "max": 0,
            },
            "truncated_examples": 0,
            "zero_loss_examples": 0,
            "error_samples": [],
        }
        input_descriptors.append(
            {
                "name": source_name,
                "kind": kind,
                "sha256": source["sha256"],
                "dataset_fingerprint": dataset_fingerprint,
                "probe_indices_sha256": source["probe_indices_sha256"],
                "probe_epochs": list(epochs),
                "serialized_row_hashes_sha256": source["serialized_row_hashes_sha256"],
                "probe_image_content_sha256": image_sha,
                "max_observed_sequence_length": maximum_length,
                "truncated_rows": 0,
            }
        )

    preprocessing = {
        "visual": source_spec.as_canonical_dict(),
        "native_text_replay_fingerprint": data.native_text_replay_fingerprint,
    }
    targets = mixture.resolved_targets()
    sampling = mixture.sampling_weights()
    audit = {
        "format": vision_alignment._JOINT_AUDIT_FORMAT,
        "version": vision_alignment._JOINT_AUDIT_VERSION,
        "status": "ok",
        "phase": "joint",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "source_catalog_version": vision_alignment.VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION,
        "auditor_implementation": {
            "path": vision_alignment._JOINT_AUDITOR_IMPLEMENTATION,
            "sha256": vision_alignment._sha256_file(
                Path(vision_alignment.__file__).parents[3]
                / vision_alignment._JOINT_AUDITOR_IMPLEMENTATION
            ),
        },
        "shared_auditor_sha256": vision_alignment._sha256_file(
            Path(vision_alignment.__file__).parents[3]
            / vision_alignment._JOINT_SHARED_AUDITOR_IMPLEMENTATION
        ),
        "catalog_path": str(catalog_path),
        "catalog_sha256": vision_alignment._sha256_file(catalog_path),
        "catalog_content_sha256": "e" * 64,
        "input_content_sha256": vision_alignment._canonical_sha256(input_descriptors),
        "source_registry_version": vision_alignment.VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": vision_alignment.joint_alignment_runtime_registry_sha256(),
        "source_implementation_inventory": vision_alignment.joint_alignment_runtime_implementation_inventory(),
        "exporter_implementation": {
            "path": vision_alignment._JOINT_EXPORTER_IMPLEMENTATION,
            "sha256": vision_alignment._sha256_file(
                Path(vision_alignment.__file__).parents[3]
                / vision_alignment._JOINT_EXPORTER_IMPLEMENTATION
            ),
        },
        "visual_projection": {
            "path": str(projection.path),
            "raw_sha256": projection.raw_sha256,
            "content_sha256": projection.content_sha256,
        },
        "native_train_manifest": {
            "path": str(native_path),
            "raw_sha256": vision_alignment._sha256_file(native_path),
            "content_fingerprint": data.native_text_replay_fingerprint,
        },
        "native_verification_receipt": {
            "path": str(receipt_path),
            "sha256": vision_alignment._sha256_file(receipt_path),
        },
        "preprocessing": preprocessing,
        "preprocessing_sha256": vision_alignment._canonical_sha256(preprocessing),
        "probe": {
            "format": vision_alignment._JOINT_PROBE_FORMAT,
            "version": vision_alignment._JOINT_PROBE_VERSION,
            "selection_algorithm": vision_alignment.VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
            "seed": vision_alignment._JOINT_PROBE_SEED,
            "visual": {
                "unique_indices": vision_alignment._JOINT_VISUAL_PROBE_INDICES,
                "epochs": list(vision_alignment._JOINT_VISUAL_PROBE_EPOCHS),
                "rows_per_source": vision_alignment._JOINT_VISUAL_PROBE_INDICES
                * len(vision_alignment._JOINT_VISUAL_PROBE_EPOCHS),
            },
            "native_text_replay": {
                "unique_indices": vision_alignment._JOINT_NATIVE_PROBE_INDICES,
                "epochs": list(vision_alignment._JOINT_NATIVE_PROBE_EPOCHS),
                "rows_per_source": vision_alignment._JOINT_NATIVE_PROBE_INDICES
                * len(vision_alignment._JOINT_NATIVE_PROBE_EPOCHS),
            },
            "sequence_length": vision_alignment._JOINT_SEQUENCE_LENGTH,
            "truncation_policy": "forbid-raw-length-above-sequence-length-v1",
        },
        "inputs": inputs,
        "target_loss_mass": targets,
        "sources": summaries,
        "mean_loss_weight": mixture.mean_loss_weight,
        "sampling_probabilities": sampling,
        "expected_loss_mass": vision_alignment.expected_loss_mass(
            sampling, mixture.mean_loss_weight
        ),
        "failures": [],
    }
    audit["fingerprint"] = vision_alignment._canonical_sha256(audit)
    audit_path = tmp_path / "joint-audit.json"
    audit_path.write_text(json.dumps(audit))
    data.source_audit_path = str(audit_path)
    data.source_audit_fingerprint = audit["fingerprint"]
    return config, audit, audit_path, projection


def _rewrite_joint_audit(vision_alignment, config, audit, audit_path):
    audit["fingerprint"] = vision_alignment._canonical_sha256(
        {key: value for key, value in audit.items() if key != "fingerprint"}
    )
    audit_path.write_text(json.dumps(audit))
    config.data.source_audit_fingerprint = audit["fingerprint"]


def test_phase_selector_is_required_before_derived_defaults():
    vision_alignment = _load_module()

    assert vision_alignment._extract_phase(["--phase=bridge"]) == "bridge"
    assert vision_alignment._extract_phase(["--phase=joint"]) == "joint"
    with pytest.raises(ValueError, match="exactly one"):
        vision_alignment._extract_phase([])
    with pytest.raises(ValueError, match="exactly one"):
        vision_alignment._extract_phase(["--phase=bridge", "--phase=joint"])
    with pytest.raises(ValueError, match="Unknown"):
        vision_alignment._extract_phase(["--phase=sft"])


@pytest.mark.parametrize(
    "override",
    [
        "--launch.cmd=[]",
        "--trainer.no_checkpoints=true",
        "--trainer.load_strategy=never",
        "--model.lm.n_layers=1",
        "--train_module.freeze_params=[]",
        "--train_module.optim.lr=1e-6",
        "--train_module.scheduler.group_name_field=wrong",
    ],
)
def test_structural_and_checkpoint_bypass_overrides_are_rejected(override):
    vision_alignment = _load_module()

    with pytest.raises(ValueError, match="audited override surface"):
        vision_alignment._validate_override_surface(["--phase=bridge", override])


@pytest.mark.parametrize("run_name", ["../escape", "nested/path", ".", "UPPERCASE"])
def test_run_name_cannot_escape_or_alias_checkpoint_root(run_name):
    vision_alignment = _load_module()

    with pytest.raises(ValueError, match="run names must match"):
        vision_alignment._validate_run_name(run_name)


def test_git_provenance_requires_owned_branch_for_local_launch(monkeypatch):
    vision_alignment = _load_module()
    config = SimpleNamespace(
        launch=SimpleNamespace(
            git=SimpleNamespace(branch="vision-moe", ref="a" * 40),
        )
    )
    monkeypatch.setattr(vision_alignment, "is_running_in_beaker_batch_job", lambda: False)

    vision_alignment._validate_git_provenance(config, runtime=False)

    config.launch.git.branch = "main"
    with pytest.raises(ValueError, match="user-owned vision-moe branch"):
        vision_alignment._validate_git_provenance(config, runtime=False)


def test_runtime_launch_imports_the_exact_gantry_checkout_source():
    vision_alignment = _load_module()
    launch = vision_alignment.BeakerLaunchConfig(
        name="vision-alignment-runtime-test",
        cmd=["true"],
        env_vars=[
            vision_alignment.BeakerEnvVar(name="PYTHONPATH", value="/stale/source"),
            vision_alignment.BeakerEnvVar(name="EXPLICIT_SETTING", value="kept"),
        ],
    )

    vision_alignment._configure_launch_runtime(launch)

    realized_env = dict(launch._get_env_vars())
    assert realized_env["PYTHONPATH"] == "/gantry-runtime/src"
    assert realized_env["EXPLICIT_SETTING"] == "kept"
    assert [item.name for item in launch.env_vars].count("PYTHONPATH") == 1


def test_git_provenance_accepts_matching_detached_beaker_checkout(monkeypatch):
    vision_alignment = _load_module()
    git_ref = "b" * 40
    config = SimpleNamespace(
        launch=SimpleNamespace(
            git=SimpleNamespace(branch=None, ref=git_ref),
        )
    )
    monkeypatch.setattr(vision_alignment, "is_running_in_beaker_batch_job", lambda: True)
    monkeypatch.setenv("GIT_BRANCH", "vision-moe")
    monkeypatch.setenv("GIT_REF", git_ref)

    vision_alignment._validate_git_provenance(config, runtime=True)


@pytest.mark.parametrize(
    ("branch", "runtime_ref", "checkout_branch", "checkout_ref", "message"),
    [
        ("main", "c" * 40, None, "c" * 40, "runtime metadata"),
        ("vision-moe", "short", None, "c" * 40, "exact GIT_REF"),
        ("vision-moe", "c" * 40, None, "d" * 40, "detached checkout"),
        ("vision-moe", "c" * 40, "main", "c" * 40, "unexpected active branch"),
    ],
)
def test_git_provenance_rejects_invalid_beaker_metadata(
    monkeypatch, branch, runtime_ref, checkout_branch, checkout_ref, message
):
    vision_alignment = _load_module()
    config = SimpleNamespace(
        launch=SimpleNamespace(
            git=SimpleNamespace(branch=checkout_branch, ref=checkout_ref),
        )
    )
    monkeypatch.setattr(vision_alignment, "is_running_in_beaker_batch_job", lambda: True)
    monkeypatch.setenv("GIT_BRANCH", branch)
    monkeypatch.setenv("GIT_REF", runtime_ref)

    with pytest.raises(ValueError, match=message):
        vision_alignment._validate_git_provenance(config, runtime=True)


def test_git_provenance_rejects_train_worker_outside_beaker(monkeypatch):
    vision_alignment = _load_module()
    git_ref = "e" * 40
    config = SimpleNamespace(
        launch=SimpleNamespace(
            git=SimpleNamespace(branch=None, ref=git_ref),
        )
    )
    monkeypatch.setattr(vision_alignment, "is_running_in_beaker_batch_job", lambda: False)

    with pytest.raises(ValueError, match="inside a Beaker batch job"):
        vision_alignment._validate_git_provenance(config, runtime=True)


@pytest.mark.parametrize(
    (
        "phase",
        "freeze_params",
        "lm_lr",
        "vision_lr",
        "microbatch_instances",
        "connector_t_max",
    ),
    [
        (
            "bridge",
            ["vision.*", "lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"],
            0.0,
            0.0,
            4,
            250,
        ),
        (
            "perception",
            ["lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"],
            0.0,
            3e-6,
            4,
            None,
        ),
        ("joint", ["lm.lm_head.w_out.weight"], 1e-6, 2e-6, 1, None),
    ],
)
def test_phase_trainability_and_lr_contract(
    phase, freeze_params, lm_lr, vision_lr, microbatch_instances, connector_t_max
):
    vision_alignment = _load_module()
    policy = vision_alignment._PHASE_POLICIES[vision_alignment.VisionAlignmentPhase(phase)]
    image_rows = [100, 101, 102, 103, 104, 105]

    config = vision_alignment._build_train_module_config(policy, image_rows)

    assert config.freeze_params == freeze_params
    assert config.train_embedding_rows == image_rows
    assert config.optim.lr == (lm_lr if lm_lr > 0 else policy.connector_lr)
    assert config.rank_microbatch_size == microbatch_instances * policy.sequence_length
    groups = {tuple(group.params): group.opts for group in config.optim.group_overrides}
    assert groups[("*lm.embeddings.weight",)]["lr"] == policy.connector_lr
    assert groups[("*connector.*",)]["lr"] == policy.connector_lr
    assert groups[("*vision.*",)]["lr"] == vision_lr
    assert config.scheduler.schedulers["connector"].t_max == connector_t_max
    assert config.scheduler.schedulers["vision"].t_max is None
    assert config.scheduler.default.t_max is None
    if phase == "bridge":
        connector_scheduler = config.scheduler.schedulers["connector"]
        assert connector_scheduler.get_lr(policy.connector_lr, 200, 500) == pytest.approx(6.5e-5)
        assert connector_scheduler.get_lr(policy.connector_lr, 250, 500) == pytest.approx(2e-5)
        assert connector_scheduler.get_lr(policy.connector_lr, 500, 500) == pytest.approx(2e-5)


@pytest.mark.parametrize(
    ("phase", "rank_batch_instances"),
    [("bridge", 4), ("perception", 4), ("joint", 1)],
)
def test_intrinsic_eval_uses_phase_supported_rank_batch(phase, rank_batch_instances):
    vision_alignment = _load_module()
    policy = vision_alignment._PHASE_POLICIES[vision_alignment.VisionAlignmentPhase(phase)]

    evaluation = vision_alignment._build_evaluation_config(policy)

    assert evaluation.rank_batch_instances == rank_batch_instances
    assert evaluation.rank_batch_instances == policy.rank_microbatch_instances


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_group",
        "zero_connector_lr",
        "disable_distributed",
        "reroute_scheduler",
        "fixed_scheduler_horizon",
        "warmup_not_shorter_than_duration",
        "horizon_longer_than_duration",
    ],
)
def test_optimizer_scheduler_contract_rejects_unsafe_overrides(mutation):
    vision_alignment = _load_module()
    policy = vision_alignment._PHASE_POLICIES[vision_alignment.VisionAlignmentPhase.bridge]
    train_module = vision_alignment._build_train_module_config(
        policy, [100, 101, 102, 103, 104, 105]
    )
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.bridge,
        perception_trainability_arm=vision_alignment.PerceptionTrainabilityArm.treatment,
        train_module=train_module,
        trainer=SimpleNamespace(max_duration=vision_alignment.Duration.steps(policy.max_steps)),
        data=SimpleNamespace(allow_unpinned_synthetic_smoke=False),
    )

    if mutation == "remove_group":
        train_module.optim.group_overrides = train_module.optim.group_overrides[:-1]
    elif mutation == "zero_connector_lr":
        train_module.optim.group_overrides[0].opts["lr"] = 0.0
    elif mutation == "disable_distributed":
        train_module.optim.use_distributed = False
    elif mutation == "reroute_scheduler":
        train_module.scheduler.group_name_field = "wrong"
    elif mutation == "fixed_scheduler_horizon":
        train_module.scheduler.schedulers["connector"].t_max = policy.max_steps
    elif mutation == "warmup_not_shorter_than_duration":
        config.trainer.max_duration = vision_alignment.Duration.steps(policy.connector_warmup)
    elif mutation == "horizon_longer_than_duration":
        assert policy.connector_t_max is not None
        config.trainer.max_duration = vision_alignment.Duration.steps(policy.connector_t_max - 1)
    else:  # pragma: no cover - parameter table is exhaustive.
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        vision_alignment._validate_optimizer_scheduler_contract(config, policy)


def test_perception_control_changes_only_vision_trainability():
    vision_alignment = _load_module()
    policy = vision_alignment._PHASE_POLICIES[vision_alignment.VisionAlignmentPhase.perception]
    image_rows = [100, 101, 102, 103, 104, 105]

    treatment_module = vision_alignment._build_train_module_config(policy, image_rows)
    treatment = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.perception,
        perception_trainability_arm=vision_alignment.PerceptionTrainabilityArm.treatment,
        train_module=treatment_module,
        trainer=SimpleNamespace(max_duration=vision_alignment.Duration.steps(policy.max_steps)),
        data=SimpleNamespace(allow_unpinned_synthetic_smoke=False),
    )
    vision_alignment._validate_optimizer_scheduler_contract(treatment, policy)
    assert treatment_module.freeze_params == list(policy.freeze_params)
    assert treatment_module.optim.group_overrides[2].opts["lr"] == policy.vision_lr

    control_module = vision_alignment._build_train_module_config(policy, image_rows)
    control_module.freeze_params = ["vision.*", *(control_module.freeze_params or [])]
    control_module.optim.group_overrides[2].opts["lr"] = 0.0
    control = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.perception,
        perception_trainability_arm=(
            vision_alignment.PerceptionTrainabilityArm.frozen_vision_control
        ),
        train_module=control_module,
        trainer=SimpleNamespace(max_duration=vision_alignment.Duration.steps(policy.max_steps)),
        data=SimpleNamespace(allow_unpinned_synthetic_smoke=False),
    )
    vision_alignment._validate_optimizer_scheduler_contract(control, policy)
    assert control_module.freeze_params == ["vision.*", *policy.freeze_params]
    assert control_module.optim.group_overrides[2].opts["lr"] == 0.0


def test_vision_alignment_uses_a_dedicated_checkpoint_namespace():
    vision_alignment = _load_module()

    root = Path(vision_alignment.VISION_ALIGNMENT_ROOT)

    assert root == Path(vision_alignment.EXPERIMENT_ROOT) / "vision-alignment"
    assert root / "checkpoints" != Path(vision_alignment.EXPERIMENT_ROOT) / "checkpoints"


def test_bridge_uses_stable_document_caption_and_transcript_formats():
    vision_alignment = _load_module()
    token_ids = Molmo2TokenIds()
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.bridge,
        perception_trainability_arm=vision_alignment.PerceptionTrainabilityArm.treatment,
        data=vision_alignment.VisionAlignmentDataConfig(),
        evaluation=vision_alignment.VisionAlignmentEvalConfig(),
    )

    caption = vision_alignment._visual_dataset_config(config, token_ids, "pixmo_caption")
    transcript = vision_alignment._visual_dataset_config(config, token_ids, "pixmo_transcript")

    assert caption.message_format == "document"
    assert caption.mode == "caption"
    assert caption.fixed_prompt == "Description:"
    assert caption.style_length_conditioning is False
    assert transcript.message_format == "document"
    assert transcript.mode == "transcript"
    assert transcript.require_transcript is True
    assert transcript.fixed_prompt == "Transcript:"
    assert transcript.style_length_conditioning is False


@pytest.mark.parametrize(
    ("phase", "expected_targets"),
    [
        ("bridge", {"pixmo_caption": 0.70, "pixmo_transcript": 0.30}),
        (
            "perception",
            {
                "pixmo_caption": 0.45,
                "pixmo_transcript": 0.20,
                "pixmo_points_basic": 0.10,
                "pixmo_points_high_frequency": 0.02,
                "cosyn_point": 0.03,
                "ocr_document": 0.10,
                "scalar_count": 0.05,
                "audited_alignment": 0.05,
            },
        ),
        (
            "joint",
            {
                "native_text_replay": 0.35,
                "pixmo_caption": 0.28,
                "pixmo_transcript": 0.12,
                "pixmo_points_basic": 0.05,
                "pixmo_points_high_frequency": 0.01,
                "cosyn_point": 0.02,
                "ocr_document": 0.08,
                "count_numeric": 0.04,
                "audited_alignment": 0.05,
            },
        ),
    ],
)
def test_canonical_data_policy_pins_phase_sources_and_ratios_but_allows_artifact_paths(
    phase, expected_targets
):
    vision_alignment = _load_module()
    config = _canonical_policy_config(vision_alignment, phase)
    config.data.pixmo_cap_path = f"/pinned/data/{phase}"
    config.data.source_audit_path = f"/pinned/audits/{phase}.json"
    config.data.source_audit_fingerprint = "a" * 64
    config.data.prefetch_workers = 0

    vision_alignment._validate_canonical_data_policy(config)

    assert config.data.mixture.resolved_targets() == expected_targets


@pytest.mark.parametrize(
    ("field_name", "drifted_value"),
    [
        ("message_format", "olmo3_chat"),
        ("loss_token_weighting", "none"),
        ("caption_prompt", "Caption:"),
        ("transcript_prompt", "Audio transcript:"),
        ("max_crops", 4),
        ("pack_sequences", False),
        ("pack_buffer_size", 12),
        ("pack_max_crops", 8),
    ],
)
def test_canonical_data_policy_rejects_structural_data_overrides(field_name, drifted_value):
    vision_alignment = _load_module()
    config = _canonical_policy_config(vision_alignment, "bridge")
    setattr(config.data, field_name, drifted_value)

    with pytest.raises(ValueError, match=field_name):
        vision_alignment._validate_canonical_data_policy(config)


def test_canonical_data_policy_rejects_source_and_target_ratio_drift():
    vision_alignment = _load_module()
    config = _canonical_policy_config(vision_alignment, "bridge")

    config.data.mixture.target_loss_mass = {
        "pixmo_caption": 0.69,
        "pixmo_transcript": 0.30,
        "unapproved_source": 0.01,
    }
    with pytest.raises(ValueError, match="canonical mixture sources"):
        vision_alignment._validate_canonical_data_policy(config)

    config = _canonical_policy_config(vision_alignment, "bridge")
    config.data.mixture.target_loss_mass = {
        "pixmo_caption": 0.60,
        "pixmo_transcript": 0.40,
    }
    with pytest.raises(ValueError, match="canonical target loss-mass ratios"):
        vision_alignment._validate_canonical_data_policy(config)


def test_canonical_data_policy_rejects_calibration_source_drift():
    vision_alignment = _load_module()
    config = _canonical_policy_config(vision_alignment, "bridge")
    config.data.mixture.mean_loss_weight.pop("pixmo_transcript")

    with pytest.raises(ValueError, match="calibration.*canonical phase sources"):
        vision_alignment._validate_canonical_data_policy(config)


def test_synthetic_smoke_uses_the_same_canonical_bridge_policy():
    vision_alignment = _load_module()
    config = _canonical_policy_config(vision_alignment, "bridge")
    config.data.pixmo_cap_path = "synthetic"
    config.data.allow_unpinned_synthetic_smoke = True

    vision_alignment._validate_canonical_data_policy(config)


def test_perception_sources_are_supported_but_require_pinned_audit():
    vision_alignment = _load_module()
    mixture = vision_alignment.VisionAlignmentMixtureConfig(phase="perception")
    targets = mixture.resolved_targets()
    mixture.mean_loss_weight = {source: 1.0 for source in targets}
    config = SimpleNamespace(
        data=SimpleNamespace(
            mixture=mixture,
            native_text_replay=None,
        )
    )

    config.phase = vision_alignment.VisionAlignmentPhase.perception
    config.data.allow_unpinned_synthetic_smoke = False
    config.data.source_audit_path = None
    config.data.source_audit_fingerprint = None

    with pytest.raises(ValueError, match="pinned successful serialized-source audit"):
        vision_alignment._build_mixture_sources(object(), Molmo2TokenIds(), config)


@pytest.mark.parametrize("phase", ["bridge", "perception"])
@pytest.mark.parametrize(
    ("scope", "field_name", "value"),
    [
        ("data", "native_text_replay", object()),
        ("data", "native_text_replay_fingerprint", "a" * 64),
        ("evaluation", "native_text_holdout", object()),
        ("evaluation", "native_text_holdout_fingerprint", "b" * 64),
    ],
)
def test_non_joint_phases_forbid_all_native_replay_configs_and_fingerprints(
    phase, scope, field_name, value
):
    vision_alignment = _load_module()
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase(phase),
        data=vision_alignment.VisionAlignmentDataConfig(),
        evaluation=vision_alignment.VisionAlignmentEvalConfig(),
    )
    setattr(getattr(config, scope), field_name, value)

    with pytest.raises(ValueError, match="forbidden outside joint"):
        vision_alignment._validate_native_artifact_phase(config)


def test_profile_owns_phase_and_forbids_a_second_selector(tmp_path):
    vision_alignment = _load_module()
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "version: 1",
                "phase: bridge",
                "overrides:",
                "  - --data.prefetch_workers=0",
            ]
        )
    )

    loaded, overrides = vision_alignment._load_profile([f"--profile={profile}"])

    assert loaded is not None
    assert overrides == ["--phase=bridge", "--data.prefetch_workers=0"]
    with pytest.raises(ValueError, match="not both"):
        vision_alignment._load_profile([f"--profile={profile}", "--phase=perception"])


def test_profile_rejects_duplicate_yaml_and_override_keys(tmp_path):
    vision_alignment = _load_module()
    duplicate_yaml = tmp_path / "duplicate-yaml.yaml"
    duplicate_yaml.write_text("version: 1\nphase: bridge\nphase: perception\n")
    with pytest.raises(ValueError, match="duplicate key"):
        vision_alignment._load_profile([f"--profile={duplicate_yaml}"])

    duplicate_override = tmp_path / "duplicate-override.yaml"
    duplicate_override.write_text(
        "\n".join(
            [
                "version: 1",
                "phase: bridge",
                "overrides:",
                "  - --data.prefetch_workers=0",
                "  - --data.prefetch_workers=1",
            ]
        )
    )
    with pytest.raises(ValueError, match="repeat a destination"):
        vision_alignment._load_profile([f"--profile={duplicate_override}"])


@pytest.mark.parametrize(
    ("phase", "allowlist_format"),
    [
        ("perception", "vision_alignment_perception_profile_allowlist"),
        ("joint", "vision_alignment_joint_profile_allowlist"),
    ],
)
def test_production_profile_requires_exact_reviewed_allowlist_entry(
    tmp_path, monkeypatch, phase, allowlist_format
):
    vision_alignment = _load_module()
    fake_script = tmp_path / "src" / "scripts" / "train" / "Vision-Alignment.py"
    profile = tmp_path / "configs" / "vision_moe" / "vision_alignment" / phase / "reviewed.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "\n".join(
            [
                "version: 1",
                "name: reviewed",
                f"phase: {phase}",
                "overrides:",
                "  - --data.prefetch_workers=0",
            ]
        )
    )
    relative = f"configs/vision_moe/vision_alignment/{phase}/reviewed.yaml"
    allowlist = profile.parent / "approved_profiles.json"
    allowlist.write_text(
        json.dumps(
            {"format": allowlist_format, "profiles": {}, "version": 1},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    monkeypatch.setattr(vision_alignment, "__file__", str(fake_script))

    with pytest.raises(ValueError, match="reviewed SHA-256 allowlist"):
        vision_alignment._load_profile([f"--profile={profile}"])

    allowlist.write_text(
        json.dumps(
            {
                "format": allowlist_format,
                "profiles": {relative: hashlib.sha256(profile.read_bytes()).hexdigest()},
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    loaded, overrides = vision_alignment._load_profile([f"--profile={profile}"])
    assert loaded is not None
    assert loaded["__reviewed_path__"] == relative
    assert overrides == [f"--phase={phase}", "--data.prefetch_workers=0"]


def test_checked_in_joint_profile_allowlist_is_empty_and_canonical():
    vision_alignment = _load_module()
    repository_root = Path(vision_alignment.__file__).resolve().parents[3]

    profiles, raw_sha256 = vision_alignment._load_approved_profiles(
        repository_root, vision_alignment.VisionAlignmentPhase.joint
    )

    assert profiles == {}
    assert len(raw_sha256) == 64


def test_profile_launch_schema_has_no_hostname_escape_hatch():
    vision_alignment = _load_module()
    launch = SimpleNamespace(
        clusters=[vision_alignment.BEAKER_CLUSTER],
        hostnames=None,
        num_nodes=2,
        num_gpus=8,
        workspace=vision_alignment.BEAKER_WORKSPACE,
        budget=vision_alignment.BEAKER_BUDGET,
        priority="normal",
        min_runtime=None,
        description=None,
    )
    config = SimpleNamespace(launch=launch)
    profile = {
        "version": 1,
        "name": "test-profile",
        "phase": "bridge",
        "launch": {"hostnames": ["holmes-cs-aus-500"]},
    }

    with pytest.raises(ValueError, match="Unknown profile launch fields"):
        vision_alignment._apply_profile_launch(config, profile)


def test_profile_name_must_match_positional_run_name():
    vision_alignment = _load_module()
    config = SimpleNamespace(launch=SimpleNamespace())
    profile = {"version": 1, "name": "reviewed-run", "phase": "bridge"}

    with pytest.raises(ValueError, match="must match positional run name"):
        vision_alignment._apply_profile_launch(config, profile, run_name="different-run")


def test_parent_output_paths_must_be_disjoint(tmp_path):
    vision_alignment = _load_module()
    output = tmp_path / "phase"

    assert vision_alignment._parent_is_inside_output(str(output / "step1"), str(output))
    assert vision_alignment._parent_is_inside_output(str(output), str(output / "child"))
    assert not vision_alignment._parent_is_inside_output(
        str(tmp_path / "bridge" / "step1"), str(tmp_path / "joint")
    )


def test_cross_phase_parent_requires_pinned_approved_quality_gate(tmp_path):
    vision_alignment = _load_module()
    parent = tmp_path / "step100"
    parent.mkdir()
    (parent / ".metadata.json").write_text(json.dumps({"ephemeral": False, "version": "2.5.0"}))
    parent_meta = {
        "phase": "bridge",
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
    }
    parent_config: Dict[str, Any] = {"vision_alignment": parent_meta}
    parent_config_sha = "c" * 64
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": 1,
        "status": "approved",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "phase": "bridge",
        "checkpoint": str(parent),
        "checkpoint_config_sha256": parent_config_sha,
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
        "global_step": 100,
        "metrics_artifact_sha256": "d" * 64,
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))
    gate_sha = vision_alignment._sha256_file(gate_path)
    config = SimpleNamespace(
        initialization=SimpleNamespace(
            parent_gate_path=str(gate_path),
            parent_gate_sha256=gate_sha,
            expected_parent_phase=vision_alignment.VisionAlignmentPhase.bridge,
        )
    )

    assert (
        vision_alignment._validate_parent_gate(
            config, str(parent), parent_config, parent_config_sha
        )
        == gate_sha
    )

    gate["status"] = "rejected"
    gate_path.write_text(json.dumps(gate))
    config.initialization.parent_gate_sha256 = vision_alignment._sha256_file(gate_path)
    with pytest.raises(ValueError, match="incompatible"):
        vision_alignment._validate_parent_gate(
            config, str(parent), parent_config, parent_config_sha
        )

    parent_config["data"] = {"allow_unpinned_synthetic_smoke": True}
    with pytest.raises(ValueError, match="synthetic-smoke"):
        vision_alignment._validate_parent_gate(
            config, str(parent), parent_config, parent_config_sha
        )


def test_production_perception_requires_v2_gate_and_exact_human_waivers(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    from olmo_core.eval import vision_alignment_promotion as promotion

    parent = tmp_path / "step500"
    parent.mkdir()
    (parent / ".metadata.json").write_text(json.dumps({"ephemeral": False, "version": "2.5.0"}))
    parent_meta = {
        "phase": "bridge",
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
    }
    parent_config = {"vision_alignment": parent_meta}
    parent_config_sha = "c" * 64
    bundle = tmp_path / "promotion-bundle.json"
    bundle.write_text(json.dumps({"created_at": "2026-08-12T00:00:00+00:00"}) + "\n")
    bundle_sha = promotion.sha256_file(bundle)
    deviations = {
        promotion.STEP250_WAIVER_ID: "d" * 64,
        promotion.STEP356_WAIVER_ID: "e" * 64,
    }
    monkeypatch.setattr(
        promotion,
        "validate_promotion_bundle",
        lambda value, **kwargs: {
            "candidate": {"checkpoint_identity_sha256": "f" * 64},
            "deviation_sha256": deviations,
        },
    )
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": 2,
        "status": "approved",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "phase": "bridge",
        "checkpoint": str(parent),
        "checkpoint_config_sha256": parent_config_sha,
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
        "global_step": 500,
        "metrics_artifact_sha256": bundle_sha,
        "promotion_bundle_path": str(bundle),
        "promotion_bundle_sha256": bundle_sha,
        "checkpoint_identity_sha256": "f" * 64,
        "approved_by": "rustin@allenai.org",
        "approved_at": "2026-08-12T00:00:00+00:00",
        "waivers": [
            {"id": waiver_id, "decision": "approved", "deviation_sha256": deviations[waiver_id]}
            for waiver_id in sorted(promotion.REQUIRED_WAIVER_IDS)
        ],
    }
    gate_path = tmp_path / "gate-v2.json"
    gate_path.write_text(json.dumps(gate))
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.perception,
        data=SimpleNamespace(allow_unpinned_synthetic_smoke=False),
        initialization=SimpleNamespace(
            parent_gate_path=str(gate_path),
            parent_gate_sha256=vision_alignment._sha256_file(gate_path),
            expected_parent_phase=vision_alignment.VisionAlignmentPhase.bridge,
        ),
    )

    assert vision_alignment._validate_parent_gate(
        config, str(parent), parent_config, parent_config_sha
    ) == vision_alignment._sha256_file(gate_path)

    gate["waivers"][0]["id"] = "free_form_waiver"
    gate_path.write_text(json.dumps(gate))
    config.initialization.parent_gate_sha256 = vision_alignment._sha256_file(gate_path)
    with pytest.raises(ValueError, match="unknown or unapproved waiver"):
        vision_alignment._validate_parent_gate(
            config, str(parent), parent_config, parent_config_sha
        )


def test_production_perception_rejects_legacy_v1_parent_gate(tmp_path):
    vision_alignment = _load_module()
    parent = tmp_path / "step500"
    parent.mkdir()
    (parent / ".metadata.json").write_text(json.dumps({"ephemeral": False, "version": "2.5.0"}))
    parent_config_sha = "c" * 64
    parent_config = {
        "vision_alignment": {
            "phase": "bridge",
            "data_contract_sha256": "a" * 64,
            "trainable_contract_sha256": "b" * 64,
        }
    }
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": 1,
        "status": "approved",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "phase": "bridge",
        "checkpoint": str(parent),
        "checkpoint_config_sha256": parent_config_sha,
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
        "global_step": 500,
        "metrics_artifact_sha256": "d" * 64,
    }
    gate_path = tmp_path / "gate-v1.json"
    gate_path.write_text(json.dumps(gate))
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.perception,
        data=SimpleNamespace(allow_unpinned_synthetic_smoke=False),
        initialization=SimpleNamespace(
            parent_gate_path=str(gate_path),
            parent_gate_sha256=vision_alignment._sha256_file(gate_path),
            expected_parent_phase=vision_alignment.VisionAlignmentPhase.bridge,
        ),
    )

    with pytest.raises(ValueError, match="requires a v2 parent gate"):
        vision_alignment._validate_parent_gate(
            config, str(parent), parent_config, parent_config_sha
        )


def _joint_v3_parent_gate_case(tmp_path, vision_alignment):
    from olmo_core.eval import vision_alignment_perception_promotion as promotion

    parent = tmp_path / "step4000"
    parent.mkdir()
    marker_path = parent / ".metadata.json"
    marker_path.write_text(json.dumps({"ephemeral": False, "version": "2.5.0"}))
    parent_config_sha = "c" * 64
    parent_meta = {
        "phase": "perception",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
    }
    parent_config = {"vision_alignment": parent_meta}
    bundle_path = tmp_path / "perception-promotion-bundle.json"
    bundle = {"format": "test-perception-promotion-bundle", "created_at": "2026-08-13T00:00:00Z"}
    bundle_path.write_text(json.dumps(bundle) + "\n")
    bundle_sha = vision_alignment._sha256_file(bundle_path)
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": 3,
        "status": "approved",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "phase": "perception",
        "checkpoint": str(parent),
        "checkpoint_config_sha256": parent_config_sha,
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
        "global_step": 4000,
        "metrics_artifact_sha256": bundle_sha,
        "promotion_bundle_path": str(bundle_path),
        "promotion_bundle_sha256": bundle_sha,
        "checkpoint_identity_sha256": "f" * 64,
        "approved_by": "rustins",
        "approved_at": "2026-08-13T21:47:16Z",
        "waivers": [
            {
                "id": promotion.TREATMENT_GUARD_WAIVER_ID,
                "decision": "approved",
                "deviation_sha256": "d" * 64,
            }
        ],
        "promotion_kind": "perception",
        "promotion_policy": promotion.PERCEPTION_PROMOTION_POLICY,
    }
    gate_path = tmp_path / "perception-parent-gate-v3.json"
    gate_path.write_text(json.dumps(gate) + "\n")
    gate_sha = vision_alignment._sha256_file(gate_path)
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.joint,
        data=SimpleNamespace(allow_unpinned_synthetic_smoke=False),
        initialization=SimpleNamespace(
            parent_gate_path=str(gate_path),
            parent_gate_sha256=gate_sha,
            expected_parent_phase=vision_alignment.VisionAlignmentPhase.perception,
        ),
    )
    return SimpleNamespace(
        bundle=bundle,
        bundle_path=bundle_path,
        config=config,
        gate=gate,
        gate_path=gate_path,
        gate_sha=gate_sha,
        marker_path=marker_path,
        parent=parent,
        parent_config=parent_config,
        parent_config_sha=parent_config_sha,
        promotion=promotion,
    )


def _allow_test_joint_v3_gate(monkeypatch, case):
    monkeypatch.setattr(
        case.promotion,
        "EXPECTED_APPROVED_PERCEPTION_PARENT_GATE_RAW_SHA256",
        case.gate_sha,
    )
    monkeypatch.setattr(
        case.promotion,
        "EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256",
        case.gate["promotion_bundle_sha256"],
    )


def test_production_joint_routes_exact_v3_gate_to_approved_perception_adapter(
    tmp_path, monkeypatch
):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    _allow_test_joint_v3_gate(monkeypatch, case)
    observed: Dict[str, Any] = {}

    def validate(bundle, **kwargs):
        observed["bundle"] = bundle
        observed.update(kwargs)

    monkeypatch.setattr(
        case.promotion,
        "validate_approved_perception_parent_gate_bundle",
        validate,
    )

    assert vision_alignment._validate_parent_gate(
        case.config,
        str(case.parent),
        case.parent_config,
        case.parent_config_sha,
    ) == vision_alignment._sha256_file(case.gate_path)
    assert observed == {
        "bundle": case.bundle,
        "gate": case.gate,
        "expected_checkpoint": case.parent.resolve(),
        "expected_checkpoint_config_sha256": case.parent_config_sha,
    }


@pytest.mark.parametrize("version", [1, 2])
def test_production_joint_rejects_legacy_parent_gate_versions(tmp_path, version):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    case.gate["version"] = version
    if version == 2:
        case.gate.pop("promotion_kind")
        case.gate.pop("promotion_policy")
    else:
        for field in (
            "promotion_bundle_path",
            "promotion_bundle_sha256",
            "checkpoint_identity_sha256",
            "approved_by",
            "approved_at",
            "waivers",
            "promotion_kind",
            "promotion_policy",
        ):
            case.gate.pop(field)
    case.gate_path.write_text(json.dumps(case.gate) + "\n")
    case.config.initialization.parent_gate_sha256 = vision_alignment._sha256_file(case.gate_path)

    with pytest.raises(ValueError, match="Production joint requires a v3 perception parent gate"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_perception_phase_rejects_v3_parent_gate(tmp_path):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    case.config.phase = vision_alignment.VisionAlignmentPhase.perception

    with pytest.raises(ValueError, match="Production perception requires a v2 parent gate"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_joint_v3_gate_must_be_exactly_allowlisted(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    monkeypatch.setattr(
        case.promotion,
        "EXPECTED_APPROVED_PERCEPTION_PARENT_GATE_RAW_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="not the exact approved perception gate"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_joint_v3_gate_rejects_bundle_raw_sha_mismatch(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    _allow_test_joint_v3_gate(monkeypatch, case)
    case.bundle_path.write_text(json.dumps({"changed": True}) + "\n")

    with pytest.raises(ValueError, match="promotion bundle SHA mismatch"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_joint_v3_gate_requires_exactly_allowlisted_bundle(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    _allow_test_joint_v3_gate(monkeypatch, case)
    monkeypatch.setattr(
        case.promotion,
        "EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="not the exact approved bundle"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_joint_v3_gate_propagates_approved_adapter_error(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    _allow_test_joint_v3_gate(monkeypatch, case)

    def reject(*args, **kwargs):
        raise case.promotion.PromotionValidationError("candidate identity differs")

    monkeypatch.setattr(
        case.promotion,
        "validate_approved_perception_parent_gate_bundle",
        reject,
    )

    with pytest.raises(ValueError, match="candidate identity differs"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_joint_v3_gate_recipe_and_formatter_bind_to_parent_not_current_launcher(
    tmp_path, monkeypatch
):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    _allow_test_joint_v3_gate(monkeypatch, case)
    monkeypatch.setattr(vision_alignment, "RECIPE_VERSION", case.gate["recipe_version"] + 1)
    monkeypatch.setattr(vision_alignment, "FORMATTER_VERSION", "future-document-formatter")
    monkeypatch.setattr(
        case.promotion,
        "validate_approved_perception_parent_gate_bundle",
        lambda *args, **kwargs: None,
    )

    assert (
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )
        == case.gate_sha
    )


def test_joint_v3_gate_rejects_invalid_approval_time_through_adapter(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    case.gate["approved_at"] = "not-a-time"
    case.gate_path.write_text(json.dumps(case.gate) + "\n")
    case.gate_sha = vision_alignment._sha256_file(case.gate_path)
    case.config.initialization.parent_gate_sha256 = case.gate_sha
    _allow_test_joint_v3_gate(monkeypatch, case)

    def reject(bundle, *, gate, **kwargs):
        assert gate["approved_at"] == "not-a-time"
        raise case.promotion.PromotionValidationError("approval timestamp is invalid")

    monkeypatch.setattr(
        case.promotion,
        "validate_approved_perception_parent_gate_bundle",
        reject,
    )

    with pytest.raises(ValueError, match="approval timestamp is invalid"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_joint_v3_gate_still_requires_permanent_checkpoint(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    _allow_test_joint_v3_gate(monkeypatch, case)
    monkeypatch.setattr(
        case.promotion,
        "validate_approved_perception_parent_gate_bundle",
        lambda *args, **kwargs: None,
    )
    case.marker_path.write_text(json.dumps({"ephemeral": True, "version": "2.5.0"}))

    with pytest.raises(ValueError, match="permanent parent checkpoint"):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda gate: gate.update(version=True), "version must be exactly integer"),
        (lambda gate: gate.update(version=3.0), "version must be exactly integer"),
        (lambda gate: gate.update(version="3"), "version must be exactly integer"),
        (lambda gate: gate.pop("promotion_kind"), "fields differ from the locked schema"),
        (lambda gate: gate.update(unexpected=True), "fields differ from the locked schema"),
    ],
)
def test_joint_v3_gate_rejects_schema_and_version_type_confusion(tmp_path, mutation, message):
    vision_alignment = _load_module()
    case = _joint_v3_parent_gate_case(tmp_path, vision_alignment)
    mutation(case.gate)
    case.gate_path.write_text(json.dumps(case.gate) + "\n")
    case.config.initialization.parent_gate_sha256 = vision_alignment._sha256_file(case.gate_path)

    with pytest.raises(ValueError, match=message):
        vision_alignment._validate_parent_gate(
            case.config,
            str(case.parent),
            case.parent_config,
            case.parent_config_sha,
        )


def test_joint_parent_recipe_is_bound_by_v3_gate_not_future_launcher_version(monkeypatch, tmp_path):
    vision_alignment = _load_module()
    parent = "/checkpoints/perception/step4000"
    parent_config_sha = "a" * 64
    provenance_path = (tmp_path / "vision-alignment-perception-provenance.json").resolve()
    provenance_path.write_bytes(b"parent provenance\n")
    provenance_sha = vision_alignment._sha256_file(provenance_path)
    parent_config = {
        "vision_alignment": {
            "phase": "perception",
            "recipe_version": 1,
            "formatter_version": "vision-alignment-document-v1",
        },
        "data": {
            "perception_provenance_path": str(provenance_path),
            "perception_provenance_sha256": provenance_sha,
        },
    }
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.joint,
        initialization=SimpleNamespace(
            checkpoint=parent,
            expected_parent_phase=vision_alignment.VisionAlignmentPhase.perception,
            parent_config_sha256=None,
            parent_gate_sha256=None,
        ),
        vision_alignment=SimpleNamespace(
            parent_config_sha256=None,
            parent_gate_sha256=None,
        ),
    )
    observed: Dict[str, Any] = {}
    monkeypatch.setattr(vision_alignment, "RECIPE_VERSION", 2)
    monkeypatch.setattr(vision_alignment, "_latest_output_checkpoint", lambda config: None)
    monkeypatch.setattr(
        vision_alignment,
        "_checkpoint_config",
        lambda checkpoint: (parent_config, parent_config_sha),
    )
    projection = SimpleNamespace(
        parent_provenance=SimpleNamespace(path=provenance_path, raw_sha256=provenance_sha)
    )
    monkeypatch.setattr(
        vision_alignment,
        "_joint_visual_projection",
        lambda config_arg, token_ids=None: projection,
    )

    def validate_gate(config, selected_parent, selected_config, selected_config_sha):
        observed.update(
            parent=selected_parent,
            config=selected_config,
            config_sha=selected_config_sha,
        )
        return "b" * 64

    monkeypatch.setattr(vision_alignment, "_validate_parent_gate", validate_gate)

    vision_alignment._validate_parent_or_resume(config)

    assert observed == {
        "parent": parent,
        "config": parent_config,
        "config_sha": parent_config_sha,
    }
    assert config.initialization.parent_config_sha256 == parent_config_sha
    assert config.initialization.parent_gate_sha256 == "b" * 64

    projection.parent_provenance.raw_sha256 = "f" * 64
    with pytest.raises(ValueError, match="differs from the approved perception checkpoint"):
        vision_alignment._validate_parent_or_resume(config)


def test_joint_resume_rechecks_parent_projection_lineage(monkeypatch, tmp_path):
    vision_alignment = _load_module()
    existing = "/checkpoints/joint/step100"
    parent = "/checkpoints/perception/step4000"
    parent_sha = "a" * 64
    gate_sha = "b" * 64
    provenance_path = (tmp_path / "vision-alignment-perception-provenance.json").resolve()
    provenance_path.write_bytes(b"parent provenance\n")
    provenance_sha = vision_alignment._sha256_file(provenance_path)
    parent_config = {
        "data": {
            "perception_provenance_path": str(provenance_path),
            "perception_provenance_sha256": provenance_sha,
        }
    }
    saved_config = {
        "vision_alignment": {
            "recipe_version": vision_alignment.RECIPE_VERSION,
            "formatter_version": vision_alignment.FORMATTER_VERSION,
            "phase": "joint",
            "lineage_id": "joint-lineage",
            "parent_checkpoint": parent,
            "parent_config_sha256": parent_sha,
            "parent_gate_sha256": gate_sha,
            "data_contract_sha256": "c" * 64,
            "trainable_contract_sha256": "d" * 64,
        }
    }
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.joint,
        initialization=SimpleNamespace(
            checkpoint=parent,
            parent_config_sha256=None,
            parent_gate_sha256=None,
        ),
        vision_alignment=SimpleNamespace(
            lineage_id="joint-lineage",
            parent_checkpoint=parent,
            parent_config_sha256=None,
            parent_gate_sha256=None,
            data_contract_sha256="c" * 64,
            trainable_contract_sha256="d" * 64,
        ),
    )
    projection = SimpleNamespace(
        parent_provenance=SimpleNamespace(path=provenance_path, raw_sha256=provenance_sha)
    )
    monkeypatch.setattr(vision_alignment, "_latest_output_checkpoint", lambda _: existing)

    def checkpoint_config(checkpoint):
        return (saved_config, "e" * 64) if checkpoint == existing else (parent_config, parent_sha)

    monkeypatch.setattr(vision_alignment, "_checkpoint_config", checkpoint_config)
    monkeypatch.setattr(
        vision_alignment,
        "_joint_visual_projection",
        lambda config_arg, token_ids=None: projection,
    )

    vision_alignment._validate_parent_or_resume(config)

    parent_config["data"]["perception_provenance_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from the approved perception checkpoint"):
        vision_alignment._validate_parent_or_resume(config)


def test_non_joint_parent_still_rejects_recipe_version_mismatch(monkeypatch):
    vision_alignment = _load_module()
    parent_config = {
        "vision_alignment": {
            "phase": "bridge",
            "recipe_version": vision_alignment.RECIPE_VERSION + 1,
        }
    }
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.perception,
        initialization=SimpleNamespace(
            checkpoint="/checkpoints/bridge/step500",
            expected_parent_phase=vision_alignment.VisionAlignmentPhase.bridge,
        ),
    )
    monkeypatch.setattr(vision_alignment, "_latest_output_checkpoint", lambda config: None)
    monkeypatch.setattr(
        vision_alignment,
        "_checkpoint_config",
        lambda checkpoint: (parent_config, "a" * 64),
    )

    with pytest.raises(ValueError, match="incompatible recipe version"):
        vision_alignment._validate_parent_or_resume(config)


def test_runtime_trainability_rejects_stale_freeze_patterns():
    vision_alignment = _load_module()
    model = nn.Module()
    model.connector = nn.Linear(2, 2)
    train_module = SimpleNamespace(
        multimodal_model=model,
        train_embedding_rows=(100, 101, 102, 103, 104, 105),
    )
    config = SimpleNamespace(
        train_module=SimpleNamespace(
            freeze_params=["lm.blocks.*"],
            train_embedding_rows=[100, 101, 102, 103, 104, 105],
        )
    )

    with pytest.raises(RuntimeError, match="freeze patterns did not match"):
        vision_alignment._validate_runtime_trainability(train_module, config)


def test_frozen_lm_phase_rejects_trainable_default_optimizer_parameters():
    vision_alignment = _load_module()
    model = nn.Module()
    model.connector = nn.Linear(2, 2)
    model.unexpected = nn.Linear(2, 2)
    image_rows = [100, 101, 102, 103, 104, 105]
    train_module = SimpleNamespace(
        multimodal_model=model,
        train_embedding_rows=tuple(image_rows),
    )
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.bridge,
        train_module=SimpleNamespace(
            freeze_params=[],
            train_embedding_rows=image_rows,
            optim=SimpleNamespace(group_overrides=[SimpleNamespace(params=["*connector.*"])]),
        ),
    )

    with pytest.raises(RuntimeError, match="fallback/default-group.*unexpected"):
        vision_alignment._validate_runtime_trainability(train_module, config)


def test_resume_contract_hashes_cover_full_data_and_train_module_config():
    vision_alignment = _load_module()
    policy = vision_alignment._PHASE_POLICIES[vision_alignment.VisionAlignmentPhase.bridge]
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.bridge,
        perception_trainability_arm=vision_alignment.PerceptionTrainabilityArm.treatment,
        data=vision_alignment.VisionAlignmentDataConfig(),
        evaluation=vision_alignment.VisionAlignmentEvalConfig(),
        collator=SimpleNamespace(as_config_dict=lambda: {"pad_sequence_length": 2560}),
        global_batch_size=128 * policy.sequence_length,
        data_seed=vision_alignment.DATA_SEED,
        model=SimpleNamespace(as_config_dict=lambda: {"model_revision": 1}),
        train_module=vision_alignment._build_train_module_config(
            policy, [100, 101, 102, 103, 104, 105]
        ),
        trainer=SimpleNamespace(max_duration=vision_alignment.Duration.steps(policy.max_steps)),
        router_lb_loss_weight=0.015,
        vision_alignment=vision_alignment.VisionAlignmentMetadataConfig(),
        reviewed_profile_path="configs/perception/control.yaml",
        reviewed_profile_sha256="a" * 64,
        reviewed_profile_allowlist_path="configs/perception/approved_profiles.json",
        reviewed_profile_allowlist_sha256="b" * 64,
    )

    vision_alignment._set_contract_hashes(config)
    original_data_hash = config.vision_alignment.data_contract_sha256
    original_train_hash = config.vision_alignment.trainable_contract_sha256
    original_preprocessing_hash = vision_alignment._preprocessing_config_sha256(config)

    config.perception_trainability_arm = (
        vision_alignment.PerceptionTrainabilityArm.frozen_vision_control
    )
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.data_contract_sha256 == original_data_hash
    assert config.vision_alignment.trainable_contract_sha256 != original_train_hash
    config.perception_trainability_arm = vision_alignment.PerceptionTrainabilityArm.treatment

    config.reviewed_profile_path = "configs/perception/treatment.yaml"
    config.reviewed_profile_sha256 = "c" * 64
    config.reviewed_profile_allowlist_sha256 = "d" * 64
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.data_contract_sha256 == original_data_hash
    assert config.vision_alignment.trainable_contract_sha256 == original_train_hash

    config.data.caption_prompt = "Caption:"
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.data_contract_sha256 != original_data_hash

    config.data.caption_prompt = "Description:"
    config.data.require_transcript = False
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.data_contract_sha256 != original_data_hash
    assert vision_alignment._preprocessing_config_sha256(config) != original_preprocessing_hash

    config.data.require_transcript = True
    config.collator = SimpleNamespace(as_config_dict=lambda: {"pad_sequence_length": 8192})
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.data_contract_sha256 != original_data_hash

    config.train_module.optim.lr = 1e-6
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.trainable_contract_sha256 != original_train_hash

    config.train_module.optim.lr = 0.0
    config.model = SimpleNamespace(as_config_dict=lambda: {"model_revision": 2})
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.trainable_contract_sha256 != original_train_hash

    config.model = SimpleNamespace(as_config_dict=lambda: {"model_revision": 1})
    config.router_lb_loss_weight = 0.02
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.trainable_contract_sha256 != original_train_hash

    config.router_lb_loss_weight = 0.015
    config.trainer.max_duration = vision_alignment.Duration.steps(policy.max_steps - 1)
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.trainable_contract_sha256 != original_train_hash


def test_real_data_requires_a_matching_pinned_source_audit(tmp_path):
    vision_alignment = _load_module()
    data = vision_alignment.VisionAlignmentDataConfig()
    data.mixture.mean_loss_weight = {
        "pixmo_caption": 2.0,
        "pixmo_transcript": 1.0,
    }
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.bridge,
        data=data,
        trainer=SimpleNamespace(max_duration=SimpleNamespace(value=1000)),
        evaluation=SimpleNamespace(interval=500),
    )
    with pytest.raises(ValueError, match="requires a pinned successful"):
        vision_alignment._validated_source_audit(config)

    probe_seed = 6198
    examples_per_source = 1024
    dataset_fingerprint = "pixmo-cap-train-fingerprint"
    dataset_size = 714985
    probe_indices = list(
        vision_alignment.select_deterministic_probe_indices(
            dataset_size,
            examples_per_source,
            seed=probe_seed,
            dataset_fingerprint=dataset_fingerprint,
        )
    )
    row_hashes = ["f" * 64] * examples_per_source
    source_inputs = {}
    for ordinal, source_name in enumerate(("pixmo_caption", "pixmo_transcript")):
        source_inputs[source_name] = {
            "format": "jsonl",
            "path": f"{source_name}.jsonl",
            "sha256": f"{ordinal + 1:064x}",
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_size": dataset_size,
            "probe_indices": probe_indices,
            "probe_indices_sha256": vision_alignment._canonical_sha256(probe_indices),
            "serialized_row_hashes": row_hashes,
            "serialized_row_hashes_sha256": vision_alignment._canonical_sha256(row_hashes),
        }
    audit: Dict[str, Any] = {
        "format": "vision_alignment_source_audit",
        "version": 2,
        "auditor_sha256": vision_alignment._sha256_file(
            Path(vision_alignment.__file__).parents[1] / "data" / "audit_vision_alignment_mix.py"
        ),
        "status": "ok",
        "phase": "bridge",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "source_catalog_version": vision_alignment.VISION_ALIGNMENT_SOURCE_CATALOG_VERSION,
        "source_registry_version": vision_alignment.VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": vision_alignment.vision_alignment_source_registry_sha256(),
        "exporter_sha256": vision_alignment._sha256_file(
            Path(vision_alignment.__file__).parents[1] / "data" / "export_vision_alignment_probe.py"
        ),
        "image_manifest_sha256": "a" * 64,
        "preprocessing_config": vision_alignment._source_spec(config).as_canonical_dict(),
        "preprocessing_config_sha256": vision_alignment._preprocessing_config_sha256(config),
        "probe": {
            "format": vision_alignment.VISION_ALIGNMENT_PROBE_FORMAT,
            "version": vision_alignment.VISION_ALIGNMENT_PROBE_VERSION,
            "selection_algorithm": vision_alignment.VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
            "seed": probe_seed,
            "epoch": 0,
            "examples_per_source": examples_per_source,
        },
        "catalog_sha256": "b" * 64,
        "input_content_sha256": "c" * 64,
        "target_loss_mass": data.mixture.resolved_targets(),
        "mean_loss_weight": data.mixture.mean_loss_weight,
        "sampling_probabilities": data.mixture.sampling_weights(),
        "expected_loss_mass": {
            "pixmo_caption": 0.7000000000000001,
            "pixmo_transcript": 0.3,
        },
        "failures": [],
        "inputs": source_inputs,
        "sources": {
            source_name: {
                "examples": {"seen": 1024, "valid": 1024, "errors": 0},
                "mean_sum_loss_masks": mean_loss_weight,
                "zero_loss_examples": 0,
                "error_samples": [],
            }
            for source_name, mean_loss_weight in data.mixture.mean_loss_weight.items()
        },
    }
    audit["fingerprint"] = vision_alignment._canonical_sha256(audit)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    data.source_audit_path = str(audit_path)
    data.source_audit_fingerprint = audit["fingerprint"]

    loaded = vision_alignment._validated_source_audit(config)

    assert loaded is not None
    assert loaded["fingerprint"] == audit["fingerprint"]
    data.mixture.mean_loss_weight["pixmo_caption"] = 3.0
    with pytest.raises(ValueError, match="mean_loss_weight differs"):
        vision_alignment._validated_source_audit(config)
    data.mixture.mean_loss_weight["pixmo_caption"] = 2.0

    audit["expected_loss_mass"]["pixmo_caption"] = 0.700001
    audit["fingerprint"] = vision_alignment._canonical_sha256(
        {key: value for key, value in audit.items() if key != "fingerprint"}
    )
    audit_path.write_text(json.dumps(audit))
    data.source_audit_fingerprint = audit["fingerprint"]
    with pytest.raises(ValueError, match="expected loss mass differs"):
        vision_alignment._validated_source_audit(config)


def test_joint_source_audit_strictly_binds_all_nine_sources(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    config, audit, audit_path, projection = _joint_audit_fixture(tmp_path, vision_alignment)
    monkeypatch.setattr(
        vision_alignment,
        "_joint_visual_projection",
        lambda config_arg, token_ids=None: projection,
    )
    native_summary = audit["sources"]["native_text_replay"]
    native_summary["zero_loss_examples"] = 1
    native_summary["summed_loss_weight"]["min"] = 0.0
    native_summary["summed_loss_weight"]["max"] = 2.0
    _rewrite_joint_audit(vision_alignment, config, audit, audit_path)

    loaded = vision_alignment._validated_source_audit(config)

    assert loaded is not None
    assert tuple(sorted(loaded["inputs"])) == vision_alignment._JOINT_SOURCE_NAMES
    assert loaded["probe"]["visual"]["epochs"] == [0, 1, 2, 3]
    assert loaded["probe"]["native_text_replay"]["epochs"] == [0]

    audit["probe"]["visual"]["epochs"] = [0, 1, 2]
    _rewrite_joint_audit(vision_alignment, config, audit, audit_path)
    with pytest.raises(ValueError, match="probe identity or epoch panel"):
        vision_alignment._validated_source_audit(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_field", "fields differ"),
        ("auditor_sha", "implementation, identity, or status"),
        ("projection_sha", "different visual projection"),
        ("calibration", "calibration differs"),
        ("native_image", "truncation/image evidence"),
        ("native_epoch_bool", "runtime probe differs"),
    ],
)
def test_joint_source_audit_rejects_self_consistent_adversarial_drift(
    tmp_path, monkeypatch, mutation, message
):
    vision_alignment = _load_module()
    config, audit, audit_path, projection = _joint_audit_fixture(tmp_path, vision_alignment)
    monkeypatch.setattr(
        vision_alignment,
        "_joint_visual_projection",
        lambda config_arg, token_ids=None: projection,
    )
    if mutation == "extra_field":
        audit["unreviewed"] = True
    elif mutation == "auditor_sha":
        audit["auditor_implementation"]["sha256"] = "f" * 64
    elif mutation == "projection_sha":
        audit["visual_projection"]["raw_sha256"] = "f" * 64
    elif mutation == "calibration":
        audit["sources"]["pixmo_caption"]["mean_sum_loss_masks"] = 1.1
    elif mutation == "native_image":
        audit["inputs"]["native_text_replay"]["probe_image_content_sha256"] = "f" * 64
    elif mutation == "native_epoch_bool":
        audit["inputs"]["native_text_replay"]["probe_epochs"] = [False]
    else:  # pragma: no cover - parameter table is exhaustive.
        raise AssertionError(mutation)
    _rewrite_joint_audit(vision_alignment, config, audit, audit_path)

    with pytest.raises(ValueError, match=message):
        vision_alignment._validated_source_audit(config)


def test_launcher_accepts_exact_pixmo_builder_artifact(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    builder = _load_pixmo_builder()
    from datasets import Dataset, DatasetDict, load_from_disk

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_paths = {}
    for name, payload in {
        "train-a": b"image-a",
        "train-b": b"image-b",
        "train-overlap": b"image-c",
        "validation-c": b"image-c",
        "validation-d": b"image-d",
    }.items():
        image_path = image_dir / name
        image_path.write_bytes(payload)
        image_paths[name] = str(image_path.resolve())

    source_path = tmp_path / "source"
    source = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "image": [
                        image_paths["train-a"],
                        image_paths["train-b"],
                        image_paths["train-overlap"],
                    ],
                    "caption": ["a", "b", "c"],
                    "transcripts": [["a"], ["b"], ["c"]],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "image": [image_paths["validation-c"], image_paths["validation-d"]],
                    "caption": ["c", "d"],
                    "transcripts": [["c"], ["d"]],
                }
            ),
        }
    )
    source.save_to_disk(source_path)
    source = load_from_disk(source_path)
    artifact_path = tmp_path / "artifact"
    builder.build_pixmo_cap_artifact(
        source_dataset_path=str(source_path),
        output_dir=str(artifact_path),
        expected_train_fingerprint=source["train"]._fingerprint,
        expected_train_examples=len(source["train"]),
        expected_validation_fingerprint=source["validation"]._fingerprint,
        expected_validation_examples=len(source["validation"]),
        workers=2,
        scan_batch_size=2,
    )

    manifest_path = artifact_path / "vision-alignment-validation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _patch_canonical_pixmo_source_policy(
        monkeypatch,
        vision_alignment,
        manifest["source"]["dataset_path"],
        manifest["source"]["splits"],
    )
    data = SimpleNamespace(
        allow_unpinned_synthetic_smoke=False,
        pixmo_cap_path=str(artifact_path / "dataset"),
    )
    evaluation = SimpleNamespace(
        examples_per_source=2,
        validation_manifest_path=str(manifest_path),
        validation_manifest_sha256=vision_alignment._sha256_file(manifest_path),
    )
    train = manifest["output"]["splits"]["train"]
    source_audit = {
        "image_manifest_sha256": manifest["inventories"]["train"]["sha256"],
        "inputs": {
            name: {
                "dataset_fingerprint": train["dataset_fingerprint"],
                "dataset_size": train["examples"],
            }
            for name in ("pixmo_caption", "pixmo_transcript")
        },
    }

    loaded = vision_alignment._validate_validation_manifest(
        SimpleNamespace(data=data, evaluation=evaluation), source_audit
    )

    assert loaded == manifest
    assert manifest["filtering"] == {
        "output_overlap_unique_images": 0,
        "removed_train_examples": 1,
        "source_overlap_unique_images": 1,
        "validation_duplicate_examples": 0,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "dataset_path",
        "train_fingerprint",
        "train_examples",
        "validation_fingerprint",
        "validation_examples",
    ],
)
def test_self_minted_v3_from_arbitrary_pixmo_source_is_rejected(tmp_path, mutation):
    vision_alignment = _load_module()
    source_splits = {
        split: {
            "dataset_fingerprint": fingerprint,
            "examples": examples,
            "row_image_paths_sha256": "a" * 64,
            "row_image_content_sha256": "b" * 64,
            "unique_image_paths": 1,
            "unique_image_content": 1,
        }
        for split, (fingerprint, examples) in (
            vision_alignment._CANONICAL_PIXMO_SOURCE_SPLITS.items()
        )
    }
    source_path = vision_alignment._CANONICAL_PIXMO_SOURCE_DATASET
    expected_error = "canonical PixMoCap source fingerprint and row count"
    if mutation == "dataset_path":
        source_path = str(tmp_path / "arbitrary-pixmo-source")
        expected_error = "canonical PixMoCap source dataset"
    else:
        split, field = mutation.split("_", maxsplit=1)
        key = "dataset_fingerprint" if field == "fingerprint" else "examples"
        source_splits[split][key] = "self-minted-source" if key == "dataset_fingerprint" else 1
    output_split = {
        "dataset_fingerprint": "self-minted-output",
        "examples": 1,
        "row_image_paths_sha256": "a" * 64,
        "row_image_content_sha256": "b" * 64,
        "unique_image_paths": 1,
        "unique_image_content": 1,
        "row_image_content_path": "row-images.sha256",
    }
    builder_path = (
        Path(vision_alignment.__file__).parents[3] / vision_alignment._PIXMO_BUILDER_SCRIPT
    )
    manifest = {
        "format": "vision_alignment_validation_manifest",
        "version": 3,
        "builder": {
            "format": "vision_alignment_pixmo_cap_builder",
            "version": 1,
            "script": vision_alignment._PIXMO_BUILDER_SCRIPT,
            "script_sha256": vision_alignment._sha256_file(builder_path),
            "filter_algorithm": vision_alignment._PIXMO_FILTER_ALGORITHM,
            "image_hash_algorithm": "sha256",
            "row_image_paths_algorithm": (
                vision_alignment.VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM
            ),
            "row_image_content_algorithm": vision_alignment._PIXMO_ROW_CONTENT_ALGORITHM,
        },
        "source": {
            "dataset_path": source_path,
            "splits": source_splits,
        },
        "output": {
            "dataset_path": "dataset",
            "splits": {"train": output_split, "validation": output_split},
        },
        "inventories": {
            split: {"path": f"{split}.sha256", "sha256": "c" * 64, "count": 1}
            for split in ("train", "validation")
        },
        "filtering": {
            "source_overlap_unique_images": 0,
            "removed_train_examples": 0,
            "validation_duplicate_examples": 0,
            "output_overlap_unique_images": 0,
        },
    }
    manifest_path = tmp_path / "vision-alignment-validation-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = SimpleNamespace(
        data=SimpleNamespace(
            allow_unpinned_synthetic_smoke=False,
            pixmo_cap_path=str(tmp_path / "dataset"),
        ),
        evaluation=SimpleNamespace(
            examples_per_source=1,
            validation_manifest_path=str(manifest_path),
            validation_manifest_sha256=vision_alignment._sha256_file(manifest_path),
        ),
    )

    with pytest.raises(ValueError, match=expected_error):
        vision_alignment._validate_validation_manifest(
            config,
            {
                "image_manifest_sha256": "c" * 64,
                "inputs": {
                    name: {
                        "dataset_fingerprint": "self-minted-output",
                        "dataset_size": 1,
                    }
                    for name in ("pixmo_caption", "pixmo_transcript")
                },
            },
            validate_live_datasets=False,
        )


def test_production_requires_canonical_builder_pixmo_artifact(tmp_path, monkeypatch):
    vision_alignment = _load_module()
    from datasets import Dataset, DatasetDict, load_from_disk

    source_path = tmp_path / "source"
    dataset_path = tmp_path / "dataset"
    source_train_paths = ["/images/a", "/images/b", "/images/c"]
    output_train_paths = source_train_paths[:2]
    validation_paths = ["/images/c", "/images/d"]
    source = DatasetDict(
        {
            "train": Dataset.from_dict({"image": source_train_paths}),
            "validation": Dataset.from_dict({"image": validation_paths}),
        }
    )
    output = DatasetDict(
        {
            "train": Dataset.from_dict({"image": output_train_paths}),
            "validation": Dataset.from_dict({"image": validation_paths}),
        }
    )
    source.save_to_disk(source_path)
    output.save_to_disk(dataset_path)
    source = load_from_disk(source_path)
    output = load_from_disk(dataset_path)

    content = {name: f"{value:064x}" for value, name in enumerate("abcd", start=1)}
    source_content = {
        "train": [content["a"], content["b"], content["c"]],
        "validation": [content["c"], content["d"]],
    }
    output_content = {
        "train": [content["a"], content["b"]],
        "validation": source_content["validation"],
    }
    inventories = {split: sorted(set(output_content[split])) for split in ("train", "validation")}
    inventory_entries = {}
    output_entries = {}
    for split in ("train", "validation"):
        inventory_path = tmp_path / f"{split}-images.sha256"
        inventory_path.write_text("\n".join(inventories[split]) + "\n")
        row_content_path = tmp_path / f"{split}-row-content.sha256"
        row_content_path.write_text("\n".join(output_content[split]) + "\n")
        inventory_entries[split] = {
            "path": inventory_path.name,
            "sha256": vision_alignment._sha256_file(inventory_path),
            "count": len(inventories[split]),
        }
        output_entries[split] = {
            "dataset_fingerprint": output[split]._fingerprint,
            "examples": len(output[split]),
            "row_image_paths_sha256": vision_alignment.pixmo_row_path_inventory(output[split])[
                "sha256"
            ],
            "row_image_content_path": row_content_path.name,
            "row_image_content_sha256": vision_alignment._sha256_file(row_content_path),
            "unique_image_paths": len(set(output[split]["image"])),
            "unique_image_content": len(inventories[split]),
        }

    source_entries = {}
    for split in ("train", "validation"):
        raw_content = ("\n".join(source_content[split]) + "\n").encode()
        source_entries[split] = {
            "dataset_fingerprint": source[split]._fingerprint,
            "examples": len(source[split]),
            "row_image_paths_sha256": vision_alignment.pixmo_row_path_inventory(source[split])[
                "sha256"
            ],
            "row_image_content_sha256": vision_alignment.hashlib.sha256(raw_content).hexdigest(),
            "unique_image_paths": len(set(source[split]["image"])),
            "unique_image_content": len(set(source_content[split])),
        }

    data = SimpleNamespace(
        allow_unpinned_synthetic_smoke=False,
        pixmo_cap_path=str(dataset_path),
    )
    evaluation = SimpleNamespace(
        examples_per_source=2,
        validation_manifest_path=None,
        validation_manifest_sha256=None,
    )
    config = SimpleNamespace(data=data, evaluation=evaluation)
    source_audit = {
        "image_manifest_sha256": inventory_entries["train"]["sha256"],
        "inputs": {
            name: {
                "dataset_fingerprint": output_entries["train"]["dataset_fingerprint"],
                "dataset_size": output_entries["train"]["examples"],
            }
            for name in ("pixmo_caption", "pixmo_transcript")
        },
    }
    builder_path = (
        Path(vision_alignment.__file__).parents[3] / vision_alignment._PIXMO_BUILDER_SCRIPT
    )
    manifest = {
        "format": "vision_alignment_validation_manifest",
        "version": 3,
        "builder": {
            "format": "vision_alignment_pixmo_cap_builder",
            "version": 1,
            "script": vision_alignment._PIXMO_BUILDER_SCRIPT,
            "script_sha256": vision_alignment._sha256_file(builder_path),
            "filter_algorithm": vision_alignment._PIXMO_FILTER_ALGORITHM,
            "image_hash_algorithm": "sha256",
            "row_image_paths_algorithm": (
                vision_alignment.VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM
            ),
            "row_image_content_algorithm": vision_alignment._PIXMO_ROW_CONTENT_ALGORITHM,
        },
        "source": {"dataset_path": "source", "splits": source_entries},
        "output": {"dataset_path": "dataset", "splits": output_entries},
        "inventories": inventory_entries,
        "filtering": {
            "source_overlap_unique_images": 1,
            "removed_train_examples": 1,
            "validation_duplicate_examples": 0,
            "output_overlap_unique_images": 0,
        },
    }
    _patch_canonical_pixmo_source_policy(
        monkeypatch,
        vision_alignment,
        source_path,
        source_entries,
    )
    manifest_path = tmp_path / "validation.json"
    manifest_path.write_text(json.dumps(manifest))
    evaluation.validation_manifest_path = str(manifest_path)
    evaluation.validation_manifest_sha256 = vision_alignment._sha256_file(manifest_path)

    vision_alignment._validate_validation_manifest(config, source_audit)

    class ValidationDataset:
        _hf = SimpleNamespace(_fingerprint=output_entries["validation"]["dataset_fingerprint"])

        def __init__(self):
            self.validated_required_annotations = False

        def __len__(self):
            return 2

        def validate_required_annotations(self):
            self.validated_required_annotations = True

    validation_dataset = ValidationDataset()
    vision_alignment._validate_live_validation_dataset(validation_dataset, manifest)
    assert validation_dataset.validated_required_annotations is True
    with pytest.raises(ValueError, match="Live validation dataset fingerprint"):
        vision_alignment._validate_live_validation_dataset(
            SimpleNamespace(
                _hf=SimpleNamespace(_fingerprint="changed"),
                __len__=lambda: 8,
            ),
            manifest,
        )

    expected_path_digest = output_entries["train"]["row_image_paths_sha256"]
    output_entries["train"]["row_image_paths_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    evaluation.validation_manifest_sha256 = vision_alignment._sha256_file(manifest_path)
    with pytest.raises(ValueError, match="Live canonical PixMoCap output train"):
        vision_alignment._validate_validation_manifest(config, source_audit)
    output_entries["train"]["row_image_paths_sha256"] = expected_path_digest

    validation_inventory_path = tmp_path / inventory_entries["validation"]["path"]
    validation_inventory_path.write_text(
        "\n".join(sorted([inventories["train"][0], inventories["validation"][1]])) + "\n"
    )
    inventory_entries["validation"]["sha256"] = vision_alignment._sha256_file(
        validation_inventory_path
    )
    manifest_path.write_text(json.dumps(manifest))
    evaluation.validation_manifest_sha256 = vision_alignment._sha256_file(manifest_path)
    with pytest.raises(ValueError, match="not disjoint"):
        vision_alignment._validate_validation_manifest(
            config, source_audit, validate_live_datasets=False
        )


def test_pixmo_row_path_inventory_has_a_portable_digest():
    vision_alignment = _load_module()
    from datasets import Dataset

    inventory = vision_alignment.pixmo_row_path_inventory(
        Dataset.from_dict({"image": ["/images/a", "/images/b", "/images/a"]})
    )

    assert inventory == {
        "algorithm": "sha256-jsonl-index-image-path-v1",
        "rows": 3,
        "unique_paths": 2,
        "sha256": "cfcbbb377fe84bb193946d59a44d0ddec6dd583a7ea1090eb569807f98743c75",
    }


def test_caller_self_attested_v2_validation_manifest_is_rejected(tmp_path):
    vision_alignment = _load_module()
    manifest_path = tmp_path / "validation.json"
    manifest_path.write_text(
        json.dumps({"format": "vision_alignment_validation_manifest", "version": 2})
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            allow_unpinned_synthetic_smoke=False,
            pixmo_cap_path=str(tmp_path / "dataset"),
        ),
        evaluation=SimpleNamespace(
            examples_per_source=1,
            validation_manifest_path=str(manifest_path),
            validation_manifest_sha256=vision_alignment._sha256_file(manifest_path),
        ),
    )

    with pytest.raises(ValueError, match="fields differ|identity is incompatible"):
        vision_alignment._validate_validation_manifest(config, {"inputs": {}})


def test_audited_dataset_binds_offline_audit_to_live_source_identity():
    vision_alignment = _load_module()

    class Dataset:
        _hf = SimpleNamespace(_fingerprint="live-fingerprint")

        def __init__(self):
            self.example = {
                "input_ids": np.array([1, 2], dtype=np.int64),
                "labels": np.array([2, -100], dtype=np.int64),
                "loss_masks": np.array([1.0, 0.0], dtype=np.float32),
                "position_ids": np.array([0, 1], dtype=np.int64),
                "token_type_ids": np.array([0, 0], dtype=np.int64),
                "images": np.zeros((1, 2, 3), dtype=np.float32),
                "pooled_patches_idx": np.zeros((1, 4), dtype=np.int64),
            }

        def __len__(self):
            return 1

        def get(self, index, epoch=0):
            assert index == 0 and epoch == 0
            return self.example

    dataset = Dataset()
    row_hash = serialized_example_sha256(dataset.example)
    audit: Dict[str, Any] = {
        "fingerprint": "a" * 64,
        "source_registry_sha256": "d" * 64,
        "exporter_sha256": "e" * 64,
        "input_content_sha256": "b" * 64,
        "inputs": {
            "pixmo_caption": {
                "sha256": "c" * 64,
                "dataset_fingerprint": "live-fingerprint",
                "dataset_size": 1,
                "probe_indices": [0],
                "probe_indices_sha256": "f" * 64,
                "serialized_row_hashes": [row_hash],
                "serialized_row_hashes_sha256": "1" * 64,
            }
        },
        "sources": {"pixmo_caption": {"examples": {"valid": 1}}},
    }

    wrapped = vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit)
    assert len(wrapped.content_fingerprint) == 64

    dataset.example["labels"][0] = 7
    with pytest.raises(ValueError, match="serialized runtime row drifted"):
        vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit)
    dataset.example["labels"][0] = 2

    audit["inputs"]["pixmo_caption"]["dataset_fingerprint"] = "different"
    with pytest.raises(ValueError, match="Live dataset fingerprint"):
        vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit)

    audit["inputs"]["pixmo_caption"]["dataset_fingerprint"] = "live-fingerprint"
    audit["inputs"]["pixmo_caption"]["dataset_size"] = 2
    with pytest.raises(ValueError, match="dataset length"):
        vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit)


def test_joint_audited_dataset_replays_epoch_order_images_and_calibration(monkeypatch):
    vision_alignment = _load_module()
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    monkeypatch.setattr(provenance, "N_PATCHES_SQ", 2)
    monkeypatch.setattr(provenance, "PATCH_DIM", 3)
    token_ids = Molmo2TokenIds(
        im_start_id=100278,
        im_end_id=100279,
        im_patch_id=100280,
        im_col_id=100281,
        low_res_im_start_id=100282,
        image_placeholder_id=100283,
        im_end_turn_id=100265,
    )

    class Dataset:
        content_fingerprint = "a" * 64

        def __init__(self):
            self.image_salt = "pinned"

        def __len__(self):
            return 2

        def get(self, index, epoch=0):
            input_ids = np.array([index, epoch, index + epoch + 1], dtype=np.int64)
            input_ids[0] = token_ids.im_patch_id
            token_type_ids = np.isin(
                input_ids,
                np.fromiter(token_ids.image_token_ids, dtype=np.int64),
            ).astype(np.int64)
            return {
                "input_ids": input_ids,
                "labels": np.array(input_ids, copy=True),
                "loss_masks": np.ones(3, dtype=np.float32),
                "position_ids": np.arange(3, dtype=np.int64),
                "token_type_ids": token_type_ids,
                "images": np.zeros((1, 2, 3), dtype=np.float32),
                "pooled_patches_idx": np.zeros((1, 4), dtype=np.int64),
            }

        def validate_image_content(self, indices):
            return vision_alignment._canonical_sha256(
                [{"index": index, "salt": self.image_salt} for index in indices]
            )

    dataset = Dataset()
    probe_indices = [0, 1]
    row_hashes = [
        serialized_example_sha256(dataset.get(index, epoch))
        for epoch in vision_alignment._JOINT_VISUAL_PROBE_EPOCHS
        for index in probe_indices
    ]
    source = {
        "sha256": "b" * 64,
        "dataset_fingerprint": dataset.content_fingerprint,
        "dataset_size": len(dataset),
        "probe_indices": probe_indices,
        "probe_indices_sha256": vision_alignment._canonical_sha256(probe_indices),
        "probe_epochs": list(vision_alignment._JOINT_VISUAL_PROBE_EPOCHS),
        "serialized_row_hashes": row_hashes,
        "serialized_row_hashes_sha256": vision_alignment._canonical_sha256(row_hashes),
        "probe_image_content_sha256": dataset.validate_image_content(probe_indices),
        "max_observed_sequence_length": 3,
    }
    audit = {
        "format": vision_alignment._JOINT_AUDIT_FORMAT,
        "fingerprint": "c" * 64,
        "source_registry_sha256": "d" * 64,
        "exporter_implementation": {
            "path": vision_alignment._JOINT_EXPORTER_IMPLEMENTATION,
            "sha256": "e" * 64,
        },
        "input_content_sha256": "f" * 64,
        "inputs": {"pixmo_caption": source},
        "sources": {
            "pixmo_caption": {
                "mean_sum_loss_masks": 3.0,
                "zero_loss_examples": 0,
            }
        },
    }

    wrapped = vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit, token_ids=token_ids)

    assert len(wrapped.content_fingerprint) == 64
    reordered = list(row_hashes)
    reordered[0], reordered[-1] = reordered[-1], reordered[0]
    source["serialized_row_hashes"] = reordered
    with pytest.raises(ValueError, match="serialized row differs"):
        vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit, token_ids=token_ids)
    source["serialized_row_hashes"] = row_hashes

    dataset.image_salt = "drifted"
    with pytest.raises(ValueError, match="image bytes"):
        vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit, token_ids=token_ids)
    dataset.image_salt = "pinned"

    audit["sources"]["pixmo_caption"]["mean_sum_loss_masks"] = 2.0
    with pytest.raises(ValueError, match="loss mass"):
        vision_alignment._AuditedDataset(dataset, "pixmo_caption", audit, token_ids=token_ids)


def test_joint_audited_native_dataset_allows_zero_rows_but_requires_positive_aggregate(
    monkeypatch,
):
    vision_alignment = _load_module()
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    monkeypatch.setattr(provenance, "N_PATCHES_SQ", 2)
    monkeypatch.setattr(provenance, "PATCH_DIM", 3)
    token_ids = Molmo2TokenIds(
        im_start_id=100278,
        im_end_id=100279,
        im_patch_id=100280,
        im_col_id=100281,
        low_res_im_start_id=100282,
        image_placeholder_id=100283,
        im_end_turn_id=100265,
    )

    class Dataset:
        content_fingerprint = "a" * 64

        def __init__(self):
            self.all_zero = False

        def __len__(self):
            return 2

        def get(self, index, epoch=0):
            assert epoch == 0
            input_ids = np.arange(vision_alignment._JOINT_SEQUENCE_LENGTH, dtype=np.int64)
            loss_masks = np.zeros(vision_alignment._JOINT_SEQUENCE_LENGTH, dtype=np.float32)
            if index == 1 and not self.all_zero:
                loss_masks[0] = 1.0
            return {
                "input_ids": input_ids,
                "labels": np.array(input_ids, copy=True),
                "loss_masks": loss_masks,
                "position_ids": np.arange(vision_alignment._JOINT_SEQUENCE_LENGTH, dtype=np.int64),
                "token_type_ids": np.zeros(vision_alignment._JOINT_SEQUENCE_LENGTH, dtype=np.int64),
                "images": np.zeros((0, 2, 3), dtype=np.float32),
                "pooled_patches_idx": np.zeros((0, 4), dtype=np.int64),
            }

    dataset = Dataset()
    probe_indices = [0, 1]
    row_hashes = [serialized_example_sha256(dataset.get(index, 0)) for index in probe_indices]
    source = {
        "sha256": "b" * 64,
        "dataset_fingerprint": dataset.content_fingerprint,
        "dataset_size": len(dataset),
        "probe_indices": probe_indices,
        "probe_indices_sha256": vision_alignment._canonical_sha256(probe_indices),
        "probe_epochs": [0],
        "serialized_row_hashes": row_hashes,
        "serialized_row_hashes_sha256": vision_alignment._canonical_sha256(row_hashes),
        "probe_image_content_sha256": vision_alignment._canonical_sha256([]),
        "max_observed_sequence_length": vision_alignment._JOINT_SEQUENCE_LENGTH,
    }
    summary = {"mean_sum_loss_masks": 0.5, "zero_loss_examples": 1}
    audit = {
        "format": vision_alignment._JOINT_AUDIT_FORMAT,
        "fingerprint": "c" * 64,
        "source_registry_sha256": "d" * 64,
        "exporter_implementation": {
            "path": vision_alignment._JOINT_EXPORTER_IMPLEMENTATION,
            "sha256": "e" * 64,
        },
        "input_content_sha256": "f" * 64,
        "inputs": {"native_text_replay": source},
        "sources": {"native_text_replay": summary},
    }

    wrapped = vision_alignment._AuditedDataset(
        dataset, "native_text_replay", audit, token_ids=token_ids
    )

    assert len(wrapped.content_fingerprint) == 64
    summary["zero_loss_examples"] = 0
    with pytest.raises(ValueError, match="loss mass"):
        vision_alignment._AuditedDataset(dataset, "native_text_replay", audit, token_ids=token_ids)
    summary["zero_loss_examples"] = 1

    source["probe_epochs"] = [False]
    with pytest.raises(ValueError, match="epoch panel"):
        vision_alignment._AuditedDataset(dataset, "native_text_replay", audit, token_ids=token_ids)
    source["probe_epochs"] = [0]

    dataset.all_zero = True
    source["serialized_row_hashes"] = [
        serialized_example_sha256(dataset.get(index, 0)) for index in probe_indices
    ]
    with pytest.raises(ValueError, match="no aggregate supervised loss mass"):
        vision_alignment._AuditedDataset(dataset, "native_text_replay", audit, token_ids=token_ids)


def test_checked_in_smoke_profile_is_cluster_only_and_calibrated():
    import yaml

    path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "vision_alignment"
        / "bridge"
        / "synthetic_smoke.yaml"
    )
    profile = yaml.safe_load(path.read_text())

    assert profile["phase"] == "bridge"
    assert profile["launch"]["cluster"] == "ai2/holmes"
    assert profile["launch"]["priority"] == "urgent"
    assert profile["launch"]["min_runtime"] == "8h"
    assert "hostnames" not in profile["launch"]
    assert "--data.allow_unpinned_synthetic_smoke=true" in profile["overrides"]
    assert any("mean_loss_weight.pixmo_caption" in value for value in profile["overrides"])
    assert any("mean_loss_weight.pixmo_transcript" in value for value in profile["overrides"])


def test_joint_evaluates_all_visual_sources_and_deterministically_shuffles_native_holdout(
    monkeypatch, tmp_path
):
    vision_alignment = _load_module()
    loader_calls: List[Dict[str, Any]] = []

    class FakeLoader:
        def __init__(self, dataset, collator, **kwargs):
            del dataset, collator
            loader_calls.append(kwargs)

    class FakeDataset:
        def __len__(self):
            return 8

        def validate_image_content(self):
            return "a" * 64

    class FakeDatasetConfig:
        def build(self, tokenizer):
            del tokenizer
            return FakeDataset()

    callbacks: Dict[str, Any] = {}
    trainer = SimpleNamespace(
        work_dir=tmp_path,
        device="cpu",
        dp_process_group=None,
        add_callback=lambda name, callback: callbacks.setdefault(name, callback),
    )
    evaluation = SimpleNamespace(
        interval=100,
        examples_per_source=8,
        rank_batch_instances=1,
        seed=6198,
        eval_on_startup=True,
        eval_on_finish=True,
        native_text_holdout=FakeDatasetConfig(),
    )
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.joint,
        data=SimpleNamespace(sequence_length=2560),
        evaluation=evaluation,
    )

    monkeypatch.setattr(vision_alignment, "MultimodalDataLoader", FakeLoader)
    monkeypatch.setattr(
        vision_alignment,
        "MultimodalLMEvaluator",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        vision_alignment,
        "MultimodalBlankImageEvaluator",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(vision_alignment, "EvaluatorCallback", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        vision_alignment,
        "_joint_visual_projection",
        lambda config, token_ids=None: object(),
    )
    monkeypatch.setattr(
        vision_alignment,
        "build_selected_joint_dataset",
        lambda *args, **kwargs: FakeDataset(),
    )

    vision_alignment._add_intrinsic_visual_evaluators(
        trainer,
        object(),
        config,
        object(),
        Molmo2TokenIds(),
        dp_world_size=2,
        dp_rank=0,
    )

    assert len(loader_calls) == 9
    assert [call["shuffle"] for call in loader_calls] == [False] * 8 + [True]
    assert loader_calls[-1]["seed"] == evaluation.seed
    assert "vision_alignment_intrinsic_validation" in callbacks
    evaluators = callbacks["vision_alignment_intrinsic_validation"]["evaluators"]
    assert len(evaluators) == 11
    assert {evaluator.name for evaluator in evaluators} == {
        *(
            f"vision-alignment-{name}-validation"
            for name in vision_alignment.JOINT_VISUAL_SOURCE_NAMES
        ),
        "vision-alignment-pixmo_caption-blank-image",
        "vision-alignment-pixmo_transcript-blank-image",
        "vision-alignment-native-text-holdout",
    }
