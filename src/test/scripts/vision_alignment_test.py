"""Focused configuration contracts for the vision-alignment training recipe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision import Molmo2TokenIds

IMAGE_TOKEN_ROWS = [100, 101, 102, 103, 104, 105]


def _load_module():
    path = Path(__file__).parents[2] / "scripts" / "train" / "Vision-Alignment.py"
    spec = importlib.util.spec_from_file_location("vision_alignment_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vision_alignment():
    return _load_module()


def _optimizer_config(vision_alignment, phase="bridge", *, train_module=None):
    phase = vision_alignment.VisionAlignmentPhase(phase)
    policy = vision_alignment._PHASE_POLICIES[phase]
    return SimpleNamespace(
        phase=phase,
        train_module=(
            train_module
            if train_module is not None
            else vision_alignment._build_train_module_config(policy, IMAGE_TOKEN_ROWS)
        ),
        trainer=SimpleNamespace(max_duration=vision_alignment.Duration.steps(policy.max_steps)),
        data=SimpleNamespace(allow_unpinned_synthetic_smoke=False),
    )


def _canonical_policy_config(vision_alignment, phase):
    phase = vision_alignment.VisionAlignmentPhase(phase)
    mixture = vision_alignment.VisionAlignmentMixtureConfig(phase=phase.value)
    mixture.mean_loss_weight = {source: 1.0 for source in mixture.resolved_targets()}
    return SimpleNamespace(
        phase=phase,
        artifacts=vision_alignment.ArtifactConfig(),
        data=vision_alignment.VisionAlignmentDataConfig(mixture=mixture),
    )


def _resume_config(vision_alignment, parent):
    phase = vision_alignment.VisionAlignmentPhase.perception
    return SimpleNamespace(
        phase=phase,
        initialization=vision_alignment.InitializationConfig(
            mode=vision_alignment.InitializationMode.checkpoint,
            checkpoint=parent,
            expected_parent_phase=vision_alignment.VisionAlignmentPhase.bridge,
        ),
        vision_alignment=vision_alignment.VisionAlignmentMetadataConfig(
            phase=phase,
            lineage_id="alignment-run",
            parent_checkpoint=parent,
            data_contract_sha256="d" * 64,
            trainable_contract_sha256="e" * 64,
        ),
    )


def _native_replay_case(monkeypatch, vision_alignment):
    artifacts = vision_alignment.ArtifactConfig()
    parent_paths = ("s3://bucket/parent-a.npy", "s3://bucket/parent-b.npy")
    remote_sources = tuple({"parent_path": path, "size_bytes": 8192 * 4} for path in parent_paths)
    pair_calls = []
    receipt = SimpleNamespace(
        version=3,
        parent_paths_sha256=artifacts.base_data_paths_sha256,
        parent_mix_sha256=vision_alignment.BASE_PARENT_MIX_SHA256,
        parent_config_sha256=artifacts.base_config_sha256,
        parent_trainer_state_sha256=vision_alignment.BASE_TRAINER_STATE_SHA256,
        parent_dataset_fingerprint=vision_alignment.BASE_DATASET_FINGERPRINT,
        remote_snapshot_sha256="1" * 64,
        compact_materialization_sha256="2" * 64,
        remote_sources=remote_sources,
        validate_pair=lambda train, holdout: pair_calls.append((train, holdout)),
    )
    common_lineage = {
        "parent_checkpoint": artifacts.base_checkpoint,
        "parent_mix": artifacts.parent_text_mix,
        "parent_paths_sha256": artifacts.base_data_paths_sha256,
        "parent_config_sha256": artifacts.base_config_sha256,
        "parent_trainer_state_sha256": vision_alignment.BASE_TRAINER_STATE_SHA256,
        "parent_dataset_fingerprint": vision_alignment.BASE_DATASET_FINGERPRINT,
        "remote_snapshot_sha256": receipt.remote_snapshot_sha256,
        "compact_materialization_sha256": receipt.compact_materialization_sha256,
        "selection_algorithm": vision_alignment._JOINT_NATIVE_REPLAY_SELECTION_ALGORITHM,
        "selection_seed": vision_alignment._JOINT_NATIVE_REPLAY_SELECTION_SEED,
    }
    train = SimpleNamespace(
        version=3,
        sequence_length=vision_alignment._JOINT_SEQUENCE_LENGTH,
        num_windows=1024,
        provenance={**common_lineage, "split": "train"},
    )
    holdout = SimpleNamespace(
        version=3,
        sequence_length=vision_alignment._JOINT_SEQUENCE_LENGTH,
        num_windows=1024,
        provenance={**common_lineage, "split": "holdout"},
    )

    def replay_config(manifest, fingerprint):
        return SimpleNamespace(
            expected_parent_checkpoint=artifacts.base_checkpoint,
            expected_parent_mix=artifacts.parent_text_mix,
            expected_parent_paths_sha256=artifacts.base_data_paths_sha256,
            expected_fingerprint=fingerprint,
            validate_source_files=True,
            verify_source_hashes=False,
            verification_receipt_path="/artifacts/replay-receipt.json",
            expected_verification_receipt_sha256="3" * 64,
            build=lambda: SimpleNamespace(manifest=manifest),
        )

    train_fingerprint = "train-fingerprint"
    holdout_fingerprint = "holdout-fingerprint"
    config = SimpleNamespace(
        artifacts=artifacts,
        data=SimpleNamespace(
            sequence_length=vision_alignment._JOINT_SEQUENCE_LENGTH,
            native_text_replay=replay_config(train, train_fingerprint),
            native_text_replay_fingerprint=train_fingerprint,
        ),
        evaluation=SimpleNamespace(
            examples_per_source=512,
            native_text_holdout=replay_config(holdout, holdout_fingerprint),
            native_text_holdout_fingerprint=holdout_fingerprint,
        ),
    )
    monkeypatch.setattr(
        vision_alignment.NativeTextReplayVerificationReceipt,
        "load",
        staticmethod(lambda *args, **kwargs: receipt),
    )
    monkeypatch.setattr(
        vision_alignment,
        "_load_pinned_native_parent_paths",
        lambda unused: parent_paths,
    )
    monkeypatch.setattr(
        vision_alignment,
        "_native_parent_dataset_fingerprint",
        lambda unused: vision_alignment.BASE_DATASET_FINGERPRINT,
    )
    return SimpleNamespace(
        config=config,
        holdout=holdout,
        pair_calls=pair_calls,
        receipt=receipt,
        train=train,
    )


def _dict_config(**values: Any):
    return SimpleNamespace(values=values, as_config_dict=lambda: dict(values))


def _contract_config(vision_alignment, phase="bridge"):
    phase = vision_alignment.VisionAlignmentPhase(phase)
    policy = vision_alignment._PHASE_POLICIES[phase]
    return SimpleNamespace(
        phase=phase,
        data=vision_alignment.VisionAlignmentDataConfig(),
        evaluation=vision_alignment.VisionAlignmentEvalConfig(),
        collator=_dict_config(pad_sequence_length=policy.sequence_length),
        global_batch_size=128 * policy.sequence_length,
        data_seed=vision_alignment.DATA_SEED,
        model=_dict_config(model_revision=1),
        train_module=vision_alignment._build_train_module_config(policy, IMAGE_TOKEN_ROWS),
        trainer=SimpleNamespace(max_duration=vision_alignment.Duration.steps(policy.max_steps)),
        router_lb_loss_weight=0.015,
        vision_alignment=vision_alignment.VisionAlignmentMetadataConfig(),
    )


def _serialized_example(index: int, epoch: int):
    values = np.asarray([index, epoch], dtype=np.int64)
    return {
        "input_ids": values,
        "labels": values.copy(),
        "loss_masks": np.ones(2, dtype=np.float32),
        "position_ids": np.arange(2, dtype=np.int64),
        "token_type_ids": np.zeros(2, dtype=np.int64),
        "images": np.zeros((1, 1, 1, 3), dtype=np.float32),
        "pooled_patches_idx": np.zeros((1,), dtype=np.int64),
    }


class _ProbeDataset:
    content_fingerprint = "live-dataset"

    def __len__(self):
        return 2

    def get(self, index, epoch=0):
        return _serialized_example(index, epoch)

    def validate_image_content(self, indices):
        assert indices == [0, 1]
        return "image-digest"


def _probe_audit(vision_alignment, phase, raw_epochs):
    epochs = (
        (0,)
        if raw_epochs is None
        else tuple(range(raw_epochs))
        if isinstance(raw_epochs, int)
        else tuple(raw_epochs)
    )
    row_hashes = [
        vision_alignment.serialized_example_sha256(_serialized_example(index, epoch))
        for epoch in epochs
        for index in (0, 1)
    ]
    source = {
        "dataset_fingerprint": "live-dataset",
        "dataset_size": 2,
        "sha256": "a" * 64,
        "probe_indices": [0, 1],
        "probe_indices_sha256": vision_alignment._canonical_sha256([0, 1]),
        "serialized_row_hashes": row_hashes,
        "serialized_row_hashes_sha256": vision_alignment._canonical_sha256(row_hashes),
    }
    if raw_epochs is not None:
        source["probe_epochs"] = raw_epochs
        source["probe_image_content_sha256"] = "image-digest"
    audit = {
        "format": phase,
        "fingerprint": "f" * 64,
        "source_registry_sha256": "b" * 64,
        "input_content_sha256": "c" * 64,
        "inputs": {"pixmo_caption": source},
    }
    if phase == vision_alignment._JOINT_AUDIT_FORMAT:
        audit["exporter_implementation"] = {"sha256": "d" * 64}
    else:
        audit["exporter_sha256"] = "d" * 64
    return audit


@pytest.mark.parametrize(
    (
        "phase",
        "initialization_mode",
        "parent_phase",
        "freeze_params",
        "sequence_length",
        "microbatch_instances",
        "max_steps",
        "connector_lr",
        "vision_lr",
        "lm_lr",
        "connector_t_max",
    ),
    [
        (
            "bridge",
            "bare",
            None,
            ["vision.*", "lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"],
            2560,
            4,
            1000,
            2e-4,
            0.0,
            0.0,
            250,
        ),
        (
            "perception",
            "checkpoint",
            "bridge",
            ["lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"],
            2560,
            4,
            4000,
            5e-5,
            3e-6,
            0.0,
            None,
        ),
        (
            "joint",
            "checkpoint",
            "perception",
            ["lm.lm_head.w_out.weight"],
            8192,
            1,
            16000,
            2e-5,
            2e-6,
            1e-6,
            None,
        ),
    ],
)
def test_phase_defaults_trainability_and_optimizer_groups(
    vision_alignment,
    phase,
    initialization_mode,
    parent_phase,
    freeze_params,
    sequence_length,
    microbatch_instances,
    max_steps,
    connector_lr,
    vision_lr,
    lm_lr,
    connector_t_max,
):
    selected = vision_alignment._extract_phase([f"--phase={phase}"])
    policy = vision_alignment._PHASE_POLICIES[selected]
    train_module = vision_alignment._build_train_module_config(policy, IMAGE_TOKEN_ROWS)
    evaluation = vision_alignment._build_evaluation_config(policy)

    assert policy.initialization_mode.value == initialization_mode
    assert getattr(policy.expected_parent_phase, "value", None) == parent_phase
    assert policy.sequence_length == sequence_length
    assert policy.max_steps == max_steps
    assert train_module.freeze_params == freeze_params
    assert train_module.train_embedding_rows == IMAGE_TOKEN_ROWS
    assert train_module.rank_microbatch_size == microbatch_instances * sequence_length
    assert evaluation.rank_batch_instances == microbatch_instances
    assert train_module.optim.lr == (lm_lr if lm_lr else connector_lr)
    assert [group.params for group in train_module.optim.group_overrides] == [
        ["*lm.embeddings.weight"],
        ["*connector.*"],
        ["*vision.*"],
    ]
    assert [group.opts["lr"] for group in train_module.optim.group_overrides] == [
        connector_lr,
        connector_lr,
        vision_lr,
    ]
    assert train_module.scheduler.schedulers["connector"].t_max == connector_t_max


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
def test_optimizer_scheduler_contract_rejects_drift(vision_alignment, mutation):
    policy = vision_alignment._PHASE_POLICIES[vision_alignment.VisionAlignmentPhase.bridge]
    config = _optimizer_config(vision_alignment)
    train_module = config.train_module
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
        config.trainer.max_duration = vision_alignment.Duration.steps(policy.connector_t_max - 1)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        vision_alignment._validate_optimizer_scheduler_contract(config, policy)


def test_profile_owns_phase_and_forbids_second_selector(tmp_path, vision_alignment):
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "version: 1",
                "name: bridge-run",
                "phase: bridge",
                "overrides:",
                "  - --data.prefetch_workers=0",
            ]
        )
    )

    loaded, overrides = vision_alignment._load_profile([f"--profile={profile}"])

    assert loaded is not None and loaded["phase"] == "bridge"
    assert overrides == ["--phase=bridge", "--data.prefetch_workers=0"]
    with pytest.raises(ValueError, match="not both"):
        vision_alignment._load_profile([f"--profile={profile}", "--phase=perception"])


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("version: 1\nphase: bridge\nphase: perception\n", "duplicate key"),
        (
            "\n".join(
                [
                    "version: 1",
                    "phase: bridge",
                    "overrides:",
                    "  - --data.prefetch_workers=0",
                    "  - --data.prefetch_workers=1",
                ]
            ),
            "repeat a destination",
        ),
    ],
)
def test_profile_rejects_duplicate_keys(tmp_path, vision_alignment, contents, message):
    profile = tmp_path / "profile.yaml"
    profile.write_text(contents)

    with pytest.raises(ValueError, match=message):
        vision_alignment._load_profile([f"--profile={profile}"])


def test_same_phase_resume_requires_saved_full_state_contract(monkeypatch, vision_alignment):
    existing = "/checkpoints/alignment-run/step100"
    parent = "/checkpoints/bridge/step500"
    parent_sha256 = "a" * 64
    config = _resume_config(vision_alignment, parent)
    saved = {
        "vision_alignment": {
            "recipe_version": vision_alignment.RECIPE_VERSION,
            "formatter_version": vision_alignment.FORMATTER_VERSION,
            "phase": "perception",
            "lineage_id": "alignment-run",
            "parent_checkpoint": parent,
            "parent_config_sha256": parent_sha256,
            "data_contract_sha256": "d" * 64,
            "trainable_contract_sha256": "e" * 64,
        }
    }
    monkeypatch.setattr(vision_alignment, "_latest_output_checkpoint", lambda unused: existing)
    monkeypatch.setattr(
        vision_alignment,
        "_checkpoint_config",
        lambda checkpoint: (saved, "f" * 64),
    )
    vision_alignment._validate_parent_or_resume(config)

    assert config.initialization.parent_config_sha256 == parent_sha256
    saved["vision_alignment"]["data_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data_contract_sha256"):
        vision_alignment._validate_parent_or_resume(_resume_config(vision_alignment, parent))


@pytest.mark.parametrize("mutation", [None, "wrong_phase", "wrong_sha", "ephemeral"])
def test_fresh_transition_binds_exact_permanent_parent(
    tmp_path, monkeypatch, vision_alignment, mutation
):
    parent_path = (tmp_path / "step500").resolve()
    parent_path.mkdir()
    marker_path = parent_path / ".metadata.json"
    marker_path.write_text(json.dumps({"ephemeral": mutation == "ephemeral"}))
    parent = str(parent_path)
    parent_sha256 = "a" * 64
    parent_config = {
        "vision_alignment": {
            "phase": "perception" if mutation == "wrong_phase" else "bridge",
            "recipe_version": vision_alignment.RECIPE_VERSION,
        }
    }
    config = _resume_config(vision_alignment, parent)
    config.initialization.parent_config_sha256 = (
        "0" * 64 if mutation == "wrong_sha" else parent_sha256
    )
    monkeypatch.setattr(vision_alignment, "_latest_output_checkpoint", lambda unused: None)
    monkeypatch.setattr(
        vision_alignment,
        "_checkpoint_config",
        lambda checkpoint: (parent_config, parent_sha256),
    )
    if mutation is None:
        vision_alignment._validate_parent_or_resume(config)
        assert config.initialization.parent_config_sha256 == parent_sha256
        assert config.vision_alignment.parent_config_sha256 == parent_sha256
    else:
        with pytest.raises(ValueError):
            vision_alignment._validate_parent_or_resume(config)


def test_joint_transition_rejects_incompatible_parent_recipe(
    tmp_path, monkeypatch, vision_alignment
):
    parent_path = (tmp_path / "step4000").resolve()
    parent_path.mkdir()
    (parent_path / ".metadata.json").write_text(json.dumps({"ephemeral": False}))
    config = _resume_config(vision_alignment, str(parent_path))
    config.phase = vision_alignment.VisionAlignmentPhase.joint
    config.initialization.expected_parent_phase = vision_alignment.VisionAlignmentPhase.perception
    config.vision_alignment.phase = vision_alignment.VisionAlignmentPhase.joint
    monkeypatch.setattr(vision_alignment, "_latest_output_checkpoint", lambda unused: None)
    monkeypatch.setattr(
        vision_alignment,
        "_checkpoint_config",
        lambda checkpoint: (
            {
                "vision_alignment": {
                    "phase": "perception",
                    "recipe_version": vision_alignment.RECIPE_VERSION + 1,
                }
            },
            "a" * 64,
        ),
    )

    with pytest.raises(ValueError, match="incompatible recipe version"):
        vision_alignment._validate_parent_or_resume(config)


def test_load_strategy_distinguishes_resume_from_model_only_transition(
    monkeypatch, vision_alignment
):
    def merge(config, overrides):
        for override in overrides:
            if override.startswith("--initialization.checkpoint="):
                config.initialization.checkpoint = override.split("=", 1)[1]
        return config

    def launch_config(**kwargs):
        return vision_alignment.BeakerLaunchConfig(
            name=kwargs["name"],
            cmd=kwargs["cmd"],
            clusters=[kwargs["cluster"]],
            workspace=kwargs["workspace"],
            budget=kwargs["budget"],
            num_nodes=kwargs["num_nodes"],
        )

    monkeypatch.setattr(vision_alignment.ExperimentConfig, "merge", merge)
    monkeypatch.setattr(
        vision_alignment,
        "_load_tokenizer",
        lambda artifacts: (SimpleNamespace(pad_token_id=0), Molmo2TokenIds()),
    )
    monkeypatch.setattr(
        vision_alignment,
        "_build_model_config",
        lambda token_ids, artifacts: SimpleNamespace(lm=object()),
    )
    monkeypatch.setattr(vision_alignment, "build_launch_config", launch_config)
    monkeypatch.setattr(vision_alignment, "_configure_launch_runtime", lambda config: None)
    monkeypatch.setattr(
        vision_alignment,
        "_configure_router_load_balancing",
        lambda lm, weight: None,
    )
    monkeypatch.setattr(vision_alignment, "_validate_phase_contract", lambda *args: None)

    resume = vision_alignment.build_config("recipe.py", "bridge-run", ["--phase=bridge"])
    transition = vision_alignment.build_config(
        "recipe.py",
        "perception-run",
        [
            "--phase=perception",
            "--initialization.checkpoint=/checkpoints/bridge/step500",
        ],
    )

    assert resume.trainer.load_strategy is vision_alignment.LoadStrategy.if_available
    assert resume.trainer.load_path is None
    assert resume.trainer.load_optim_state is None
    assert resume.trainer.load_trainer_state is None
    assert transition.trainer.load_strategy is vision_alignment.LoadStrategy.always
    assert transition.trainer.load_path == "/checkpoints/bridge/step500"
    assert transition.trainer.load_optim_state is False
    assert transition.trainer.load_trainer_state is False


@pytest.mark.parametrize(
    ("phase", "source_count", "has_native"),
    [("bridge", 2, False), ("perception", 8, False), ("joint", 9, True)],
)
def test_source_config_and_phase_mixture_are_canonical(
    vision_alignment, phase, source_count, has_native
):
    config = _canonical_policy_config(vision_alignment, phase)
    config.data.native_text_replay_fingerprint = "native-fingerprint" if phase == "joint" else None

    vision_alignment._validate_canonical_data_policy(config)
    spec = vision_alignment._source_spec(config)

    targets = config.data.mixture.resolved_targets()
    assert len(targets) == source_count
    assert ("native_text_replay" in targets) is has_native
    assert spec.phase == phase
    assert spec.sequence_length == config.data.sequence_length
    assert (spec.message_format, spec.caption_prompt, spec.transcript_prompt) == (
        "document",
        "Description:",
        "Transcript:",
    )
    assert spec.native_text_replay_fingerprint == config.data.native_text_replay_fingerprint
    assert len(spec.preprocessing_sha256) == 64
    original_fingerprint = spec.preprocessing_sha256
    config.data.source_audit_path = "/artifacts/audit.json"
    assert vision_alignment._source_spec(config).preprocessing_sha256 == original_fingerprint
    config.data.caption_prompt = "Caption:"
    assert vision_alignment._source_spec(config).preprocessing_sha256 != original_fingerprint


@pytest.mark.parametrize(
    "mutation",
    [
        "message_format",
        "loss_token_weighting",
        "caption_prompt",
        "transcript_prompt",
        "max_crops",
        "pack_sequences",
        "pack_buffer_size",
        "pack_max_crops",
        "source_set",
        "target_ratio",
        "calibration",
    ],
)
def test_source_config_rejects_structural_or_ratio_drift(vision_alignment, mutation):
    config = _canonical_policy_config(vision_alignment, "bridge")
    drifted = {
        "message_format": "olmo3_chat",
        "loss_token_weighting": "none",
        "caption_prompt": "Caption:",
        "transcript_prompt": "Audio transcript:",
        "max_crops": 4,
        "pack_sequences": False,
        "pack_buffer_size": 12,
        "pack_max_crops": 8,
    }
    if mutation in drifted:
        setattr(config.data, mutation, drifted[mutation])
    elif mutation == "source_set":
        config.data.mixture.target_loss_mass = {
            "pixmo_caption": 0.69,
            "pixmo_transcript": 0.30,
            "unapproved_source": 0.01,
        }
    elif mutation == "target_ratio":
        config.data.mixture.target_loss_mass = {
            "pixmo_caption": 0.60,
            "pixmo_transcript": 0.40,
        }
    elif mutation == "calibration":
        config.data.mixture.mean_loss_weight.pop("pixmo_transcript")
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        vision_alignment._validate_canonical_data_policy(config)


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
def test_native_replay_artifacts_are_joint_only(vision_alignment, phase, scope, field_name, value):
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase(phase),
        data=vision_alignment.VisionAlignmentDataConfig(),
        evaluation=vision_alignment.VisionAlignmentEvalConfig(),
    )
    setattr(getattr(config, scope), field_name, value)

    with pytest.raises(ValueError, match="forbidden outside joint"):
        vision_alignment._validate_native_artifact_phase(config)


@pytest.mark.parametrize(
    "mutation",
    [
        "train_fingerprint",
        "receipt_path",
        "parent_mix",
        "receipt_lineage",
        "manifest_lineage",
        "pair_overlap",
    ],
)
def test_joint_native_replay_rejects_binding_drift(monkeypatch, vision_alignment, mutation):
    case = _native_replay_case(monkeypatch, vision_alignment)
    vision_alignment._validate_native_replay_pair(case.config)
    assert case.pair_calls == [(case.train, case.holdout)]
    case.pair_calls.clear()
    if mutation == "train_fingerprint":
        case.config.data.native_text_replay_fingerprint = "changed"
    elif mutation == "receipt_path":
        case.config.evaluation.native_text_holdout.verification_receipt_path = "/other.json"
    elif mutation == "parent_mix":
        case.config.data.native_text_replay.expected_parent_mix = "changed"
    elif mutation == "receipt_lineage":
        case.receipt.parent_config_sha256 = "0" * 64
    elif mutation == "manifest_lineage":
        case.train.provenance["parent_config_sha256"] = "0" * 64
    elif mutation == "pair_overlap":
        case.receipt.validate_pair = lambda *args: (_ for _ in ()).throw(
            OLMoConfigurationError("overlap in parent path")
        )
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises((ValueError, OLMoConfigurationError)):
        vision_alignment._validate_native_replay_pair(case.config)


@pytest.mark.parametrize(
    ("audit_format", "probe_epochs"),
    [
        ("vision_alignment_source_audit", None),
        ("vision_alignment_perception_source_audit", 2),
        ("vision_alignment_joint_source_audit", [0, 1]),
    ],
)
def test_audited_dataset_replays_epoch_panel_and_preserves_fingerprint(
    vision_alignment, audit_format, probe_epochs
):
    audit = _probe_audit(vision_alignment, audit_format, probe_epochs)
    wrapped = vision_alignment._AuditedDataset(_ProbeDataset(), "pixmo_caption", audit)
    source = audit["inputs"]["pixmo_caption"]
    expected_payload = {
        "audit_fingerprint": audit["fingerprint"],
        "source_registry_sha256": audit["source_registry_sha256"],
        "input_content_sha256": audit["input_content_sha256"],
        "source": "pixmo_caption",
        "source_sha256": source["sha256"],
        "probe_indices_sha256": source["probe_indices_sha256"],
        "serialized_row_hashes_sha256": source["serialized_row_hashes_sha256"],
        "runtime_dataset_fingerprint": "live-dataset",
        "runtime_dataset_length": 2,
        (
            "exporter_implementation"
            if audit_format == vision_alignment._JOINT_AUDIT_FORMAT
            else "exporter_sha256"
        ): audit[
            (
                "exporter_implementation"
                if audit_format == vision_alignment._JOINT_AUDIT_FORMAT
                else "exporter_sha256"
            )
        ],
    }
    if probe_epochs is not None:
        expected_payload["probe_image_content_sha256"] = "image-digest"
        expected_payload["probe_epochs"] = probe_epochs

    assert wrapped.content_fingerprint == vision_alignment._canonical_sha256(expected_payload)
    source["serialized_row_hashes"][0] = "0" * 64
    source["serialized_row_hashes_sha256"] = vision_alignment._canonical_sha256(
        source["serialized_row_hashes"]
    )
    with pytest.raises(ValueError, match="serialized row differs"):
        vision_alignment._AuditedDataset(_ProbeDataset(), "pixmo_caption", audit)


def test_source_audit_consumer_binds_config_and_probe_digests(
    tmp_path, monkeypatch, vision_alignment
):
    config = _canonical_policy_config(vision_alignment, "bridge")
    config.trainer = SimpleNamespace(max_duration=vision_alignment.Duration.steps(500))
    config.evaluation = vision_alignment.VisionAlignmentEvalConfig(interval=100)
    preprocessing_sha = "e" * 64
    monkeypatch.setattr(
        vision_alignment, "_preprocessing_config_sha256", lambda unused: preprocessing_sha
    )
    targets = config.data.mixture.resolved_targets()
    means = config.data.mixture.mean_loss_weight
    sampling = config.data.mixture.sampling_weights()
    indices = [0]
    hashes = ["1" * 64]
    inputs = {
        name: {
            "dataset_fingerprint": f"{name}-dataset",
            "dataset_size": 1,
            "sha256": "2" * 64,
            "probe_indices": indices,
            "probe_indices_sha256": vision_alignment._canonical_sha256(indices),
            "serialized_row_hashes": hashes,
            "serialized_row_hashes_sha256": vision_alignment._canonical_sha256(hashes),
        }
        for name in targets
    }
    audit = {
        "format": "vision_alignment_source_audit",
        "version": 2,
        "status": "ok",
        "phase": "bridge",
        "recipe_version": vision_alignment.RECIPE_VERSION,
        "formatter_version": vision_alignment.FORMATTER_VERSION,
        "preprocessing_config_sha256": preprocessing_sha,
        "target_loss_mass": targets,
        "mean_loss_weight": means,
        "sampling_probabilities": sampling,
        "expected_loss_mass": vision_alignment.expected_loss_mass(sampling, means),
        "inputs": inputs,
        "sources": {
            name: {"mean_sum_loss_masks": means[name], "error_samples": []} for name in targets
        },
        "failures": [],
        "source_registry_sha256": "3" * 64,
        "input_content_sha256": "4" * 64,
        "exporter_sha256": "5" * 64,
    }
    audit["fingerprint"] = vision_alignment._canonical_sha256(audit)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    config.data.source_audit_path = str(path)
    config.data.source_audit_fingerprint = audit["fingerprint"]

    assert vision_alignment._validated_source_audit(config) == audit
    audit["preprocessing_config_sha256"] = "0" * 64
    unsigned = dict(audit)
    unsigned.pop("fingerprint")
    audit["fingerprint"] = vision_alignment._canonical_sha256(unsigned)
    path.write_text(json.dumps(audit))
    config.data.source_audit_fingerprint = audit["fingerprint"]
    with pytest.raises(ValueError, match="preprocessing differs"):
        vision_alignment._validated_source_audit(config)


def test_joint_preprocessing_identity_includes_native_replay(monkeypatch, vision_alignment):
    visual = {"phase": "joint", "sequence_length": 8192}
    replay_fingerprint = "a" * 64
    config = SimpleNamespace(
        phase=vision_alignment.VisionAlignmentPhase.joint,
        data=SimpleNamespace(native_text_replay_fingerprint=replay_fingerprint),
    )
    monkeypatch.setattr(
        vision_alignment,
        "_joint_visual_projection",
        lambda unused: SimpleNamespace(
            source_spec=SimpleNamespace(as_canonical_dict=lambda: visual)
        ),
    )

    assert vision_alignment._preprocessing_config_sha256(
        config
    ) == vision_alignment._canonical_sha256(
        {"visual": visual, "native_text_replay_fingerprint": replay_fingerprint}
    )


def test_validation_manifest_consumer_binds_bridge_split(tmp_path, vision_alignment):
    artifact_root = tmp_path / "artifact"
    dataset_path = artifact_root / "dataset"
    dataset_path.mkdir(parents=True)
    fields: dict[str, Any] = {
        "dataset_fingerprint": "validation-data",
        "examples": 512,
        "row_image_paths_sha256": "1" * 64,
        "row_image_content_sha256": "2" * 64,
        "unique_image_paths": 500,
        "unique_image_content": 500,
    }
    output_train = {**fields, "dataset_fingerprint": "train-data", "examples": 1000}
    manifest: dict[str, Any] = {
        "format": "vision_alignment_validation_manifest",
        "version": 3,
        "source": {
            "splits": {
                "train": output_train,
                "validation": {**fields, "dataset_fingerprint": "source-validation-data"},
            }
        },
        "output": {
            "dataset_path": "dataset",
            "splits": {"train": output_train, "validation": fields},
        },
        "filtering": {"output_overlap_unique_images": 0},
        "inventories": {"train": {"sha256": "3" * 64}},
    }
    path = artifact_root / "manifest.json"
    raw = json.dumps(manifest).encode()
    path.write_bytes(raw)
    config = SimpleNamespace(
        data=SimpleNamespace(
            allow_unpinned_synthetic_smoke=False,
            pixmo_cap_path=str(dataset_path),
        ),
        evaluation=SimpleNamespace(
            validation_manifest_path=str(path),
            validation_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            examples_per_source=512,
        ),
    )
    audit = {
        "image_manifest_sha256": "3" * 64,
        "inputs": {
            name: {"dataset_fingerprint": "train-data", "dataset_size": 1000}
            for name in ("pixmo_caption", "pixmo_transcript")
        },
    }

    assert vision_alignment._validate_validation_manifest(config, audit) == manifest
    manifest["filtering"]["output_overlap_unique_images"] = 1
    raw = json.dumps(manifest).encode()
    path.write_bytes(raw)
    config.evaluation.validation_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="not disjoint"):
        vision_alignment._validate_validation_manifest(config, audit)


@pytest.mark.parametrize(
    ("phase", "has_legacy_treatment_key"),
    [("bridge", False), ("perception", True), ("joint", True)],
)
def test_trainable_contract_preserves_phase_specific_checkpoint_schema(
    vision_alignment, phase, has_legacy_treatment_key
):
    config = _contract_config(vision_alignment, phase)
    vision_alignment._set_contract_hashes(config)
    payload = {
        "model": config.model.as_config_dict(),
        "train_module": config.train_module.as_config_dict(),
        "router_lb_loss_weight": config.router_lb_loss_weight,
        "max_duration": {
            "value": config.trainer.max_duration.value,
            "unit": config.trainer.max_duration.unit.value,
        },
    }
    if has_legacy_treatment_key:
        payload["perception_trainability_arm"] = "treatment"
    assert config.vision_alignment.trainable_contract_sha256 == (
        vision_alignment._canonical_sha256(payload)
    )


@pytest.mark.parametrize(
    ("mutation", "changed_contract"),
    [
        ("data", "data"),
        ("collator", "data"),
        ("model", "trainable"),
        ("optimizer", "trainable"),
    ],
)
def test_config_contract_hashes_are_stable_and_sensitive(
    vision_alignment, mutation, changed_contract
):
    config = _contract_config(vision_alignment)
    vision_alignment._set_contract_hashes(config)
    original_data = config.vision_alignment.data_contract_sha256
    original_trainable = config.vision_alignment.trainable_contract_sha256
    vision_alignment._set_contract_hashes(config)
    assert config.vision_alignment.data_contract_sha256 == original_data
    assert config.vision_alignment.trainable_contract_sha256 == original_trainable

    if mutation == "data":
        config.data.caption_prompt = "Caption:"
    elif mutation == "collator":
        config.collator.values["pad_sequence_length"] = 8192
    elif mutation == "model":
        config.model.values["model_revision"] = 2
    elif mutation == "optimizer":
        config.train_module.optim.lr = 1e-6
    else:  # pragma: no cover
        raise AssertionError(mutation)

    vision_alignment._set_contract_hashes(config)
    if changed_contract == "data":
        assert config.vision_alignment.data_contract_sha256 != original_data
        assert config.vision_alignment.trainable_contract_sha256 == original_trainable
    else:
        assert config.vision_alignment.data_contract_sha256 == original_data
        assert config.vision_alignment.trainable_contract_sha256 != original_trainable
