"""Focused contracts for the frozen-vision midtraining recipe."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from torch import nn

from olmo_core.data import DataMix
from olmo_core.data.multimodal import (
    CoSynPointDatasetConfig,
    NumpyFSLTextDatasetConfig,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.vision import (
    Molmo2TokenIds,
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
)
from olmo_core.train import Duration, LoadStrategy


def _load_module():
    path = Path(__file__).parents[2] / "scripts" / "train" / "Vision-Midtraining.py"
    spec = importlib.util.spec_from_file_location("vision_midtraining_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vision_midtraining():
    return _load_module()


@pytest.fixture(scope="module")
def smoke_config(vision_midtraining):
    overrides = [
        "--data.synthetic_smoke=true",
        "--data.prefetch_workers=0",
        "--optimization.scheduler_warmup_tokens=0",
        "--max_tokens=1048576",
        "--checkpoint_interval=1",
        "--ephemeral_checkpoint_interval=null",
        "--max_checkpoints=1",
        "--wandb_enabled=false",
    ]
    run_name = "vision-midtraining-synthetic-smoke-v1"
    config = vision_midtraining.build_config("Vision-Midtraining.py", run_name, overrides)
    vision_midtraining._validate_contract(config, run_name, "dry_run")
    return config


def _tiny_multimodal_config() -> MultimodalLMConfig:
    lm = TransformerConfig.olmo2_1M(vocab_size=128)
    vision = VisionEncoderConfig.siglip_b16_224()
    connector = VisionConnectorConfig.from_vision_encoder(
        vision, output_dim=lm.d_model, mlp_hidden_size=64
    )
    return MultimodalLMConfig(
        lm=lm,
        vision=vision,
        connector=connector,
        image_patch_token_id=120,
    )


@pytest.fixture
def parent_fixture(tmp_path, vision_midtraining):
    model = _tiny_multimodal_config()
    checkpoint = tmp_path / "step12000"
    state_dir = checkpoint / "model_and_optim"
    state_dir.mkdir(parents=True)
    config_path = checkpoint / "config.json"
    marker_path = checkpoint / ".metadata.json"
    dcp_path = state_dir / ".metadata"
    config_path.write_text(
        json.dumps(
            {
                "model": model.as_config_dict(),
                "vision_alignment": {"phase": "joint"},
                "train_module": {"freeze_params": ["lm.lm_head.w_out.weight"]},
            },
            sort_keys=True,
        )
    )
    marker_path.write_text(json.dumps({"ephemeral": False}, sort_keys=True))
    dcp_path.write_bytes(b"test-dcp-metadata")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    artifact = vision_midtraining.ParentArtifactConfig(
        checkpoint=str(checkpoint),
        config_sha256=digest(config_path),
        marker_sha256=digest(marker_path),
        dcp_metadata_sha256=digest(dcp_path),
    )
    return artifact, model


def test_parent_loader_deserializes_exact_checkpoint_model(vision_midtraining, parent_fixture):
    artifact, expected = parent_fixture

    actual, raw = vision_midtraining._load_parent_model_config(artifact)

    assert actual.as_config_dict() == expected.as_config_dict()
    assert raw["model"] == expected.as_config_dict()


def test_parent_loader_checks_all_three_pinned_files(vision_midtraining, parent_fixture):
    artifact, _ = parent_fixture
    artifact.config_sha256 = "0" * 64

    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        vision_midtraining._load_parent_model_config(artifact)


def test_live_parent_constants_identify_locked_treatment(vision_midtraining):
    parent = vision_midtraining.ParentArtifactConfig()
    model, raw = vision_midtraining._load_parent_model_config(parent)

    assert raw["vision_alignment"]["phase"] == "joint"
    assert raw["vision_alignment"]["lineage_id"] == "vision-alignment-joint-v1"
    assert model.image_patch_token_id == raw["model"]["image_patch_token_id"]
    assert "vision.*" not in raw["train_module"]["freeze_params"]


def test_fresh_transition_and_same_folder_resume_contract(smoke_config, vision_midtraining):
    trainer = smoke_config.trainer

    assert trainer.load_path == vision_midtraining.PARENT_CHECKPOINT
    assert trainer.load_strategy is LoadStrategy.always
    assert trainer.load_optim_state is False
    assert trainer.load_trainer_state is False
    assert trainer.save_overwrite is True
    assert trainer.checkpointer.load_thread_count == 8
    assert trainer.max_duration == Duration.tokens(smoke_config.max_tokens)
    # Trainer.fit checks save_folder first with full trainer/optimizer state before load_path.
    assert trainer.save_folder.endswith("/vision-midtraining-synthetic-smoke-v1")


def test_trainability_is_frozen_vision_full_lm_and_connector(vision_midtraining):
    config = vision_midtraining._build_train_module_config(
        vision_midtraining.OptimizationConfig(), sequence_length=8192
    )

    assert config.freeze_params == ["vision.*"]
    assert config.train_embedding_rows is None
    assert not config.vision_activation_checkpointing
    assert config.connector_activation_checkpointing
    assert config.dp_config.name is DataParallelType.ddp
    assert config.ep_config.degree == 8
    assert config.optim.lr == pytest.approx(1e-5)
    assert config.optim.eps == pytest.approx(1e-8)
    assert config.optim.weight_decay == pytest.approx(0.1)
    groups = {tuple(group.params): group.opts for group in config.optim.group_overrides}
    assert groups[("*vision.*",)]["lr"] == 0.0
    assert groups[("*connector.*",)]["lr"] == pytest.approx(2e-5)
    assert groups[("*connector.*",)]["weight_decay"] == 0.0


def test_optimizer_groups_cover_output_head_embeddings_and_connector(vision_midtraining):
    config = vision_midtraining._build_train_module_config(
        vision_midtraining.OptimizationConfig(), sequence_length=8192
    )
    model = nn.Module()
    model.lm = nn.Module()
    model.lm.embeddings = nn.Embedding(16, 4)
    model.lm.embedding_norm = nn.LayerNorm(4)
    block = nn.Module()
    block.foo_norm = nn.LayerNorm(4)
    block.body = nn.Linear(4, 4, bias=False)
    model.lm.blocks = nn.ModuleList([block])
    model.lm.lm_head = nn.Module()
    model.lm.lm_head.norm = nn.LayerNorm(4)
    model.lm.lm_head.w_out = nn.Linear(4, 16, bias=False)
    model.connector = nn.Linear(4, 4)
    model.vision = nn.Linear(4, 4)
    model.vision.requires_grad_(False)

    groups = config.optim.build_groups([model])
    grouped = {name: group for group in groups for name in group["named_params"]}

    assert not any(name.startswith("vision.") for name in grouped)
    assert grouped["lm.embeddings.weight"]["weight_decay"] == 0.0
    assert "lm.lm_head.w_out.weight" in grouped
    assert grouped["connector.weight"]["lr"] == pytest.approx(2e-5)
    assert "lm.blocks.0.body.weight" in grouped


def test_real_source_configs_use_official_text_and_all_stage1_visual_adapters(
    vision_midtraining, smoke_config
):
    config = copy.deepcopy(smoke_config)
    config.data.synthetic_smoke = False
    sources = vision_midtraining._build_source_configs(config, Molmo2TokenIds())

    assert tuple(sorted(sources)) == vision_midtraining.SOURCE_NAMES
    assert isinstance(sources["text_midtraining"], NumpyFSLTextDatasetConfig)
    assert (
        sources["text_midtraining"].dataset.mix
        == DataMix.OLMo_midtraining_mix_0925_ingredient1_100B
    )
    assert isinstance(sources["pixmo_caption"], PixMoCapDatasetConfig)
    assert isinstance(sources["pixmo_transcript"], PixMoCapDatasetConfig)
    assert isinstance(sources["pixmo_points_basic"], PixMoPointsDatasetConfig)
    assert isinstance(sources["pixmo_points_high_frequency"], PixMoPointsDatasetConfig)
    assert isinstance(sources["pixmo_count"], PixMoCountDatasetConfig)
    assert isinstance(sources["cosyn_point"], CoSynPointDatasetConfig)
    assert all("tulu" not in name.lower() for name in sources)
    assert all(
        source.message_format == "document"
        for name, source in sources.items()
        if name != "text_midtraining"
    )
    assert all(
        source.loss_token_weighting == "none"
        for name, source in sources.items()
        if name != "text_midtraining"
    )
    assert config.data.max_crops == 8
    assert config.data.pack_max_crops == 16


def test_reviewed_pilot_science_is_explicit_in_config_defaults(vision_midtraining):
    experiment_fields = vision_midtraining.ExperimentConfig.__dataclass_fields__
    optimization = vision_midtraining.OptimizationConfig()
    data = vision_midtraining.VisionMidtrainingDataConfig()

    assert experiment_fields["max_tokens"].default == 10_485_760_000
    assert experiment_fields["global_batch_size"].default == 1_048_576
    assert optimization.lm_lr == pytest.approx(1e-5)
    assert optimization.connector_lr == pytest.approx(2e-5)
    assert data.target_loss_mass["text_midtraining"] == pytest.approx(0.50)
    assert sum(data.target_loss_mass.values()) == pytest.approx(1.0)
    assert data.max_crops == 8
    assert data.pack_max_crops == 16


def test_loss_mass_calibration_reconstructs_targets(vision_midtraining, smoke_config):
    weights = vision_midtraining._sampling_weights(smoke_config)
    means = {name: 1.0 for name in vision_midtraining.SOURCE_NAMES}
    delivered = vision_midtraining.expected_loss_mass(weights, means)
    targets = vision_midtraining._normalized(smoke_config.data.target_loss_mass)

    assert set(weights) == set(vision_midtraining.SOURCE_NAMES)
    assert delivered == pytest.approx(targets)


def test_loss_mass_contract_rejects_non_unit_targets(vision_midtraining, smoke_config):
    config = copy.deepcopy(smoke_config)
    config.data.target_loss_mass["text_midtraining"] = 0.49
    config.train_module.source_loss_mass_targets = dict(config.data.target_loss_mass)

    with pytest.raises(ValueError, match="sum to one"):
        vision_midtraining._validate_contract(
            config, "vision-midtraining-synthetic-smoke-v1", "dry_run"
        )


def test_launch_replays_overrides_and_real_training_requires_receipt(
    vision_midtraining, smoke_config
):
    assert "--data.synthetic_smoke=true" in smoke_config.launch.cmd
    assert "--max_tokens=1048576" in smoke_config.launch.cmd
    vision_midtraining._validate_contract(
        smoke_config,
        "vision-midtraining-synthetic-smoke-v1",
        "train",
    )
    real = copy.deepcopy(smoke_config)
    real.data.synthetic_smoke = False
    with pytest.raises(ValueError, match="requires a pinned source-mean audit"):
        vision_midtraining._validate_contract(
            real, "vision-midtraining-synthetic-smoke-v1", "train"
        )


def test_same_folder_resume_requires_exact_run_contract(
    tmp_path, monkeypatch, vision_midtraining, smoke_config
):
    checkpoint = tmp_path / "step1"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    saved = smoke_config.as_config_dict()
    config_path.write_text(json.dumps(saved))
    monkeypatch.setattr(
        vision_midtraining.Checkpointer,
        "latest_checkpoint",
        classmethod(lambda cls, folder: str(checkpoint)),
    )

    vision_midtraining._validate_output_resume_contract(smoke_config)
    saved["vision_midtraining"]["run_contract_sha256"] = "0" * 64
    config_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="different run contract"):
        vision_midtraining._validate_output_resume_contract(smoke_config)


def test_audit_indices_are_deterministic_unique_and_bounded(vision_midtraining):
    first = vision_midtraining._audit_indices(101, 50, 6198)
    second = vision_midtraining._audit_indices(101, 50, 6198)

    assert first == second
    assert len(first) == len(set(first)) == 50
    assert min(first) >= 0 and max(first) < 101


def test_real_run_loads_means_from_matching_audit_summary(
    tmp_path, vision_midtraining, smoke_config
):
    config = copy.deepcopy(smoke_config)
    config.data.synthetic_smoke = False
    sources = {
        name: {"mean_loss_weight": float(index + 1)}
        for index, name in enumerate(vision_midtraining.SOURCE_NAMES)
    }
    receipt = {
        "version": 1,
        "source_contract_sha256": config.vision_midtraining.source_contract_sha256,
        "sources": sources,
    }
    path = tmp_path / "means.json"
    path.write_text(json.dumps(receipt, sort_keys=True))
    config.data.mean_loss_weight_receipt = str(path)
    config.data.mean_loss_weight_receipt_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    assert vision_midtraining._validated_mean_receipt(config) == {
        name: value["mean_loss_weight"] for name, value in sources.items()
    }
    receipt["source_contract_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt, sort_keys=True))
    config.data.mean_loss_weight_receipt_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="different data contract"):
        vision_midtraining._validated_mean_receipt(config)
