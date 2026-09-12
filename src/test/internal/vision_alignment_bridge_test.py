import importlib
import json

import pytest

from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.vision_alignment import VisionAlignmentExperimentConfig
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    build_visual_sources,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.train import Duration, LoadStrategy
from olmo_core.train.callbacks.metric_saver import MetricSaverCallback
from olmo_core.train.callbacks.restore_metrics import RestoreMetricsCallback

recipe_tests = importlib.import_module("test.internal.vision_alignment_test")
alignment_recipe = recipe_tests.alignment_recipe

BRIDGE_MEANS = {
    "pixmo_caption": 28.829104933305643,
    "pixmo_transcript": 29.77422616124386,
}


def test_bridge_geometry_and_loader_defaults(alignment_recipe):
    config = alignment_recipe.build()
    loader = config.data_loader
    assert loader.sequence_length == config.train_module.max_sequence_length == 8192
    assert loader.global_batch_size == 128 * 8192
    assert config.train_module.rank_microbatch_size == 4 * 8192
    assert loader.pack and loader.pack_max_crops == 64
    assert loader.pack_buffer_size == 48
    assert loader.prefetch_workers == 8
    assert loader.prefetch_max_in_flight is None
    assert loader.defer_packed_image_copy is None
    assert loader.max_consecutive_data_errors == loader.max_total_data_errors == 0
    assert loader.seed == 95818
    assert config.init_seed == 6198
    assert not config.train_module.trim_microbatch_image_padding
    assert list(config.dataset.sources) == ["pixmo_caption", "pixmo_transcript"]
    assert all(source.max_sequence_length == 8192 for source in config.dataset.sources.values())
    assert not any(
        isinstance(source, PretrainingReplayConfig) for source in config.dataset.sources.values()
    )


def test_bridge_default_loss_calibration(alignment_recipe):
    config = alignment_recipe.build(
        include_means=False,
        overrides=[f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}"],
    )
    assert ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"] == config.dataset.mean_loss_weight == BRIDGE_MEANS
    assert config.dataset.target_loss_mass == {"pixmo_caption": 0.7, "pixmo_transcript": 0.3}
    assert config.train_module.source_loss_mass_targets == config.dataset.target_loss_mass
    unnormalized = {
        name: target / BRIDGE_MEANS[name]
        for name, target in config.dataset.target_loss_mass.items()
    }
    expected = {name: value / sum(unnormalized.values()) for name, value in unnormalized.items()}
    assert config.dataset.sampling_weights() == pytest.approx(expected)


def test_bridge_nondefault_context_requires_calibration(alignment_recipe):
    overrides = [
        f"--recipe.artifact_root={DEFAULT_ALIGNMENT_ARTIFACT_ROOT}",
        "--recipe.sequence_length=2560",
    ]
    with pytest.raises(OLMoConfigurationError, match="calibrated dataset.mean_loss_weight"):
        alignment_recipe.build(include_means=False, overrides=overrides)
    short_means = {"pixmo_caption": 29.144221610185923, "pixmo_transcript": 29.720018727071874}
    config = alignment_recipe.build(
        include_means=False,
        overrides=[*overrides, f"--dataset.mean_loss_weight={json.dumps(short_means)}"],
    )
    assert config.dataset.mean_loss_weight == short_means
    assert config.data_loader.sequence_length == 2560
    assert ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"] == BRIDGE_MEANS


@pytest.mark.parametrize("split", ["train", "validation"])
def test_bridge_visual_source_definitions(split):
    sources = build_visual_sources("bridge", 8192, split=split)
    assert list(sources) == ["pixmo_caption", "pixmo_transcript"]
    for name, source in sources.items():
        assert (
            source.dataset_path
            == f"{DEFAULT_ALIGNMENT_ARTIFACT_ROOT}/pixmo-cap-content-disjoint-v1/dataset"
        )
        assert source.split == split and source.require_split
        assert source.max_sequence_length == 8192 and source.max_crops == 8
        assert source.loss_token_weighting == "root_subsegments_root_tokens"
        assert source.message_format == "document"
        assert not source.style_length_conditioning
        assert source.seed == 0
        assert source.mode == ("caption" if name == "pixmo_caption" else "transcript")
    assert sources["pixmo_caption"].fixed_prompt == "Description:"
    assert sources["pixmo_transcript"].fixed_prompt == "Transcript:"
    assert sources["pixmo_transcript"].require_transcript


def test_bridge_model_preserves_parent_architecture(alignment_recipe):
    original = recipe_tests._set_pretrained_router_coefficients(alignment_recipe, 0.02)
    config = alignment_recipe.build()
    expected = original.copy()
    expected.recompute_each_block = True
    expected.recompute_all_blocks_by_chunk = False
    expected.two_batch_overlap = False
    for block in [expected.block, *expected.block_overrides.values()]:
        block.sequence_mixer.backend = AttentionBackendName.flex
        block.ep.path = ExpertParallelPath.rowwise_nvshmem
        block.ep.capacity_factor = 8
        block.ep.share_dispatch_out = True
        block.routed_experts_router.lb_loss_weight = 0.0
    assert config.model.lm == expected
    assert config.model.image_patch_token_id == alignment_recipe.token_ids.im_patch_id
    saved = json.loads((alignment_recipe.base / "config.json").read_text())
    assert saved["model"] == json.loads(json.dumps(original.as_config_dict()))


@pytest.mark.parametrize("weight", [0.0, 0.025])
def test_bridge_explicit_lb_override(alignment_recipe, weight):
    original = recipe_tests._set_pretrained_router_coefficients(alignment_recipe, 0.02)
    config = alignment_recipe.build(overrides=[f"--recipe.router_lb_loss_weight={weight}"])
    for block, parent in zip(
        config.model.lm.resolved_block_configs, original.resolved_block_configs
    ):
        assert block.routed_experts_router.lb_loss_weight == weight
        assert (
            block.routed_experts_router.z_loss_weight == parent.routed_experts_router.z_loss_weight
        )
        assert block.routed_experts_router.top_k == parent.routed_experts_router.top_k
        assert block.ep.capacity_factor == 8 and block.ep.share_dispatch_out


def test_bridge_optimizer_freezing_and_row_mask(alignment_recipe):
    config = alignment_recipe.build()
    module = config.train_module
    assert module.freeze_params == [
        "vision.*",
        "lm.embedding_norm.*",
        "lm.blocks.*",
        "lm.lm_head.*",
    ]
    assert module.train_embedding_rows == recipe_tests.vision_alignment._image_token_rows(
        alignment_recipe.token_ids
    )
    assert not module.vision_activation_checkpointing
    assert module.connector_activation_checkpointing and module.response_logits_only
    assert module.compile_model and module.ep_config.degree == 8
    assert module.z_loss_multiplier == 1e-4
    optim = module.optim
    assert optim.lr == 2e-4
    assert optim.betas == (0.9, 0.95)
    assert optim.eps == 1e-6 and optim.weight_decay == 0.0
    assert optim.max_grad_norm == module.max_grad_norm == 1.0
    assert optim.clip_grad_norm_by_scheduler_group and optim.check_nan_inf_grad
    assert optim.sigma_factor == 12 and optim.foreach_chunk_size == 50_000_000
    assert not optim.compile
    assert [(group.params, group.opts) for group in optim.group_overrides] == [
        (
            ["*lm.embeddings.weight"],
            {"lr": 2e-4, "weight_decay": 0.0, "scheduler_name": "connector"},
        ),
        (["*connector.*"], {"lr": 2e-4, "weight_decay": 0.0, "scheduler_name": "connector"}),
        (["*vision.*"], {"lr": 0.0, "weight_decay": 0.0, "scheduler_name": "vision"}),
    ]


def test_bridge_schedule_decays_before_training_stop(alignment_recipe):
    config = alignment_recipe.build()
    assert config.trainer.max_duration == Duration.steps(500)
    scheduler = config.train_module.scheduler
    connector = scheduler.schedulers["connector"]
    assert connector.warmup == 100 and connector.t_max == 250
    assert connector.alpha_f == 0.1
    assert connector.get_lr(2e-4, 50, 500) == pytest.approx(1e-4)
    assert connector.get_lr(2e-4, 100, 500) == pytest.approx(2e-4)
    assert connector.get_lr(2e-4, 250, 500) == pytest.approx(2e-5)
    assert connector.get_lr(2e-4, 500, 500) == pytest.approx(2e-5)
    assert scheduler.default.warmup == scheduler.schedulers["vision"].warmup == 100
    assert scheduler.default.t_max == scheduler.schedulers["vision"].t_max == 500
    shorter = alignment_recipe.build(overrides=["--trainer.max_duration.value=250"])
    assert shorter.train_module.scheduler == scheduler


def test_bridge_uses_native_checkpoint_resume_and_callbacks(alignment_recipe):
    config = alignment_recipe.build()
    assert config.trainer.load_strategy == LoadStrategy.if_available
    assert config.trainer.load_path is None
    assert config.trainer.load_optim_state is config.trainer.load_trainer_state is None
    assert not config.train_module.reset_optimizer_states_on_load
    assert not config.train_module.reset_optimizer_states_on_resume
    initializer = config.trainer.callbacks["initialize_multimodal"]
    assert initializer.language_checkpoint == str(alignment_recipe.base)
    assert initializer.seed == config.init_seed
    assert config.pretraining_checkpoint == str(alignment_recipe.base)
    assert config.trainer.callbacks["beaker"] is not None
    assert config.trainer.callbacks["wandb"].auto_resume
    assert all(
        type(callback).__module__.startswith("olmo_core.")
        for callback in config.trainer.callbacks.values()
    )
    restored = VisionAlignmentExperimentConfig.from_dict(
        json.loads(json.dumps(config.as_config_dict()))
    )
    assert restored == config


def test_bridge_checkpoint_and_native_evaluation_policy(alignment_recipe):
    config = alignment_recipe.build()
    checkpointer = config.trainer.callbacks["checkpointer"]
    assert checkpointer.enabled and not checkpointer.save_async
    assert checkpointer.save_interval == 500
    assert checkpointer.ephemeral_save_interval == 50
    assert checkpointer.max_checkpoints == 2
    assert checkpointer.pre_train_checkpoint is False
    assert checkpointer.fixed_steps is None
    evaluator = config.trainer.callbacks["multimodal_evaluator"]
    assert evaluator.eval_interval == 500
    assert evaluator.eval_on_startup and evaluator.eval_on_finish
    assert evaluator.sequence_length == 8192 and evaluator.rank_batch_size == 4
    assert evaluator.examples_per_source == evaluator.matched_image_examples == 64
    assert evaluator.matched_image_candidates == 512
    assert evaluator.early_response_tokens == 8
    assert (
        evaluator.blank_image_sources
        == evaluator.matched_image_sources
        == [
            "pixmo_caption",
            "pixmo_transcript",
        ]
    )
    for source in evaluator.eval_dataset.sources.values():
        assert source.max_sequence_length == 2560
        assert source.max_crops == 8
        assert source.split == "validation"
    callbacks = config.trainer.callbacks
    savers = {
        name: item for name, item in callbacks.items() if isinstance(item, MetricSaverCallback)
    }
    restorers = [item for item in callbacks.values() if isinstance(item, RestoreMetricsCallback)]
    assert len(savers) == len(restorers) == 1
    assert restorers[0].metrics_callback in savers


def test_perception_and_joint_geometry_stays_unchanged(alignment_recipe):
    bridge = alignment_recipe.save(alignment_recipe.build())
    perception = alignment_recipe.build("perception", bridge)
    joint = alignment_recipe.build("joint", alignment_recipe.save(perception))
    for config, context, microbatch, steps in [
        (perception, 2560, 4, 4000),
        (joint, 8192, 1, 16000),
    ]:
        assert config.data_loader.sequence_length == context
        assert config.data_loader.pack_max_crops == 9
        assert config.data_loader.pack_buffer_size == 48
        assert config.data_loader.prefetch_workers == 8
        assert config.train_module.rank_microbatch_size == microbatch * context
        assert config.trainer.max_duration == Duration.steps(steps)
        assert config.train_module.scheduler.default.t_max == steps
        assert config.train_module.scheduler.schedulers["connector"].t_max == steps
