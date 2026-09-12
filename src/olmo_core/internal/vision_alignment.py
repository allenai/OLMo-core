"""Vision-alignment recipe using the shared internal experiment runner."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url, normalize_path, resource_path
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import (
    Molmo2TokenIds,
    MultimodalLMConfig,
    load_molmo2_hf_vision_config,
    multimodal_config_from_molmo2_vision,
)
from olmo_core.optim import (
    CosWithWarmup,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    PerGroupScheduler,
)
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    MetricSaverCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.multimodal import (
    InitializeMultimodalModelCallback,
    MultimodalEvaluatorCallbackConfig,
)
from olmo_core.train.callbacks.restore_metrics import RestoreMetricsCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
)

from .common import build_launch_config
from .experiment import CliContext, ExperimentConfig
from .vision_alignment_data import DEFAULT_ALIGNMENT_ARTIFACT_ROOT, build_visual_sources

_DOLMA2_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"


class AlignmentPhase(StrEnum):
    """Successive training phases before mixed vision/text midtraining."""

    bridge = "bridge"
    perception = "perception"
    joint = "joint"


@dataclass
class VisionAlignmentRecipeConfig(Config):
    """Inputs used to construct an alignment experiment's ordinary component configs.

    Set ``pretraining_checkpoint`` for bridge and ``parent_checkpoint`` for subsequent
    phases. The latter inherit the multimodal model and original text-data ancestry.
    Component-level CLI overrides are applied after these defaults are constructed.
    """

    phase: AlignmentPhase = AlignmentPhase.bridge
    sequence_length: int | None = None
    """Context length for sources, packing, training, and evaluation.

    ``None`` uses 8,192 tokens for bridge/joint and 2,560 for perception. The global batch
    remains 128 sequences, and phase microbatch instance counts are unchanged. Changing
    this length requires fresh visual loss-weight calibration; it does not modify RoPE.
    """
    pretraining_checkpoint: str | None = None
    parent_checkpoint: str | None = None
    artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT
    output_root: str = (
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints"
    )
    work_dir: str = "/weka/oe-training-default/rustin/dataset-cache/vision-alignment"
    hf_cache_dir: str | None = "/weka/oe-training-default/rustin/hf-cache/hub"
    tokenizer_revision: str | None = None
    molmo2_model_id: str = "allenai/Molmo2-4B"
    molmo2_revision: str = "042abfa7a38879a376cec03d949eff0aefaa0600"
    vision_model_id: str = "google/siglip2-so400m-patch14-384"
    vision_revision: str = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    text_validation_size: int = 1024
    """Native replay windows withheld in joint; zero requires separate text validation."""
    text_validation_seed: int = 6198
    """Seed shared by the complementary native replay training and validation splits."""
    router_lb_loss_weight: float | None = None
    """Override routed experts' load-balancing coefficients.

    ``None`` disables balancing for bridge and inherits it from the parent in later phases.

    Zero disables the load-balancing objective without changing router z-loss or CE.
    This model-config override persists in checkpoints, including across phase handoffs.
    Later phases must explicitly set a positive value to re-enable disabled balancing.
    """
    restore_pretraining_router_lb: bool = False
    """Restore original pretrained per-layer balancing coefficients after phase inheritance.

    This opt-in policy reads only checkpoint config metadata, validates the resolved LM/router
    architecture, and copies only routed-router ``lb_loss_weight`` values (including zero and
    ``None``). It preserves dispatch capacity and other auxiliary objectives. Mutually exclusive
    with ``router_lb_loss_weight`` or explicit model-level balancing-coefficient overrides.
    """


@dataclass
class VisionAlignmentExperimentConfig(ExperimentConfig):
    """An alignment experiment with portable pretraining-checkpoint ancestry."""

    model: MultimodalLMConfig
    dataset: MultimodalMixtureConfig
    data_loader: MixtureDataLoaderConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    recipe: VisionAlignmentRecipeConfig = field(default_factory=VisionAlignmentRecipeConfig)
    pretraining_checkpoint: str = ""


@dataclass(frozen=True)
class _PhaseDefaults:
    sequence_length: int
    microbatch_instances: int
    steps: int
    freeze_params: tuple[str, ...]
    connector_lr: float
    vision_lr: float
    lm_lr: float
    connector_warmup: int = 200
    vision_warmup: int = 500
    lm_warmup: int = 500
    pack_max_crops: int = 9
    validation_sequence_length: int | None = None


_PHASES = {
    AlignmentPhase.bridge: _PhaseDefaults(
        sequence_length=8192,
        microbatch_instances=4,
        steps=500,
        freeze_params=("vision.*", "lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"),
        connector_lr=2e-4,
        vision_lr=0.0,
        lm_lr=0.0,
        connector_warmup=100,
        vision_warmup=100,
        lm_warmup=100,
        pack_max_crops=64,
        validation_sequence_length=2560,
    ),
    AlignmentPhase.perception: _PhaseDefaults(
        sequence_length=2560,
        microbatch_instances=4,
        steps=4000,
        freeze_params=("lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"),
        connector_lr=5e-5,
        vision_lr=3e-6,
        lm_lr=0.0,
    ),
    AlignmentPhase.joint: _PhaseDefaults(
        sequence_length=8192,
        microbatch_instances=1,
        steps=16000,
        freeze_params=("lm.lm_head.w_out.weight",),
        connector_lr=2e-5,
        vision_lr=2e-6,
        lm_lr=1e-6,
    ),
}


def _read_checkpoint_config(checkpoint: str) -> dict:
    import json

    with resource_path(checkpoint, "config.json").open() as stream:
        return json.load(stream)


def _image_token_rows(ids: Molmo2TokenIds) -> list[int]:
    return [
        ids.im_start_id,
        ids.im_end_id,
        ids.im_patch_id,
        ids.im_col_id,
        ids.low_res_im_start_id,
        ids.image_placeholder_id,
    ]


def _resolve_parent(recipe: VisionAlignmentRecipeConfig) -> tuple[str, dict | None]:
    if recipe.phase == AlignmentPhase.bridge:
        if recipe.parent_checkpoint is not None or not recipe.pretraining_checkpoint:
            raise OLMoConfigurationError(
                "Bridge requires recipe.pretraining_checkpoint and no recipe.parent_checkpoint"
            )
        return recipe.pretraining_checkpoint, None
    if not recipe.parent_checkpoint:
        raise OLMoConfigurationError("Set recipe.parent_checkpoint to the previous alignment phase")
    parent = _read_checkpoint_config(recipe.parent_checkpoint)
    previous = parent.get("recipe", {}).get("phase", parent.get("phase"))
    expected = "bridge" if recipe.phase == AlignmentPhase.perception else "perception"
    if previous != expected:
        raise OLMoConfigurationError(
            f"{recipe.phase} requires a {expected} checkpoint, got {previous}"
        )
    checkpoint = parent.get("pretraining_checkpoint") or parent.get("artifacts", {}).get(
        "base_checkpoint"
    )
    if not checkpoint:
        raise OLMoConfigurationError("Parent does not record its original pretraining checkpoint")
    if recipe.pretraining_checkpoint not in (None, checkpoint):
        raise OLMoConfigurationError("Requested pretraining checkpoint differs from phase parent")
    return checkpoint, parent


def _resolve_tokenizer_revision(
    recipe: VisionAlignmentRecipeConfig, parent: dict | None, tokenizer: TokenizerConfig
) -> str | None:
    if parent is not None:
        parent_dataset = parent.get("dataset", {})
        parent_artifacts = parent.get("artifacts", {})
        parent_tokenizer = parent_dataset.get("tokenizer")
        if parent_tokenizer is not None:
            if TokenizerConfig.from_dict(parent_tokenizer) != tokenizer:
                raise OLMoConfigurationError(
                    "Phase parent tokenizer differs from its pretraining checkpoint"
                )
        elif parent_artifacts.get("tokenizer_id", tokenizer.identifier) != tokenizer.identifier:
            raise OLMoConfigurationError(
                "Phase parent tokenizer differs from its pretraining checkpoint"
            )
        if "tokenizer_revision" in parent_dataset or "tokenizer_revision" in parent_artifacts:
            revision = parent_dataset.get(
                "tokenizer_revision", parent_artifacts.get("tokenizer_revision")
            )
            if recipe.tokenizer_revision is not None and recipe.tokenizer_revision != revision:
                raise OLMoConfigurationError(
                    "Requested tokenizer revision differs from the phase parent"
                )
            return revision
    return recipe.tokenizer_revision or (
        _DOLMA2_REVISION if tokenizer.identifier == "allenai/dolma2-tokenizer" else None
    )


def _build_model(
    recipe: VisionAlignmentRecipeConfig,
    checkpoint: str,
    parent: dict | None,
    token_ids: Molmo2TokenIds,
) -> MultimodalLMConfig:
    if parent is not None:
        model = MultimodalLMConfig.from_dict(parent["model"])
    else:
        lm = OLMoDDPModelConfig.from_dict(_read_checkpoint_config(checkpoint)["model"])
        if not isinstance(lm, OLMoDDPModelConfig):
            raise OLMoConfigurationError("This alignment recipe currently requires an OLMoDDP LM")
        hf_config = load_molmo2_hf_vision_config(
            recipe.molmo2_model_id,
            revision=recipe.molmo2_revision,
            cache_dir=recipe.hf_cache_dir,
        )
        model = multimodal_config_from_molmo2_vision(
            hf_config, lm, image_patch_token_id=token_ids.im_patch_id
        )
    if not isinstance(model.lm, OLMoDDPModelConfig):
        raise OLMoConfigurationError("This alignment recipe currently requires an OLMoDDP LM")
    lb_loss_weight = recipe.router_lb_loss_weight
    if recipe.phase == AlignmentPhase.bridge and lb_loss_weight is None:
        lb_loss_weight = 0.0
    for block in [model.lm.block, *(model.lm.block_overrides or {}).values()]:
        if not isinstance(block, OLMoDDPTransformerBlockConfig):
            raise OLMoConfigurationError("Alignment requires OLMoDDP transformer blocks")
        if isinstance(block.sequence_mixer, AttentionConfig):
            block.sequence_mixer.backend = AttentionBackendName.flex
        if block.ep is not None:
            block.ep.path = ExpertParallelPath.rowwise_nvshmem
        if block.routed_experts_router is not None:
            if lb_loss_weight is not None:
                block.routed_experts_router.lb_loss_weight = lb_loss_weight
            if recipe.phase == AlignmentPhase.bridge and block.ep is not None:
                block.ep.capacity_factor = 8
                block.ep.share_dispatch_out = True
    model.lm.recompute_each_block = True
    model.lm.recompute_all_blocks_by_chunk = False
    model.lm.two_batch_overlap = False
    return model


def _build_train_module(
    phase: AlignmentPhase, token_ids: Molmo2TokenIds, sequence_length: int
) -> MultimodalOLMoDDPTrainModuleConfig:
    policy = _PHASES[phase]
    return MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=policy.microbatch_instances * sequence_length,
        max_sequence_length=sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=policy.lm_lr or policy.connector_lr,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["*lm.embeddings.weight"],
                    opts={
                        "lr": policy.connector_lr,
                        "weight_decay": 0.0,
                        "scheduler_name": "connector",
                    },
                ),
                OptimGroupOverride(
                    params=["*connector.*"],
                    opts={
                        "lr": policy.connector_lr,
                        "weight_decay": 0.0,
                        "scheduler_name": "connector",
                    },
                ),
                OptimGroupOverride(
                    params=["*vision.*"],
                    opts={"lr": policy.vision_lr, "weight_decay": 0.0, "scheduler_name": "vision"},
                ),
            ],
            compile=False,
            foreach_chunk_size=50_000_000,
            sigma_factor=12,
            max_grad_norm=1.0,
            clip_grad_norm_by_scheduler_group=True,
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        freeze_params=list(policy.freeze_params),
        train_embedding_rows=_image_token_rows(token_ids),
        vision_activation_checkpointing=phase != AlignmentPhase.bridge,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        diagnostics_interval=1 if phase == AlignmentPhase.bridge else 100,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=True,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(
                    warmup=policy.connector_warmup,
                    alpha_f=0.1,
                    t_max=250 if phase == AlignmentPhase.bridge else policy.steps,
                ),
                "vision": CosWithWarmup(
                    warmup=policy.vision_warmup, alpha_f=0.1, t_max=policy.steps
                ),
            },
            default=CosWithWarmup(warmup=policy.lm_warmup, alpha_f=0.1, t_max=policy.steps),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_dtype=DType.float32,
            only_allreduce_last_microbatch=True,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=8),
    )


def _restore_pretraining_router_lb(
    model: OLMoDDPModelConfig, pretrained: OLMoDDPModelConfig
) -> None:
    """Restore effective per-layer balancing coefficients without changing the LM function."""
    model_fields = (
        "name",
        "d_model",
        "n_layers",
        "vocab_size",
        "embedding_norm",
        "embed_scale",
        "tie_word_embeddings",
    )
    if any(getattr(model, name) != getattr(pretrained, name) for name in model_fields):
        raise OLMoConfigurationError(
            "Pretraining router LB restoration requires matching LM topology"
        )

    def topology(block: OLMoDDPTransformerBlockConfig):
        value = block.as_config_dict()
        # Alignment changes execution choices, not learned architecture. Do not restore any
        # of these fields, and permit larger fixed capacities or checkpointing policies.
        for name in (
            "ep",
            "rowwise_fp8",
            "checkpoint_attn",
            "checkpoint_permute_moe_unpermute",
            "checkpoint_second_unpermute",
        ):
            value.pop(name, None)
        if isinstance(block.sequence_mixer, AttentionConfig):
            value["sequence_mixer"].pop("backend", None)
        for name in ("routed_experts_router", "shared_experts_router"):
            if (router := value.get(name)) is not None:
                for coefficient in ("lb_loss_weight", "z_loss_weight", "orth_loss_weight"):
                    router.pop(coefficient, None)
        return value

    current_blocks = model.resolved_block_configs
    pretrained_blocks = pretrained.resolved_block_configs
    for index, (current, original) in enumerate(zip(current_blocks, pretrained_blocks)):
        if not isinstance(current, OLMoDDPTransformerBlockConfig) or not isinstance(
            original, OLMoDDPTransformerBlockConfig
        ):
            raise OLMoConfigurationError("Router LB restoration requires OLMoDDP block topology")
        if topology(current) != topology(original):
            raise OLMoConfigurationError(
                f"Pretraining router LB restoration requires matching block/router topology "
                f"at layer {index}"
            )
        router = original.routed_experts_router
        if (
            router is not None
            and router.lb_loss_weight is not None
            and (not isfinite(router.lb_loss_weight) or router.lb_loss_weight < 0)
        ):
            raise OLMoConfigurationError(
                f"Invalid pretrained router LB coefficient at layer {index}"
            )

    # A default block can be shared by several effective layers. Preserve that layout unless
    # differing original coefficients require a new layer override; never overwrite an earlier
    # layer through the same shared config object.
    assigned: dict[int, float | None] = {}
    for index, (current, original) in enumerate(zip(current_blocks, pretrained_blocks)):
        assert isinstance(current, OLMoDDPTransformerBlockConfig)
        assert isinstance(original, OLMoDDPTransformerBlockConfig)
        if current.routed_experts_router is None:
            continue
        assert original.routed_experts_router is not None
        coefficient = original.routed_experts_router.lb_loss_weight
        if id(current) in assigned and assigned[id(current)] != coefficient:
            current = current.copy()
            model.block_overrides = dict(model.block_overrides or {})
            model.block_overrides[index] = current
        assigned[id(current)] = coefficient
        assert current.routed_experts_router is not None
        current.routed_experts_router = current.routed_experts_router.copy()
        current.routed_experts_router.lb_loss_weight = coefficient


def _build_recipe(cli: CliContext) -> VisionAlignmentRecipeConfig:
    recipe = VisionAlignmentRecipeConfig().merge(
        [arg.replace("--recipe.", "--", 1) for arg in cli.overrides if arg.startswith("--recipe.")]
    )
    if recipe.restore_pretraining_router_lb and (
        recipe.router_lb_loss_weight is not None
        or any(
            arg.startswith("--model.") and arg.partition("=")[0].endswith(".lb_loss_weight")
            for arg in cli.overrides
        )
    ):
        raise OLMoConfigurationError(
            "restore_pretraining_router_lb is mutually exclusive with explicit router LB overrides"
        )
    if recipe.router_lb_loss_weight is not None and (
        not isfinite(recipe.router_lb_loss_weight) or recipe.router_lb_loss_weight < 0
    ):
        raise OLMoConfigurationError("recipe.router_lb_loss_weight must be finite and nonnegative")
    return recipe


def _sequence_length(recipe: VisionAlignmentRecipeConfig) -> int:
    policy = _PHASES[recipe.phase]
    sequence_length = (
        policy.sequence_length if recipe.sequence_length is None else recipe.sequence_length
    )
    if type(sequence_length) is not int or sequence_length < 2:
        raise OLMoConfigurationError("recipe.sequence_length must be an integer of at least two")
    return sequence_length


def _build_datasets(
    recipe: VisionAlignmentRecipeConfig,
    checkpoint: str,
    parent: dict | None,
    sequence_length: int,
) -> tuple[MultimodalMixtureConfig, MultimodalMixtureConfig, Molmo2TokenIds]:
    from .vision_alignment_data import (
        ALIGNMENT_LOSS_TARGETS,
        ALIGNMENT_MEAN_LOSS_WEIGHTS,
    )

    phase = recipe.phase
    policy = _PHASES[phase]
    replay = PretrainingReplayConfig(
        checkpoint=checkpoint, sequence_length=sequence_length, work_dir=recipe.work_dir
    )
    text = replay.resolve_dataset()
    lm_config = _read_checkpoint_config(checkpoint)["model"]
    dataset = MultimodalMixtureConfig(
        tokenizer=text.tokenizer,
        tokenizer_revision=_resolve_tokenizer_revision(recipe, parent, text.tokenizer),
        tokenizer_cache_dir=recipe.hf_cache_dir,
        model_vocab_size=lm_config["vocab_size"],
        sources=build_visual_sources(phase, sequence_length, recipe.artifact_root),
        target_loss_mass=ALIGNMENT_LOSS_TARGETS[phase].copy(),
        mean_loss_weight=ALIGNMENT_MEAN_LOSS_WEIGHTS[phase].copy(),
    )
    if phase == AlignmentPhase.joint:
        if recipe.text_validation_size < 0:
            raise OLMoConfigurationError("recipe.text_validation_size cannot be negative")
        if recipe.text_validation_size:
            replay.split = "train"
            replay.validation_size = recipe.text_validation_size
            replay.split_seed = recipe.text_validation_seed
        dataset.sources["native_text_replay"] = replay
        dataset.sources = dict(sorted(dataset.sources.items()))
    if (
        text.tokenizer != TokenizerConfig.dolma2()
        or dataset.tokenizer_revision != _DOLMA2_REVISION
        or recipe.artifact_root != DEFAULT_ALIGNMENT_ARTIFACT_ROOT
        or sequence_length != policy.sequence_length
    ):
        dataset.mean_loss_weight = {}
    if phase == AlignmentPhase.joint:
        if text.label_mask_paths is None:
            dataset.mean_loss_weight["native_text_replay"] = float(sequence_length - 1)
        else:
            dataset.mean_loss_weight.pop("native_text_replay", None)
    _, token_ids = dataset.build_tokenizer()
    validation = dataset.copy()
    validation_sequence_length = (
        sequence_length
        if policy.validation_sequence_length is None
        else min(sequence_length, policy.validation_sequence_length)
    )
    validation.sources = build_visual_sources(
        phase, validation_sequence_length, recipe.artifact_root, split="validation"
    )
    validation.target_loss_mass = {name: 1.0 for name in validation.sources}
    validation.mean_loss_weight = {}
    return dataset, validation, token_ids


def _build_data_loader(
    cli: CliContext, recipe: VisionAlignmentRecipeConfig, sequence_length: int
) -> MixtureDataLoaderConfig:
    data_loader = MixtureDataLoaderConfig(
        global_batch_size=128 * sequence_length,
        sequence_length=sequence_length,
        seed=95818,
        work_dir=f"{recipe.work_dir}/{cli.run_name}",
        pack=True,
        pack_buffer_size=48,
        pack_max_crops=_PHASES[recipe.phase].pack_max_crops,
        prefetch_workers=8,
    )
    if recipe.phase == AlignmentPhase.bridge:
        data_loader.max_consecutive_data_errors = 0
        data_loader.max_total_data_errors = 0
    return data_loader


def _build_trainer(
    cli: CliContext,
    recipe: VisionAlignmentRecipeConfig,
    checkpoint: str,
    validation: MultimodalMixtureConfig,
    token_ids: Molmo2TokenIds,
    sequence_length: int,
) -> TrainerConfig:
    phase = recipe.phase
    policy = _PHASES[phase]
    is_bridge = phase == AlignmentPhase.bridge
    trainer = (
        TrainerConfig(
            save_folder=f"{recipe.output_root}/{cli.run_name}",
            work_dir=f"{recipe.work_dir}/{cli.run_name}",
            save_overwrite=False,
            load_path=recipe.parent_checkpoint,
            load_strategy=LoadStrategy.if_available if is_bridge else LoadStrategy.always,
            load_optim_state=None if is_bridge else False,
            load_trainer_state=None if is_bridge else False,
            metrics_collect_interval=1 if is_bridge else 5,
            cancel_check_interval=5,
            max_duration=Duration.steps(policy.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=500 if is_bridge else 1000,
                ephemeral_save_interval=50 if is_bridge else 250,
                save_async=False,
                pre_train_checkpoint=False if is_bridge else None,
                max_checkpoints=2 if is_bridge else 6,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "wandb",
            WandBCallback(name=cli.run_name, project="vision-alignment", auto_resume=True),
        )
        .with_callback(
            "multimodal_evaluator",
            MultimodalEvaluatorCallbackConfig(
                eval_dataset=validation,
                sequence_length=sequence_length,
                rank_batch_size=policy.microbatch_instances,
                examples_per_source=64 if is_bridge else 512,
                eval_interval=500,
                eval_on_startup=True,
                eval_on_finish=True,
                blank_image_sources=["pixmo_caption", "pixmo_transcript"],
                matched_image_sources=["pixmo_caption", "pixmo_transcript"],
            ),
        )
    )
    if is_bridge:
        trainer = trainer.with_callback(
            "metrics",
            MetricSaverCallback(save_interval=1, final_metrics_fname="metrics-final.json"),
        ).with_callback("restore_metrics", RestoreMetricsCallback(metrics_callback="metrics"))
        trainer = trainer.with_callback(
            "initialize_multimodal",
            InitializeMultimodalModelCallback(
                language_checkpoint=checkpoint,
                vision_model_id=recipe.vision_model_id,
                vision_revision=recipe.vision_revision,
                cache_dir=recipe.hf_cache_dir,
                image_token_ids=_image_token_rows(token_ids),
                seed=6198,
            ),
        )
    return trainer


def _build_launch(cli: CliContext, phase: AlignmentPhase) -> BeakerLaunchConfig | None:
    if cli.cluster == "local":
        return None
    preset = get_preset("olmo-ddp")
    launch = build_launch_config(
        name=cli.run_name,
        cmd=cli.remote_cmd,
        cluster=cli.cluster,
        root_dir="/weka/oe-training-default",
        workspace="ai2/molmofication",
        num_nodes=2,
    )
    if preset.beaker_image is not None:
        launch.beaker_image = preset.beaker_image
    launch.post_setup = preset.post_setup
    launch.env_vars.extend(BeakerEnvVar(name=k, value=v) for k, v in preset.env_vars)
    launch.env_vars.extend(
        BeakerEnvVar(name=k, value=v)
        for k, v in {
            "OLMO_USE_OWN_SYMM_MEM": "1",
            "OLMO_EP_MP_HIGH_PRIORITY_GROUP": "1",
            "OLMO_OWN_SYMM_PREWARM": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            "TORCH_LOGS": "-dynamo",
        }.items()
    )
    if phase == AlignmentPhase.bridge:
        launch.priority = "urgent"
        launch.min_runtime = "8h"
        launch.shared_memory = "32GiB"
    return launch


def build_config(cli: CliContext) -> VisionAlignmentExperimentConfig:
    """Build one alignment phase from checkpoint metadata and ordinary CLI overrides."""
    recipe = _build_recipe(cli)
    sequence_length = _sequence_length(recipe)
    checkpoint, parent = _resolve_parent(recipe)
    dataset, validation, token_ids = _build_datasets(recipe, checkpoint, parent, sequence_length)
    config = VisionAlignmentExperimentConfig(
        run_name=cli.run_name,
        launch=_build_launch(cli, recipe.phase),
        model=_build_model(recipe, checkpoint, parent, token_ids),
        dataset=dataset,
        data_loader=_build_data_loader(cli, recipe, sequence_length),
        train_module=_build_train_module(recipe.phase, token_ids, sequence_length),
        trainer=_build_trainer(cli, recipe, checkpoint, validation, token_ids, sequence_length),
        recipe=recipe,
        pretraining_checkpoint=checkpoint,
        init_seed=6198,
    ).merge(cli.overrides)
    _validate_config(config, cli, dataset, token_ids, checkpoint)
    return config


def _validate_config(
    config: VisionAlignmentExperimentConfig,
    cli: CliContext,
    dataset: MultimodalMixtureConfig,
    token_ids: Molmo2TokenIds,
    checkpoint: str,
) -> None:
    recipe = config.recipe
    phase = recipe.phase
    policy = _PHASES[phase]
    sequence_length = _sequence_length(recipe)
    if config.recipe.restore_pretraining_router_lb:
        if not isinstance(config.model.lm, OLMoDDPModelConfig):
            raise OLMoConfigurationError("Router LB restoration requires an OLMoDDP LM")
        pretrained = OLMoDDPModelConfig.from_dict(_read_checkpoint_config(checkpoint)["model"])
        _restore_pretraining_router_lb(config.model.lm, pretrained)
    if (
        config.dataset.tokenizer != dataset.tokenizer
        or config.dataset.tokenizer_revision != dataset.tokenizer_revision
    ):
        raise OLMoConfigurationError(
            "Select the tokenizer through recipe.pretraining_checkpoint and recipe.tokenizer_revision"
        )
    changed_sources = [
        name
        for name, source in config.dataset.sources.items()
        if source != dataset.sources.get(name)
        and not any(
            arg.startswith((f"--dataset.mean_loss_weight.{name}=", "--dataset.mean_loss_weight="))
            for arg in cli.overrides
        )
    ]
    if changed_sources:
        raise OLMoConfigurationError(
            f"Supply calibrated dataset.mean_loss_weight for changed sources: {changed_sources}"
        )
    for name, source in config.dataset.sources.items():
        if isinstance(source, PretrainingReplayConfig):
            replay_dataset = source.resolve_dataset()
            if replay_dataset.sequence_length != config.data_loader.sequence_length:
                raise OLMoConfigurationError("Replay and data-loader sequence lengths must agree")
            if replay_dataset.tokenizer != config.dataset.tokenizer:
                raise OLMoConfigurationError("Replay and visual tokenizers must agree")
            if replay_dataset.label_mask_paths is None:
                expected_mean = float(replay_dataset.sequence_length - 1)
                if config.dataset.mean_loss_weight.get(name, expected_mean) != expected_mean:
                    raise OLMoConfigurationError(
                        f"Unmasked replay source {name!r} has exact mean_loss_weight "
                        f"{expected_mean} (sequence_length - 1)"
                    )
                config.dataset.mean_loss_weight[name] = expected_mean
    if sequence_length != policy.sequence_length:
        missing = sorted(set(config.dataset.sources) - set(config.dataset.mean_loss_weight))
        if missing:
            raise OLMoConfigurationError(
                "Changing recipe.sequence_length requires calibrated dataset.mean_loss_weight "
                f"for sources: {missing}. Use MultimodalMixtureConfig.estimate_mean_loss_weights() "
                "for a bounded calibration sample."
            )
    config.dataset.sampling_weights()
    targets = config.dataset.target_loss_mass
    config.train_module.source_loss_mass_targets = {
        k: v / sum(targets.values()) for k, v in targets.items()
    }
    if config.trainer.load_path != recipe.parent_checkpoint:
        raise OLMoConfigurationError(
            "Use recipe.parent_checkpoint to select a phase's initial model"
        )
    source_path = normalize_path(recipe.parent_checkpoint or checkpoint).rstrip("/")
    output_path = normalize_path(config.trainer.save_folder).rstrip("/")
    if not is_url(source_path):
        source_path = str(Path(source_path).resolve())
    if not is_url(output_path):
        output_path = str(Path(output_path).resolve())
    if (
        source_path == output_path
        or source_path.startswith(output_path + "/")
        or output_path.startswith(source_path + "/")
    ):
        raise OLMoConfigurationError("Use a separate output folder for the new alignment phase")
    if config.pretraining_checkpoint != checkpoint:
        raise OLMoConfigurationError("Pretraining ancestry must match the selected phase parent")
    if config.model.lm.tie_word_embeddings:
        raise OLMoConfigurationError(
            "Alignment's row-masked image embeddings require untied LM input and output weights"
        )
    if config.model.image_patch_token_id != token_ids.im_patch_id or (
        config.train_module.train_embedding_rows != _image_token_rows(token_ids)
    ):
        raise OLMoConfigurationError(
            "Model image tokens and trainable rows must match the tokenizer"
        )
    if config.dataset.model_vocab_size != config.model.lm.vocab_size:
        raise OLMoConfigurationError("Dataset and model vocabulary sizes must agree")
    if config.data_loader.sequence_length != config.train_module.max_sequence_length:
        raise OLMoConfigurationError("Data-loader and train-module sequence lengths must agree")
    evaluator = config.trainer.callbacks["multimodal_evaluator"]
    if (
        evaluator.eval_dataset.tokenizer != config.dataset.tokenizer
        or evaluator.eval_dataset.tokenizer_revision != config.dataset.tokenizer_revision
        or evaluator.eval_dataset.model_vocab_size != config.model.lm.vocab_size
    ):
        raise OLMoConfigurationError("Evaluation tokenizer and vocabulary must match training")
    if evaluator.sequence_length != config.data_loader.sequence_length:
        raise OLMoConfigurationError("Evaluation and training sequence lengths must agree")
    if phase == AlignmentPhase.joint and recipe.text_validation_size:
        training_replay = config.dataset.sources.get("native_text_replay")
        if (
            not isinstance(training_replay, PretrainingReplayConfig)
            or training_replay.split != "train"
        ):
            raise OLMoConfigurationError("Native replay validation requires the training split")
        if training_replay.validation_size < evaluator.examples_per_source:
            raise OLMoConfigurationError("Native replay holdout must cover examples_per_source")
        # Derive after CLI overrides so corpus, window boundaries, and exclusion seed agree.
        holdout = training_replay.copy()
        holdout.split = "validation"
        evaluator.eval_dataset.sources["native_text_holdout"] = holdout
        evaluator.eval_dataset.target_loss_mass["native_text_holdout"] = 1.0
