"""Molmo2 stage-2 SFT for the s002 OLMo3 MoE language model.

Fine-tunes the connector, vision encoder, and MoE LM on mm_olmo's 43-source
``image-only-v9`` mixture. The production topology is OLMoDDP with EP=8, either on a
local eight-GPU node or two Beaker nodes. Stage 2 must load model weights from a completed
stage-1 run; trainer and optimizer state are intentionally reset.

The default ``debug`` mixture contains ``tulu4``, ``text_vqa``, and
``chart_qa_weighted``. Set ``--mixture=image-only-v9`` for the production mixture.
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import yaml

from olmo_core.config import Config, DType
from olmo_core.data.multimodal import (
    PIXMO_DATASETS,
    MixtureDataLoader,
    MultimodalCollatorConfig,
    MultimodalDataLoader,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    VALIDATION_MIXTURES,
    build_image_only_v9_mixture,
)
from olmo_core.data.multimodal.sft_common import (
    MaxSequenceLengthDataset,
    SftMessageFormat,
    validate_sft_message_format,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import Evaluator, MultimodalLMEvaluator
from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.nn.vision import Molmo2TokenIds, MultimodalLMConfig
from olmo_core.optim import (
    CosWithWarmup,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    PerGroupScheduler,
)
from olmo_core.train import (
    Duration,
    LoadStrategy,
    ReduceType,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    ConsoleLoggerCallback,
    EvaluatorCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

BASE_CHECKPOINT = "/weka/oe-training-default/robertb/s002-step125500"
VISION_MODEL_ID = "allenai/Molmo2-4B"
VISION_REVISION = "042abfa7a38879a376cec03d949eff0aefaa0600"
TOKENIZER_ID = (
    "/weka/oe-training-default/robertb/olmo3moe-post-training/checkpoints/"
    "s002-olmo3moe-instruct-sft-resume-to1000-fused-20260727-hf"
)
HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
EXPERIMENT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-moe"
SEQUENCE_LENGTH = 16384  # mm_olmo image-only-v9: --seq_len 16384
USE_FLEX_ATTN = True
PACK_SEQUENCES = True
COMPILE_MODEL = True
RESPONSE_LOGITS_ONLY = True
DATA_PREFETCH_WORKERS = 4
MAX_CONSECUTIVE_DATA_ERRORS = 10
MAX_TOTAL_DATA_ERRORS = 1000
MAX_CROPS = 8
# Released Stage 2 permits up to five images, disables the high-resolution branch for
# multi-image examples, and allows eight tiled crops plus one global crop per image.
# Single-image pointing can still use its independent 25-crop high-resolution path.
PACK_MAX_CROPS = 5 * (1 + MAX_CROPS)
PACK_BUFFER_SIZE = 48
PACK_IMAGE_WEIGHT = 30.0
EP_DEGREE = 8
STAGE2_MOE_CAPACITY_FACTOR = 2.0
DIAGNOSTICS_INTERVAL = 100
EST_TOKENS_PER_EXAMPLE = 1500  # packed 16k sequences; tune if batch counts look off
FAST_VISION_EVAL_INTERVAL = 2000
# At 16k, 32 examples are 524k padded tokens per task, close to Stage 1's 655k.
FAST_VISION_EVAL_EXAMPLES = 32
FAST_VISION_EVAL_RANK_BATCH_INSTANCES = 1
FAST_VISION_EVAL_SEED = 6198

# Optional PR #806 reasoning sources. Both remain off in the official image-only-v9
# baseline and must be enabled explicitly as isolated mixture ablations.
MMFINEREASON_RATE = 0.0
FINEVISION_CONFIGS = (
    "visualwebinstruct(filtered)",
    "mavis_math_rule_geo",
    "mavis_math_metagen",
    "geo170k(align)",
    "geo170k(qa)",
)
FINEVISION_MIN_VISUAL_DEPENDENCY: Optional[int] = None

# mm_olmo train_image_video_sft.py uses global 128. One sequence per forward is the
# conservative starting point for the much larger s002 MoE language model.
GLOBAL_BATCH_INSTANCES = 128
RANK_MICROBATCH_INSTANCES = 1
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo SFT).
CONNECTOR_LR = 5e-6
VISION_LR = 5e-6
LLM_LR = 1e-5
COMPONENT_WARMUP = 200
ALPHA_F = 0.1

MAX_STEPS = 30_000

# Default transition: latest checkpoint under the canonical stage-1 run folder.
DEFAULT_LOAD_PATH = str(
    Path(EXPERIMENT_ROOT)
    / "checkpoints"
    / "s002-stage1-corrected-clean-32k-b300-20260807"
    / "step32000"
)

# Beaker.
BEAKER_CLUSTER = "ai2/holmes"
NUM_NODES = 2
BEAKER_WORKSPACE = "ai2/molmofication"
BEAKER_BUDGET = "ai2/oe-other"

WANDB_PROJECT: Optional[str] = "molmo2-stage2"
WANDB_ENTITY: Optional[str] = None

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    trainer: TrainerConfig
    token_ids: Molmo2TokenIds
    base_checkpoint: str = BASE_CHECKPOINT
    vision_model_id: str = VISION_MODEL_ID
    vision_revision: str = VISION_REVISION
    tokenizer_id: str = TOKENIZER_ID
    hf_cache_dir: str = HF_CACHE_DIR
    data_seed: int = 50189
    init_seed: int = 6198
    global_batch_size: int = GLOBAL_BATCH_SIZE
    global_batch_instances: int = GLOBAL_BATCH_INSTANCES
    """Expected number of packed sequences per optimizer step."""
    rank_microbatch_instances: int = RANK_MICROBATCH_INSTANCES
    """Expected number of packed sequences per rank forward pass."""
    mixture: str = "debug"
    """Mixture tier — see ``VALIDATION_MIXTURES`` in ``image_only_v9.py``."""
    message_format: SftMessageFormat = "olmo3_chat"
    """Serialization shared by every Stage 2 source. The s002 run uses OLMo 3 chat."""
    pack_sequences: bool = PACK_SEQUENCES
    pack_max_crops: int = PACK_MAX_CROPS
    pack_buffer_size: int = PACK_BUFFER_SIZE
    pack_image_weight: float = PACK_IMAGE_WEIGHT
    mmfinereason_rate: float = MMFINEREASON_RATE
    finevision_rate: float = 0.0
    fast_vision_eval_interval: Optional[int] = FAST_VISION_EVAL_INTERVAL
    """Run held-out caption, counting, and pointing validation at this interval."""
    fast_vision_eval_examples: int = FAST_VISION_EVAL_EXAMPLES
    """Number of deterministic examples per fast vision evaluator."""
    fast_vision_eval_seed: int = FAST_VISION_EVAL_SEED
    max_consecutive_data_errors: int = MAX_CONSECUTIVE_DATA_ERRORS
    """Stop when more than this many source examples fail consecutively on a rank."""
    max_total_data_errors: int = MAX_TOTAL_DATA_ERRORS
    """Stop when more than this many source examples fail cumulatively on a rank."""


@dataclass
class _DataErrorMonitorCallback(Callback):
    """Expose bounded mixture-loader skips as a cumulative cross-rank metric."""

    def post_train_batch(self):
        data_loader = self.trainer.data_loader
        if not isinstance(data_loader, MixtureDataLoader):
            return
        self.trainer.record_metric(
            "data/errors total",
            data_loader.total_data_errors,
            reduce_type=ReduceType.sum,
        )


def _build_model_config(token_ids: Molmo2TokenIds) -> MultimodalLMConfig:
    """Compose the native s002 MoE LM with the Molmo2 vision tower."""
    import json

    from olmo_core.nn.attention import AttentionConfig
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
    from olmo_core.nn.transformer import OLMoDDPModelConfig
    from olmo_core.nn.vision import (
        load_molmo2_hf_vision_config,
        multimodal_config_from_molmo2_vision,
    )

    with (Path(BASE_CHECKPOINT) / "config.json").open() as checkpoint_config:
        lm_config = OLMoDDPModelConfig.from_dict(json.load(checkpoint_config)["model"])

    attention_backend = AttentionBackendName.flex if USE_FLEX_ATTN else AttentionBackendName.torch
    block_configs = [lm_config.block, *(lm_config.block_overrides or {}).values()]
    for block_config in block_configs:
        if isinstance(block_config.sequence_mixer, AttentionConfig):
            block_config.sequence_mixer.backend = attention_backend
        if block_config.ep is not None:
            block_config.ep.path = ExpertParallelPath.rowwise_nvshmem
            # Robert's successful s002 instruction SFT used capacity 2.0 on every MoE
            # block. Heterogeneous Stage 2 batches are more skewed than pretraining, and
            # rowwise EP otherwise tail-drops overflow routes.
            block_config.ep.capacity_factor = STAGE2_MOE_CAPACITY_FACTOR

    lm_config.recompute_each_block = True
    lm_config.recompute_all_blocks_by_chunk = False
    lm_config.two_batch_overlap = False

    hf_config = load_molmo2_hf_vision_config(
        VISION_MODEL_ID,
        revision=VISION_REVISION,
        cache_dir=HF_CACHE_DIR,
    )
    return multimodal_config_from_molmo2_vision(
        hf_config,
        lm_config,
        image_patch_token_id=token_ids.im_patch_id,
    )


def _mixture_dataset_names(mixture: str) -> Optional[Sequence[str]]:
    if mixture not in VALIDATION_MIXTURES:
        known = ", ".join(sorted(VALIDATION_MIXTURES))
        raise ValueError(f"Unknown mixture {mixture!r}; use one of: {known}")
    return VALIDATION_MIXTURES[mixture]


def _build_train_module_config(
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    rank_microbatch_size: int = RANK_MICROBATCH_SIZE,
    ep_degree: int = EP_DEGREE,
    compile_model: bool = COMPILE_MODEL,
) -> MultimodalOLMoDDPTrainModuleConfig:
    return MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=LLM_LR,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["*connector.*"],
                    opts=dict(lr=CONNECTOR_LR, weight_decay=0.0, scheduler_name="connector"),
                ),
                OptimGroupOverride(
                    params=["*vision.*"],
                    opts=dict(lr=VISION_LR, weight_decay=0.0, scheduler_name="vision"),
                ),
            ],
            compile=False,
            foreach_chunk_size=50_000_000,
            max_grad_norm=1.0,
            sigma_factor=12,
            clip_grad_norm_by_scheduler_group=True,
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        vision_activation_checkpointing=True,
        connector_activation_checkpointing=True,
        response_logits_only=RESPONSE_LOGITS_ONLY,
        diagnostics_interval=DIAGNOSTICS_INTERVAL,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=compile_model,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(
                    warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F, t_max=MAX_STEPS
                ),
                "vision": CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F, t_max=MAX_STEPS),
            },
            default=CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F, t_max=MAX_STEPS),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_dtype=DType.float32,
            only_allreduce_last_microbatch=True,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=ep_degree),
    )


def _validate_batch_geometry(config: ExperimentConfig, dp_world_size: Optional[int] = None) -> None:
    """Reject partial sequence-length overrides and invalid accumulation geometry.

    Stage 2 expresses batch sizes in tokens. An 8k versus 16k experiment must therefore
    update the collator, train-module horizon, global batch tokens, and rank microbatch
    tokens together. Failing early prevents a nominal 8k arm from silently retaining a
    16k collator or changing the number of sequences per optimizer step.

    :param config: Fully merged experiment config.
    :param dp_world_size: Optional runtime data-parallel world size.

    :raises ValueError: If any sequence or batch dimensions are inconsistent.
    """
    sequence_length = config.train_module.max_sequence_length
    if sequence_length <= 0:
        raise ValueError("train_module.max_sequence_length must be positive")
    if config.collator.pad_sequence_length != sequence_length:
        raise ValueError(
            "Stage 2 sequence-length overrides must update both "
            "collator.pad_sequence_length and train_module.max_sequence_length: "
            f"{config.collator.pad_sequence_length} != {sequence_length}"
        )
    for name, size in (
        ("global_batch_size", config.global_batch_size),
        ("train_module.rank_microbatch_size", config.train_module.rank_microbatch_size),
    ):
        if size <= 0 or size % sequence_length:
            raise ValueError(
                f"{name}={size} must be a positive multiple of sequence length "
                f"{sequence_length}"
            )
    global_instances = config.global_batch_size // sequence_length
    if global_instances != config.global_batch_instances:
        raise ValueError(
            "Stage 2 sequence-length overrides must preserve the global batch of "
            f"{config.global_batch_instances} sequences; got {global_instances}. "
            "Override global_batch_instances too when changing this intentionally."
        )
    rank_microbatch_instances = config.train_module.rank_microbatch_size // sequence_length
    if rank_microbatch_instances != config.rank_microbatch_instances:
        raise ValueError(
            "Stage 2 sequence-length overrides must preserve the rank microbatch of "
            f"{config.rank_microbatch_instances} sequence(s); got {rank_microbatch_instances}. "
            "Override rank_microbatch_instances too when changing this intentionally."
        )
    if dp_world_size is not None:
        if config.global_batch_size % dp_world_size:
            raise ValueError(
                f"global_batch_size={config.global_batch_size} must divide evenly across "
                f"DP world size {dp_world_size}"
            )
        rank_batch_size = config.global_batch_size // dp_world_size
        if rank_batch_size % config.train_module.rank_microbatch_size:
            raise ValueError(
                f"Rank batch size {rank_batch_size} must be divisible by rank microbatch "
                f"size {config.train_module.rank_microbatch_size}"
            )


def _validate_fixed_artifacts(config: ExperimentConfig) -> None:
    """Reject overrides that would disagree with the already-built model and tokenizer."""
    expected = {
        "base_checkpoint": BASE_CHECKPOINT,
        "vision_model_id": VISION_MODEL_ID,
        "vision_revision": VISION_REVISION,
        "tokenizer_id": TOKENIZER_ID,
        "hf_cache_dir": HF_CACHE_DIR,
    }
    for name, value in expected.items():
        configured = getattr(config, name)
        if configured != value:
            raise ValueError(
                f"{name} is fixed when constructing the Stage 2 model; "
                f"expected {value!r}, got {configured!r}"
            )


def _build_console_logger() -> ConsoleLoggerCallback:
    """Retain standard metrics and expose multimodal/component diagnostics."""
    callback = ConsoleLoggerCallback()
    callback.metrics.extend(
        [
            "data/*",
            "multimodal/*",
            "optim/* grad norm",
            "optim/* clip coefficient",
        ]
    )
    return callback


def _add_fast_vision_validation_callback(
    trainer,
    tokenizer,
    config: ExperimentConfig,
    collator,
    *,
    dp_world_size: int,
    dp_rank: int,
) -> None:
    """Add low-cost held-out caption, counting, and pointing CE/PPL signals."""
    interval = config.fast_vision_eval_interval
    if interval is None:
        return
    if interval <= 0:
        raise ValueError("fast_vision_eval_interval must be positive or None")
    if config.fast_vision_eval_examples <= 0:
        raise ValueError("fast_vision_eval_examples must be positive")
    if dp_world_size <= 0:
        raise ValueError("dp_world_size must be positive")

    sequence_length = config.train_module.max_sequence_length
    if sequence_length <= 0:
        raise ValueError("train_module.max_sequence_length must be positive")
    if collator.pad_sequence_length != sequence_length:
        raise ValueError(
            "Fast vision validation requires collator padding to match the configured "
            f"sequence length: {collator.pad_sequence_length} != {sequence_length}"
        )

    global_eval_instances = FAST_VISION_EVAL_RANK_BATCH_INSTANCES * dp_world_size
    if config.fast_vision_eval_examples % global_eval_instances != 0:
        raise ValueError(
            f"fast_vision_eval_examples ({config.fast_vision_eval_examples}) must be divisible "
            f"by the global validation batch ({global_eval_instances} instances)"
        )
    eval_batches = config.fast_vision_eval_examples // global_eval_instances

    common: Dict[str, Any] = {
        "max_crops": MAX_CROPS,
        "loss_token_weighting": "none",
        "token_ids": config.token_ids,
        "message_format": config.message_format,
        "seed": config.fast_vision_eval_seed,
    }
    dataset_configs: List[Tuple[str, Any]] = [
        (
            "pixmo-cap-caption-validation",
            PixMoCapDatasetConfig(
                dataset_path=f"{PIXMO_DATASETS}/cap",
                split="validation",
                mode="caption",
                max_sequence_length=sequence_length,
                **common,
            ),
        ),
        (
            "pixmo-count-validation",
            PixMoCountDatasetConfig(split="validation", counting="both", **common),
        ),
        (
            "pixmo-points-validation",
            PixMoPointsDatasetConfig(
                split="validation",
                kind="basic",
                counting="both",
                both_mode="duplicate",
                **common,
            ),
        ),
    ]

    evaluators: List[Evaluator] = []
    for name, dataset_config in dataset_configs:
        dataset = MaxSequenceLengthDataset(
            dataset_config.build(tokenizer),
            sequence_length,
            token_ids=config.token_ids,
        )
        loader = MultimodalDataLoader(
            dataset,
            collator,
            work_dir=trainer.work_dir / name,
            global_batch_size=global_eval_instances * sequence_length,
            seed=config.fast_vision_eval_seed,
            shuffle=False,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
        evaluators.append(
            MultimodalLMEvaluator(
                name=name,
                batches=loader,
                device=trainer.device,
                process_group=trainer.dp_process_group,
                deterministic=True,
            )
        )

    trainer.add_callback(
        "fast_vision_validation",
        EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=interval,
            eval_duration=Duration.steps(eval_batches),
            eval_on_startup=False,
            eval_on_finish=False,
            log_interval=max(eval_batches // 4, 1),
        ),
    )


def _configure_launch_runtime(launch_config: BeakerLaunchConfig) -> None:
    """Apply the OLMoDDP runtime without mutating the image's pinned Python dependencies."""
    from olmo_core.launch.beaker_presets import get_preset

    preset = get_preset("olmo-ddp")
    if preset.beaker_image is not None:
        launch_config.beaker_image = preset.beaker_image
    env = {item.name: item.value for item in launch_config.env_vars}
    env.update(dict(preset.env_vars))
    env.update(
        {
            "OLMO_USE_OWN_SYMM_MEM": "1",
            "OLMO_EP_MP_HIGH_PRIORITY_GROUP": "1",
            "OLMO_OWN_SYMM_PREWARM": "1",
            # Eight ranks times Inductor's default worker count oversubscribes a B300 node
            # during the compile warmup. Match the validated Stage 1 launch behavior.
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            # The launcher default enables verbose graph-break/recompile diagnostics. They are
            # useful when debugging compilation but overwhelm normal multi-rank training logs.
            "TORCH_LOGS": "-dynamo",
        }
    )
    launch_config.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
    launch_config.priority = "urgent"
    launch_config.min_runtime = "8h"
    # The image already pins datasets/pyarrow versions compatible with ai2-olmo-eval. The
    # multimodal compatibility loader reads datasets-5-authored Arrow files on those pins.
    launch_config.post_setup = preset.post_setup


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    tokenizer, token_ids = _load_tokenizer()
    model_config = _build_model_config(token_ids)

    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer {TOKENIZER_ID!r} does not define a pad token")
    collator_config = MultimodalCollatorConfig(
        pad_token_id=int(tokenizer.pad_token_id),
        label_ignore_index=-100,
        pad_sequence_length=SEQUENCE_LENGTH,
    )

    train_module_config = _build_train_module_config()

    trainer_config = (
        TrainerConfig(
            save_folder=f"{EXPERIMENT_ROOT}/checkpoints/{run_name}",
            save_overwrite=True,
            load_path=DEFAULT_LOAD_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=2000,
                ephemeral_save_interval=500,
                max_checkpoints=2,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                enabled=WANDB_PROJECT is not None,
                cancel_check_interval=10,
                auto_resume=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback("console_logger", _build_console_logger())
    )

    launch_config = build_launch_config(
        name=run_name,
        root_dir=root_dir,
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=NUM_NODES,
    )
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    launch_config.env_secrets = [
        s for s in launch_config.env_secrets if s.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    _configure_launch_runtime(launch_config)

    config = ExperimentConfig(
        model=model_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
        token_ids=token_ids,
    ).merge(overrides)
    _validate_fixed_artifacts(config)
    _validate_batch_geometry(config)
    return config


def _load_beaker_test_config(
    overrides: List[str],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Load a checked-in test profile without forwarding its path to the worker command."""
    prefix = "--beaker-test-config="
    profile_args = [value for value in overrides if value.startswith(prefix)]
    if len(profile_args) > 1:
        raise ValueError("At most one --beaker-test-config may be supplied")
    if not profile_args:
        return None, overrides

    profile_path = Path(profile_args[0][len(prefix) :])
    with profile_path.open() as f:
        profile = yaml.safe_load(f)
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError(f"Invalid Beaker test config {profile_path}: expected version: 1")
    unknown = set(profile) - {"version", "name", "description", "launch", "overrides"}
    if unknown:
        raise ValueError(f"Unknown keys in Beaker test config {profile_path}: {sorted(unknown)}")
    profile_overrides = profile.get("overrides", [])
    if not isinstance(profile_overrides, list) or not all(
        isinstance(value, str) and value.startswith("--") for value in profile_overrides
    ):
        raise ValueError(f"{profile_path}: overrides must be a list of '--key=value' strings")
    cli_overrides = [value for value in overrides if not value.startswith(prefix)]
    return profile, [*profile_overrides, *cli_overrides]


def _apply_beaker_test_config(
    config: ExperimentConfig, profile: Optional[Dict[str, Any]]
) -> ExperimentConfig:
    """Apply launch-only fields from a checked-in Beaker test profile."""
    if profile is None:
        return config
    launch = profile.get("launch", {})
    if not isinstance(launch, dict):
        raise ValueError("Beaker test config 'launch' must be a mapping")
    unknown = set(launch) - {
        "num_nodes",
        "num_gpus",
        "workspace",
        "cluster",
        "budget",
        "priority",
        "min_runtime",
    }
    if unknown:
        raise ValueError(f"Unknown Beaker test launch keys: {sorted(unknown)}")

    config.launch.num_nodes = int(launch.get("num_nodes", config.launch.num_nodes))
    config.launch.num_gpus = int(launch.get("num_gpus", config.launch.num_gpus))
    config.launch.workspace = launch.get("workspace")
    cluster = launch.get("cluster")
    config.launch.clusters = [] if cluster is None else [str(cluster)]
    config.launch.budget = launch.get("budget", config.launch.budget)
    config.launch.priority = str(launch.get("priority", config.launch.priority))
    config.launch.min_runtime = launch.get("min_runtime", config.launch.min_runtime)
    config.launch.description = profile.get("description")
    return config


def _load_tokenizer(
    identifier: str = TOKENIZER_ID,
    cache_dir: str = HF_CACHE_DIR,
):
    from transformers import GPT2Tokenizer

    from olmo_core.nn.vision import prepare_molmo2_tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(identifier, cache_dir=cache_dir)
    token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=100352)
    return tokenizer, token_ids


def _build_mixture(tokenizer, config: ExperimentConfig):
    max_sequence_length = config.train_module.max_sequence_length
    names_filter = _mixture_dataset_names(config.mixture)
    datasets, weights, names = build_image_only_v9_mixture(
        tokenizer,
        seed=config.data_seed,
        dataset_names=names_filter,
        max_sequence_length=max_sequence_length,
        token_ids=config.token_ids,
        message_format=config.message_format,
        return_names=True,
    )
    datasets, weights, names = _append_extra_sft_sources(
        config, tokenizer, datasets, weights, names
    )
    log.info(
        "Mixture %s sources / weights: %s",
        config.mixture,
        list(zip(names, [round(w, 4) for w in weights])),
    )
    return datasets, weights, names


def _append_extra_sft_sources(
    config: ExperimentConfig,
    tokenizer,
    datasets,
    weights,
    names,
):
    """Append optional PR #806 reasoning data without changing the default mixture."""
    from olmo_core.data.multimodal import (
        FineVisionDatasetConfig,
        MMFineReasonDatasetConfig,
    )

    if config.mmfinereason_rate < 0 or config.finevision_rate < 0:
        raise ValueError("Optional Stage 2 source rates must be non-negative")
    extra_total = config.mmfinereason_rate + config.finevision_rate
    if extra_total == 0:
        return datasets, weights, names
    if extra_total >= 1:
        raise ValueError(f"Optional Stage 2 source rates sum to {extra_total}; expected < 1")

    max_sequence_length = config.train_module.max_sequence_length
    datasets = list(datasets)
    weights = [weight * (1.0 - extra_total) for weight in weights]
    names = list(names)
    if config.mmfinereason_rate:
        datasets.append(
            MMFineReasonDatasetConfig(
                max_crops=MAX_CROPS,
                max_sequence_length=max_sequence_length,
                token_ids=config.token_ids,
                message_format=config.message_format,
                seed=config.data_seed,
            ).build(tokenizer)
        )
        weights.append(config.mmfinereason_rate)
        names.append("mmfinereason")

    if config.finevision_rate:
        per_config_rate = config.finevision_rate / len(FINEVISION_CONFIGS)
        for config_name in FINEVISION_CONFIGS:
            datasets.append(
                FineVisionDatasetConfig(
                    config_name=config_name,
                    max_crops=MAX_CROPS,
                    max_sequence_length=max_sequence_length,
                    min_visual_dependency=FINEVISION_MIN_VISUAL_DEPENDENCY,
                    token_ids=config.token_ids,
                    message_format=config.message_format,
                    seed=config.data_seed,
                ).build(tokenizer)
            )
            weights.append(per_config_rate)
            names.append(f"finevision[{config_name}]")
    return datasets, weights, names


def train(config: ExperimentConfig):
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    seed_all(config.init_seed)

    _validate_fixed_artifacts(config)

    tokenizer, token_ids = _load_tokenizer(config.tokenizer_id, config.hf_cache_dir)
    if token_ids != config.token_ids:
        raise ValueError(
            "Tokenizer image-token IDs do not match the serialized data/model config: "
            f"{token_ids} != {config.token_ids}"
        )
    validate_sft_message_format(
        config.message_format,
        tokenizer=tokenizer,
        token_ids=config.token_ids,
    )
    if not config.trainer.load_path:
        raise ValueError("Stage 2 requires trainer.load_path pointing to a stage-1 checkpoint")

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    collator = config.collator.build()

    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)
    _validate_batch_geometry(config, dp_world_size)

    datasets, weights, dataset_names = _build_mixture(tokenizer, config)
    log.info(
        "Stage 2 packing: pack=%s buffer=%d pack_max_crops=%d image_weight=%g "
        "vit_crop_microbatch=%s",
        config.pack_sequences,
        config.pack_buffer_size,
        config.pack_max_crops,
        config.pack_image_weight,
        os.environ.get("VIT_CROP_MICROBATCH", "16"),
    )
    data_loader = MixtureDataLoader(
        datasets,
        weights,
        collator,
        work_dir=config.trainer.save_folder,
        global_batch_size=config.global_batch_size,
        seed=config.data_seed,
        pack=config.pack_sequences,
        pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
        pack_buffer_size=config.pack_buffer_size if config.pack_sequences else 0,
        pack_image_weight=config.pack_image_weight,
        est_tokens_per_example=EST_TOKENS_PER_EXAMPLE,
        prefetch_workers=DATA_PREFETCH_WORKERS,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        dataset_names=dataset_names,
        # The released mixture contains a small number of malformed or unreadable rows.
        # Skip them deterministically while retaining strict bounds so a broken source still
        # fails quickly instead of silently changing the training mixture.
        max_consecutive_data_errors=config.max_consecutive_data_errors,
        max_total_data_errors=config.max_total_data_errors,
    )

    trainer = config.trainer.build(train_module, data_loader)
    trainer.add_callback("data_error_monitor", _DataErrorMonitorCallback())
    _add_fast_vision_validation_callback(
        trainer,
        tokenizer,
        config,
        collator,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict

    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} [dry_run|launch|train] RUN_NAME [OVERRIDES...]

  * dry_run: Print out the final config after applying overrides and exit.
  * launch:  Launch the script on Beaker as a batch job for training.
  * train:   Run training locally (usually under torchrun).

Examples
========

Print the config:
› python {sys.argv[0]} dry_run molmo2-stage2-debug

Local 8-GPU debug smoke (use a completed stage-1 run):
› torchrun --nproc-per-node=8 {sys.argv[0]} train smoke \\
      --trainer.load_path=/path/to/stage1/run --trainer.max_duration.value=1 \\
      --mixture=debug --train_module.compile_model=false

Full image-only-v9 mixture:
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run \\
      --trainer.load_path=/path/to/stage1/run --mixture=image-only-v9

Print the fresh two-node pilot config:
› python {sys.argv[0]} dry_run s002-stage2-image-only-v9-pilot \\
      --beaker-test-config=configs/vision_moe/stage2_ep8_2node_image_only_v9_to50.yaml

Launch on two Beaker nodes:
› python {sys.argv[0]} launch molmo2-stage2 \\
      --trainer.load_path=/path/to/stage1/run --mixture=image-only-v9
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        prepare_training_environment(timeout=timedelta(minutes=60))
    else:
        prepare_cli_environment()

    beaker_test_config, overrides = _load_beaker_test_config(overrides)
    config = build_config(script, run_name, overrides)
    config = _apply_beaker_test_config(config, beaker_test_config)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        pass
    else:
        print(usage)
        sys.exit(1)
