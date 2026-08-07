"""
Molmo2 "stage 1" caption-pretraining with an s002 OLMo3 MoE language model.

Trains the connector + LM + vision encoder on PixMoCap captions/transcripts, using binary
response-token loss weights and per-component learning rates / warmups. A deterministic
held-out PixMoCap slice reports response-token loss and perplexity during training.

Run without arguments for usage. Quick local smoke test on synthetic data::

    torchrun --nproc-per-node=8 src/scripts/train/Molmo2-Stage1.py train smoke \\
        --dataset.dataset_path=synthetic --trainer.max_duration.value=5 \\
        --trainer.max_duration.unit=steps

.. note::
    The production topology is PP=1, EP=8, with DDP over all ranks. A two-node Beaker
    launch therefore has EP-DP=2. TP/CP/PP and two-batch overlap are intentionally disabled.
"""

import logging
import os
import sys
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import yaml

from olmo_core.config import Config, DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.multimodal import (
    CoSynPointDatasetConfig,
    MixtureDataLoader,
    MultimodalCollatorConfig,
    MultimodalDataLoader,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
    Tulu4DatasetConfig,
)
from olmo_core.data.multimodal.paths import PIXMO_DATASETS
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import MultimodalLMEvaluator
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
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    ConsoleLoggerCallback,
    DownstreamEvaluatorCallbackConfig,
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
MOLMO2_CONFIG_MODEL_ID = "allenai/Molmo2-4B"
MOLMO2_CONFIG_REVISION = "042abfa7a38879a376cec03d949eff0aefaa0600"
VISION_MODEL_ID = "google/siglip2-so400m-patch14-384"
VISION_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
VISION_FINGERPRINT = "9d9257ea672527b2e37cae7f61734afdf9280d3e77680f2c2d13d4da60aba6bf"
TOKENIZER_ID = "allenai/dolma2-tokenizer"
HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
EXPERIMENT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-moe"
SEQUENCE_LENGTH = 2560  # released 32k Molmo2 stage-1 artifact sequence length
USE_FLEX_ATTN = True  # fused FlexAttention backend for the multimodal masks (~+8% MFU)
PACK_SEQUENCES = True  # pack several examples into each fixed-length training sequence
PACK_BUFFER_SIZE = 48  # released Molmo2 dynamic-packing lookahead
PACK_MAX_CROPS = 16  # released Molmo2 composite image/video packing budget
ROUTER_LB_LOSS_WEIGHT: Optional[float] = 0.015  # native s002 pretraining objective
COMPILE_MODEL = True  # torch.compile the LM (fuses pointwise ops; one-time compile warmup)
DATA_PREFETCH_WORKERS = 8  # B300 gate: prevents deterministic packing-stream load stalls
DIAGNOSTICS_INTERVAL = 100  # connector/input scales and per-component gradient norms
MAX_CROPS = 8
LOSS_TOKEN_WEIGHTING = "none"
EP_DEGREE = 8
EVAL_INTERVAL = 1000
EVAL_EXAMPLES = 2048
EVAL_RANK_BATCH_INSTANCES = 4
EVAL_SEED = 6198
FAST_VISION_EVAL_INTERVAL = 2000
FAST_VISION_EVAL_EXAMPLES = 256
FAST_LANGUAGE_EVAL_INTERVAL = 4000
# The complete repository "fast" group took about 2.6 hours on the step-4k s002 artifact.
# These four low-cost sentinels retain MCQA, generative BPB, arithmetic, and task-format
# coverage while full OLMES remains an external checkpoint evaluation.
FAST_LANGUAGE_EVAL_TASKS = (
    "arc_challenge_test_mc_5shot_fast",
    "basic_skills_arithmetic_rc_5shot",
    "copycolors_10way_fast",
    "hellaswag_bpb_5shot",
)

# KNOWN DELTA vs mm_olmo stage-1 captioner: `response_residual_dropout=0.1`.
# mm_olmo applies 0.1 dropout to the residual stream of RESPONSE tokens only (input/image
# tokens get 0.0), via a per-token drop mask in its LM block (olmo/nn/llm.py: `Dropout`
# with `mask_p`). OLMo-core's `TransformerBlock` has a single uniform `nn.Dropout` (default
# 0.0) with no per-token/response path, so this regularizer is intentionally NOT applied
# here — adding it would require threading a response drop-mask through the core transformer
# block. This remains a known fidelity delta for the 32k run and should receive a separate,
# targeted core design and validation rather than an approximate uniform-dropout substitute.
# (The other mm_olmo delta, the `style_and_length_v2` length conditioning, IS implemented;
# see PixMoCapDataset.style_length_conditioning.)

# Instance-based batching, expressed in tokens. Keep released Molmo2's global batch 128 and
# device microbatch 4. On the two-node EP8 topology this gives each rank eight sequences in two
# microbatches, while increasing the routed-token population per forward toward s002 pretraining.
GLOBAL_BATCH_INSTANCES = 128
RANK_MICROBATCH_INSTANCES = 4
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo train_captioner.py).
CONNECTOR_LR = 2e-4
VISION_LR = 6e-6
LLM_LR = 2e-5
CONNECTOR_WARMUP = 200
VISION_WARMUP = 2000
LLM_WARMUP = 2000
ALPHA_F = 0.1

# Data: the canonical PixMoCap "cap" dataset (HF DatasetDict, load_from_disk). Override as needed.
DATASET_PATH = f"{PIXMO_DATASETS}/cap"
MAX_STEPS = 32000

# Stage-1 mixture rates (mm_olmo train_captioner --pointing/--nlp). Caption gets the
# remainder (1 - POINTING_RATE - NLP_RATE). Set both to 0.0 for a caption-only run.
POINTING_RATE = 0.30
NLP_RATE = 0.10

# Beaker.
BEAKER_CLUSTER = "ai2/holmes"
NUM_NODES = 2
BEAKER_WORKSPACE = "ai2/molmofication"
BEAKER_BUDGET = "ai2/oe-other"

# Logging. Set WANDB_PROJECT to None to disable W&B (requires the WANDB_API_KEY secret
# in the Beaker workspace). Metrics always go to the console regardless.
# WANDB_ENTITY=None uses the API key's default entity (personal account), avoiding 403s
# from writing to a team the key lacks access to; set it to a team you can write to.
WANDB_PROJECT: Optional[str] = "molmo2-stage1"
WANDB_ENTITY: Optional[str] = None

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    dataset: PixMoCapDatasetConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    trainer: TrainerConfig
    base_checkpoint: str = BASE_CHECKPOINT
    molmo2_config_model_id: str = MOLMO2_CONFIG_MODEL_ID
    molmo2_config_revision: str = MOLMO2_CONFIG_REVISION
    vision_model_id: str = VISION_MODEL_ID
    vision_revision: str = VISION_REVISION
    vision_fingerprint: str = VISION_FINGERPRINT
    tokenizer_id: str = TOKENIZER_ID
    hf_cache_dir: str = HF_CACHE_DIR
    checkpoint_load_threads: int = 8
    data_seed: int = 95818
    init_seed: int = 6198
    global_batch_size: int = GLOBAL_BATCH_SIZE
    """Global batch in *tokens* (= global instances × seq len). Override to scale the batch;
    pair with ``--train_module.rank_microbatch_size`` to set sequences/forward (GEMM size)."""
    pointing_rate: float = POINTING_RATE
    """Fraction of mixture samples drawn from pointing and counting sources."""
    nlp_rate: float = NLP_RATE
    """Fraction of mixture samples drawn from Tulu4 text-only SFT."""
    pack_sequences: bool = PACK_SEQUENCES
    """Whether to pack multiple examples into each training sequence."""
    pack_buffer_size: int = PACK_BUFFER_SIZE
    """Examples considered by Molmo2's buffered two-constraint packing solver."""
    pack_max_crops: int = PACK_MAX_CROPS
    """Maximum total image crops in one packed sequence."""
    data_prefetch_workers: int = DATA_PREFETCH_WORKERS
    """Background data-preprocessing threads per rank; zero disables prefetching."""
    router_lb_loss_weight: Optional[float] = ROUTER_LB_LOSS_WEIGHT
    """Stage-1 router load-balancing weight; defaults to the native s002 value."""
    eval_interval: Optional[int] = EVAL_INTERVAL
    """Run held-out PixMoCap loss/PPL every this many steps; ``None`` disables it."""
    eval_examples: int = EVAL_EXAMPLES
    """Number of deterministic validation examples per evaluation."""
    eval_rank_batch_instances: int = EVAL_RANK_BATCH_INSTANCES
    """Validation instances per DP rank and forward pass."""
    eval_seed: int = EVAL_SEED
    fast_vision_eval_interval: Optional[int] = FAST_VISION_EVAL_INTERVAL
    """Run caption-only, counting, and pointing validation every this many steps."""
    fast_vision_eval_examples: int = FAST_VISION_EVAL_EXAMPLES
    """Deterministic examples per fast vision evaluator."""
    fast_language_eval_interval: Optional[int] = FAST_LANGUAGE_EVAL_INTERVAL
    """Run the compact language-retention sentinel every this many steps."""


def _configure_router_load_balancing(lm_config, weight: Optional[float]) -> int:
    """Set the routed-expert load-balancing weight on every s002 block-config variant."""
    if weight is not None and weight < 0:
        raise ValueError("router_lb_loss_weight must be non-negative or None")

    configured = 0
    block_configs = [lm_config.block, *(lm_config.block_overrides or {}).values()]
    for block_config in block_configs:
        router = getattr(block_config, "routed_experts_router", None)
        if router is not None:
            router.lb_loss_weight = weight
            configured += 1
    if configured == 0:
        raise ValueError("The s002 Stage-1 language model has no routed-expert router configs")
    return configured


def _build_model_config(token_ids: Molmo2TokenIds) -> MultimodalLMConfig:
    """Compose the native s002 LM with Molmo2's connector and SigLIP2 architecture."""
    import json

    from olmo_core.nn.attention import AttentionConfig
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
    from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
    from olmo_core.nn.transformer import OLMoDDPModelConfig
    from olmo_core.nn.vision import (
        load_molmo2_hf_vision_config,
        multimodal_config_from_molmo2_vision,
    )

    config_path = Path(BASE_CHECKPOINT) / "config.json"
    with config_path.open() as f:
        lm_config = OLMoDDPModelConfig.from_dict(json.load(f)["model"])

    attention_backend = AttentionBackendName.flex if USE_FLEX_ATTN else AttentionBackendName.torch
    block_configs = [lm_config.block, *(lm_config.block_overrides or {}).values()]
    for block_config in block_configs:
        if not isinstance(block_config, OLMoDDPTransformerBlockConfig):
            raise TypeError(
                "The s002 checkpoint must use OLMoDDPTransformerBlockConfig for every block"
            )
        if isinstance(block_config.sequence_mixer, AttentionConfig):
            block_config.sequence_mixer.backend = attention_backend
        if block_config.ep is not None:
            block_config.ep.path = ExpertParallelPath.rowwise_nvshmem

    # PP is unavailable for the composite model, so checkpoint each LM block to keep the
    # full sequence on one rank without retaining all 31 blocks' activations.
    lm_config.recompute_each_block = True
    lm_config.recompute_all_blocks_by_chunk = False
    lm_config.two_batch_overlap = False

    hf_config = load_molmo2_hf_vision_config(
        MOLMO2_CONFIG_MODEL_ID,
        revision=MOLMO2_CONFIG_REVISION,
        cache_dir=HF_CACHE_DIR,
    )
    return multimodal_config_from_molmo2_vision(
        hf_config,
        lm_config,
        image_patch_token_id=token_ids.im_patch_id,
    )


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
            sigma_factor=12,
            max_grad_norm=1.0,
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        freeze_params=None,
        vision_activation_checkpointing=True,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        diagnostics_interval=DIAGNOSTICS_INTERVAL,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=compile_model,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(warmup=CONNECTOR_WARMUP, alpha_f=ALPHA_F),
                "vision": CosWithWarmup(warmup=VISION_WARMUP, alpha_f=ALPHA_F),
            },
            default=CosWithWarmup(warmup=LLM_WARMUP, alpha_f=ALPHA_F),
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
            # during the compile warmup. Eight workers per rank keeps compilation parallel
            # without starving the training and data-loader processes.
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            # The launcher default enables verbose graph-break/recompile diagnostics. They are
            # useful when debugging compilation but overwhelm normal multi-rank training logs.
            "TORCH_LOGS": "-dynamo",
        }
    )
    launch_config.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
    # The image already pins datasets/pyarrow versions compatible with ai2-olmo-eval. The
    # multimodal compatibility loader reads datasets-5-authored Arrow files on those pins.
    launch_config.post_setup = preset.post_setup


def _build_console_logger() -> ConsoleLoggerCallback:
    """Retain OLMo's standard console metrics and expose Stage 1 diagnostics."""
    callback = ConsoleLoggerCallback()
    callback.metrics.extend(
        [
            "data/*",
            "multimodal/*",
            "optim/* grad norm",
        ]
    )
    return callback


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)

    tokenizer, token_ids = _load_tokenizer()
    model_config = _build_model_config(token_ids)

    dataset_config = PixMoCapDatasetConfig(
        dataset_path=DATASET_PATH,
        mode="transcript_and_caption",
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting=LOSS_TOKEN_WEIGHTING,
        # Molmo2 derives augmentation from (example index, source epoch). The mixture loader's
        # separate data_seed controls source sampling and permutations.
        seed=0,
        token_ids=token_ids,
    )

    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer {TOKENIZER_ID!r} does not define a pad token")
    # Fixed-length padding keeps every batch compatible with the token-based Trainer.
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
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            # Synchronous checkpointing: avoids the async checkpoint thread pool whose
            # teardown raced/failed on this cluster ("cannot schedule new futures after
            # interpreter shutdown"). Saves block briefly but complete reliably.
            CheckpointerCallback(save_interval=2000, ephemeral_save_interval=500, save_async=False),
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
    # Stage-1 reads data and writes checkpoints on weka, so no S3 / GCS secrets are required.
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    # Only request env secrets that exist in the (debug) workspace; drop optional ones
    # (COMET / R2 / WEKA / SLACK) that aren't provisioned there.
    launch_config.env_secrets = [
        s for s in launch_config.env_secrets if s.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    # Reuse the repository's OLMoDDP image/setup contract (NVSHMEM + symmetric memory).
    _configure_launch_runtime(launch_config)

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
    ).merge(overrides)
    configured_routers = _configure_router_load_balancing(
        config.model.lm, config.router_lb_loss_weight
    )
    log.info(
        "Configured %d routed-expert block variants with lb_loss_weight=%s; "
        "router z-loss remains checkpoint-native",
        configured_routers,
        config.router_lb_loss_weight,
    )
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

    # Dolma2 publishes tokenizer assets without a model config. Loading the declared GPT-2
    # tokenizer class directly also works from an offline/shared HF cache.
    tokenizer = GPT2Tokenizer.from_pretrained(identifier, cache_dir=cache_dir)
    token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=100352)
    return tokenizer, token_ids


def _checkpoint_state_dir(checkpoint: str) -> str:
    nested = Path(checkpoint) / "model_and_optim"
    return str(nested if nested.is_dir() else Path(checkpoint))


def _build_mixture_sources(tokenizer, config: ExperimentConfig):
    """Build the caption + pointing + NLP sources and their sampling weights (mm_olmo
    SubMixture): caption gets ``1 - pointing_rate - nlp_rate``; the pointing group shares
    ``pointing_rate`` split by sqrt(size); NLP gets ``nlp_rate``."""
    import numpy as np

    p, n = config.pointing_rate, config.nlp_rate
    sources: List[Tuple[str, Any, float]] = [
        (
            "pixmo_cap_with_transcripts",
            config.dataset.build(tokenizer),
            max(1.0 - p - n, 0.0),
        )
    ]

    if p > 0:
        pointing: List[Any] = [
            PixMoPointsDatasetConfig(
                kind="basic",
                max_crops=MAX_CROPS,
                loss_token_weighting=config.dataset.loss_token_weighting,
                token_ids=config.dataset.token_ids,
            ).build(tokenizer),
            PixMoCountDatasetConfig(
                max_crops=MAX_CROPS,
                loss_token_weighting=config.dataset.loss_token_weighting,
                token_ids=config.dataset.token_ids,
            ).build(tokenizer),
            PixMoPointsDatasetConfig(
                kind="high_frequency",
                max_crops=MAX_CROPS,
                loss_token_weighting=config.dataset.loss_token_weighting,
                token_ids=config.dataset.token_ids,
            ).build(tokenizer),
            CoSynPointDatasetConfig(
                max_crops=MAX_CROPS,
                loss_token_weighting=config.dataset.loss_token_weighting,
                token_ids=config.dataset.token_ids,
            ).build(tokenizer),
        ]
        frac = np.sqrt(np.array([len(d) for d in pointing], dtype=np.float64))
        frac = frac / frac.sum()
        sources.extend(
            (name, dataset, p * float(weight))
            for name, dataset, weight in zip(
                [
                    "pixmo_points_train",
                    "pixmo_count_train",
                    "pixmo_points_high_freq_train",
                    "cosyn_point",
                ],
                pointing,
                frac,
            )
        )

    if n > 0:
        sources.append(
            (
                "tulu4_max_2304",
                Tulu4DatasetConfig(
                    max_sequence_length=config.dataset.max_sequence_length,
                    loss_token_weighting=config.dataset.loss_token_weighting,
                    token_ids=config.dataset.token_ids,
                ).build(tokenizer),
                n,
            )
        )

    # mm_olmo sorts the flattened KwargsMixture by dataset name before handing the rates to
    # IterableDatasetMixture. Source index is part of the seeded multinomial stream, so retain
    # that order exactly rather than relying on construction order.
    sources.sort(key=lambda source: source[0])
    names = [name for name, _, _ in sources]
    datasets = [dataset for _, dataset, _ in sources]
    weights = [weight for _, _, weight in sources]

    log.info(
        "Mixture sources / weights: %s",
        [(name, type(d).__name__, round(w, 3)) for name, d, w in zip(names, datasets, weights)],
    )
    return datasets, weights, names


def _add_validation_callback(
    trainer,
    tokenizer,
    config: ExperimentConfig,
    collator,
    *,
    dp_world_size: int,
    dp_rank: int,
) -> None:
    if config.eval_interval is None:
        return
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be positive or None")
    if config.eval_examples <= 0:
        raise ValueError("eval_examples must be positive")
    if config.eval_rank_batch_instances <= 0:
        raise ValueError("eval_rank_batch_instances must be positive")
    if collator.pad_sequence_length is None:
        raise ValueError("Stage 1 validation requires fixed-length padding")

    # Keep the dataset's example/epoch augmentation seed unchanged. The released Molmo2
    # evaluator's seed controls the loader; mm_olmo's DeterministicDataset derives example
    # augmentation independently from (index, epoch).
    eval_dataset_config = replace(config.dataset, split="validation")
    global_eval_instances = config.eval_rank_batch_instances * dp_world_size
    if config.eval_examples % global_eval_instances != 0:
        raise ValueError(
            f"eval_examples ({config.eval_examples}) must be divisible by the global "
            f"validation batch ({global_eval_instances} instances)"
        )
    eval_batches = config.eval_examples // global_eval_instances
    eval_loader = MultimodalDataLoader(
        eval_dataset_config.build(tokenizer),
        collator,
        work_dir=trainer.work_dir / "pixmo_cap_validation",
        global_batch_size=(global_eval_instances * collator.pad_sequence_length),
        seed=config.eval_seed,
        shuffle=False,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    evaluator = MultimodalLMEvaluator(
        name="pixmo-cap-validation",
        batches=eval_loader,
        device=trainer.device,
        process_group=trainer.dp_process_group,
        deterministic=True,
    )
    trainer.add_callback(
        "pixmo_cap_validation",
        EvaluatorCallback(
            evaluators=[evaluator],
            eval_interval=config.eval_interval,
            eval_duration=Duration.steps(eval_batches),
            eval_on_startup=False,
            eval_on_finish=False,
            log_interval=max(eval_batches // 4, 1),
        ),
    )


def _add_fast_vision_validation_callback(
    trainer,
    tokenizer,
    config: ExperimentConfig,
    collator,
    *,
    dp_world_size: int,
    dp_rank: int,
) -> None:
    """Add low-cost held-out caption, counting, and pointing health signals."""
    interval = config.fast_vision_eval_interval
    if interval is None:
        return
    if interval <= 0:
        raise ValueError("fast_vision_eval_interval must be positive or None")
    if config.fast_vision_eval_examples <= 0:
        raise ValueError("fast_vision_eval_examples must be positive")
    if config.eval_rank_batch_instances <= 0:
        raise ValueError("eval_rank_batch_instances must be positive")
    if collator.pad_sequence_length is None:
        raise ValueError("Stage 1 validation requires fixed-length padding")

    global_eval_instances = config.eval_rank_batch_instances * dp_world_size
    if config.fast_vision_eval_examples % global_eval_instances != 0:
        raise ValueError(
            f"fast_vision_eval_examples ({config.fast_vision_eval_examples}) must be divisible "
            f"by the global validation batch ({global_eval_instances} instances)"
        )
    eval_batches = config.fast_vision_eval_examples // global_eval_instances

    common = {
        "max_crops": config.dataset.max_crops,
        "loss_token_weighting": config.dataset.loss_token_weighting,
        "token_ids": config.dataset.token_ids,
        "seed": config.dataset.seed,
    }
    dataset_configs = [
        (
            "pixmo-cap-caption-validation",
            replace(config.dataset, split="validation", mode="caption"),
        ),
        (
            "pixmo-count-validation",
            PixMoCountDatasetConfig(split="validation", counting="both", **common),
        ),
        (
            "pixmo-points-validation",
            PixMoPointsDatasetConfig(split="validation", kind="basic", counting="both", **common),
        ),
    ]

    evaluators = []
    for name, dataset_config in dataset_configs:
        loader = MultimodalDataLoader(
            dataset_config.build(tokenizer),
            collator,
            work_dir=trainer.work_dir / name,
            global_batch_size=(global_eval_instances * collator.pad_sequence_length),
            seed=config.eval_seed,
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


def _add_fast_language_validation_callback(trainer, config: ExperimentConfig) -> None:
    """Add a compact OLMES language-retention sentinel to in-loop reporting."""
    interval = config.fast_language_eval_interval
    if interval is None:
        return
    if interval <= 0:
        raise ValueError("fast_language_eval_interval must be positive or None")

    callback = DownstreamEvaluatorCallbackConfig(
        tasks=list(FAST_LANGUAGE_EVAL_TASKS),
        tokenizer=TokenizerConfig.dolma2(),
        eval_interval=interval,
        eval_on_startup=False,
        eval_on_finish=False,
        log_interval=20,
        lazy=True,
    ).build(trainer)
    if callback is None:
        raise RuntimeError("Fast language evaluator callback was unexpectedly disabled")
    trainer.add_callback("fast_language_validation", callback)


def train(config: ExperimentConfig):
    if config.data_prefetch_workers < 0:
        raise ValueError("data_prefetch_workers must be non-negative")

    # These are harmless when already set by the Beaker OLMoDDP preset and make the same
    # rowwise-NVSHMEM path explicit for local torchrun launches.
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    seed_all(config.init_seed)

    tokenizer, token_ids = _load_tokenizer(config.tokenizer_id, config.hf_cache_dir)
    if token_ids != config.dataset.token_ids:
        raise ValueError(
            "Tokenizer image-token IDs do not match the serialized dataset/model config: "
            f"{token_ids} != {config.dataset.token_ids}"
        )

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)

    import torch.distributed as dist

    state_dir = _checkpoint_state_dir(config.base_checkpoint)
    log.info("Loading s002 language-model weights from %s", state_dir)
    train_module.load_state_dict_direct(
        state_dir,
        process_group=dist.group.WORLD,
        thread_count=config.checkpoint_load_threads,
        load_optim_state=False,
    )

    from olmo_core.nn.vision import (
        load_siglip_hf_vision_state_dict,
        siglip_hf_state_dict_to_vision,
        vision_state_fingerprint,
    )

    log.info(
        "Loading pristine SigLIP2 vision weights from %s at revision %s",
        config.vision_model_id,
        config.vision_revision,
    )
    hf_vision_state = load_siglip_hf_vision_state_dict(
        config.vision_model_id,
        revision=config.vision_revision,
        cache_dir=config.hf_cache_dir,
    )
    vision_state = siglip_hf_state_dict_to_vision(
        hf_vision_state, train_module.multimodal_model.cfg.vision
    )
    fingerprint = vision_state_fingerprint(vision_state)
    if fingerprint != config.vision_fingerprint:
        raise ValueError(
            "SigLIP2 vision checkpoint fingerprint mismatch: "
            f"expected {config.vision_fingerprint}, got {fingerprint}"
        )
    log.info(
        "Verified SigLIP2 vision fingerprint %s; patch RMS=%.6f, position RMS=%.6f",
        fingerprint,
        vision_state["patch_embedding.weight"].float().square().mean().sqrt().item(),
        vision_state["positional_embedding"].float().square().mean().sqrt().item(),
    )
    train_module.load_vision_state_dict(vision_state)
    train_module.assert_vision_optimizer_state_synced()
    del hf_vision_state, vision_state

    # The six image-format tokens occupy previously padded s002 vocabulary rows. Reset both
    # embedding and LM-head rows, then synchronize those changes into optimizer main state.
    train_module.reset_image_token_rows(
        [
            token_ids.im_start_id,
            token_ids.im_end_id,
            token_ids.im_patch_id,
            token_ids.im_col_id,
            token_ids.low_res_im_start_id,
            token_ids.image_placeholder_id,
        ],
        seed=config.init_seed,
    )

    collator = config.collator.build()
    # Derive the data-parallel world size / rank from the train module's DP process
    # group so each rank reads its own shard (must match the trainer's DP degree).
    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)

    data_loader: DataLoaderBase
    if config.pointing_rate > 0 or config.nlp_rate > 0:
        datasets, weights, dataset_names = _build_mixture_sources(tokenizer, config)
        log.info(
            "Stage 1 packing: pack=%s buffer_size=%d max_crops=%d prefetch_workers=%d",
            config.pack_sequences,
            config.pack_buffer_size,
            config.pack_max_crops,
            config.data_prefetch_workers,
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
            prefetch_workers=config.data_prefetch_workers,
            dataset_names=dataset_names,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
    else:
        data_loader = MultimodalDataLoader(
            config.dataset.build(tokenizer),
            collator,
            work_dir=config.trainer.save_folder,
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

    trainer = config.trainer.build(train_module, data_loader)
    _add_validation_callback(
        trainer,
        tokenizer,
        config,
        collator,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    _add_fast_vision_validation_callback(
        trainer,
        tokenizer,
        config,
        collator,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    _add_fast_language_validation_callback(trainer, config)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict

    trainer.fit()


def launch(config: ExperimentConfig):
    if not config.launch.workspace or not config.launch.clusters:
        raise RuntimeError(
            "Beaker workspace and cluster are unset. Fill the approved target in the test config "
            "before launching; no experiment was submitted."
        )
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
› python {sys.argv[0]} dry_run molmo2-stage1

Print a submission-safe two-node test config:
› python {sys.argv[0]} dry_run molmo2-stage1-gate \
      --beaker-test-config=configs/vision_moe/stage1_ep8_2node_real_1step.yaml

Local synthetic smoke test:
› torchrun --nproc-per-node=8 {sys.argv[0]} train smoke \\
      --dataset.dataset_path=synthetic --trainer.max_duration.value=5

Launch on Beaker:
› python {sys.argv[0]} launch molmo2-stage1
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        # Use a generous process-group timeout (gloo + NCCL). The default 15 min was the
        # exact watchdog timeout that aborted runs when a rank lagged on a collective
        # during checkpointing / bookkeeping (and W&B network stalls can add latency).
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
