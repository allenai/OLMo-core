"""
Molmo2 "stage 1" caption-pretraining with an s002 OLMo3 MoE language model.

Trains the connector + LM on PixMoCap captions/transcripts with the vision encoder
**frozen**, using the float ``root_subsegments``-weighted loss and per-component
learning rates / warmups. In-loop evaluation is intentionally omitted.

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
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, cast

from olmo_core.config import Config, DType
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
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.common import (
    build_launch_config,
    get_root_dir,
)
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
TOKENIZER_ID = "allenai/dolma2-tokenizer"
HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
EXPERIMENT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-moe"
SEQUENCE_LENGTH = 4096  # fixed pad length; mm_olmo stage 1 uses ~5248
USE_FLEX_ATTN = True  # fused FlexAttention backend for the multimodal masks (~+8% MFU)
PACK_SEQUENCES = True  # pack several examples per sequence (most are ~1.4k of 4096 tokens)
COMPILE_MODEL = True  # torch.compile the LM (fuses pointwise ops; one-time compile warmup)
DATA_PREFETCH_WORKERS = 4  # background threads preprocessing examples (0 = synchronous)
MAX_CROPS = 8
EP_DEGREE = 8

# KNOWN DELTA vs mm_olmo stage-1 captioner: `response_residual_dropout=0.1`.
# mm_olmo applies 0.1 dropout to the residual stream of RESPONSE tokens only (input/image
# tokens get 0.0), via a per-token drop mask in its LM block (olmo/nn/llm.py: `Dropout`
# with `mask_p`). OLMo-core's `TransformerBlock` has a single uniform `nn.Dropout` (default
# 0.0) with no per-token/response path, so this regularizer is intentionally NOT applied
# here — adding it would require threading a response drop-mask through the core transformer
# block. Low impact for the short benchmark runs; revisit for a full-fidelity stage-1
# reproduction. (The other mm_olmo delta, the `style_and_length_v2` length-conditioning
# system prompt, IS implemented — see PixMoCapDataset.style_length_conditioning.)

# Instance-based batching (mm_olmo: global 8, device microbatch 1), expressed in tokens.
# Sixteen keeps one instance/rank on the normal two-node (16 GPU) topology and
# accumulates two one-instance microbatches/rank for an eight-GPU local run.
GLOBAL_BATCH_INSTANCES = 16
RANK_MICROBATCH_INSTANCES = 1
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo train_captioner.py).
CONNECTOR_LR = 2e-4
LLM_LR = 2e-5
CONNECTOR_WARMUP = 200
LLM_WARMUP = 2000
ALPHA_F = 0.1

# Data: the canonical PixMoCap "cap" dataset (HF DatasetDict, load_from_disk). Override as needed.
DATASET_PATH = "/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets/cap"
MAX_STEPS = 4000

# Stage-1 mixture rates (mm_olmo train_captioner --pointing/--nlp). Caption gets the
# remainder (1 - POINTING_RATE - NLP_RATE). Set both to 0.0 for a caption-only run.
POINTING_RATE = 0.30
NLP_RATE = 0.10

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
NUM_NODES = 2
BEAKER_WORKSPACE = "ai2/OLMo-core"
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
    vision_model_id: str = VISION_MODEL_ID
    vision_revision: str = VISION_REVISION
    tokenizer_id: str = TOKENIZER_ID
    hf_cache_dir: str = HF_CACHE_DIR
    checkpoint_load_threads: int = 8
    data_seed: int = 34521
    init_seed: int = 12536
    global_batch_size: int = GLOBAL_BATCH_SIZE
    """Global batch in *tokens* (= global instances × seq len). Override to scale the batch;
    pair with ``--train_module.rank_microbatch_size`` to set sequences/forward (GEMM size)."""
    pointing_rate: float = POINTING_RATE
    """Fraction of mixture samples drawn from pointing and counting sources."""
    nlp_rate: float = NLP_RATE
    """Fraction of mixture samples drawn from Tulu4 text-only SFT."""
    pack_sequences: bool = PACK_SEQUENCES
    """Whether to pack multiple examples into each training sequence."""


def _build_model_config(token_ids: Molmo2TokenIds) -> MultimodalLMConfig:
    """Compose the exact native s002 LM architecture with the Molmo2 vision tower."""
    import json

    from transformers import AutoConfig

    from olmo_core.nn.attention import AttentionConfig
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
    from olmo_core.nn.transformer import OLMoDDPModelConfig
    from olmo_core.nn.vision import multimodal_config_from_molmo2_vision

    config_path = Path(BASE_CHECKPOINT) / "config.json"
    with config_path.open() as f:
        lm_config = OLMoDDPModelConfig.from_dict(json.load(f)["model"])

    attention_backend = AttentionBackendName.flex if USE_FLEX_ATTN else AttentionBackendName.torch
    block_configs = [lm_config.block, *(lm_config.block_overrides or {}).values()]
    for block_config in block_configs:
        if isinstance(block_config.sequence_mixer, AttentionConfig):
            block_config.sequence_mixer.backend = attention_backend
        if block_config.ep is not None:
            block_config.ep.path = ExpertParallelPath.rowwise_nvshmem

    # PP is unavailable for the composite model, so checkpoint each LM block to keep the
    # full sequence on one rank without retaining all 31 blocks' activations.
    lm_config.recompute_each_block = True
    lm_config.recompute_all_blocks_by_chunk = False
    lm_config.two_batch_overlap = False

    hf_config = AutoConfig.from_pretrained(
        VISION_MODEL_ID,
        revision=VISION_REVISION,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
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
            ],
            compile=False,
            foreach_chunk_size=50_000_000,
            max_grad_norm=1.0,
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        freeze_params=["vision.*"],
        response_logits_only=True,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=compile_model,
        scheduler=PerGroupScheduler(
            schedulers={"connector": CosWithWarmup(warmup=CONNECTOR_WARMUP, alpha_f=ALPHA_F)},
            default=CosWithWarmup(warmup=LLM_WARMUP, alpha_f=ALPHA_F),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_dtype=DType.float32,
            only_allreduce_last_microbatch=True,
            reduce_grads_in_fp32=False,
            accumulate_grads_in_fp32=False,
        ),
        ep_config=TransformerExpertParallelConfig(degree=ep_degree),
    )


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)

    tokenizer, token_ids = _load_tokenizer()
    model_config = _build_model_config(token_ids)

    dataset_config = PixMoCapDatasetConfig(
        dataset_path=DATASET_PATH,
        mode="transcript_and_caption",
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="root_subsegments",
        seed=34521,
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
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
    )  # NOTE: no in-loop eval callbacks (out of scope for stage 1).

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
    # Reuse the repository's OLMoDDP image/setup contract (NVSHMEM + the symmetric-memory
    # extension), then add the dataset-version requirement specific to these Arrow files.
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
        }
    )
    launch_config.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
    post_setup = [preset.post_setup, "pip install -U 'datasets>=4,<6'"]
    launch_config.post_setup = " && ".join(step for step in post_setup if step)

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
    ).merge(overrides)


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
    SubMixture): caption gets ``1 - POINTING_RATE - NLP_RATE``; the pointing group shares
    ``POINTING_RATE`` split by sqrt(size); NLP gets ``NLP_RATE``."""
    import numpy as np

    p, n = config.pointing_rate, config.nlp_rate
    datasets: List = [config.dataset.build(tokenizer)]  # caption
    weights: List[float] = [max(1.0 - p - n, 0.0)]

    if p > 0:
        pointing = [
            PixMoPointsDatasetConfig(
                kind="basic", max_crops=MAX_CROPS, token_ids=config.dataset.token_ids
            ).build(tokenizer),
            PixMoCountDatasetConfig(max_crops=MAX_CROPS, token_ids=config.dataset.token_ids).build(
                tokenizer
            ),
            PixMoPointsDatasetConfig(
                kind="high_frequency",
                max_crops=MAX_CROPS,
                token_ids=config.dataset.token_ids,
            ).build(tokenizer),
            CoSynPointDatasetConfig(max_crops=MAX_CROPS, token_ids=config.dataset.token_ids).build(
                tokenizer
            ),
        ]
        frac = np.sqrt(np.array([len(d) for d in pointing], dtype=np.float64))
        frac = frac / frac.sum()
        datasets += pointing
        weights += [p * float(f) for f in frac]

    if n > 0:
        datasets.append(
            Tulu4DatasetConfig(
                max_sequence_length=config.dataset.max_sequence_length,
                token_ids=config.dataset.token_ids,
            ).build(tokenizer)
        )
        weights.append(n)

    log.info(
        "Mixture sources / weights: %s",
        [(type(d).__name__, round(w, 3)) for d, w in zip(datasets, weights)],
    )
    return datasets, weights


def train(config: ExperimentConfig):
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

    from olmo_core.nn.vision import load_molmo2_hf_vision_state_dict

    log.info(
        "Loading Molmo2 vision weights from %s at revision %s",
        config.vision_model_id,
        config.vision_revision,
    )
    vision_state = load_molmo2_hf_vision_state_dict(
        config.vision_model_id,
        revision=config.vision_revision,
        cache_dir=config.hf_cache_dir,
    )
    train_module.load_molmo2_vision_state_dict(vision_state)
    del vision_state

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

    if config.pointing_rate > 0 or config.nlp_rate > 0:
        datasets, weights = _build_mixture_sources(tokenizer, config)
        data_loader = MixtureDataLoader(
            datasets,
            weights,
            collator,
            work_dir=config.trainer.save_folder,
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            pack=config.pack_sequences,
            prefetch_workers=DATA_PREFETCH_WORKERS,
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

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

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
› python {sys.argv[0]} dry_run molmo2-stage1

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

    config = build_config(script, run_name, overrides)
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
