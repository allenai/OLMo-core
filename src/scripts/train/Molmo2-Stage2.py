"""Molmo2 stage-2 SFT for the s002 OLMo3 MoE language model.

Fine-tunes the connector, vision encoder, and MoE LM on mm_olmo's 32-source
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
from typing import List, Optional, Sequence, cast

from olmo_core.config import Config, DType
from olmo_core.data.multimodal import MixtureDataLoader, MultimodalCollatorConfig
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    VALIDATION_MIXTURES,
    build_image_only_v9_mixture,
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
    LoadStrategy,
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
SEQUENCE_LENGTH = 16384  # mm_olmo image-only-v9: --seq_len 16384
USE_FLEX_ATTN = True
PACK_SEQUENCES = True
COMPILE_MODEL = True
RESPONSE_LOGITS_ONLY = True
DATA_PREFETCH_WORKERS = 4
MAX_CROPS = 8
EP_DEGREE = 8
EST_TOKENS_PER_EXAMPLE = 1500  # packed 16k sequences; tune if batch counts look off

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
DEFAULT_LOAD_PATH = f"{EXPERIMENT_ROOT}/checkpoints/molmo2-stage1"

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
NUM_NODES = 2
BEAKER_WORKSPACE = "ai2/OLMo-core"
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
    mixture: str = "debug"
    """Mixture tier — see ``VALIDATION_MIXTURES`` in ``image_only_v9.py``."""
    pack_sequences: bool = PACK_SEQUENCES
    pack_max_crops: int = MAX_CROPS


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


def _full_image_only_v9_names(
    tokenizer,
    token_ids: Molmo2TokenIds,
    seed: int,
    max_sequence_length: int = SEQUENCE_LENGTH,
) -> List[str]:
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        IMAGE_ONLY_V9_SUBMIXTURES,
        build_image_only_v9_datasets,
        compute_flat_mixture_weights,
    )

    datasets_map = build_image_only_v9_datasets(
        tokenizer,
        seed,
        max_sequence_length=max_sequence_length,
        token_ids=token_ids,
    )
    lengths = {name: len(datasets_map[name]) for name in datasets_map.keys()}
    flat = compute_flat_mixture_weights(IMAGE_ONLY_V9_SUBMIXTURES, lengths)
    return [name for name, _ in flat]


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
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        vision_activation_checkpointing=True,
        connector_activation_checkpointing=True,
        response_logits_only=RESPONSE_LOGITS_ONLY,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=compile_model,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
                "vision": CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
            },
            default=CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
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
            # The launcher default enables verbose graph-break/recompile diagnostics. They are
            # useful when debugging compilation but overwhelm normal multi-rank training logs.
            "TORCH_LOGS": "-dynamo",
        }
    )
    launch_config.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
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

    return ExperimentConfig(
        model=model_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
        token_ids=token_ids,
    ).merge(overrides)


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
    datasets, weights = build_image_only_v9_mixture(
        tokenizer,
        seed=config.data_seed,
        dataset_names=names_filter,
        max_sequence_length=max_sequence_length,
        token_ids=config.token_ids,
    )
    names = (
        _full_image_only_v9_names(
            tokenizer,
            config.token_ids,
            config.data_seed,
            max_sequence_length=max_sequence_length,
        )
        if names_filter is None
        else list(names_filter)
    )
    log.info(
        "Mixture %s sources / weights: %s",
        config.mixture,
        list(zip(names, [round(w, 4) for w in weights])),
    )
    return datasets, weights, names


def train(config: ExperimentConfig):
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    seed_all(config.init_seed)

    tokenizer, token_ids = _load_tokenizer(config.tokenizer_id, config.hf_cache_dir)
    if token_ids != config.token_ids:
        raise ValueError(
            "Tokenizer image-token IDs do not match the serialized data/model config: "
            f"{token_ids} != {config.token_ids}"
        )
    if not config.trainer.load_path:
        raise ValueError("Stage 2 requires trainer.load_path pointing to a stage-1 checkpoint")

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    collator = config.collator.build()

    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)

    datasets, weights, dataset_names = _build_mixture(tokenizer, config)
    log.info(
        "Stage 2 packing: pack=%s pack_max_crops=%d vit_crop_microbatch=%s",
        config.pack_sequences,
        config.pack_max_crops,
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
        est_tokens_per_example=EST_TOKENS_PER_EXAMPLE,
        prefetch_workers=DATA_PREFETCH_WORKERS,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        dataset_names=dataset_names,
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
› python {sys.argv[0]} dry_run molmo2-stage2-debug

Local 8-GPU debug smoke (use a completed stage-1 run):
› torchrun --nproc-per-node=8 {sys.argv[0]} train smoke \\
      --trainer.load_path=/path/to/stage1/run --trainer.max_duration.value=1 \\
      --mixture=debug --train_module.compile_model=false

Full image-only-v9 mixture:
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run \\
      --trainer.load_path=/path/to/stage1/run --mixture=image-only-v9

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
