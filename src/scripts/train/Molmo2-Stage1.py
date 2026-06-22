"""
Molmo2 "stage 1" caption-pretraining (reproduction of ``mm_olmo``'s captioner).

Trains the connector + LM on PixMoCap captions/transcripts with the vision encoder
**frozen**, using the float ``root_subsegments``-weighted loss and per-component
learning rates / warmups. In-loop evaluation is intentionally omitted.

Run without arguments for usage. Quick local smoke test on synthetic data::

    torchrun --nproc-per-node=1 src/scripts/train/Molmo2-Stage1.py train smoke \\
        --dataset.dataset_path=synthetic --trainer.max_duration.value=5 \\
        --trainer.max_duration.unit=steps

.. note::
    Single-GPU, DDP, and FSDP/HSDP data parallelism are supported. TP/CP/PP/EP of the
    multimodal model are out of scope.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import timedelta
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
    get_beaker_username,
    get_root_dir,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.vision import MultimodalLM, MultimodalLMConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride, PerGroupScheduler
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
    MultimodalTransformerTrainModuleConfig,
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

MODEL_ID = "allenai/Molmo2-4B"  # HF checkpoint to initialise from (also provides the tokenizer)
SEQUENCE_LENGTH = 4096  # fixed pad length; mm_olmo stage 1 uses ~5248
MAX_CROPS = 8

# Instance-based batching (mm_olmo: global 8, device microbatch 1), expressed in tokens.
GLOBAL_BATCH_INSTANCES = 8
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
NUM_NODES = 1
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
    train_module: MultimodalTransformerTrainModuleConfig
    trainer: TrainerConfig
    model_id: str = MODEL_ID
    data_seed: int = 34521
    init_seed: int = 12536


def _build_model_config() -> MultimodalLMConfig:
    """Build the :class:`MultimodalLMConfig` from the HF Molmo2 config (no weights)."""
    from transformers import AutoConfig

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_config_from_hf_config,
    )

    ensure_default_rope_registered()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    return molmo2_config_from_hf_config(hf_config)


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    model_config = _build_model_config()

    dataset_config = PixMoCapDatasetConfig(
        dataset_path=DATASET_PATH,
        mode="transcript_and_caption",
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="root_subsegments",
        seed=34521,
    )

    # Pad token: Molmo2/Qwen2.5 EOS (151643). Fixed-length padding so every batch has a
    # constant token count for the token-based Trainer.
    collator_config = MultimodalCollatorConfig(
        pad_token_id=151643,
        label_ignore_index=-100,
        pad_sequence_length=SEQUENCE_LENGTH,
    )

    train_module_config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LLM_LR,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["connector.*"],
                    opts=dict(lr=CONNECTOR_LR, weight_decay=0.0, scheduler_name="connector"),
                ),
            ],
        ),
        freeze_params=["vision.*"],
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        autocast_precision=DType.bfloat16,
        scheduler=PerGroupScheduler(
            schedulers={"connector": CosWithWarmup(warmup=CONNECTOR_WARMUP, alpha_f=ALPHA_F)},
            default=CosWithWarmup(warmup=LLM_WARMUP, alpha_f=ALPHA_F),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
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

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
    ).merge(overrides)


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def _init_weights_from_hf(model: MultimodalLM, model_cfg: MultimodalLMConfig) -> None:
    """Load converted HF Molmo2 weights into the (meta-initialised) model."""
    from transformers import AutoModelForImageTextToText

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_hf_state_dict_to_multimodal_lm,
        reinit_rope_buffers,
    )

    ensure_default_rope_registered()
    log.info(f"Loading HF weights from {MODEL_ID} ...")
    hf = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True)
    reinit_rope_buffers(hf)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), model_cfg)
    del hf
    model.to_empty(device=get_default_device())
    model.load_state_dict(converted, strict=False)
    del converted


def _build_mixture_sources(tokenizer, config: ExperimentConfig):
    """Build the caption + pointing + NLP sources and their sampling weights (mm_olmo
    SubMixture): caption gets ``1 - POINTING_RATE - NLP_RATE``; the pointing group shares
    ``POINTING_RATE`` split by sqrt(size); NLP gets ``NLP_RATE``."""
    import numpy as np

    p, n = POINTING_RATE, NLP_RATE
    datasets: List = [config.dataset.build(tokenizer)]  # caption
    weights: List[float] = [max(1.0 - p - n, 0.0)]

    if p > 0:
        pointing = [
            PixMoPointsDatasetConfig(kind="basic", max_crops=MAX_CROPS).build(tokenizer),
            PixMoCountDatasetConfig(max_crops=MAX_CROPS).build(tokenizer),
            PixMoPointsDatasetConfig(kind="high_frequency", max_crops=MAX_CROPS).build(tokenizer),
            CoSynPointDatasetConfig(max_crops=MAX_CROPS).build(tokenizer),
        ]
        frac = np.sqrt(np.array([len(d) for d in pointing], dtype=np.float64))
        frac = frac / frac.sum()
        datasets += pointing
        weights += [p * float(f) for f in frac]

    if n > 0:
        datasets.append(Tulu4DatasetConfig().build(tokenizer))
        weights.append(n)

    log.info(
        "Mixture sources / weights: %s",
        [(type(d).__name__, round(w, 3)) for d, w in zip(datasets, weights)],
    )
    return datasets, weights


def train(config: ExperimentConfig):
    seed_all(config.init_seed)

    tokenizer = _load_tokenizer()

    model = config.model.build(init_device="meta")
    _init_weights_from_hf(model, config.model)

    train_module = config.train_module.build(model)

    collator = config.collator.build()
    # Derive the data-parallel world size / rank from the train module's DP process
    # group so each rank reads its own shard (must match the trainer's DP degree).
    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)

    if POINTING_RATE > 0 or NLP_RATE > 0:
        datasets, weights = _build_mixture_sources(tokenizer, config)
        data_loader = MixtureDataLoader(
            datasets,
            weights,
            collator,
            work_dir=config.trainer.save_folder,
            global_batch_size=GLOBAL_BATCH_SIZE,
            seed=config.data_seed,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
    else:
        data_loader = MultimodalDataLoader(
            config.dataset.build(tokenizer),
            collator,
            work_dir=config.trainer.save_folder,
            global_batch_size=GLOBAL_BATCH_SIZE,
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
› torchrun --nproc-per-node=1 {sys.argv[0]} train smoke \\
      --dataset.dataset_path=synthetic --trainer.max_duration.value=5

Launch on Beaker:
› python {sys.argv[0]} launch molmo2-stage1 --launch.num_nodes=1
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
