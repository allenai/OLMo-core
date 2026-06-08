from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkInstanceSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# SFT for Qwen3-4B + LANDMARK attention.
#
# This mirrors the Qwen3-4B landmark *pretraining* script but swaps in an SFT
# data pipeline and SFT hyperparameters, and initializes from a landmark
# *pretrained* checkpoint (so the landmark token embedding is already learned).
#
# Data pipeline (composable):
#   PackingInstanceSource(token_ids + labels_mask)   # bin-pack SFT conversations, carry the loss mask
#     -> LandmarkInstanceSource(mem_freq, mem_id)     # insert a landmark token every MEM_FREQ tokens
#
# The landmark source preserves the upstream SFT label_mask and additionally
# masks landmark positions out of the loss.
#
# NOTE ON INTRA-DOCUMENT MASKING:
#   Landmark attention does NOT support intra-document (cu_doc_lens) masking
#   (see LandmarkAttention.forward, which raises if cu_doc_lens is given). So we
#   deliberately leave ``generate_doc_lengths=False``: packed conversations in a
#   sequence attend across each other. This matches how the landmark model was
#   pretrained (ConcatAndChunk concatenates documents without masking), so it is
#   consistent with the pretraining regime, but it differs from the standard
#   (non-landmark) OLMo SFT path, which relies on generate_doc_lengths=True.
# ---------------------------------------------------------------------------

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 65536  # final instance length (must be divisible by BLOCK_SIZE)
# Content length fed to the packer (landmark tokens are inserted afterwards):
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 64512

LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

# --- FILL THESE IN -----------------------------------------------------------
# Path to the tokenized SFT dataset (Qwen3 tokenizer). Expected to contain:
#   token_ids_part_*.npy  and  labels_mask_*.npy
# produced by open-instruct's convert_sft_data_for_olmocore.py. The landmark
# token (LANDMARK_TOKEN_ID) must NOT already appear in the tokenized data.
DATASET_PATH = "/weka/oe-training-default/ai2-llm/PATH/TO/sft-dataset"  # TODO

# Landmark *pretrained* checkpoint to initialize from (point at the
# `model_and_optim` dir). Only model weights are loaded (load_trainer_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/PATH/TO/landmark-ckpt/model_and_optim"
)  # TODO
# -----------------------------------------------------------------------------

NUM_NODES = 4  # 4 nodes x 8 GPUs = 32 GPUs; cp_degree=8 -> 4 DP replicas

# Global batch in *tokens* (counting landmark tokens). With rank_microbatch =
# SEQUENCE_LENGTH and 4 DP replicas, the trainer derives grad-accum steps from
# this. SEQUENCE_LENGTH * 16 = 1048576 (~1M tokens) -> 4 grad-accum steps.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 16

# SFT hyperparameters (cf. the OLMo SFT scripts): no weight decay, low LR,
# linear decay to zero, a few epochs.
LR = 5e-5
NUM_EPOCHS = 3


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=NUM_NODES,
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-4B with landmark attention (block_size=64, fused Triton kernel). No YaRN.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        landmark=True,
        mem_freq=MEM_FREQ,
        landmark_use_kernel=True,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # 1 instance per rank with CP
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,  # NOTE: different from pretraining
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        # Ulysses CP: LandmarkAttention.forward does its own cp2hp/hp2cp all-to-all so each rank
        # gathers the full sequence (with n_heads/8 heads) before the grouped softmax. Qwen3-4B:
        # n_heads=32, n_kv_heads=8 -> 4 q-heads and 1 kv-head per CP rank (both divisible by degree=8).
        cp_config=TransformerContextParallelConfig.ulysses(degree=8),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,  # disabled for SFT (cf. OLMo SFT scripts)
        max_grad_norm=1.0,
    )

    # Composable SFT data pipeline:
    #   PackingInstanceSource (token_ids_part_*.npy + labels_mask_*.npy, packed to CONTENT length)
    #     -> LandmarkInstanceSource (insert landmark token every MEM_FREQ tokens -> SEQUENCE_LENGTH)
    clean_path = DATASET_PATH.rstrip("/")
    instance_source_config = LandmarkInstanceSourceConfig(
        source=PackingInstanceSourceConfig.from_npy(
            f"{clean_path}/token_ids_part_*.npy",
            tokenizer=tokenizer_config,
            sequence_length=CONTENT_SEQUENCE_LENGTH,
            label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
            expand_glob=True,
            long_doc_strategy=LongDocStrategy.truncate,  # truncate docs over CONTENT_SEQUENCE_LENGTH
        ),
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # generate_doc_lengths left False: landmark attention does not support
        # intra-document masking (see module docstring above).
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            # Initialize from the landmark pretrained checkpoint, weights only. The trainer first
            # tries to resume from save_folder; if nothing is there it falls back to load_path.
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(NUM_EPOCHS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(name=run_name_with_ts, enabled=False),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=[instance_source_config],
        data_loader=data_loader_config,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    SFT Qwen3-4B + landmark attention. Fill in DATASET_PATH and BASE_CHECKPOINT above first.

        python src/scripts/train/sft/Qwen3-4B-landmark-SFT.py \\
            launch my-sft-run ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
