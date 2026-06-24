from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkInstanceSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# Contradiction (n=20) task SFT for Qwen3-0.6B + standard FAST LANDMARK attention.
#
# A small-scale sibling of Qwen3-4B-fast-landmark-longctx-SFT.py: same fast_landmark mixer and
# composable data pipeline, but the 0.6B model, a single short task (contradiction, n=20), and a
# much shorter sequence length. n=20 contradiction instances are short (max ~1.6k tokens; p99
# ~1.3k), so SEQUENCE_LENGTH=2048 keeps 100% of instances with little padding -- no context
# parallelism or activation checkpointing is needed (cf. the 4B 64k run which used CP=8 + AC).
#
# Weights are initialized from the RAW Qwen3-0.6B olmo-core base checkpoint (NOT a landmark
# pretrain), so the landmark token (id 151860) embedding is untrained -- a quick exploratory run
# to see how well the small model learns the task with fast landmark attention.
#
# Data pipeline (composable, no packing):
#   PadToLengthInstanceSource(token_ids + labels_mask)   # one example/instance, padded to CONTENT len
#     -> LandmarkInstanceSource(mem_freq, mem_id)         # insert a landmark token every MEM_FREQ
#
# The token_ids_part_*.npy / labels_mask_*.npy shards are produced by
# src/scripts/data/convert_longctx_tasks_to_sft.py on contradiction_train_pubmed_both_n20_k3.jsonl
# (Qwen3 chat template, query_position=both, plain answers / --cot-mode none).
# ---------------------------------------------------------------------------

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 2048  # final instance length (must be divisible by BLOCK_SIZE)
# Content length each example is padded to (landmark tokens are inserted afterwards):
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 2016

LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

# Tokenized SFT dataset (Qwen3 tokenizer): token_ids_part_*.npy + labels_mask_*.npy, produced by
# src/scripts/data/convert_longctx_tasks_to_sft.py on the n=20 contradiction train jsonl.
DATA_PATH = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/longctx_sft_qwen/contradiction_n20"
)

# Raw Qwen3-0.6B olmo-core base checkpoint to initialize from (model weights only).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3-0.6B-olmocore/model_and_optim"
)

NUM_NODES = 1  # 1 node x 8 GPUs; no CP -> DP-only over the (tiny) 0.6B model
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8  # 8 instances/step
LR = 5e-5
NUM_EPOCHS = 3


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/prasanns/{cli_context.run_name}"

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
    # The SFT shards are separated by SINGLE EOS tokens (see convert_longctx_tasks_to_sft.py).
    # qwen3 sets bos_token_id == eos_token_id, so drop BOS for document-boundary detection.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3-0.6B with the standard FAST landmark attention mixer (AttentionType.fast_landmark).
    model_config = TransformerConfig.qwen3_0_6B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_landmark=True,
        mem_freq=MEM_FREQ,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one instance per rank per micro-step
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
        # No CP: at SEQUENCE_LENGTH=2048 the 0.6B model + full local sequence fit comfortably, and
        # the fast-landmark grouped softmax runs over the whole sequence on each rank. DP-only:
        # shard_degree=1 -> pure replicas (DDP-like) across the 8 GPUs.
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,  # disabled for SFT (cf. OLMo SFT scripts)
        max_grad_norm=1.0,
    )

    # Composable SFT data pipeline (no packing -- landmark attention can't intra-document mask):
    #   PadToLengthInstanceSource (one EOS-terminated example/instance, padded to CONTENT len)
    #     -> LandmarkInstanceSource (insert landmark token every MEM_FREQ tokens -> SEQUENCE_LENGTH)
    instance_source_config = LandmarkInstanceSourceConfig(
        source=PadToLengthInstanceSourceConfig.from_npy(
            f"{DATA_PATH}/token_ids_part_*.npy",
            tokenizer=doc_tokenizer_config,
            sequence_length=CONTENT_SEQUENCE_LENGTH,
            label_mask_paths=[f"{DATA_PATH}/labels_mask_*.npy"],
            expand_glob=True,
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
        # generate_doc_lengths left False: landmark attention does not support intra-document masking.
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            # Initialize from the raw Qwen3-0.6B base checkpoint, weights only. The trainer first
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
                ephemeral_save_interval=250,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="prasanns-allen-institute-for-ai",
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
    Contradiction (n=20) task SFT of Qwen3-0.6B + standard fast-landmark attention, initialized
    from the raw Qwen3-0.6B olmo-core base checkpoint. Example::

        python src/scripts/train/sft/Qwen3-0.6B-fast-landmark-contradiction-n20-SFT.py \\
            dry_run q06b-fast-contra-n20-sft ai2/jupiter-cirrascale-2
        python src/scripts/train/sft/Qwen3-0.6B-fast-landmark-contradiction-n20-SFT.py \\
            launch q06b-fast-contra-n20-sft ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
