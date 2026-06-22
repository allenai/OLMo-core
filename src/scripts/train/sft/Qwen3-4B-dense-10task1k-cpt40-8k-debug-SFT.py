from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# DEBUG run: small/fast 8k version of Qwen3-4B-dense-10task1k-cpt40-64k-SFT.py.
#
# Same data pipeline as the 64k long run -- the 10-task / 1000-per-task SFT mix + 40% CPT, mixed
# then ConcatAndChunk -- but tokenized at an 8k cap and trained on 1 node with no context
# parallelism, full activation checkpointing (no compile), and a tiny step budget. The point is to
# shake out the data path (60/40 mixing, completion vs full-loss masks, 8k chunking, doc-boundary
# masking) cheaply BEFORE committing 2 nodes at 64k.
#
# At 8k (< Qwen3's native 32k) no YaRN is needed. Every SFT instance is <= 8192 tokens
# (converter --max-seq-len 8192), so no packed window is all-prompt -> no loss-free-microbatch NaN.
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 8192  # short, fast packed window (well within Qwen3's native 32k -> no YaRN)

# 10-task / 1000-per-task SFT shards, tokenized at an 8k cap.
SFT_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/suite_it_sft_qwen/combined_10task_1000_8k"
# Qwen3-tokenized dolma3longmino CPT sample (part-*.npy) + all-True masks (mask-*.npy).
CPT_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/dolma3_longmino_qwen3_sample"
# Token fraction of CPT in the mix (matches the 64k long run).
CPT_FRAC = 0.40

# Dense CPT checkpoint to initialize from (same base as the full dense unified SFT).
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-dense-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 1

# Global batch in *tokens*. 1 node = 8 GPUs, no CP (8 DP replicas), rank_microbatch=SEQUENCE_LENGTH
# -> SEQUENCE_LENGTH * 8 ~ 65k tokens/step.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8

LR = 5e-5
# Debug: just enough steps to exercise data loading, mixing, masking, and a few optim steps.
NUM_STEPS = 40


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
    # SFT shards are separated by single EOS tokens; qwen3 sets bos==eos, so drop bos for doc splitting.
    from dataclasses import replace

    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # No YaRN: 8k is well within Qwen3's native 32k, so native rope is correct.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=False,  # debug: skip compile startup cost
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            # Full FSDP across the node's 8 GPUs (a 4B model + AdamW state OOMs replicated on 1x80GB).
            shard_degree=8,
        ),
        # Full activation checkpointing (every block) -- keeps 8k activations small without compile
        # (the 'budget' AC mode requires compile, which we disable here for fast startup).
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # 60/40 token-fraction mix of SFT (completion-masked) + CPT (full-loss) documents, then
    # ConcatAndChunk into 8k windows -- same pipeline as the 64k long run.
    sft = SFT_DATA_ROOT.rstrip("/")
    cpt = CPT_DATA_ROOT.rstrip("/")
    sft_doc_source = NumpyDocumentSourceConfig(
        source_paths=[f"{sft}/token_ids_part_*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{sft}/labels_mask_*.npy"],
        expand_glob=True,
    )
    cpt_doc_source = NumpyDocumentSourceConfig(
        source_paths=[f"{cpt}/part-*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{cpt}/mask-*.npy"],  # explicit all-True => full-sequence CPT loss
        expand_glob=True,
    )
    mixed_source = MixingDocumentSourceConfig(
        source_specs=[
            MixingDocumentSourceSpecConfig(
                source=sft_doc_source, ratio=1.0 - CPT_FRAC, max_repetition_factor=8.0, label="sft_10task"
            ),
            MixingDocumentSourceSpecConfig(
                source=cpt_doc_source, ratio=CPT_FRAC, max_repetition_factor=8.0, label="cpt_longmino"
            ),
        ],
    )
    instance_source_config = ConcatAndChunkInstanceSourceConfig(
        sources=[mixed_source],
        sequence_length=SEQUENCE_LENGTH,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        generate_doc_lengths=True,  # block-diagonal masking at EOS doc boundaries (dense flash varlen)
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(NUM_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=40,
                ephemeral_save_interval=20,
                max_checkpoints=2,
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
                cancel_check_interval=5,
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
    Fast 8k 1-node debug of the 10-task / 1000-per-task + 40% CPT SFT data pipeline.

        python src/scripts/train/sft/Qwen3-4B-dense-10task1k-cpt40-8k-debug-SFT.py \\
            launch q4b-dense-10task1k-cpt40-8k-debug ai2/jupiter
    """
    main(config_builder=build_experiment_config)
