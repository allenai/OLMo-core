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
from olmo_core.nn.rope import YaRNRoPEScalingConfig
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
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# LONG RUN: dense Qwen3-4B SFT on the 10-task / 1000-per-task suite mix at a 64k packed window,
# with 40% continued-pretraining (dolma3longmino) mixed in to counter long-context (RULER)
# forgetting.
#
# This is the "real" version of the no-ruler CPT-mix probes: instead of 17 tasks x ~59 short
# examples capped at 4-5k, the SFT data is 10 tasks x 1000 examples tokenized at a 64k cap, so the
# full per-task length spread (oolong ctx1k->64k, narrativeqa L8k/L32k/L64k, etc.) actually survives
# into training. CPT_FRAC=0.30 recovered most of the RULER loss at 5k; this pushes the mix to 0.40
# AND trains at the lengths RULER tests.
#
# Mirrors Qwen3-4B-dense-unified-SFT.py (flash-attn 2 + YaRN factor 2 -> 64k, CP=8 ulysses, budget
# AC + compile, shard_degree=1) EXCEPT:
#   * 2 nodes (16 GPUs): CP=8 -> 2 DP replicas. GLOBAL_BATCH = SEQ*16 (~1.05M tok/step) keeps the
#     per-replica accumulation (8) identical to the 4-node headline recipe.
#   * Instance source is a 60/40 token-fraction MIX: SFT docs (completion-masked) + CPT docs
#     (all-True masks => full-sequence loss), mixed then ConcatAndChunk into 64k windows.
#   * Fixed step budget (the mix has no natural "epoch").
# Every SFT instance is <= 65536 tokens (converter --max-seq-len 65536), so no packed window is
# all-prompt -> no loss-free-microbatch NaN.
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 65536  # packed window; YaRN extends Qwen3's native 32k to this.

# 10-task / 1000-per-task SFT shards, tokenized at a 64k cap (build_combined_suite_jsonl.py
# --per-task-budget 1000 --cot plain over the 10 tasks + convert_unified_to_sft.py --max-seq-len 65536).
SFT_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/suite_it_sft_qwen/combined_10task_1000_64k"
# Qwen3-tokenized dolma3longmino CPT sample (part-*.npy) + all-True masks (mask-*.npy).
CPT_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/dolma3_longmino_qwen3_sample"
# Token fraction of CPT in the mix.
CPT_FRAC = 0.40

# Dense CPT checkpoint to initialize from (same base as the full dense unified SFT).
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-dense-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 2

# Global batch in *tokens*. 2 nodes = 16 GPUs, CP=8 -> 2 DP replicas, rank_microbatch=SEQUENCE_LENGTH
# -> SEQUENCE_LENGTH * 16 ~ 1.05M tokens/step (8 grad-accum steps/replica, matching the headline).
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 16

LR = 5e-5
# The 10x1000 64k-capped SFT set is ~9.8k instances (~130M tokens); with 40% CPT mixed in the
# packed stream is ~220M tokens (~3.3k 64k windows). At 16 windows/step that's ~210 steps/pass.
# Train a fixed 250 steps (~1.2 passes) -- a substantial long-context run, not a probe.
NUM_STEPS = 250


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

    # YaRN factor 2 extends native 32k -> 64k (required at this seq len), matching the headline recipe.
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
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
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=8),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # 60/40 token-fraction mix of SFT (completion-masked) + CPT (full-loss) documents, then
    # ConcatAndChunk into 64k windows. Each document carries its own mask so a window straddling an
    # SFT and a CPT doc is masked correctly (SFT: completion-only; CPT: all tokens).
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
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(NUM_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100,
                ephemeral_save_interval=50,
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
    Long-context (64k) dense SFT of Qwen3-4B on the 10-task / 1000-per-task suite mix + 40% CPT,
    2 nodes with CP=8.

        python src/scripts/train/sft/Qwen3-4B-dense-10task1k-cpt40-64k-SFT.py \\
            launch q4b-dense-10task1k-cpt40-64k ai2/jupiter --launch.num_nodes=2
    """
    main(config_builder=build_experiment_config)
