import os
import sys
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import ComposableDataLoaderConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))

from longmino_512k_mix import build_longmino_512k_mix  # noqa: E402

# Qwen3.5-4B DENSE baseline (hybrid Gated DeltaNet + full attention, 3:1) at 512k context on the
# custom longmino-512k mix, 10B tokens, 8 H100 nodes. Paired with
# Qwen3.5-4B-fast-compressive-landmark-longmino512k.py.
#
# Parallelism is almost fully determined by the architecture, not by choice:
#   * TP=1     -- GatedDeltaNet.apply_tp raises NotImplementedError.
#   * Ulysses  -- ring/zigzag CP is rejected by both GatedDeltaNet and landmark attention.
#   * CP<=4    -- Ulysses requires cp_degree to divide n_kv_heads, and qwen3_5_4B has n_kv_heads=4.
#                 (The Qwen3-4B scripts use cp_degree=8 because that model has n_kv_heads=8; that
#                 value raises here, which is most likely why the earlier Qwen3.5 scripts concluded
#                 "no CP available" and fell back to shard_degree=8 at 64k. CP=4 is supported and
#                 covered by test_context_parallel_gdn_ulysses.)
# So: cp=4, tp=1, dp=16 on 64 GPUs.
SEQUENCE_LENGTH = 524288  # 512k
CP_DEGREE = 4  # max allowed: n_kv_heads=4
NUM_NODES = 8  # 8 x 8 = 64 GPUs -> DP = 64 / 4 = 16

# 1 sequence per DP rank per micro-step; DP=16 -> the smallest global batch reachable at 512k.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 16  # ~8.4M tokens
MAX_TOKENS = 10_000_000_000  # 10B -> ~1192 steps
LR = 3.2e-4  # NB: carried over from the 64k/4M-batch runs; batch here is ~2x, so revisit.

DATA_ROOT = "/weka/oe-training-default/amandab/longmino_512k"

# The Qwen3.5-4B olmo-core base checkpoint. Override with --trainer.load_path=
CHECKPOINT_PATH = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3.5-4B-olmocore/model_and_optim"
)


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

    tokenizer_config = TokenizerConfig.qwen3_5()

    # flash_3 (Hopper FA3) rather than flash_2. At 512k, attention is ~81% of total FLOPs for this
    # model even though only 8 of 32 layers are full attention, so the attention kernel dominates
    # wall-clock and FA3 is the single largest speed lever available on H100. It is marked beta in
    # AttentionBackendName -- if it misbehaves, fall back with --model.block.attn.sequence_mixer.backend=flash_2
    # (costs roughly a third of throughput). Note the landmark variant cannot use this at all: it
    # never touches self.backend.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )

    # Mandatory at 512k, not merely an optimization: with CP=4 each rank holds 131,072 tokens, and
    # dense logits would be 131072 x 248320 x 2B = ~65GB in bf16 alone (double that once
    # cross-entropy upcasts to fp32, plus an equal-sized grad). Liger's fused linear cross-entropy
    # never materializes them.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one full sequence per DP rank, split across CP
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        # ~1192 steps at this batch size (vs ~2500 at 64k/4M), so warmup is scaled down to keep it
        # at a comparable fraction of the run.
        scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
        # GatedDeltaNet layers use custom kernels; compile stays off. This also rules out 'budget'
        # activation checkpointing, which only sets torch._functorch.config.activation_memory_budget
        # and is a no-op without compile.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            # Shard params/grads/optim across all 16 DP ranks (~64GB of fp32 master + Adam state ->
            # ~4GB/rank). Unlike the 64k scripts there is no replication to trade away here.
            shard_degree=16,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        # Full AC. Estimated peak ~45GB of 80: 21.5GB of block-input checkpoints (32 blocks x 131072
        # tokens x 2560 x 2B) + ~13.5GB for the one block being recomputed + ~4GB optimizer shard +
        # LM head and comm buffers. The remaining headroom cannot be spent usefully: un-checkpointing
        # even one block costs ~13GB (a GDN block's activation set is ~99KB/token at this width), and
        # the finer-grained 'budget' mode needs compile.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Per-stratum longmino-512k mix (qwen35 tree), composed by ratio. No landmark insertion.
    instance_source_config = build_longmino_512k_mix(
        tokenizer=tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        tree="qwen35",
        root=DATA_ROOT,
        seed=1234,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=CHECKPOINT_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=250,  # ~1192 steps total, so 1000 would checkpoint once
                ephemeral_save_interval=None,
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
    Qwen3.5-4B hybrid dense baseline at 512k on the longmino-512k mix, 10B tokens, 8 nodes.

        python src/scripts/train/Qwen3/Qwen3.5-4B-dense-longmino512k.py \\
            dry_run test-512k-dense ai2/jupiter-cirrascale-2

        python src/scripts/train/Qwen3/Qwen3.5-4B-dense-longmino512k.py \\
            launch q35-4b-dense-512k ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
