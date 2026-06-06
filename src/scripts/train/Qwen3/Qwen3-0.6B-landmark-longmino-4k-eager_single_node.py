from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    LandmarkInstanceSourceConfig,
    NumpyDocumentSourceMixConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.transformer import (
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
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

# CONTROL EXPERIMENT (debugging landmark + longmino loss collapse) -- EAGER variant.
#
# Logic: landmark attention sees a strict SUBSET of what dense attention sees (full attention only
# within the current block; compressed history via landmarks), so a *correct* landmark model can
# never achieve LOWER next-token loss than dense. Yet landmark collapses to ~0 (below dense ~2) on
# longmino, which is only possible via an information leak. The EAGER grouped-softmax path is
# provably causal (verified), so this run uses it to localize the leak:
#   - This script is identical to the fused-kernel 4k control EXCEPT landmark_use_kernel=False.
#   - Compare eager@4k vs kernel@4k on the same longmino data:
#       * eager matches kernel (same "some ~0" degenerate-table windows) -> kernel does NOT leak at
#         4k; those ~0 instances are genuine degenerate data, and the 64k collapse needs the
#         direct kernel causality probe (eager can't reach 64k -- it is O(T^2) in memory).
#       * kernel mean-loss dips BELOW eager mean-loss -> the fused kernel leaks even at 4k.
# NOTE: eager is O(T^2); 4k is feasible (~1GB attn tensor), 64k is not (~275GB).
#
# Landmark attention block structure:
#   block_size = mem_freq + 1 = 64; SEQUENCE_LENGTH must be divisible by block_size.
#   LandmarkInstanceSource inserts one landmark token after every MEM_FREQ content tokens, so the
#   upstream source must produce sequences of CONTENT_SEQUENCE_LENGTH tokens.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 4096  # SHORT control context (must be divisible by BLOCK_SIZE)
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 4032

# Qwen3 reserved token used as the landmark (memory) token.
# 151860 is in the reserved range [151644, 151935] and does not appear in normal text.
LANDMARK_TOKEN_ID = 151860

GLOBAL_BATCH_SIZE = 32768  # 8 sequences/step at 4k
MAX_TOKENS = 10_000_000_000  # 10B (watch wandb; cancel once the trend is clear)
LR = 2e-5


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
        num_nodes=1,
        num_gpus=2,
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-0.6B with landmark attention (block_size=64), EAGER grouped-softmax path (no kernel).
    # No attn_backend: LandmarkAttention has its own forward and ignores flash backends.
    model_config = TransformerConfig.qwen3_0_6B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        landmark=True,
        mem_freq=MEM_FREQ,
        landmark_use_kernel=True,
    )
    # Use the eager (dense O(T^2) grouped-softmax) path instead of the fused Triton kernel.
    model_config.block.sequence_mixer.landmark_use_kernel = False

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # 1 sequence per rank per micro-step
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=10),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Composable data pipeline (identical to the long-context landmark run except SEQUENCE_LENGTH):
    #   NumpyDocumentSourceMix (longmino/qwen)
    #     -> ConcatAndChunkInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=4032)
    #     -> LandmarkInstanceSource (inserts landmark token every MEM_FREQ tokens -> seq_len=4096)
    instance_source_config = LandmarkInstanceSourceConfig(
        source=ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer_config,
                    mix=DataMix.longmino_qwen,
                    mix_base_dir="s3://ai2-llm",
                    source_group_size=1,
                )
            ],
            sequence_length=CONTENT_SEQUENCE_LENGTH,
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
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path="/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3-0.6B-olmocore/model_and_optim",
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
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
    Control: Qwen3-0.6B + landmark attention (EAGER grouped softmax, no kernel) on longmino-qwen at
    4k context. Pair with the fused-kernel 4k control to localize the landmark loss collapse: the
    eager path is provably causal, so if kernel mean-loss dips below eager's the fused kernel is
    leaking. (Eager is O(T^2) in memory, so 4k is the practical ceiling; the 64k regime where the
    kernel broadly collapses needs the direct kernel causality probe instead.)

    Examples:
        Render the config and exit:
            python src/scripts/train/Qwen3/Qwen3-0.6B-landmark-longmino-4k-eager_single_node.py \\
                dry_run my-run ai2/jupiter-cirrascale-2

        Launch on Beaker (urgent priority is set in the script):
            python src/scripts/train/Qwen3/Qwen3-0.6B-landmark-longmino-4k-eager_single_node.py \\
                launch my-run ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
