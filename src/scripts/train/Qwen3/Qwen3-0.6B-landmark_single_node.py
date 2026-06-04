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

# Landmark attention block structure:
#   block_size = mem_freq + 1 = 64; SEQUENCE_LENGTH must be divisible by block_size.
#   LandmarkInstanceSource inserts one landmark token after every MEM_FREQ content tokens, so the
#   upstream source must produce sequences of CONTENT_SEQUENCE_LENGTH tokens.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 1024  # 64k model context (must be divisible by BLOCK_SIZE)
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 64512

# Qwen3 reserved token used as the landmark (memory) token.
# 151860 is in the reserved range [151644, 151935] and does not appear in normal text.
LANDMARK_TOKEN_ID = 151860

GLOBAL_BATCH_SIZE = 1024 #* 64  # ~4M tokens
MAX_TOKENS = 10_000_000_000  # 10B
# StepFun optimal LR (Li et al. 2025): 1.79 * n^-0.713 * d^0.307
# n ≈ 3.65B non-embedding params, d = 10B tokens → ~3.2e-4
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
        num_nodes=1,  # 4 nodes × 8 GPUs = 32 GPUs; cp_degree=8 → 4 DP replicas
        num_gpus=1,
    )

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-0.6B with landmark attention (block_size=64, fused Triton kernel).
    # No attn_backend: LandmarkAttention has its own forward and ignores flash backends.
    model_config = TransformerConfig.qwen3_0_6B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        landmark=True,
        mem_freq=MEM_FREQ,
        landmark_use_kernel=True,
    )

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
        scheduler=LinearWithWarmup(warmup=4, alpha_f=0.0),
        compile_model=True,
        # dp_config=TransformerDataParallelConfig(
        #     name=DataParallelType.hsdp,
        #     param_dtype=DType.bfloat16,
        #     reduce_dtype=DType.float32,
        #     wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        #     shard_degree=1,
        # ),
        # Ulysses CP: LandmarkAttention.forward performs the cp2hp/hp2cp all-to-all itself so that
        # each rank gathers the full sequence T (with n_heads/8 heads) before the grouped softmax,
        # which must see every preceding block's landmark. Ring/zigzag CP (which splits T) is
        # incompatible and is rejected by LandmarkAttention.apply_cp.
        # Qwen3-4B: n_heads=32, n_kv_heads=8 → 4 q-heads and 1 kv-head per CP rank (both divisible
        # by degree=8, as Ulysses requires).
        #cp_config=TransformerContextParallelConfig.ulysses(degree=8),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Composable data pipeline:
    #   NumpyDocumentSourceMix (longmino/qwen)
    #     → ConcatAndChunkInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=64512)
    #     → LandmarkInstanceSource (inserts landmark token every MEM_FREQ tokens → seq_len=65536)
    instance_source_config = LandmarkInstanceSourceConfig(
        source=ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer_config,
                    mix=DataMix.longmino_qwen,
                    mix_base_dir="s3://ai2-llm",
                    source_group_size=8,
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
    Qwen3-4B with landmark attention (Mohtashami & Jaggi, 2023) at 64k context.

    Landmark attention inserts a special token every MEM_FREQ=63 content tokens (block_size=64).
    The fused Triton kernel provides O(T)-memory attention, avoiding the quadratic cost of
    full attention at 64k. No context parallelism is needed; the landmark mechanism handles
    long-range dependencies via block-gated softmax.

    The data pipeline inserts LANDMARK_TOKEN_ID (Qwen3 reserved token 151860) as the landmark
    token via LandmarkInstanceSource. Upstream content sequences of 64512 tokens are expanded
    to 65536-token model inputs (1024 blocks × 64 tokens/block).

    Training starts from a pre-trained Qwen3-4B checkpoint (dense attention weights are
    reused; the landmark token embedding is learned from scratch).

    Examples:
        Render the config and exit:
            python src/scripts/train/Qwen3/Qwen3-4B-landmark.py dry_run my-run ai2/jupiter-cirrascale-2

        Launch on Beaker with 4 nodes:
            python src/scripts/train/Qwen3/Qwen3-4B-landmark.py launch my-run ai2/jupiter-cirrascale-2

        Override LR or landmark block size:
            python src/scripts/train/Qwen3/Qwen3-4B-landmark.py launch my-run ai2/jupiter-cirrascale-2 \\
                --train_module.optim.lr=1e-4

        Load from a specific checkpoint:
            python src/scripts/train/Qwen3/Qwen3-4B-landmark.py launch my-run ai2/jupiter-cirrascale-2 \\
                --trainer.load_path=gs://ai2-llm/checkpoints/.../stepXXXXX \\
                --trainer.load_strategy=if_available \\
                --trainer.load_trainer_state=false
    """
    main(config_builder=build_experiment_config)
