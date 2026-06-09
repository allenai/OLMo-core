from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    LandmarkInstanceSourceConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName, AttentionType
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# Qwen3.5-4B (hybrid Gated DeltaNet + full attention, 3:1) with LANDMARK attention (fused kernel)
# replacing the full-attention layers, at 64k on the 15B-token dolma3_longmino sample.
#
# Gated attention is PRESERVED: the full-attention block keeps its elementwise output gate
# (gate=GateConfig(elementwise) from qwen3_5_4B). LandmarkAttention now supports the gate, so the
# attention output is gated (att * sigmoid(w_g(x))) exactly as in the gated Qwen3.5 base, and the
# gate weights (attention.w_g) load straight from the converted Qwen3.5 checkpoint. The GDN
# (linear-attention) layers are unchanged.
#
# Notes:
#   * qwen3_5_4B has no `landmark=` option, so we swap the "attn" block's mixer to landmark below.
#   * The data pipeline inserts landmark tokens every MEM_FREQ tokens. The landmark-attention layers
#     use them (positional is_mem); the GDN layers see them as ordinary tokens (label_mask still
#     excludes them from the loss).
#   * No Ulysses CP (incompatible with the GDN recurrence): each rank processes the full 64k.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 65536  # 64k (must be divisible by BLOCK_SIZE)
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 64512

LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

DATA_DIR = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen"
)

GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
MAX_TOKENS = 10_000_000_000  # 10B
LR = 3.2e-4

# The Qwen3.5-4B olmo-core base checkpoint (gated full attention). Override with --trainer.load_path=
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
        num_nodes=4,  # 4 nodes × 8 GPUs = 32 GPUs
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3.5-4B hybrid, then swap the full-attention layers to landmark attention.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
    attn_mixer.name = AttentionType.landmark
    attn_mixer.mem_freq = MEM_FREQ
    attn_mixer.landmark_use_kernel = True
    # NB: keep attn_mixer.gate (the elementwise gate from qwen3_5_4B) -- landmark attention now
    # applies it, so gated-attention functionality is preserved and w_g loads from the checkpoint.

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
        scheduler=LinearWithWarmup(warmup=400, alpha_f=0.0),
        # GatedDeltaNet layers use custom kernels; keep compile off.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        # No Ulysses CP: incompatible with the GatedDeltaNet recurrence; each rank handles full 64k.
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Composable data pipeline on the new dolma3_longmino sample:
    #   NumpyDocumentSource (part-*.npy, Qwen3 uint32, EOS-separated)
    #     -> ConcatAndChunkInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=64512)
    #     -> LandmarkInstanceSource (insert landmark token every MEM_FREQ tokens -> seq_len=65536)
    instance_source_config = LandmarkInstanceSourceConfig(
        source=ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=[f"{DATA_DIR}/part-*.npy"],
                    tokenizer=tokenizer_config,
                    expand_glob=True,
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
    Qwen3.5-4B (hybrid) with LANDMARK attention (fused kernel) on the full-attn layers (gate
    preserved), at 64k on the 15B dolma3_longmino sample (4 nodes, urgent). dry_run must be run on a
    node (GatedDeltaNet needs `fla`, and the fused landmark kernel needs Triton + CUDA).

        python src/scripts/train/Qwen3/Qwen3.5-4B-landmark-dolma3longmino.py \\
            launch my-run ai2/jupiter-cirrascale-2 --trainer.load_path=/weka/.../model_and_optim
    """
    main(config_builder=build_experiment_config)
