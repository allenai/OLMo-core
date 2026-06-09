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
from olmo_core.nn.transformer import TransformerConfig
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

# Qwen3-0.6B + LANDMARK attention on the freshly-tokenized 15B-token sample of
# allenai/dolma3_longmino_mix-100B-1125 (Qwen3 tokenizer, EOS-separated uint32 on weka).
# Paired with Qwen3-0.6B-dense-dolma3longmino_single_node.py (identical config, no landmark) so the
# two are a clean landmark-vs-dense comparison on the same clean data.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 4096  # must be divisible by BLOCK_SIZE; override for the long-context run
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 4032

# Qwen3 reserved token used as the landmark (memory) token.
LANDMARK_TOKEN_ID = 151860

# Tokenized sample produced by src/scripts/data/tokenize_dolma3_longmino_sample.py
DATA_DIR = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen"
)

GLOBAL_BATCH_SIZE = 32768  # 8 sequences/step at 4k
MAX_TOKENS = 10_000_000_000  # 10B
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

    # Qwen3-0.6B with landmark attention (block_size=64, fused Triton kernel).
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

    # Composable data pipeline on the new dolma3_longmino sample:
    #   NumpyDocumentSource (part-*.npy, Qwen3 uint32, EOS-separated)
    #     -> ConcatAndChunkInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=4032)
    #     -> LandmarkInstanceSource (inserts landmark token every MEM_FREQ tokens -> seq_len=4096)
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
    Qwen3-0.6B + landmark attention on the 15B-token dolma3_longmino sample (paired with the dense
    script of the same name). Examples::

        python src/scripts/train/Qwen3/Qwen3-0.6B-landmark-dolma3longmino_single_node.py \\
            dry_run my-run ai2/jupiter-cirrascale-2
        python src/scripts/train/Qwen3/Qwen3-0.6B-landmark-dolma3longmino_single_node.py \\
            launch my-run ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
