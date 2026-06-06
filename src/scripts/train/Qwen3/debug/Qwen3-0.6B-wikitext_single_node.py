from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
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

# Plain Qwen3-0.6B (dense attention, no landmark tokens). Matches the landmark variant's
# 4096-position context so the two runs are directly comparable on WikiText.
SEQUENCE_LENGTH = 4096

# Raw uint32 token array uploaded to weka. 557K tokens of WikiText tokenized with the Qwen3
# tokenizer. The same source feeds Qwen3-0.6B-landmark-wikitext_single_node.py.
WIKITEXT_PATH = "/weka/oe-training-default/ai2-llm/checkpoints/amandab/npys/wikitext_tokens.npy"

GLOBAL_BATCH_SIZE = 32768  # ~32K tokens
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

    tokenizer_config = TokenizerConfig.qwen3()

    # Qwen3-0.6B with standard dense attention.
    model_config = TransformerConfig.qwen3_0_6B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
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

    # Single local WikiText source. Use the plain concat-and-chunk FSL dataset (NOT the packed
    # variant): the packed dataset does per-document bin-packing and, without doc-length metadata,
    # treats the whole file as one document and truncates to a single sequence. Concat-and-chunk
    # streams the tokens and slices them into fixed-length sequences, matching the landmark
    # pipeline's ConcatAndChunkInstanceSource.
    dataset_config = NumpyFSLDatasetConfig(
        paths=[WIKITEXT_PATH],
        tokenizer=tokenizer_config,
        dtype=NumpyDatasetDType.uint32,
        work_dir=work_dir,
        sequence_length=SEQUENCE_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
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
        dataset=dataset_config,
        data_loader=data_loader_config,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Plain Qwen3-0.6B (dense attention, no landmark) trained on WikiText.

    The non-landmark counterpart to Qwen3-0.6B-landmark-wikitext_single_node.py: same model size,
    sequence length (4096), batch size, optimizer, and starting checkpoint, but with standard
    attention and a packed FSL dataset instead of the landmark composable pipeline. The data is a
    single local raw-uint32 token array (wikitext_tokens.npy, ~557K Qwen3 tokens) on weka, so the
    loader cycles the data many times over the 10B-token schedule (small overfit / sanity setup).

    Training starts from the pre-trained Qwen3-0.6B checkpoint.

    Examples:
        Render the config and exit:
            python src/scripts/train/Qwen3/Qwen3-0.6B-wikitext_single_node.py dry_run my-run ai2/jupiter-cirrascale-2

        Launch on Beaker (single node):
            python src/scripts/train/Qwen3/Qwen3-0.6B-wikitext_single_node.py launch my-run ai2/jupiter-cirrascale-2

        Override LR:
            python src/scripts/train/Qwen3/Qwen3-0.6B-wikitext_single_node.py launch my-run ai2/jupiter-cirrascale-2 \\
                --train_module.optim.lr=1e-4
    """
    main(config_builder=build_experiment_config)
