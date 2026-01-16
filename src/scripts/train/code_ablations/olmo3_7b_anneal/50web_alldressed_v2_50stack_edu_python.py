from datetime import datetime
import sys
from pathlib import Path
from typing import Optional

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureDatasetConfig,
    SourceMixtureList,
    SourceMixtureConfig,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main as olmo_core_main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig


# Change these to match the config you want to use
SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
MAX_TOKENS = 10_000_000_000  # 10B
LR = 0.00020712352850360292
SEED = 1337
TOKENIZER_CONFIG = TokenizerConfig.dolma2()
PRIORITY = "high"
WORKSPACE = "ai2/olmo4"
BUDGET = "ai2/oe-base"
NUM_NODES = 4

MODEL_CONFIG = TransformerConfig.olmo3_7B(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())

DATASET_CONFIG = SourceMixtureList(
    sources=[
        SourceMixtureConfig(
            source_name="all-dressed-snazzy2-v0.1-150b",
            target_ratio=0.5,
            paths=[
                f"s3://ai2-llm/preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER_CONFIG.identifier}/all-dressed-snazzy2/*/*.npy"
            ],
        ),
        SourceMixtureConfig(
            source_name="stack-edu-python-v0.1-150b",
            target_ratio=0.5,
            paths=[
                f"s3://ai2-llm/preprocessed/stack-edu/{TOKENIZER_CONFIG.identifier}/Python/*.npy"
            ],
        ),
    ]
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
        workspace=WORKSPACE,
        num_nodes=NUM_NODES,
        nccl_debug=True,
    )
    beaker_launch_config.priority = PRIORITY

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH * 2,
        learning_rate=LR,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=0, alpha_f=0.0),
        activation_memory_budget=0.5,
    )

    DATASET_CONFIG.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=DATASET_CONFIG,
            requested_tokens=MAX_TOKENS,
            global_batch_size=GLOBAL_BATCH_SIZE,
            processes=16,
            seed=SEED,
        ),
        tokenizer=TOKENIZER_CONFIG,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path="gs://ai2-llm/checkpoints/OLMo25/step1413814",
        load_trainer_state=False,
        load_optim_state=True,
        max_duration=Duration.tokens(MAX_TOKENS),
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    ).with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts, wandb_group_name=cli_context.run_name
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=MODEL_CONFIG,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


def main():
    # TWO CASES:
    # 1. len(sys.argv) < 4: definitely i have to add the run name here
    # 2. len(sys.argv) >= 4: depends on whether the first 4 all start NOT
    #                        with --: if they do, then i already have the
    #                        run name. but if any of the first 4 start
    #                        with --, then it means that some are already
    #                        overrides, so i need to add the run name in.
    if len(sys.argv) < 4 or any(arg.startswith("--") for arg in sys.argv[:4]):
        sys.argv.insert(2, Path(__file__).name)

    # now i can just call the main function
    return olmo_core_main(config_builder=build_experiment_config)


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:


        To launch a training run on Augusta w/ 8 nodes:
            python src/scripts/train/OLMo3/OLMo3-7B-midtraining.py launch my_run ai2/augusta \
                --launch.num_nodes=8 \
                --launch.priority=high
    """
    main()
