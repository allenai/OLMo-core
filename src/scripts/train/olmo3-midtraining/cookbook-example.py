from datetime import datetime
from typing import Optional

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import WSD, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 4096 * 8192
SEED = 12536


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{run_name_with_ts}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        root_dir=root_dir,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        beaker_image="petew/olmo-core-tch270cu128",
        workspace="ai2/oe-data",
        num_nodes=1,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo2_30M(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    max_duration = Duration.chinchilla_tokens(
        2.0, model_params=model_config.num_active_non_embedding_params
    )
    # max_duration=Duration.tokens(500000)
    global_batch_size = (
        cookbook.estimate_critical_batch_size(duration=max_duration, sequence_length=SEQ_LENGTH)
        * SEQ_LENGTH
    )

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH * 4,
        learning_rate=4.4e-5 * (4**0.5),
        scheduler=WSD(units=SchedulerUnits.steps, warmup=2000),
    )

    source_mix_config = SourceMixtureDatasetConfig(
        requested_tokens=100_000_000_000,  # 1B
        global_batch_size=global_batch_size,
        source_list=SourceMixtureList(
            sources=[
                SourceMixtureConfig(
                    source_name="code_fim",
                    target_ratio=0.12352010809861039,
                    max_repetition_ratio=1.0,
                    paths=[
                        "gs://ai2-llm/preprocessed/stack-edu/sample-fim-weighted-pl-edu-score-decon/**/**/*.npy"
                    ],
                ),
                # ...
            ]
        ),
        seed=SEED,
    )

    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=source_mix_config,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
        generate_doc_lengths=True,  # providing doc lengths enables intra-document masking
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path="gs://...",  # optional
        max_duration=max_duration,
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    )
    trainer_config.add_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts,
            wandb_group_name=cli_context.run_name,
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    return experiment_config


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/scripts/train/olmo3-midtraining/cookbook-example.py dry_run debug_run ai2/augusta

        To launch a training run on Augusta w/ 8 nodes:
        python src/scripts/train/olmo3-midtraining/cookbook-example.py launch my_run ai2/augusta \
            --launch.num_nodes=8 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
