from datetime import datetime
from typing import Optional

from olmo_core.internal.common import get_root_dir, get_work_dir, build_launch_config
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.internal.experiment import main, CliContext, ExperimentConfig
from olmo_core.internal import cookbook
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, LoadStrategy
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.optim.scheduler import WSD, SchedulerUnits
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
    NumpyDatasetDType,
)
from olmo_core.data.utils import infer_token_dtype
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureConfig


SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 4096 * 8192
MAX_TOKENS = 100_000_000_000
SEED = 12536


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_suffix = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{run_name_with_suffix}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        root_dir=root_dir,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        workspace="ai2/oe-data",
        num_nodes=4,
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo2_1B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        global_batch_size=GLOBAL_BATCH_SIZE,
        instances_per_rank_microbatch=4,
        optim=SkipStepAdamWConfig(
            lr=4.4e-5 * (4**0.5),
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=WSD(units=SchedulerUnits.steps, warmup=2000, decay=2000, decay_fraction=None),
    )

    trainer_config = TrainerConfig(
        max_duration=Duration.chinchilla_tokens(
            2.0, model_params=model_config.num_active_non_embedding_params
        ),
        # max_duration=Duration.tokens(500000),
        save_folder=save_dir,
        save_overwrite=True,
        load_strategy=LoadStrategy.always,
    )
    callbacks = cookbook.configure_default_callbacks(
        run_name=cli_context.run_name,
        launch_config=beaker_launch_config,
    )
    for name, callback in callbacks.items():
        trainer_config.add_callback(name, callback)

    config = [
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
    source_mix = SourceMixtureDatasetConfig(
        source_configs=config,
        max_tokens=MAX_TOKENS,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,
    )
    # source_mix = SourceMixtureDatasetConfig.from_yaml("./my-mix.yaml")

    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=source_mix,
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
        generate_doc_lengths=True,  # providing doc lenghts enables intra-document masking
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    return ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )


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
