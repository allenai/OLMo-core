from datetime import datetime
from typing import Optional

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import SourceMixtureConfig, SourceMixtureDatasetConfig
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2097152
SEED = 1337


def build_olmo25_7B(vocab_size: int) -> TransformerConfig:
    config = TransformerConfig.olmo2_7B(vocab_size=vocab_size)
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[4096, 4096, 4096, -1],
    )
    config.block.attention.use_flash = True
    return config


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{run_name_with_ts}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        beaker_image="petew/olmo-core-tch270cu128",
        workspace="ai2/olmo-3-microanneals",
        num_nodes=16,
        nccl_debug=True,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = build_olmo25_7B(vocab_size=tokenizer_config.padded_vocab_size())

    max_duration = Duration.tokens(100_000_000_000)
    global_batch_size = (
        cookbook.estimate_critical_batch_size(duration=max_duration, sequence_length=SEQ_LENGTH)
        * SEQ_LENGTH
    )

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH * 2,
        learning_rate=0.00020712352850360292,
        scheduler=LinearWithWarmup(
            units=SchedulerUnits.steps,
            warmup=0,
            alpha_f=0.1,  # annealing.enabled=True from cookbook
        ),
        activation_checkpointing_enabled=True,
    )

    source_mix_config = SourceMixtureDatasetConfig(
        requested_tokens=100_000_000_000,  # 1B
        global_batch_size=global_batch_size,
        source_configs=[
            SourceMixtureConfig(
                source_name="code_fim",
                target_ratio=0.12352010809861039,
                max_repetition_ratio=1.0,
                paths=[
                    "gs://ai2-llm/preprocessed/stack-edu/sample-fim-weighted-pl-edu-score-decon/**/**/*.npy"
                ],
            ),
            # ...
        ],
        seed=SEED,
    )
    # source_mix = SourceMixtureDatasetConfig.from_yaml("./my-mix.yaml")

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
        load_path="gs://ai2-llm/checkpoints/OLMo25/step1413814",  # load_state = False
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
