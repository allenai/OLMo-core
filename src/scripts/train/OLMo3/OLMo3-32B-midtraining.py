from datetime import datetime
from typing import Optional

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

from olmo_core.nn.attention import SlidingWindowAttentionConfig

SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**22  # ~2M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.00020712352850360292 #TO BE UPDATED
SEED = 1337


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
        workspace="ai2/olmo-3-microanneals",
        num_nodes=64,
        nccl_debug=True,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()
    # model_config = TransformerConfig.olmo3_7B(vocab_size=tokenizer_config.padded_vocab_size())
    model_config = TransformerConfig.olmo2_32B(vocab_size=tokenizer_config.padded_vocab_size())
    model_config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[4096, 4096, 4096, -1],
    )
    model_config.block.attention.use_flash = True

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH,
        learning_rate=LR,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=0, alpha_f=0.0),
        activation_memory_budget=0.5,
        dp_shard_degree=64,
    )

    source_list = SourceMixtureList.from_yaml(
        "src/olmo_core/data/source_mixtures/OLMo3-32B-midtraining.yaml"
    )
    source_list.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=MAX_TOKENS,
            global_batch_size=GLOBAL_BATCH_SIZE,
            processes=16,
            seed=SEED,
        ),
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path="gs://ai2-llm/checkpoints/OLMo25/step1413814", #TO BE UPDATED
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
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/scripts/train/OLMo3/OLMo3-7B-midtraining.py dry_run debug_run ai2/augusta

        To launch a training run on Augusta w/ 8 nodes:
        python src/scripts/train/OLMo3/OLMo3-7B-midtraining.py launch my_run ai2/augusta \
            --launch.num_nodes=8 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
