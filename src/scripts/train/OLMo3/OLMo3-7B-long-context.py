from datetime import datetime
from typing import Optional

from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

SEQ_LENGTH = 65536
GLOBAL_BATCH_SIZE = 2**22  # ~4M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.00020712352850360292
SEED = 4123


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
        # beaker_image="petew/olmo-core-tch270cu128",
        workspace="ai2/long-contexts",
        num_nodes=16,
        nccl_debug=True,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo3_7B(vocab_size=tokenizer_config.padded_vocab_size())
    assert not isinstance(model_config.block, dict)
    assert isinstance(model_config.block.sequence_mixer, AttentionConfig), (
        "Sequence mixer must be an attention config for RoPE scaling"
    )
    # Drop RoPE on all layers
    model_config.block.sequence_mixer.rope = None

    # # Drop RoPE on global attention layers only, keep it on sliding window layers.
    # sliding_window_cfg = model_config.block.sequence_mixer.sliding_window
    # assert sliding_window_cfg is not None
    # overrides: dict[int, TransformerBlockConfig] = {}
    # for i in range(model_config.n_layers):
    #     if not sliding_window_cfg.should_use_swa(i, model_config.n_layers):
    #         block_copy = model_config.block.copy()
    #         assert isinstance(block_copy.sequence_mixer, AttentionConfig)
    #         block_copy.sequence_mixer.rope = None
    #         overrides[i] = block_copy
    # model_config.block_overrides = overrides or None

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH,
        learning_rate=LR,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
        float8_enabled=True,
        activation_memory_budget=0.7,
        cp_degree=8,
        dp_shard_degree=1,
    )

    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        "/weka/oe-training-default/ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
        generate_doc_lengths=True,  # enables intra-document masking
        source_group_size=8,
        source_permutation_seed=123,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=16, prefetch_factor=10
    )

    trainer_config = cookbook.configure_trainer(
        load_path="https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684/",
        load_trainer_state=False,
        load_optim_state=True,
        max_duration=Duration.tokens(MAX_TOKENS),
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    )
    trainer_config.add_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts,
            wandb_group_name=cli_context.run_name,
            wandb_project="olmo3-7b-long-context",
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
            python src/scripts/train/OLMo3/OLMo3-7B-long-context.py dry_run debug_run ai2/augusta

        To launch a training run on Augusta w/ 8 nodes:
        python src/scripts/train/OLMo3/OLMo3-7B-long-context.py launch my_run ai2/augusta \
            --launch.num_nodes=8 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
