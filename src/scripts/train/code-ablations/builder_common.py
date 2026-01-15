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


SEED = 1337
WORKSPACE = "ai2/olmo4"
PRIORITY = "high"
BUDGET = "ai2/oe-base"


def common_build_experiment_config(
    cli_context: CliContext,
    seq_length: int,
    global_batch_size: int,
    max_tokens: int,
    lr: float,
    source_list: SourceMixtureList,
    tokenizer_config: TokenizerConfig,
    model_config: TransformerConfig,
    load_path: Optional[str] = None,
    seed: int = 1337,
    workspace: str = WORKSPACE,
    priority: str = PRIORITY,
) -> ExperimentConfig:
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
        workspace=workspace,
        num_nodes=16,
        nccl_debug=True,
    )
    beaker_launch_config.priority = priority

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=seq_length,
        rank_microbatch_size=seq_length * 2,
        learning_rate=lr,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=0, alpha_f=0.0),
        activation_memory_budget=0.5,
    )

    source_list.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=max_tokens,
            global_batch_size=global_batch_size,
            processes=16,
            seed=seed,
        ),
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=seq_length,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=seed, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path=load_path,
        load_trainer_state=False,
        load_optim_state=True if load_path else False,
        max_duration=Duration.tokens(max_tokens),
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    ).with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts,
            wandb_group_name=cli_context.run_name
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
        init_seed=seed,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config
