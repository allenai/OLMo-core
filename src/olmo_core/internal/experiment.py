import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, cast

import torch
from rich import print

from olmo_core.config import Config, StrEnum
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyFSLDatasetConfig
from olmo_core.distributed.utils import get_local_rank
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SlackNotifierCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.utils import prepare_cli_environment, seed_all

from .common import build_launch_config, get_beaker_username, get_root_dir, get_work_dir

log = logging.getLogger(__name__)


@dataclass
class CliContext(Config):
    script: str
    cmd: "SubCmd"
    run_name: str
    cluster: str
    overrides: List[str]


@dataclass
class CommonComponents(Config):
    run_name: str
    root_dir: str
    work_dir: str
    save_folder: str
    launch: Optional[BeakerLaunchConfig]


@dataclass
class DataComponents(Config):
    tokenizer: TokenizerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig


@dataclass
class ExperimentConfig(Config):
    run_name: str
    launch: Optional[BeakerLaunchConfig]
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536
    backend: Optional[str] = "cpu:gloo,cuda:nccl"


class SubCmd(StrEnum):
    launch = "launch"
    train = "train"
    train_single = "train_single"
    prep = "prep"
    launch_prep = "launch_prep"
    dry_run = "dry_run"

    def prepare_environment(self, config: ExperimentConfig):
        if self in (SubCmd.launch, SubCmd.dry_run, SubCmd.prep, SubCmd.launch_prep):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment(backend=config.backend)
        elif self == SubCmd.train_single:
            prepare_training_environment(backend=None)
        else:
            raise NotImplementedError(self)

    def run(self, config: ExperimentConfig):
        if get_local_rank() == 0:
            print(config)
            print(
                "\n"
                f"[b blue]Total parameters:[/]         {config.model.num_params:,d} ({config.model.num_active_params:,d} active)\n"
                f"[b blue]Non-embedding parameters:[/] {config.model.num_non_embedding_params:,d} ({config.model.num_active_non_embedding_params:,d} active)"
            )

        if self == SubCmd.launch:
            launch(config)
        elif self == SubCmd.dry_run:
            pass
        elif self == SubCmd.train:
            train(config)
            teardown_training_environment()
        elif self == SubCmd.train_single:
            if config.train_module.dp_config is not None:
                log.warning(
                    "'dp_config' is set to %s, but you can't use data parallelism when running on a single node. Disabling.",
                    config.train_module.dp_config,
                )
                config.train_module.dp_config = None
            if config.train_module.tp_config is not None:
                log.warning(
                    "'tp_config' is set to %s, but you can't use tensor parallelism when running on a single node. Disabling.",
                    config.train_module.dp_config,
                )
                config.train_module.tp_config = None
            train(config)
            teardown_training_environment()
        elif self == SubCmd.prep:
            prep(config)
        elif self == SubCmd.launch_prep:
            launch_prep(config)
        else:
            raise NotImplementedError(self)


ConfigBuilder = Callable[[CliContext], ExperimentConfig]
"""
Type alias for a function that builds an ExperimentConfig based on a CliContext.
"""


def build_common_components(
    cli_context: CliContext,
    *,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
    use_hostname_constraints: bool = False,
    num_execution_units: Optional[int] = None,
) -> CommonComponents:
    root_dir = get_root_dir(cli_context.cluster)

    # TODO: can this be factored better? Why are launch commands relevant here?
    cmd_to_launch = SubCmd.train
    if cli_context.cmd == SubCmd.launch_prep:
        cmd_to_launch = SubCmd.prep

    beaker_user = get_beaker_username()
    launch_config: Optional[BeakerLaunchConfig] = None
    if beaker_user is not None:
        launch_config = build_launch_config(
            name=f"{cli_context.run_name}-{cmd_to_launch}",
            root_dir=root_dir,
            cmd=[
                cli_context.script,
                cmd_to_launch,
                cli_context.run_name,
                cli_context.cluster,
                *cli_context.overrides,
            ],
            cluster=cli_context.cluster,
            nccl_debug=True,
            beaker_image=beaker_image,
            num_nodes=num_nodes,
            workspace=beaker_workspace,
            use_hostname_constraints=use_hostname_constraints,
            num_execution_units=num_execution_units,
        )

    if beaker_user is not None:
        save_folder = f"{root_dir}/checkpoints/{beaker_user.lower()}/{cli_context.run_name}"
    else:
        save_folder = f"{root_dir}/checkpoints/{cli_context.run_name}"

    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        save_folder=save_folder,
        launch=launch_config,
    )


def build_default_data_components(
    common: CommonComponents,
    global_batch_size: int,
    max_sequence_length: int = 8192,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    tokenizer_config = TokenizerConfig.dolma2()

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=common.root_dir,
        sequence_length=max_sequence_length,
        max_target_sequence_length=max_sequence_length,
        work_dir=common.work_dir,
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=34521, num_workers=4
    )

    return DataComponents(
        tokenizer=tokenizer_config, dataset=dataset_config, data_loader=data_loader_config
    )


def build_required_callbacks(common: CommonComponents) -> Dict[str, Callback]:
    callbacks = {
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "slack_notifier": SlackNotifierCallback(name=common.run_name, enabled=False),
    }
    if common.launch is not None:
        callbacks["beaker"] = BeakerCallback()
    if torch.cuda.is_available():
        callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()
    return callbacks


def build_default_eval_callbacks(
    common: CommonComponents, data: DataComponents
) -> Dict[str, Callback]:
    return {
        "lm_evaluator": LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                mix_base_dir=common.root_dir,
                sequence_length=data.dataset.effective_sequence_length,
                tokenizer=data.tokenizer,
                work_dir=common.work_dir,
            ),
            eval_interval=1000,
        ),
        "downstream_evaluator": DownstreamEvaluatorCallbackConfig(
            tasks=["hellaswag"],
            tokenizer=data.tokenizer,
            eval_interval=1000,
        ),
    }


def _set_beaker_execution_units(config: ExperimentConfig):
    # When running on Augusta with hostname constraints enabled, setting more beaker
    # execution units than model replicas may result in the replicas being split across
    # Augusta hardware blocks.
    if (
        config.launch
        and config.launch.use_hostname_constraints
        and any("augusta" in cluster for cluster in config.launch.clusters)
        and (dp_config := config.train_module.dp_config) is not None
    ):
        if dp_config.num_replicas is not None:
            num_model_replicas = dp_config.num_replicas
        elif dp_config.shard_degree is not None:
            nodes_per_replica = max(1, dp_config.shard_degree // config.launch.num_gpus)
            num_model_replicas = config.launch.num_nodes // nodes_per_replica
        else:
            return

        if config.launch.num_execution_units is None:
            log.info(f"Setting number of execution units to {num_model_replicas}.")
            config.launch.num_execution_units = num_model_replicas
        elif config.launch.num_execution_units > num_model_replicas:
            log.warning(
                f"Number of execution units {config.launch.num_execution_units} exceeds number of model replicas {num_model_replicas}. "
                "On Augusta, this may result in suboptimal performance due to model replicas being split "
                "across hardware blocks. To resolve, decrease num_execution_units in beaker launch config, "
                "increase number of model replicas or disable use_hostname_constraints in beaker launch config."
            )


def build_config(
    cli_context: CliContext,
    *,
    common_config_builder: Callable[..., CommonComponents] = build_common_components,
    data_config_builder: Callable[..., DataComponents] = build_default_data_components,
    model_config_builder: Callable[[TokenizerConfig], TransformerConfig],
    train_module_config_builder: Callable[[CommonComponents], TransformerTrainModuleConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
    use_hostname_constraints: bool = False,
    num_execution_units: Optional[int] = None,
    global_batch_size: int,
    max_sequence_length: int = 4096,
    include_default_evals: bool = False,
    **kwargs,
) -> ExperimentConfig:
    common = common_config_builder(
        cli_context,
        beaker_image,
        num_nodes,
        beaker_workspace,
        use_hostname_constraints,
        num_execution_units,
    )

    data = data_config_builder(common, global_batch_size, max_sequence_length, **kwargs)

    model = model_config_builder(data.tokenizer)

    trainer = trainer_config_builder(common)
    callbacks_to_add = build_required_callbacks(common)
    if include_default_evals:
        default_evals = build_default_eval_callbacks(common, data)
        callbacks_to_add.update(default_evals)
    for name, cb in callbacks_to_add.items():
        if name not in trainer.callbacks:
            trainer.add_callback(name, cb)

    train_module = train_module_config_builder(
        common
    )  # requires data knowledge -> effective_sequence_length

    config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=common.launch,
        model=model,
        dataset=data.dataset,
        data_loader=data.data_loader,
        train_module=train_module,
        trainer=trainer,
    )

    config = config.merge(cli_context.overrides)

    _set_beaker_execution_units(config)

    if finalize_config is not None:
        finalize_config(config)

    return config


def launch(config: ExperimentConfig):
    log.info(config)
    assert config.launch is not None
    config.launch.launch(follow=True)


def launch_prep(config: ExperimentConfig):
    assert config.launch is not None
    config.launch.num_gpus = 0
    config.launch.num_nodes = 1
    log.info(config)
    config.launch.launch(follow=True, torchrun=False)


def prep(config: ExperimentConfig):
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset)
    data_loader.reshuffle(epoch=1)


def train(config: ExperimentConfig):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


def main(*, config_builder: ConfigBuilder):
    """
    Main entry point for running Ai2-internal experiments.

    This function handles command-line argument parsing and executes the appropriate
    subcommand (launch, train, prep, etc.) based on user input.

    :param config_builder: A function that builds an ExperimentConfig from command-line
        arguments. It should accept (script, cmd, run_name, cluster, overrides)
        and return an ExperimentConfig instance.
    """

    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]prep:[/]        Prepare the dataset ahead of training to save GPU time.
[b magenta]launch_prep:[/] Launch the script on Beaker with the [b magenta]prep[/] subcommand.
[b magenta]dry_run:[/]     Pretty print the config and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} run01 ai2/pluto-cirrascale --launch.num_nodes=2[/]
    """.strip()

    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv
    cmd = SubCmd(cmd)
    cli_context = CliContext(script, cmd, run_name, cluster, overrides)

    config: ExperimentConfig = config_builder(cli_context)

    cmd.prepare_environment(config)
    cmd.run(config)
