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
    NumpyDatasetType,
    TokenizerConfig,
    VSLCurriculumConfig,
    VSLCurriculumType,
)
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
class CommonComponents(Config):
    run_name: str
    save_folder: str
    launch: Optional[BeakerLaunchConfig]
    tokenizer: TokenizerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    callbacks: Dict[str, Callback]


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


class SubCmd(StrEnum):
    launch = "launch"
    train = "train"
    train_single = "train_single"
    prep = "prep"
    launch_prep = "launch_prep"
    dry_run = "dry_run"

    def prepare_environment(self):
        if self in (SubCmd.launch, SubCmd.dry_run, SubCmd.prep, SubCmd.launch_prep):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment()
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
            try:
                train(config)
            finally:
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
            try:
                train(config)
            finally:
                teardown_training_environment()
        elif self == SubCmd.prep:
            prep(config)
        elif self == SubCmd.launch_prep:
            launch_prep(config)
        else:
            raise NotImplementedError(self)


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: List[str],
    *,
    global_batch_size: int,
    sequence_length: int = 4096,
    include_default_evals: bool = True,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
) -> CommonComponents:
    root_dir = get_root_dir(cluster)

    cmd_to_launch = SubCmd.train
    if cmd == SubCmd.launch_prep:
        cmd_to_launch = SubCmd.prep

    beaker_user = get_beaker_username()
    launch_config: Optional[BeakerLaunchConfig] = None
    if beaker_user is not None:
        launch_config = build_launch_config(
            name=f"{run_name}-{cmd_to_launch}",
            root_dir=root_dir,
            cmd=[script, cmd_to_launch, run_name, cluster, *overrides],
            cluster=cluster,
            nccl_debug=False,
            beaker_image=beaker_image,
            num_nodes=num_nodes,
            workspace=beaker_workspace,
        )

    tokenizer_config = TokenizerConfig.dolma2()

    dataset_config = NumpyDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir="s3:///oe-training-default/ai2-llm", #root_dir,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        min_sequence_length=min(256, sequence_length),
        max_sequence_length=max(8192, sequence_length),
        vsl_curriculum=VSLCurriculumConfig(
            name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False
        ),
        work_dir=get_work_dir(root_dir),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13,
            repetition_min_period=1,
            repetition_max_count=32,
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=34521, num_workers=4
    )

    callbacks: Dict[str, Callback] = {
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "slack_notifier": SlackNotifierCallback(name=run_name, enabled=False),
    }
    if beaker_user is not None:
        callbacks["beaker"] = BeakerCallback()

    if torch.cuda.is_available():
        callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()

    if include_default_evals:
        callbacks["lm_evaluator"] = LMEvaluatorCallbackConfig(
            eval_dataset=NumpyDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                name=NumpyDatasetType.padded_fsl,
                mix_base_dir=root_dir,
                sequence_length=dataset_config.effective_sequence_length,
                tokenizer=tokenizer_config,
                work_dir=get_work_dir(root_dir),
            ),
            eval_interval=1000,
        )
        callbacks["downstream_evaluator"] = DownstreamEvaluatorCallbackConfig(
            tasks=["hellaswag"],
            tokenizer=tokenizer_config,
            eval_interval=1000,
        )

    save_folder: str
    if beaker_user is not None:
        save_folder = f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}"
    else:
        save_folder = f"{root_dir}/checkpoints/{run_name}"

    return CommonComponents(
        run_name=run_name,
        save_folder=save_folder,
        launch=launch_config,
        tokenizer=tokenizer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        callbacks=callbacks,
    )


def build_config(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: List[str],
    *,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    train_module_config_builder: Callable[[CommonComponents], TransformerTrainModuleConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
    init_seed: int = 12536,
    **kwargs,
) -> ExperimentConfig:
    common = build_common_components(script, cmd, run_name, cluster, overrides, **kwargs)

    model = model_config_builder(common)

    trainer = trainer_config_builder(common)
    for name, cb in common.callbacks.items():
        if name not in trainer.callbacks:
            trainer.add_callback(name, cb)

    config = ExperimentConfig(
        run_name=run_name,
        launch=common.launch,
        model=model,
        dataset=common.dataset,
        data_loader=common.data_loader,
        train_module=train_module_config_builder(common),
        trainer=trainer,
        init_seed=init_seed,
    )

    config = config.merge(overrides)

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


def main(
    *,
    global_batch_size: int,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    train_module_config_builder: Callable[[CommonComponents], TransformerTrainModuleConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
    sequence_length: int = 4096,
    include_default_evals: bool = True,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
):
    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{'|'.join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]

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
    cmd.prepare_environment()

    config = build_config(
        script,
        cmd,
        run_name,
        cluster,
        overrides,
        global_batch_size=global_batch_size,
        model_config_builder=model_config_builder,
        train_module_config_builder=train_module_config_builder,
        trainer_config_builder=trainer_config_builder,
        finalize_config=finalize_config,
        sequence_length=sequence_length,
        include_default_evals=include_default_evals,
        intra_document_masking=intra_document_masking,
        include_instance_filter=include_instance_filter,
        beaker_image=beaker_image,
        num_nodes=num_nodes,
        beaker_workspace=beaker_workspace,
    )

    cmd.run(config)
