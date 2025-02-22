import logging
import sys
from dataclasses import dataclass
from typing import Callable, List, cast

from rich import print

from olmo_core.config import Config, StrEnum
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig
from olmo_core.distributed.utils import get_local_rank
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.model_ladder import ModelLadder, ModelSize, RunDuration
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import CometCallback, ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

from .common import build_launch_config, get_gpu_type, get_root_dir

log = logging.getLogger(__name__)


@dataclass
class LadderRunConfig(Config):
    launch: BeakerLaunchConfig
    ladder: ModelLadder
    model: TransformerConfig
    optim: OptimConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig


class SubCmd(StrEnum):
    launch = "launch"
    train = "train"
    train_single = "train_single"
    dry_run = "dry_run"

    def prepare_environment(self):
        if self in (SubCmd.launch, SubCmd.dry_run):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment()
        elif self == SubCmd.train_single:
            prepare_training_environment(backend=None)
        else:
            raise NotImplementedError(self)

    def run(self, config: LadderRunConfig):
        if get_local_rank() == 0:
            print(config)

        if self == SubCmd.launch:
            config.launch.launch(follow=True)
        elif self == SubCmd.dry_run:
            pass
        elif self in (SubCmd.train, SubCmd.train_single):
            if self == SubCmd.train_single:
                if config.model.dp_config is not None:
                    log.warning(
                        "dp_config is set to %s, but you can't use data parallelism when running on a single node. Disabling.",
                        config.model.dp_config,
                    )
                    config.model.dp_config = None
                if config.model.tp_config is not None:
                    log.warning(
                        "tp_config is set to %s, but you can't use tensor parallelism when running on a single node. Disabling.",
                        config.model.dp_config,
                    )
                    config.model.tp_config = None

            try:
                # Set RNG states on all devices.
                seed_all(config.ladder.init_seed)

                device = get_default_device()

                # Build mesh, if needed.
                world_mesh = config.model.build_mesh(device=device)

                # Build components.
                model = config.model.build(
                    init_device="meta" if self == SubCmd.train else str(device),
                    device=device,
                    max_seq_len=config.dataset.sequence_length,
                    mesh=world_mesh,
                )
                optim = config.optim.build(model)
                dataset = config.dataset.build()
                data_loader = config.data_loader.build(dataset, mesh=world_mesh)
                trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

                # Record the config to W&B/Comet and each checkpoint dir.
                config_dict = config.as_config_dict()
                cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
                cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
                cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

                # Train.
                trainer.fit()
            finally:
                teardown_training_environment()
        else:
            raise NotImplementedError(self)


def build_config(
    ladder: ModelLadder,
    script: str,
    size: ModelSize,
    run_duration: RunDuration,
    cmd: SubCmd,
    cluster: str,
    overrides: List[str],
) -> LadderRunConfig:
    del cmd

    root_dir = get_root_dir(cluster)
    launch = build_launch_config(
        name=f"{ladder.name}-{size}-{run_duration}",
        root_dir=root_dir,
        cmd=[script, SubCmd.train, size, run_duration, cluster, *overrides],
        cluster=cluster,
    ).merge(overrides, strict=False)

    dp_world_size = launch.num_nodes * launch.num_gpus
    gpu_type = get_gpu_type(cluster)

    model = ladder.get_model_config(size=size)
    optim = ladder.get_optim_config()
    dataset = ladder.get_dataset_config()
    data_loader = ladder.get_data_loader_config()
    trainer = ladder.get_trainer_config(
        size=size, run_duration=run_duration, gpu_type=gpu_type, dp_world_size=dp_world_size
    )
    if gpu_type in ("cpu", "mps"):
        del trainer.callbacks["gpu_monitor"]

    return LadderRunConfig(
        launch=launch,
        ladder=ladder,
        model=model,
        optim=optim,
        dataset=dataset,
        data_loader=data_loader,
        trainer=trainer,
    ).merge(overrides)


def main(ladder_builder: Callable[[str], ModelLadder]):
    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{'|'.join(SubCmd)}[/] [i b]SIZE RUN_DURATION CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]dry_run:[/]     Pretty print the config to run and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} 1B 1xC ai2/pluto-cirrascale --launch.num_nodes=2[/]
    """.strip()

    try:
        script, cmd, size, run_duration, cluster, overrides = (
            sys.argv[0],
            SubCmd(sys.argv[1]),
            ModelSize(sys.argv[2]),
            RunDuration(sys.argv[3]),
            sys.argv[4],
            sys.argv[5:],
        )
    except (IndexError, ValueError):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    cmd.prepare_environment()

    # Build ladder config.
    ladder = ladder_builder(get_root_dir(cluster))
    ladder = ladder.merge(overrides, prefix="ladder")

    # Build run config.
    config = build_config(ladder, script, size, run_duration, cmd, cluster, overrides)
    config.ladder.validate()

    # Run the cmd.
    cmd.run(config)
