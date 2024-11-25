import sys
from dataclasses import dataclass
from typing import Callable, List, cast

from beaker import Beaker
from rich import print

from olmo_core.config import Config, StrEnum
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig
from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.model_ladder import ModelLadder, ModelSize
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import CometCallback, ConfigSaverCallback, WandBCallback
from olmo_core.utils import (
    generate_uuid,
    get_default_device,
    prepare_cli_environment,
    seed_all,
)


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
    dry_run = "dry_run"

    def prepare_environment(self):
        if self in (SubCmd.launch, SubCmd.dry_run):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment()
        else:
            raise NotImplementedError(self)

    def run(self, size: ModelSize, config: LadderRunConfig):
        print(config)

        if self == SubCmd.launch:
            config.launch.launch(follow=True)
        elif self == SubCmd.dry_run:
            pass
        elif self == SubCmd.train:
            try:
                # Set RNG states on all devices.
                seed_all(config.ladder.init_seed)

                # Build components.
                model = config.model.build(
                    init_device="meta",
                    device=get_default_device(),
                    max_seq_len=config.dataset.sequence_length,
                    dp_mesh=config.ladder.get_dp_mesh(size=size),
                )
                optim = config.optim.build(model)
                dataset = config.dataset.build()
                data_loader = config.data_loader.build(dataset)
                trainer = config.trainer.build(model, optim, data_loader)

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


def get_root_dir(cluster: str) -> str:
    root_dir: str = "weka://oe-training-default/ai2-llm"
    if "jupiter" in cluster:
        root_dir = "/weka/oe-training-default/ai2-llm"
    elif "augusta" in cluster:
        root_dir = "gs://ai2-llm"
    return root_dir


def build_config(
    ladder: ModelLadder,
    script: str,
    size: ModelSize,
    cmd: SubCmd,
    cluster: str,
    overrides: List[str],
) -> LadderRunConfig:
    del cmd

    root_dir = get_root_dir(cluster)
    weka_buckets: List[BeakerWekaBucket] = []
    if root_dir.startswith("/weka/"):
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    beaker_user = (Beaker.from_env().account.whoami().name).upper()

    launch = BeakerLaunchConfig(
        name=f"{ladder.name}-{size}-{generate_uuid()[:8]}",
        budget="ai2/oe-training",
        cmd=[script, SubCmd.train, size, cluster, *overrides],
        task_name="train",
        workspace="ai2/OLMo-core",
        clusters=[cluster],
        weka_buckets=weka_buckets,
        beaker_image=OLMoCoreBeakerImage.nightly,  # some features require nightly at the moment
        num_nodes=1,
        num_gpus=8,
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
            BeakerEnvSecret(name="COMET_API_KEY", secret=f"{beaker_user}_COMET_API_KEY"),
            BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
            BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
            BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
            BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
        ],
        setup_steps=[
            # Clone repo.
            'git clone "$REPO_URL" .',
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            # Setup python environment.
            "conda shell.bash activate base",
            "pip install -e '.[all]'",
            "pip freeze",
            # Move AWS credentials from env to relevant files
            "mkdir -p ~/.aws",
            "printenv AWS_CONFIG > ~/.aws/config",
            "printenv AWS_CREDENTIALS > ~/.aws/credentials",
        ],
    ).merge(overrides, strict=False)

    dp_world_size = launch.num_nodes * launch.num_gpus
    gpu_type = "h100"

    model = ladder.get_model_config(size=size)
    optim = ladder.get_optim_config(size=size)
    dataset = ladder.get_dataset_config()
    data_loader = ladder.get_data_loader_config(size=size, dp_world_size=dp_world_size)
    trainer = ladder.get_trainer_config(size=size, gpu_type=gpu_type)

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
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{'|'.join(SubCmd)}[/] [i b]SIZE CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]dry_run:[/]     Pretty print the config to run and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} 1B ai2/pluto-cirrascale --launch.num_nodes=2[/]
    """.strip()

    try:
        script, cmd, size, cluster, overrides = (
            sys.argv[0],
            SubCmd(sys.argv[1]),
            ModelSize(sys.argv[2]),
            sys.argv[3],
            sys.argv[4:],
        )
    except (IndexError, ValueError):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    cmd.prepare_environment()

    # Build ladder config.
    ladder = ladder_builder(get_root_dir(cluster))
    ladder.merge(overrides, prefix="ladder")

    # Build run config.
    config = build_config(ladder, script, size, cmd, cluster, overrides)

    # Run the cmd.
    cmd.run(size, config)
