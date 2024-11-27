import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, cast

from rich import print
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import Config, StrEnum
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
    VSLCurriculumConfig,
    VSLCurriculumType,
)
from olmo_core.distributed.utils import get_num_nodes, init_hybrid_shard_mesh
from olmo_core.float8 import Float8Config
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    Callback,
    CometCallback,
    ConfigSaverCallback,
    Float8HandlerCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

from .common import build_launch_config, get_beaker_username, get_root_dir, get_work_dir

log = logging.getLogger(__name__)


@dataclass
class CommonComponents(Config):
    run_name: str
    save_folder: str
    launch: BeakerLaunchConfig
    tokenizer: TokenizerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    callbacks: Dict[str, Callback]


class DPMeshType(StrEnum):
    full = "full"
    hybrid = "hybrid"


@dataclass
class DPMeshConfig(Config):
    name: DPMeshType = DPMeshType.hybrid

    def build(self) -> Optional[DeviceMesh]:
        if get_num_nodes() == 1 or self.name == DPMeshType.full:
            return None
        else:
            return init_hybrid_shard_mesh()


@dataclass
class ExperimentConfig(Config):
    run_name: str
    launch: BeakerLaunchConfig
    model: TransformerConfig
    optim: OptimConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    dp_mesh: DPMeshConfig
    init_seed: int = 12536


class SubCmd(StrEnum):
    launch = "launch"
    train = "train"
    prep = "prep"
    launch_prep = "launch_prep"
    dry_run = "dry_run"

    def prepare_environment(self):
        if self in (SubCmd.launch, SubCmd.dry_run, SubCmd.prep, SubCmd.launch_prep):
            prepare_cli_environment()
        elif self == SubCmd.train:
            prepare_training_environment()
        else:
            raise NotImplementedError(self)

    def run(self, config: ExperimentConfig):
        print(config)
        print(
            "\n"
            f"[b blue]Total parameters:[/]                {config.model.num_params:,d}\n"
            f"[b blue]Non-embedding parameters:[/]        {config.model.num_non_embedding_params:,d}"
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
) -> CommonComponents:
    root_dir = get_root_dir(cluster)

    cmd_to_launch = SubCmd.train
    if cmd == SubCmd.launch_prep:
        cmd_to_launch = SubCmd.prep

    launch_config = build_launch_config(
        name=f"{run_name}-{cmd_to_launch}",
        root_dir=root_dir,
        cmd=[script, cmd_to_launch, run_name, cluster, *overrides],
        cluster=cluster,
    )

    beaker_user = get_beaker_username()

    tokenizer_config = TokenizerConfig.dolma2()

    dataset_config = NumpyDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        sequence_length=4096,
        max_target_sequence_length=8192,
        min_sequence_length=256,
        max_sequence_length=8192,
        vsl_curriculum=VSLCurriculumConfig(
            name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False
        ),
        work_dir=get_work_dir(root_dir),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=34521, num_workers=4
    )

    callbacks: Dict[str, Callback] = {
        "lr_scheduler": SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=2000)),
        "gpu_monitor": GPUMemoryMonitorCallback(),
        "grad_clipper": GradClipperCallback(max_grad_norm=1.0),
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "lm_evaluator": LMEvaluatorCallbackConfig(
            eval_dataset=NumpyDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                name=NumpyDatasetType.padded_fsl,
                mix_base_dir=root_dir,
                sequence_length=dataset_config.effective_sequence_length,
                tokenizer=tokenizer_config,
                work_dir=get_work_dir(root_dir),
            ),
            eval_interval=1000,
        ),
    }

    return CommonComponents(
        run_name=run_name,
        save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
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
    global_batch_size: int,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    optim_config_builder: Callable[[CommonComponents], OptimConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
) -> ExperimentConfig:
    common = build_common_components(
        script, cmd, run_name, cluster, overrides, global_batch_size=global_batch_size
    )

    model = model_config_builder(common)
    if model.float8_config is None:
        model.float8_config = Float8Config(compile=model.compile, enabled=False)

    trainer = trainer_config_builder(common)
    if trainer.load_key_mapping is None:
        trainer.load_key_mapping = {
            # For backwards compatibility when loading older checkpoints.
            "lm_head.w_out.weight": "w_out.weight",
            "lm_head.norm.weight": "norm.weight",
        }
    for name, cb in common.callbacks.items():
        if name not in trainer.callbacks:
            trainer.add_callback(name, cb)

    config = ExperimentConfig(
        run_name=run_name,
        launch=common.launch,
        model=model,
        optim=optim_config_builder(common),
        dataset=common.dataset,
        data_loader=common.data_loader,
        trainer=trainer,
        dp_mesh=DPMeshConfig(),
    )

    if finalize_config is not None:
        finalize_config(config)

    config = config.merge(overrides)

    if config.model.float8_config is not None and config.model.float8_config.enabled:
        config.trainer.add_callback(
            "float8_handler", Float8HandlerCallback(config=config.model.float8_config)
        )

    return config


def launch(config: ExperimentConfig):
    log.info(config)
    config.launch.launch(follow=True)


def launch_prep(config: ExperimentConfig):
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
    model = config.model.build(
        init_device="meta",
        device=get_default_device(),
        max_seq_len=config.dataset.sequence_length,
        dp_mesh=config.dp_mesh.build(),
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


def main(
    *,
    global_batch_size: int,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    optim_config_builder: Callable[[CommonComponents], OptimConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
):
    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{'|'.join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
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
        optim_config_builder=optim_config_builder,
        trainer_config_builder=trainer_config_builder,
        finalize_config=finalize_config,
    )

    cmd.run(config)
