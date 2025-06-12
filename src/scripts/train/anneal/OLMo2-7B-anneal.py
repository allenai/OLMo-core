"""
This script can be used to launch an annealing run for the 7B model on Beaker.
Run the script without any arguments to see usage info.
"""

import json
import logging
import sys
from dataclasses import dataclass
from typing import List, Tuple, cast

import rich
import torch
from rich import print

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMixBase,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    TokenizerConfig,
    TokenizerName,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.io import resource_path
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    LinearWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    WandBCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

EVAL_INTERVAL = 1000
SEQUENCE_LENGTH = 4096


class AnnealingDataMix(DataMixBase):
    """
    Defines the annealing mixes. To create a new mix, make a new file in this folder and add its
    name (without the '.txt' extension) below.
    """

    dolmino50 = "dolmino50"
    dolmino100 = "dolmino100"
    dolmino300 = "dolmino300"
    jallyrun = "jallyrun"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        assert tokenizer == TokenizerName.dolma2
        paths = []
        labels = []

        with open(f"src/scripts/train/anneal/{self}.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(f"{base_dir}{line}")
                labels.append(line.split("/")[1])

        return paths, labels


@dataclass
class AnnealingConfig(Config):
    """
    Custom config class for the annealing run.

    Making config classes isn't strictly necessary for OLMo-core, but it gives us a nice way to
    capture all of the hyperparameters for a run and an easy way to override those options from
    the command line without configuring a complicated command line parser.
    """

    run_name: str
    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    load_path: str
    init_seed: int = 12536

    @classmethod
    def build(
        cls,
        *,
        script: str,
        cmd: str,
        run_name: str,
        cluster: str,
        overrides: List[str],
        load_path: str,

        
    ) -> "AnnealingConfig":
        root_dir = get_root_dir(cluster)

        tokenizer_config = TokenizerConfig.dolma2()

    
        config = AnnealingConfig(
            run_name=run_name,
            load_path=load_path,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[script, cmd, run_name, cluster, *overrides],
                cluster=cluster,
                nccl_debug=False,
            ),
            model=TransformerConfig.olmo2_7B(vocab_size=tokenizer_config.padded_vocab_size()),
            dataset=NumpyDatasetConfig.from_data_mix(
                AnnealingDataMix.dolmino50,
                tokenizer=tokenizer_config,
                mix_base_dir=root_dir,
                sequence_length=SEQUENCE_LENGTH,
                work_dir=get_work_dir(root_dir),
            ),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=1024 * 4096,  # NOTE: this is specified in TOKENS, not instances.
                seed=34521,  # NOTE: can update this to change data order.
                num_workers=4,
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=2 * 4096,  # NOTE: again this is specified in tokens.
                max_sequence_length=SEQUENCE_LENGTH,
                z_loss_multiplier=1e-5,
                compile_model=True,
                optim=SkipStepAdamWConfig(
                    lr=6.1499e-05,
                    weight_decay=0.1,
                    betas=(0.9, 0.95),
                    group_overrides=[
                        OptimGroupOverride(
                            params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                        )
                    ],
                    compile=True,
                ),
                dp_config=TransformerDataParallelConfig(
                    name=DataParallelType.hsdp,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                    wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
                ),
                scheduler=LinearWithWarmup(
                    warmup_steps=0,
                    alpha_f=0.0,
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"gs://ai2-llm/checkpoints/peteish-merge/{run_name}",
                load_strategy=LoadStrategy.always,
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                load_path=load_path,
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.tokens(int(50e9)),
            )
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=1000,
                    ephemeral_save_interval=500,
                    save_async=True,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    entity="ai2-llm",
                    project="olmo-merge",
                    enabled=True,
                    cancel_check_interval=10,
                ),
            )
            .with_callback(
                "gpu_monitor",
                GPUMemoryMonitorCallback(),
            )
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_recommended_evals(
                tokenizer_config, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
            ),
        ).merge(overrides)

        # Make sure this is an 'AnnealingDataMix' instance.
        config.dataset.mix = AnnealingDataMix(config.dataset.mix)
        return config


def train(config: AnnealingConfig):
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
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    USAGE = f"""

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]launch|train|dry_run[/] [i b]RUN_NAME PRETRAIN_CHECKPOINT CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use the [b magenta]launch[/] cmd to submit it to Beaker or run it via torchrun if you know what you're doing.
[b magenta]dry_run:[/]     Print the config for debugging.

[b]Examples[/]
$ [i]python {sys.argv[0]} launch run01 gs://ai2-llm/checkpoints/peteish32/step419000 ai2/jupiter-cirrascale-2 --launch.num_nodes=2[/]
""".strip()

    # Parse command line arguments.
    if len(sys.argv) < 5 or sys.argv[1] not in ("launch", "train", "dry_run"):
        rich.get_console().print(USAGE, highlight=False)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv

    # Prepare the environment for the given command.
    if cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif cmd == "train":
        prepare_training_environment()
    else:
        raise NotImplementedError(cmd)

    # Build the config, applying any overrides.
    config = AnnealingConfig.build(
        script=script,
        cmd="train",
        run_name=run_name,
        cluster=cluster,
        overrides=overrides,
        load_path="gs://ai2-llm/checkpoints/shanea/OLMo-medium/peteish7/step928646/model_and_optim/"
        
    )

    # Print the config for debugging and then execute the command.
    if get_local_rank() == 0:
        print(config)

    if cmd == "dry_run":
        pass
    elif cmd == "launch":
        config.launch.launch(follow=True)
    elif cmd == "train":
        try:
            train(config)
        finally:
            teardown_training_environment()
    else:
        raise NotImplementedError(cmd)
