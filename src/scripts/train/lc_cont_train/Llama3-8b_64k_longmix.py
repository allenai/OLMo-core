"""
This script can be used to launch phase of continued long-context training on Beaker.
Run the script without any arguments to see usage info.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, cast

import rich
import torch
from rich import print

from olmo_core.config import Config, DType
from olmo_core.float8 import Float8Config
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
from olmo_core.nn.transformer import (
    TransformerConfig,
)
from olmo_core.optim import (
    CosWithWarmup,
    OptimConfig,
    OptimGroupOverride,
    AdamWConfig,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTensorParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    WandBCallback, 
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    # GradClipperCallback,
    # SchedulerCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

# The max number of pretraining steps configured for the purpose of setting the learning rate
# schedule. I'm hard-coding this here based on the number found in the logs. It only changes
# if batch size changes, which we're not planning on changing that over the course of the run.
# TODO: pull this from the checkpoint when https://github.com/allenai/OLMo-core/pull/143 merges.


CONTEXT_LENGTH = 4 * 16384


class AnnealingDataMix(DataMixBase):
    """
    Defines the annealing mixes. To create a new mix, make a new file in this folder and add its
    name (without the '.txt' extension) below.
    """

    datamix = "llama3_64k_longmix"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        assert tokenizer == TokenizerName.llama3
        paths = []
        labels = []

        with open(f"src/scripts/train/lc_cont_train/{self}.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(f"{base_dir}{line}")
                labels.append(line.split("/")[1])

        return paths, labels


@dataclass
class LcContTrain(Config):
    """
    Custom config class for the annealing run.

    Making config classes isn't strictly necessary for OLMo-core, but it gives us a nice way to
    capture all of the hyperparameters for a run and an easy way to override those options from
    the command line without configuring a complicated command line parser.
    """

    run_name: str
    launch: BeakerLaunchConfig
    model: TransformerConfig
    train_module: TransformerTrainModuleConfig
    # optim: OptimConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
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
        load_path: str,
        cluster: str,
        overrides: List[str],
    ) -> "LcContTrain":
        root_dir = get_root_dir(cluster)

        tokenizer_config = TokenizerConfig.llama3()


        return LcContTrain(
            run_name=run_name,
            load_path=load_path,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[script, cmd, run_name, cluster, *overrides],
                cluster=cluster,
                nccl_debug=False,
            ),
            train_module = TransformerTrainModuleConfig(
                rank_microbatch_size= 1 * CONTEXT_LENGTH,
                 optim=AdamWConfig(
                    lr= 1 * 1e-5,
                    weight_decay=0.1,
                    betas=(0.9, 0.95),
                    group_overrides=[
                        OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                    ],
                    fused=True,
                ),
                max_sequence_length=CONTEXT_LENGTH,
                compile_model=True,
                # compile_loss=True,
                z_loss_multiplier=1e-5,
                dp_config=TransformerDataParallelConfig(
                    name=DataParallelType.fsdp,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                    wrapping_strategy=TransformerDataParallelWrappingStrategy.fine_grained,
                ),
                #tp_config=TransformerTensorParallelConfig(
                #    degree=2,
                    #loss_parallel=True,
                #),
                ac_config=TransformerActivationCheckpointingConfig(),
                float8_config=Float8Config(enabled=False),  # TODO (epwalsh): broken with TP
                max_grad_norm=1.0,
                scheduler=CosWithWarmup(warmup_steps=475, alpha_f=0.0),
            ),
            model=TransformerConfig.llama3_8B(
                vocab_size=tokenizer_config.padded_vocab_size(),
                use_flash=True,
                rope_theta = 8 * 10 **6,
            ),
            dataset=NumpyDatasetConfig.from_data_mix(
                AnnealingDataMix.datamix,
                tokenizer=tokenizer_config,
                mix_base_dir=root_dir,
                generate_doc_lengths=True,
                sequence_length=CONTEXT_LENGTH,
                work_dir=get_work_dir(root_dir),
            ),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size= 64 * CONTEXT_LENGTH,  # NOTE: this is specified in TOKENS, not instances.
                seed=34521,  # NOTE: can update this to change data order.
                num_workers=4,
            ),
            trainer=TrainerConfig(
                save_folder=f"gs://ai2-llm/checkpoints/amandab/{run_name}",
                # save_folder=f"/weka/oe-training-default/ai2-llm/checkpoints/dustins/{run_name}",
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                load_path=load_path,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                hard_stop=Duration.tokens(int(10e8)), # stop at 1B tokens for this run 
                max_duration=Duration.tokens(int(20e9)),
            )
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=100,
                    save_async=True,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    entity="ai2-llm",
                    project="long-contexts",
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
        ).merge(overrides)


def train(config: LcContTrain):
    # Set RNG states on all devices.
    # seed_all(config.init_seed)

    device = get_default_device()

    # Build mesh, if needed.
    # world_mesh = config.model.build_mesh(device=device)

    # Build components.
    model = config.model.build(
        init_device="meta",
    )
    dataset = config.dataset.build()
    train_module = config.train_module.build(model, device)
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    USAGE = f"""
LC extend a 7B model.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]launch|train|dry_run[/] [i b]RUN_NAME PRETRAIN_CHECKPOINT CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use the [b magenta]launch[/] cmd to submit it to Beaker or run it via torchrun if you know what you're doing.
[b magenta]dry_run:[/]     Print the config for debugging.

[b]Examples[/]
$ [i]python {sys.argv[0]} launch run01  --launch.num_nodes=2[/]
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
    config = LcContTrain.build(
        script=script,
        cmd="train",
        run_name=run_name,
        load_path="gs://ai2-llm/checkpoints/amandab/Meta-Llama-3-8B-Base-redone/",
        cluster=cluster,
        overrides=overrides,
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
