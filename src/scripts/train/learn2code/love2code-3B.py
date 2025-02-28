"""
Train a 3B for a first coding run. Run this script without any arguments to see usage info.
"""

import sys
from typing import Callable, List, Optional, Tuple

from olmo_core.config import DType
from olmo_core.data import (
    DataMixBase,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    TokenizerName,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_root_dir, get_work_dir
from olmo_core.internal.experiment import (
    CommonComponents,
    ExperimentConfig,
    SubCmd,
    build_common_components,
    main,
)
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import (
    AdamWConfig,
    LinearWithWarmup,
    OptimConfig,
    OptimGroupOverride,
)
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    SchedulerCallback,
    WandBCallback,
)
from olmo_core.train.common import Duration

SEQUENCE_LENGTH = 2048


# =========================================================
# =                 "COMMON COMPONENTS" STUFF             =
# =========================================================


class Love2CodeDataMix(DataMixBase):
    """Defines love2code mix. To create a new mix, make a new file in this folder
        and its name (without the '.txt' extension) below.

    NOTE FOR DG/PW: I based this off of the OLMo2-32B-anneal.py AnnealingDataMix class
    """

    love2code_mix = "love2code_data_XB.txt"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        assert tokenizer == TokenizerName.dolma2
        paths = []
        labels = []

        with open(f"src/scripts/train/learn2code/{self}.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(f"{base_dir}{line}")
                labels.append(line.split("/")[1])

        return paths, labels


def build_love2code_common(
    script: str, cmd: SubCmd, run_name: str, overrides: List[str], *, global_batch_size: int
) -> CommonComponents:
    """Note for DG/PW:

    I took the original internal.experiment.build_common_components and ran it
    and then built the dataset config and hotswapped it. Maybe not canonical, but seems easiest?

    Also hotswapped the CosWithWarmup scheduler to a LinearWithWarmup
    """
    og_common = build_common_components(
        script, cmd, run_name, cluster, overrides, global_batch_size=global_batch_size
    )
    tokenizer_config = og_common.tokenizer
    root_dir = get_root_dir(cluster)

    dataset = NumpyDatasetConfig.from_data_mix(
        Love2CodeDataMix.love2code_mix,
        tokenizer=tokenizer,
        mix_base_dir=root_dir,
        sequence_length=SEQUENCE_LENGTH,
        work_dir=get_work_dir(root_dir),
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=34521, num_workers=16
    )

    og_common.data_loader = data_loader_config

    og_common.callbacks["lr_scheduler"] = SchedulerCallback(
        scheduler=LinearWithWarmup(warmup_steps=2000)
    )
    return og_common


# =========================================================
# =                       MODEL STUFF                     =
# =========================================================


def build_model_config(common: CommonComponents) -> TransformerConfig:
    """
    Note for DG/PW: Created a custom model (basically just copied what Pete sent me in slack)
    """
    return TransformerConfig.love2code_3B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        compile=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )


# =========================================================
# =                      TRAINING STUFF                   =
# =========================================================


def build_optim_config(common: CommonComponents) -> AdamWConfig:
    # Note for DG/PW: Completely left unchanged from 1B training script
    del common
    return AdamWConfig(
        lr=12e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Note for DG/PW: The only think I changed here was to add the max_duration in TrainerConfig"""

    num_ne_params = model_config_builder(common).num_non_embedding_params
    CHINCHILLA_5X_DURATION = Duration(num_ne_params * 20 * 5)

    return (
        TrainerConfig(
            save_folder=common.save_folder,
            rank_microbatch_size=8 * 4096,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            z_loss_multiplier=1e-5,
            compile_loss=True,
            max_duration=CHINCHILLA_5X_DURATION,  # <--- this line
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="love2code-3B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="love2code-3B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


# =============================================================
# =                    FULL CONFIG BUILDER                    =
# =============================================================


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
    """The only change here is that I used my custom CommonComponents
    with the correct love2code dataset
    """
    common = build_love2code_common(  # <--- this line
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
    )

    if finalize_config is not None:
        finalize_config(config)

    config = config.merge(overrides)

    if config.model.float8_config is not None and config.model.float8_config.enabled:
        config.trainer.add_callback(
            "float8_handler", Float8HandlerCallback(config=config.model.float8_config)
        )

    return config


# ========================================================
# =                      MAIN BLOCK                      =
# ========================================================

""" Completely unchanged from internal.experiment.main
    (but implicitly uses the `build_config` call I defined just above)
"""


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
        optim_config_builder=optim_config_builder,
        trainer_config_builder=trainer_config_builder,
        finalize_config=finalize_config,
    )

    cmd.run(config)


if __name__ == "__main__":
    main(
        global_batch_size=1024 * 4096,
        model_config_builder=build_model_config,
        optim_config_builder=build_optim_config,
        trainer_config_builder=build_trainer_config,
    )
