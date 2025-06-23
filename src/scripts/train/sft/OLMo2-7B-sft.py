"""
This script can be used to launch an SFT run for the 7B model on Beaker.
Run the script without any arguments to see usage info.

Getting started:
1. Ensure that the beaker workspace you are using (defaults to ai2/olmo-instruct) has the following secrets configured.
    * {beaker_username}_BEAKER_TOKEN
    * {beaker_username}_AWS_CREDENTIALS
    * {beaker_username}_AWS_CONFIG
    * {beaker_username}_WANDB_API_KEY  (optional)
    * {beaker_username}_COMET_API_KEY  (optional)

2. Ensure that the dataset and model you want to use are available on the correct storage bucket for the cluster you are using.
    * For example, if you are using ai2/jupiter-cirrascale-2, the dataset and model should be available on the /weka/oe-training-default/ai2-llm bucket.
    * If you are using ai2/augusta-google-1, the dataset and model should be available on the gs://ai2-llm bucket.
    * Data can be copied to/from gcs/weka using the `gsutil` command line tool. E.g., `gsutil cp -r /path/to/dataset gs://ai2-llm/path/to/dataset`

Training:
1. Launch this script as per the instructions below. Some notable differences from open-instruct:
    * Olmo-core loads data differently. This script is configured to minimize the amount of padding
      tokens processed by packing documents together into a single sequence, and masking attention so that
      they don't attend to each other. This results in ~0.5% of the total tokens being padding tokens.
    *

Evaluation:
# TODO: instructions for hf conversion and launching oe-eval
"""

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, cast

import rich
from rich import print

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    TokenizerConfig,
    TokenizerName,
)
from olmo_core.data.types import LongDocStrategy, NumpyDatasetType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank
from olmo_core.internal.common import (
    CLUSTER_TO_GPU_TYPE,
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

# TODO: make seq len, batch size, and datasets more configurable
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 64 * SEQUENCE_LENGTH

NUM_GPUS = 8
NUM_NODES = math.ceil(NUM_GPUS / 8)


def build_sft_dataset(root_dir: str, tokenizer_config: TokenizerConfig) -> NumpyDatasetConfig:
    if tokenizer_config.identifier == TokenizerName.dolma2:
        tokenizer_id = TokenizerName.dolma2.split("/")[-1]  # eg "dolma2-tokenizer"
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_config.identifier}")

    # NOTE: dataset path can be configured relative to root_dir
    # root_dir is /weka/oe-training-default/ai2-llm or gs://ai2-llm depending on the cluster
    sft_datasets = [
        "tylerr/sft/jacobmorrison-OpenThoughts3-1.2M-no-cot",
    ]

    paths, label_mask_paths = [], []
    root_path = Path(root_dir)
    for sft_dataset in sft_datasets:
        dataset_dir = root_path / sft_dataset / tokenizer_id
        token_files = sorted(dataset_dir.glob("token_ids*.npy"))
        label_files = sorted(dataset_dir.glob("labels*.npy"))

        paths.extend([str(f) for f in token_files])
        label_mask_paths.extend([str(f) for f in label_files])

    intra_document_masking = True
    return NumpyDatasetConfig(
        # general config
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        paths=paths,
        label_mask_paths=label_mask_paths,
        # how to handle long docs?
        name=NumpyDatasetType.packed_fsl,  # concatenated short docs into a single sequence...
        generate_doc_lengths=intra_document_masking,  # ...and mask attention so that they don't attend to each other
        long_doc_strategy=LongDocStrategy.truncate,  # truncate docs...
        sequence_length=SEQUENCE_LENGTH,  # ...over this length
    )


@dataclass
class SFTConfig(Config):
    """
    Custom config class for the sft run.

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
    init_seed: int = 53184

    @classmethod
    def build(
        cls,
        *,
        script: str,
        cmd: str,
        run_name: str,
        checkpoint: str,
        cluster: str,
        overrides: List[str],
    ) -> "SFTConfig":
        root_dir = get_root_dir(cluster)
        user_name = get_beaker_username()
        tokenizer_config = TokenizerConfig.dolma2()

        rank_microbatch_size = GLOBAL_BATCH_SIZE // NUM_GPUS // 2
        if "B200" in CLUSTER_TO_GPU_TYPE.get(cluster, "unknown"):
            rank_microbatch_size *= 2

        config = SFTConfig(
            run_name=run_name,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[script, cmd, run_name, checkpoint, cluster, *overrides],
                cluster=cluster,
                num_nodes=NUM_NODES,
                budget="ai2/oe-adapt",
                workspace="ai2/olmo-instruct",
            ),
            model=TransformerConfig.olmo2_7B(  # Based on https://github.com/allenai/OLMo-core/blob/dustins/anneal-repro/src/scripts/train/lc_cont_train/OLMo2-7B-lc_anneal_tp4.py
                vocab_size=tokenizer_config.padded_vocab_size(),
                use_flash=True,
                rope_theta=8 * 10**6,
            ),
            dataset=build_sft_dataset(root_dir, tokenizer_config),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=GLOBAL_BATCH_SIZE,  # specified in tokens, not instances/sequences
                seed=34521,
                num_workers=4,
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=rank_microbatch_size,
                max_sequence_length=SEQUENCE_LENGTH,
                z_loss_multiplier=1e-5,
                compile_model=True,
                optim=SkipStepAdamWConfig(
                    lr=8e-05,
                    weight_decay=0.0,  # NOTE: different from pretraining
                    betas=(0.9, 0.95),
                    compile=False,
                ),
                dp_config=TransformerDataParallelConfig(
                    name=DataParallelType.fsdp,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                ),
                ac_config=TransformerActivationCheckpointingConfig(
                    mode=TransformerActivationCheckpointingMode.selected_modules,
                    modules=["blocks.*.feed_forward"],
                ),
                scheduler=LinearWithWarmup(
                    warmup_fraction=0.03,
                    alpha_f=0.0,  # lr drops all the way to 0.0 at the end
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"/weka/oe-training-default/ai2-llm/checkpoints/{user_name}/olmo2-7B-sft/{run_name}",
                load_strategy=LoadStrategy.never,  # we manually load the checkpoint below
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.epochs(3),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=1000, ephemeral_save_interval=500, save_async=True
                ),
            )
            .with_callback(
                "downstream_evaluator",
                DownstreamEvaluatorCallbackConfig(
                    # WARN: this is hacked together w/o really comparing to the evals configured in open-instruct
                    # In order to actually evaluate the SFT model, use oe-eval directly.
                    tasks=[
                        # MMLU MC
                        "mmlu_stem_mc_5shot",
                        "mmlu_humanities_mc_5shot",
                        "mmlu_social_sciences_mc_5shot",
                        "mmlu_other_mc_5shot",
                        "mmlu_stem_mc_5shot_test",
                        "mmlu_humanities_mc_5shot_test",
                        "mmlu_social_sciences_mc_5shot_test",
                        "mmlu_other_mc_5shot_test",
                        # Gen tasks BPB
                        "gsm8k_gold_bpb_5shot",
                        "minerva_math_algebra_gold_bpb_0shot",
                        "minerva_math_counting_and_probability_gold_bpb_0shot",
                        "minerva_math_geometry_gold_bpb_0shot",
                        "minerva_math_intermediate_algebra_gold_bpb_0shot",
                        "minerva_math_number_theory_gold_bpb_0shot",
                        "minerva_math_prealgebra_gold_bpb_0shot",
                        "minerva_math_precalculus_gold_bpb_0shot",
                        "codex_humaneval_gold_bpb_3shot",
                        "codex_mbpp_gold_bpb_3shot",
                    ],
                    tokenizer=tokenizer_config,
                    eval_interval=500,
                    enabled=True,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    entity="ai2-llm",
                    project=f"{user_name}-7B-sft",
                    enabled=False,
                    cancel_check_interval=10,
                ),
            )
            .with_callback(
                "comet",
                CometCallback(
                    name=run_name,
                    workspace="ai2",
                    project=f"{user_name}-7B-sft",
                    enabled=False,
                    cancel_check_interval=10,
                ),
            ),
        ).merge(overrides)

        return config


def train(checkpoint: str, config: SFTConfig):
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
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
    log.info("Loading checkpoint...")
    if not trainer.maybe_load_checkpoint(trainer.save_folder):
        log.info(
            f"No checkpoint found in save folder '{trainer.save_folder}', attempting to load from pretraining checkpoint '{checkpoint}'"
        )
        trainer.load_checkpoint(checkpoint, load_trainer_state=False)
    else:
        log.info(f"Loaded checkpoint from save folder '{trainer.save_folder}'")

    # Train.
    trainer.fit()


if __name__ == "__main__":
    USAGE = f"""
SFT the 7B model.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]launch|train|dry_run[/] [i b]RUN_NAME PRETRAIN_CHECKPOINT CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use the [b magenta]launch[/] cmd to submit it to Beaker or run it via torchrun if you know what you're doing.
[b magenta]dry_run:[/]     Print the config for debugging.

[b]Arguments[/]
[b]RUN_NAME:[/]            The name of the run. Used for the run name in W&B/Comet and the checkpoint dir.
[b]PRETRAIN_CHECKPOINT:[/] Path to the pretraining checkpoint to load.
[b]CLUSTER:[/]             The Beaker cluster to use (e.g., 'ai2/jupiter-cirrascale-2').
[b]OVERRIDES:[/]           Optional key=value overrides for the config. Use dot notation for nested fields.
                           E.g., `--launch.priority=high`, `--trainer.callbacks.wandb.enabled=True`, or `--train_module.optim.lr=1e-5`

[b]Examples[/]
$ [i]python {sys.argv[0]} dry_run test /path/to/ckpt ai2/cluster
$ [i]python {sys.argv[0]} launch run01 /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921 ai2/jupiter-cirrascale-2 --launch.priority=high[/]
""".strip()

    # Parse command line arguments.
    if len(sys.argv) < 5 or sys.argv[1] not in ("launch", "train", "dry_run"):
        rich.get_console().print(USAGE, highlight=False)
        sys.exit(1)

    script, cmd, run_name, checkpoint, cluster, *overrides = sys.argv

    # Prepare the environment for the given command.
    if cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif cmd == "train":
        prepare_training_environment()
    else:
        raise NotImplementedError(cmd)

    # Build the config, applying any overrides.
    config = SFTConfig.build(
        script=script,
        cmd="train",
        run_name=run_name,
        checkpoint=checkpoint,
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
            train(checkpoint, config)
        finally:
            teardown_training_environment()
    else:
        raise NotImplementedError(cmd)
