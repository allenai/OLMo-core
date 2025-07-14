"""
Official training script for OLMo-2-0325-32B-anneal, meant to be launched with torchrun.
"""

import json
import logging
import sys
from dataclasses import dataclass
from typing import List, Tuple, cast

import torch

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMixBase,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    TokenizerConfig,
    TokenizerName,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.io import resource_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
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
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

SEQUENCE_LENGTH = 4096

# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
DATA_ROOT = "http://olmo-data.org"
WORK_DIR = "/tmp/olmo-core/dataset-cache"
SAVE_ROOT = "/tmp/olmo-core/runs"  # NOTE: change this to what you want


class AnnealingDataMix(DataMixBase):
    """
    Defines the annealing mixes. To create a new mix, make a new file in this folder and add its
    name (without the '.txt' extension) below.
    """

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
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536

    @classmethod
    def build(
        cls,
        *,
        run_name: str,
        checkpoint: str,
        overrides: List[str],
    ) -> "AnnealingConfig":
        tokenizer_config = TokenizerConfig.dolma2()

        # Get step number and max steps to infer where the learning rate left off.
        train_state = torch.load(
            resource_path(f"{checkpoint}/train", "rank0.pt"), weights_only=False
        )
        last_pretrain_step: int = train_state["global_step"]
        max_pretrain_steps: int = train_state.get("max_steps", 774861)  # default found in logs
        log.info(
            f"Will anneal from checkpoint at step {last_pretrain_step:,d} of {max_pretrain_steps:,d}"
        )

        # Now infer the learning rate.
        with resource_path(checkpoint, "config.json").open() as f:
            config = json.load(f)
        base_lr = config["optim"]["lr"]
        scheduler_config = config["trainer"]["callbacks"]["lr_scheduler"]["scheduler"]
        # assert scheduler_config.pop("_CLASS_") == LinearWithWarmup.__name__
        assert scheduler_config.pop("_CLASS_").endswith(LinearWithWarmup.__name__)
        scheduler = LinearWithWarmup(**scheduler_config)
        starting_lr = float(scheduler.get_lr(base_lr, last_pretrain_step, max_pretrain_steps))

        run_name = f"peteish32-from{last_pretrain_step}-{run_name}"

        config = AnnealingConfig(
            run_name=f"olmo2-anneal-{run_name}",
            model=TransformerConfig.olmo2_32B(vocab_size=tokenizer_config.padded_vocab_size()),
            dataset=NumpyDatasetConfig.from_data_mix(
                AnnealingDataMix.dolmino100,
                tokenizer=tokenizer_config,
                mix_base_dir=DATA_ROOT,
                sequence_length=SEQUENCE_LENGTH,
                work_dir=WORK_DIR,
            ),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=2048
                * SEQUENCE_LENGTH,  # NOTE: this is specified in TOKENS, not instances.
                seed=34521,  # NOTE: can update this to change data order.
                num_workers=4,
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=2
                * SEQUENCE_LENGTH,  # NOTE: again this is specified in tokens.
                max_sequence_length=SEQUENCE_LENGTH,
                z_loss_multiplier=1e-5,
                compile_model=True,
                optim=SkipStepAdamWConfig(
                    lr=starting_lr,
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
                    name=DataParallelType.fsdp,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                    num_replicas=get_world_size() // 64,  # NOTE: tune this,
                ),
                ac_config=TransformerActivationCheckpointingConfig(
                    mode=TransformerActivationCheckpointingMode.selected_modules,
                    modules=["blocks.*.feed_forward"],
                ),
                scheduler=LinearWithWarmup(
                    warmup_steps=0,
                    alpha_f=0.0,
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"{SAVE_ROOT}/{run_name}",
                load_strategy=LoadStrategy.always,
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=8, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.tokens(int(100e9)),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=1000,
                    ephemeral_save_interval=500,
                    save_async=True,
                ),
            )
            .with_callback(
                "comet",
                CometCallback(
                    name=run_name,
                    enabled=False,  # NOTE: change to true to enable
                    cancel_check_interval=10,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    enabled=False,  # NOTE: change to true to enable
                    cancel_check_interval=10,
                ),
            )
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "downstream_evaluator",
                DownstreamEvaluatorCallbackConfig(
                    tasks=[
                        # MMLU for backwards compatibility
                        "mmlu_stem_mc_5shot",
                        "mmlu_humanities_mc_5shot",
                        "mmlu_social_sciences_mc_5shot",
                        "mmlu_other_mc_5shot",
                        # MMLU test
                        "mmlu_stem_mc_5shot_test",
                        "mmlu_humanities_mc_5shot_test",
                        "mmlu_social_sciences_mc_5shot_test",
                        "mmlu_other_mc_5shot_test",
                        ## Core 12 tasks for backwards compatibility
                        # "arc_challenge",
                        # "arc_easy",
                        # "basic_arithmetic",
                        # "boolq",
                        # "commonsense_qa",
                        # "copa",
                        # "hellaswag",
                        # "openbook_qa",
                        # "piqa",
                        # "sciq",
                        # "social_iqa",
                        # "winogrande",
                        ## Core 12 tasks 5-shot
                        # "arc_challenge_rc_5shot",
                        # "arc_easy_rc_5shot",
                        ## "basic_arithmetic_rc_5shot",  # doesn't exist
                        ## "boolq_rc_5shot",  # we don't like it
                        # "csqa_rc_5shot",
                        ## "copa_rc_5shot",  # doesn't exist
                        # "hellaswag_rc_5shot",
                        # "openbookqa_rc_5shot",
                        # "piqa_rc_5shot",
                        ## "sciq_rc_5shot",  # doesn't exist
                        # "socialiqa_rc_5shot",
                        # "winogrande_rc_5shot",
                        ## New in-loop evals
                        # "arc_challenge_val_rc_5shot",
                        # "arc_challenge_val_mc_5shot",
                        "arc_challenge_test_rc_5shot",
                        # "arc_challenge_test_mc_5shot",
                        # "arc_easy_val_rc_5shot",
                        # "arc_easy_val_mc_5shot",
                        "arc_easy_test_rc_5shot",
                        # "arc_easy_test_mc_5shot",
                        # "boolq_val_rc_5shot",
                        # "boolq_val_mc_5shot",
                        "csqa_val_rc_5shot",
                        # "csqa_val_mc_5shot",
                        "hellaswag_val_rc_5shot",
                        # "hellaswag_val_mc_5shot",
                        # "openbookqa_val_rc_5shot",
                        # "openbookqa_val_mc_5shot",
                        "openbookqa_test_rc_5shot",
                        # "openbookqa_test_mc_5shot",
                        "piqa_val_rc_5shot",
                        # "piqa_val_mc_5shot",
                        "socialiqa_val_rc_5shot",
                        # "socialiqa_val_mc_5shot",
                        # "winogrande_val_rc_5shot",
                        # "winogrande_val_mc_5shot",
                        # "mmlu_stem_val_rc_5shot",
                        # "mmlu_stem_val_mc_5shot",
                        # "mmlu_humanities_val_rc_5shot",
                        # "mmlu_humanities_val_mc_5shot",
                        # "mmlu_social_sciences_val_rc_5shot",
                        # "mmlu_social_sciences_val_mc_5shot",
                        # "mmlu_other_val_rc_5shot",
                        # "mmlu_other_val_mc_5shot",
                    ],
                    tokenizer=tokenizer_config,
                    eval_interval=1000,
                ),
            ),
        ).merge(overrides)

        # Make sure this is an 'AnnealingDataMix' instance.
        config.dataset.mix = AnnealingDataMix(config.dataset.mix)
        return config


def train(checkpoint: str, config: AnnealingConfig):
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

    # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
    if not trainer.maybe_load_checkpoint(trainer.save_folder):
        trainer.load_checkpoint(checkpoint, load_trainer_state=False)

    # Train.
    trainer.fit()


def main():
    if len(sys.argv) < 3:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} run_name checkpoint [OVERRIDES...]")
        sys.exit(1)

    run_name, checkpoint, *overrides = sys.argv[1:]
    prepare_training_environment()
    try:
        config = AnnealingConfig.build(
            run_name=run_name,
            checkpoint=checkpoint,
            overrides=overrides,
        )
        train(checkpoint, config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
