import math
import sys
from datetime import datetime
from pathlib import Path
from typing import override

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import (
    CliContext,
    ExperimentConfig,
    get_beaker_username,
)
from olmo_core.internal.experiment import (
    main as olmo_core_main,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.model_ladder.wsds_chinchilla_run_configurator import WSDSChinchillaRunConfigurator
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import DurationUnit
from olmo_core.train.train_module import TransformerTrainModuleConfig

# Change these to match the config you want to use
SEQ_LENGTH = 8192
CHINCHILLA_MULTIPLE = 5
SEED = 1337
TOKENIZER_CONFIG = TokenizerConfig.dolma2()
PRIORITY = "high"
WORKSPACE = "ai2/olmo4"
BUDGET = "ai2/oe-base"
NUM_NODES = 4
LOAD_PATH = "gs://ai2-llm/checkpoints/OLMo25/step1413814"
BASE_SAVE_DIR = "s3://ai2-llm/checkpoints"


MODEL_CONFIG = TransformerConfig.olmo2_1B_v2(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())

DATASET_CONFIG = SourceMixtureList(
    sources=[
        SourceMixtureConfig(
            source_name="all-dressed-snazzy2-v0.1-150b",
            target_ratio=0.5,
            paths=[
                f"s3://ai2-llm/preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER_CONFIG.identifier}/all-dressed-snazzy2/**/*.npy"
            ],
        ),
        SourceMixtureConfig(
            source_name="spring2code-countup_criteria_v2-python",
            target_ratio=0.5,
            max_repetition_ratio=5,
            paths=[
                f"s3://ai2-llm/preprocessed/the-stack-v2/spring2code_v2/minhash_v2_annotated_reshard_qc_tagged_filtered/Python/{TOKENIZER_CONFIG.identifier}/*.npy"
            ],
        ),
    ]
)


class AnyChinchillaRunConfigurator(WSDSChinchillaRunConfigurator):
    def __post_init__(self):
        # this allows setting whatever multiplier we want, not just power of 2.
        pass

    @override
    def configure_chinchilla_periods(self, num_params: int) -> tuple[int, list[float]]:
        return num_params, [self.chinchilla_multiple]


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{BASE_SAVE_DIR}/{get_beaker_username()}/{cli_context.run_name}"

    beaker_launch_config: BeakerLaunchConfig | None = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace=WORKSPACE,
        num_nodes=NUM_NODES,
        nccl_debug=True,
    )
    beaker_launch_config.priority = PRIORITY

    model_num_params = MODEL_CONFIG.num_non_embedding_params
    chinchilla_config = AnyChinchillaRunConfigurator(chinchilla_multiple=CHINCHILLA_MULTIPLE)
    chinchilla_global_batch_size = chinchilla_config.configure_target_batch_size(model_num_params)

    # adjust global batch size to closest power of 2
    rounded_global_batch_size = 2 ** math.ceil(math.log2(chinchilla_global_batch_size))

    max_duration = chinchilla_config.configure_duration(
        num_params=model_num_params,
        batch_size=chinchilla_global_batch_size,
    )
    assert max_duration.unit == DurationUnit.tokens, "Duration unit should be tokens!"

    DATASET_CONFIG.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=DATASET_CONFIG,
            requested_tokens=max_duration.value,
            global_batch_size=rounded_global_batch_size,
            processes=16,
            seed=SEED,
        ),
        tokenizer=TOKENIZER_CONFIG,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=rounded_global_batch_size, seed=SEED, num_workers=4
    )

    optim = chinchilla_config.configure_optimizer(
        num_params=model_num_params, batch_size=chinchilla_global_batch_size
    )
    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH * 2,
        learning_rate=optim.lr,
        scheduler=chinchilla_config.configure_lr_scheduler(
            num_params=model_num_params, batch_size=chinchilla_global_batch_size
        ),
        activation_memory_budget=0.5,
    )

    trainer_config = cookbook.configure_trainer(
        max_duration=max_duration,
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    ).with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts,
            wandb_group_name=cli_context.run_name,
            checkpoint_save_interval=None,
            ephemeral_checkpoint_save_interval=250,
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=MODEL_CONFIG,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


def main():
    # TWO CASES:
    # 1. len(sys.argv) < 4: definitely i have to add the run name here
    # 2. len(sys.argv) >= 4: depends on whether the first 4 all start NOT
    #                        with --: if they do, then i already have the
    #                        run name. but if any of the first 4 start
    #                        with --, then it means that some are already
    #                        overrides, so i need to add the run name in.
    if len(sys.argv) < 4 or any(arg.startswith("--") for arg in sys.argv[:4]):
        sys.argv.insert(2, Path(__file__).stem)

    # now i can just call the main function
    return olmo_core_main(config_builder=build_experiment_config)


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            uv run src/.../50web_alldressed_v2_50stack_edu_python.py dry_run

        To launch a training run on Augusta w/ 8 nodes:
            uv run src/.../50web_alldressed_v2_50stack_edu_python.py launch ai2/augusta \
                --launch.num_nodes=8 \
                --launch.priority=high
    """
    main()
