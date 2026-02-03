import sys
from datetime import datetime
from pathlib import Path

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
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

# Change these to match the config you want to use
SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
MAX_TOKENS = 10_000_000_000  # 10B
LR = 0.00020712352850360292 / 2  # halfing the LR
SEED = 1337
TOKENIZER_CONFIG = TokenizerConfig.dolma2()
PRIORITY = "high"
WORKSPACE = "ai2/olmo4"
BUDGET = "ai2/oe-base"
NUM_NODES = 4
LOAD_PATH = "gs://ai2-llm/checkpoints/OLMo25/step1413814"
BASE_SAVE_DIR = "s3://ai2-llm/checkpoints"


DOLMA_3_PREFIX = "s3://ai2-llm/preprocessed/dolma2-0625/v0.1-150b"
SPRING2CODE_PREFIX = "s3://ai2-llm/preprocessed/the-stack-v2/spring2code_v2/minhash_filter_v2_2026_stack_edu_redux_tagged_partitioned"

MODEL_CONFIG = TransformerConfig.olmo3_7B(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())

DATASET_CONFIG = SourceMixtureList(
    sources=[
        SourceMixtureConfig(
            source_name="all-dressed-snazzy2-v0.1-150b",
            target_ratio=0.5,
            paths=[f"{DOLMA_3_PREFIX}/{TOKENIZER_CONFIG.identifier}/all-dressed-snazzy2/**/*.npy"],
        ),
        SourceMixtureConfig(
            source_name="stack-edu-v0.1-150b",
            target_ratio=0.5,
            paths=[
                # C: about 18.75 GB, a little bit less than 19.15 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/C/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # C++: about 31.64 GB, a little bit more than 29.23 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/C++/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C++/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C++/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # C-Sharp: about 50.54 GB, a little bit less than 50.71 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p50/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p55/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/C-Sharp/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Go: about 5.42 GB, a little bit less than 5.66 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Go/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Go/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Go/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Go/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Java: about 131.78 GB, a little bit more than 127.03 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Java/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Java/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Java/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Java/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Java/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Java/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Java/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # JavaScript: about 35.13 GB, a little bit less than 36.03 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/JavaScript/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/JavaScript/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/JavaScript/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/JavaScript/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/JavaScript/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Markdown: about 119.48 GB, a little bit more than 116.42 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p30/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p35/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p40/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p45/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p50/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p55/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Markdown/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # PHP: about 30.79 GB, a little bit more than 29.94 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/PHP/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/PHP/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/PHP/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/PHP/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Python: about 71.65 GB, a little bit less than 72.41 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Python/quality_p55/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Python/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Ruby: about 5.20 GB, a little bit less than 5.65 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Ruby/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Ruby/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Ruby/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Ruby/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Ruby/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Rust: about 5.80 GB, a little bit more than 5.71 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Rust/quality_p50/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p55/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Rust/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Shell: about 9.76 GB, a little bit less than 10.32 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Shell/quality_p15/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p20/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p25/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p30/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p35/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p40/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p45/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p50/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p55/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Shell/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # SQL: about 28.22 GB, a little bit less than 28.34 GB in stack-edu (repeat twice)
                f"{SPRING2CODE_PREFIX}/SQL/quality_p20/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p20/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p25/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p25/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p30/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p30/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p35/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p35/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p40/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p40/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p45/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p45/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p50/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p50/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p55/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p55/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/SQL/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # Swift: about 6.68 GB, a little bit more than 6.13 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/Swift/quality_p60/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Swift/quality_p65/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Swift/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Swift/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Swift/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Swift/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Swift/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/Swift/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
                # TypeScript: about 10.61 GB, a little bit less than 10.14 GB in stack-edu
                f"{SPRING2CODE_PREFIX}/TypeScript/quality_p70/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/TypeScript/quality_p75/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/TypeScript/quality_p80/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/TypeScript/quality_p85/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/TypeScript/quality_p90/{TOKENIZER_CONFIG.identifier}/*.npy",
                f"{SPRING2CODE_PREFIX}/TypeScript/quality_p95/{TOKENIZER_CONFIG.identifier}/*.npy",
            ],
        ),
    ]
)


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

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH * 2,
        learning_rate=LR,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
        activation_memory_budget=0.5,
    )

    DATASET_CONFIG.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=DATASET_CONFIG,
            requested_tokens=MAX_TOKENS,
            global_batch_size=GLOBAL_BATCH_SIZE,
            processes=16,
            seed=SEED,
        ),
        tokenizer=TOKENIZER_CONFIG,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path=LOAD_PATH,
        load_trainer_state=False,
        load_optim_state=True,
        max_duration=Duration.tokens(MAX_TOKENS),
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
