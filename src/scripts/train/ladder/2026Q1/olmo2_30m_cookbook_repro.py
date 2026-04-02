"""
Reproduce the original cookbook 30M runs using olmo-core directly.

Model architecture, optimizer, LR schedule, and batch size match the cookbook exactly.
In-loop evals use the same setup as the model ladder (v3_small_ppl_validation + downstream).

Two data pipeline options:
  --pipeline=new  composable pipeline (MixingInstanceSource) [default]
  --pipeline=old  NumpyFSL pipeline with SourceMixture + chunk_based_mixture (matches cookbook)

Usage:
  python <script> dry_run  <run_name> --treatment=... --budget=... [--pipeline=...] [OVERRIDES...]
  python <script> launch   <run_name> --treatment=... --budget=... [--pipeline=...] [OVERRIDES...]

Examples:
  python <script> launch my-baseline-1xC --treatment=baseline --budget=1xC --pipeline=old
  python <script> launch my-icl-overlap-3xC --treatment=icl_overlap --budget=3xC --pipeline=old
"""

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    InstanceSourceConfig,
    MixingInstanceSourceConfig,
    MixingInstanceSourceSpecConfig,
    NumpyDocumentSourceConfig,
    set_composable_seed,
)
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    CosWithWarmupAndLinearDecay,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    MetricSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all
import olmo_core.eval.task_groups as task_groups

log = logging.getLogger(__name__)

# ── Model (matches cookbook olmo_30m / olmo-core olmo2_30M) ──
SEQUENCE_LENGTH = 2048
TOKENIZER = TokenizerConfig.dolma2()
MODEL_CONFIG = TransformerConfig.olmo2_30M(vocab_size=TOKENIZER.padded_vocab_size())

# ── Optimizer (matches cookbook: SkipStepAdamW, wd=0.033, betas=(0.9, 0.95)) ──
LEARNING_RATE = 0.007276622186288963
OPTIMIZER = SkipStepAdamWConfig(
    lr=LEARNING_RATE,
    weight_decay=0.033,
    betas=(0.9, 0.95),
    group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
)

# ── Schedule (matches cookbook: CosWithWarmupAndLinearDecay, 400 warmup) ──
SCHEDULER = CosWithWarmupAndLinearDecay(warmup_steps=400)

# ── Batch size (matches cookbook: 131072 tokens = 64 sequences of 2048) ──
GLOBAL_BATCH_SIZE = 131072
RANK_MICROBATCH_SIZE = 8 * SEQUENCE_LENGTH  # 16384 tokens (fits 8-GPU DP)

# ── Token budgets ──
TOKENS_1XC = 582_046_720
TOKEN_BUDGETS = {"1xC": TOKENS_1XC, "3xC": 3 * TOKENS_1XC, "5xC": 5 * TOKENS_1XC}

# ── Data paths ──
ICL_OVERLAP_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v01/icl-overlap-max-suffix-2048-eos-fix-500B-sample/allenai/dolma2-tokenizer/*.npy",
]

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/s2pdf_redacted/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]

# ── Beaker ──
CLUSTER = "ai2/saturn"
NUM_NODES = 1
SEED = 1337
EVAL_INTERVAL = 1000


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: TransformerConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = SEED
    # New composable pipeline (populated when --pipeline=new)
    instance_sources: Optional[List[InstanceSourceConfig]] = None
    composable_data_loader: Optional[ComposableDataLoaderConfig] = None
    # Old NumpyFSL pipeline (populated when --pipeline=old)
    dataset: Optional[NumpyFSLDatasetConfig] = None
    numpy_data_loader: Optional[NumpyDataLoaderConfig] = None


def _build_composable_data(treatment: str):
    if treatment == "baseline":
        sources = [
            ConcatAndChunkInstanceSourceConfig(
                sources=[NumpyDocumentSourceConfig(source_paths=DOLMA2_BASELINE_PATHS, tokenizer=TOKENIZER)],
                sequence_length=SEQUENCE_LENGTH,
            ),
        ]
    else:
        sources = [
            MixingInstanceSourceConfig(
                source_specs=[
                    MixingInstanceSourceSpecConfig(
                        source=ConcatAndChunkInstanceSourceConfig(
                            sources=[NumpyDocumentSourceConfig(source_paths=ICL_OVERLAP_PATHS, tokenizer=TOKENIZER)],
                            sequence_length=SEQUENCE_LENGTH,
                        ),
                        ratio=0.5,
                        label="icl-overlap",
                    ),
                    MixingInstanceSourceSpecConfig(
                        source=ConcatAndChunkInstanceSourceConfig(
                            sources=[NumpyDocumentSourceConfig(source_paths=DOLMA2_BASELINE_PATHS, tokenizer=TOKENIZER)],
                            sequence_length=SEQUENCE_LENGTH,
                        ),
                        ratio=0.5,
                        label="baseline",
                    ),
                ],
            ),
        ]
    loader = ComposableDataLoaderConfig(num_workers=8, instance_filter_config=InstanceFilterConfig())
    return sources, loader


def _build_numpy_fsl_data(treatment: str, token_budget: int, work_dir: str):
    """Matches the cookbook's NumpyFSL pipeline with chunk_based_mixture for mixtures."""
    if treatment == "baseline":
        dataset = NumpyFSLDatasetConfig.glob(
            *DOLMA2_BASELINE_PATHS,
            sequence_length=SEQUENCE_LENGTH,
            tokenizer=TOKENIZER,
            work_dir=work_dir,
        )
    else:
        src_mix = SourceMixtureDatasetConfig(
            source_list=SourceMixtureList(sources=[
                SourceMixtureConfig(
                    source_name="icl-overlap",
                    paths=ICL_OVERLAP_PATHS,
                    target_ratio=0.5,
                ),
                SourceMixtureConfig(
                    source_name="baseline",
                    paths=DOLMA2_BASELINE_PATHS,
                    target_ratio=0.5,
                ),
            ]),
            requested_tokens=token_budget,
            global_batch_size=GLOBAL_BATCH_SIZE,
            seed=SEED,
        )
        dataset = NumpyFSLDatasetConfig.from_src_mix(
            src_mix,
            sequence_length=SEQUENCE_LENGTH,
            tokenizer=TOKENIZER,
            work_dir=work_dir,
            chunk_based_mixture=True,
        )
    loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=SEED,
        num_workers=4,
        work_dir=work_dir,
    )
    return dataset, loader


def build_config(
    script: str, run_name: str, overrides: List[str],
    treatment: str, pipeline: str, budget: str,
) -> ExperimentConfig:
    root_dir = get_root_dir(CLUSTER)
    work_dir = get_work_dir(root_dir)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    token_budget = TOKEN_BUDGETS[budget]
    save_folder = f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}"
    steps_per_1xc = TOKENS_1XC // GLOBAL_BATCH_SIZE

    instance_sources = None
    composable_data_loader = None
    dataset = None
    numpy_data_loader = None

    if pipeline == "new":
        instance_sources, composable_data_loader = _build_composable_data(treatment)
    else:
        dataset, numpy_data_loader = _build_numpy_fsl_data(treatment, token_budget, work_dir)

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=OPTIMIZER,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=SCHEDULER,
    )

    trainer_config = TrainerConfig(
        save_folder=save_folder,
        work_dir=work_dir,
        save_overwrite=True,
        metrics_collect_interval=10,
        cancel_check_interval=5,
        max_duration=Duration.tokens(token_budget),
        callbacks={
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "config_saver": ConfigSaverCallback(),
            "garbage_collector": GarbageCollectorCallback(),
            "beaker": BeakerCallback(),
            "checkpointer": CheckpointerCallback(
                save_interval=steps_per_1xc,
                ephemeral_save_interval=250,
                save_async=True,
            ),
            "wandb": WandBCallback(
                name=run_name,
                group="suffix-train-30m-repro",
                project="suffix-train-30m-repro",
                cancel_check_interval=10,
                enabled=True,
            ),
            "lm_evaluator": LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=root_dir,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=TOKENIZER,
                    work_dir=work_dir,
                ),
                eval_interval=EVAL_INTERVAL,
            ),
            "downstream_evaluator": DownstreamEvaluatorCallbackConfig(
                tokenizer=TOKENIZER,
                tasks=sorted(task_groups.FULL_TASKS),
                eval_interval=EVAL_INTERVAL,
            ),
            "metric_saver": MetricSaverCallback(
                save_interval=EVAL_INTERVAL,
            ),
        },
    )

    config = ExperimentConfig(
        model=MODEL_CONFIG,
        train_module=train_module_config,
        trainer=trainer_config,
        instance_sources=instance_sources,
        composable_data_loader=composable_data_loader,
        dataset=dataset,
        numpy_data_loader=numpy_data_loader,
        launch=build_launch_config(
            name=run_name,
            root_dir=root_dir,
            cmd=[
                script, "train", run_name,
                f"--treatment={treatment}", f"--pipeline={pipeline}", f"--budget={budget}",
                *overrides,
            ],
            cluster=CLUSTER,
            workspace="ai2/oe-t-ladder",
            budget="ai2/oe-base",
            num_nodes=NUM_NODES,
        ),
    )
    config.launch.priority = "high"
    config.launch.preemptible = True
    return config.merge(overrides)


def train(config: ExperimentConfig):
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)

    if config.dataset is not None:
        dataset = config.dataset.build()
        data_loader = config.numpy_data_loader.build(
            dataset, dp_process_group=train_module.dp_process_group,
        )
    else:
        set_composable_seed(config.init_seed)
        work_dir = config.trainer.work_dir or "./dataset-cache"
        instance_sources = [s.build(work_dir=work_dir) for s in config.instance_sources]
        data_loader = config.composable_data_loader.build(
            *instance_sources,
            work_dir=work_dir,
            global_batch_size=GLOBAL_BATCH_SIZE,
            tokenizer=TOKENIZER,
        )

    trainer = config.trainer.build(train_module, data_loader)
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.step_soft_timeout = None
    config.launch.launch(follow=False)


if __name__ == "__main__":
    usage = f"""
Usage: python {sys.argv[0]} [dry_run|launch|train] RUN_NAME [OPTIONS...] [OVERRIDES...]

Options:
  --treatment=baseline|icl_overlap  Data treatment (required)
  --pipeline=new|old                Data pipeline (default: new)
  --budget=1xC|3xC|5xC             Token budget (default: 1xC)

Token budgets:
  1xC = {TOKENS_1XC:,} tokens
  3xC = {3*TOKENS_1XC:,} tokens
  5xC = {5*TOKENS_1XC:,} tokens
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script = sys.argv[0]
    cmd = sys.argv[1]
    run_name = sys.argv[2]

    treatment = "baseline"
    pipeline = "new"
    budget = "1xC"
    overrides = []
    for arg in sys.argv[3:]:
        if arg.startswith("--treatment="):
            treatment = arg.split("=", 1)[1]
        elif arg.startswith("--pipeline="):
            pipeline = arg.split("=", 1)[1]
        elif arg.startswith("--budget="):
            budget = arg.split("=", 1)[1]
        else:
            overrides.append(arg)

    valid_treatments = ("baseline", "icl_overlap")
    if treatment not in valid_treatments:
        print(f"Unknown treatment: {treatment}. Must be one of: {valid_treatments}")
        sys.exit(1)
    if pipeline not in ("new", "old"):
        print(f"Unknown pipeline: {pipeline}. Must be 'new' or 'old'")
        sys.exit(1)
    if budget not in TOKEN_BUDGETS:
        print(f"Unknown budget: {budget}. Must be one of: {list(TOKEN_BUDGETS.keys())}")
        sys.exit(1)

    if cmd == "train":
        prepare_training_environment()
    else:
        prepare_cli_environment()

    config = build_config(script, run_name, overrides, treatment, pipeline, budget)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        import rich
        rich.print(config)
    else:
        print(usage)
        sys.exit(1)
