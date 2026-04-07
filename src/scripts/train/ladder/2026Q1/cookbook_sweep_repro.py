"""
Reproduce the cookbook scaling sweep using olmo-core directly.

Supports the same model sizes (190M, 370M, 600M), compute budgets (0.5xC, 1xC, 2xC),
and treatments (baseline, icl_overlap) as the cookbook sweep.

Hyperparameter formulas match generate_configs.py exactly:
  LR:            0.0047 * (N / 108M) ** (-1/3)
  Batch size:    round(SEQ_LEN * 160 * (N / 108M) ** (2/3)), rounded to gpus*rmbs
  Token budget:  20 * N * budget_mult, rounded to batch multiple
  Warmup:        max(round(steps * 0.10), 100)
  Scheduler:     CosWithWarmupAndLinearDecay
  Optimizer:     SkipStepAdamW with step_increment_bugfix=False (matching overlap-pretrain branch)

Data pipeline uses NumpyFSL + SourceMixture + chunk_based_mixture (--pipeline=old).

Usage:
  python <script> dry_run  <run_name> --size=190M --treatment=baseline --budget=1xC
  python <script> launch   <run_name> --size=190M --treatment=icl_overlap --budget=0.5xC
  python <script> train    <run_name> --size=190M --treatment=baseline --budget=1xC
"""

import logging
import math
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
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig
from gantry.api import GitRepoState
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

# ── Constants ──
SEQ_LEN = 2048
TOKENIZER = TokenizerConfig.dolma2()
RANK_MICROBATCH_SIZE = 16 * SEQ_LEN  # 32768, matching cookbook
SEED = 1337
CLUSTER = "ai2/saturn"
NUM_NODES = 1
# OLMo-core branch for gantry to clone. Update OLMO_CORE_REF after subtree push.
OLMO_CORE_BRANCH = "ianm/overlap-pretrain-ladder"
OLMO_CORE_REF = ""  # Set via --olmo-core-ref= flag or hardcoded after subtree push

# ── Model specs (from generate_configs.py) ──
MODEL_SPECS = {
    "190M": {"non_emb": 190_354_176, "gpus": 4, "config_fn": TransformerConfig.olmo2_190M},
    "370M": {"non_emb": 371_262_464, "gpus": 8, "config_fn": TransformerConfig.olmo2_370M},
    "600M": {"non_emb": 462_466_368, "gpus": 8, "config_fn": TransformerConfig.olmo2_600M},
}

BUDGET_MULTS = {"0.5xC": 0.5, "1xC": 1.0, "2xC": 2.0}

# ── Data paths ──
ICL_OVERLAP_PATHS_500B = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v01/icl-overlap-max-suffix-2048-eos-fix-500B-sample/allenai/dolma2-tokenizer/*.npy",
]

ICL_OVERLAP_PATHS_3B = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v01/icl-overlap-max-suffix-2048-eos-fix/allenai/dolma2-tokenizer/*.npy",
]

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/s2pdf_redacted/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]


def compute_hyperparams(size: str, budget: str) -> dict:
    """Compute hyperparameters matching cookbook's generate_configs.py exactly."""
    spec = MODEL_SPECS[size]
    N = spec["non_emb"]
    gpus = spec["gpus"]
    budget_mult = BUDGET_MULTS[budget]

    lr = 0.0047 * (N / 108_000_000) ** (-1 / 3)

    # Global batch size: round to nearest multiple of (gpus * rank_microbatch_size)
    raw_batch = round(SEQ_LEN * 160 * (N / 108_000_000) ** (2 / 3))
    unit = gpus * RANK_MICROBATCH_SIZE
    global_batch_size = max(round(raw_batch / unit), 1) * unit

    # Token budget: 20 * N * c, rounded to batch multiple
    raw_tokens = int(20 * N * budget_mult)
    steps = max(round(raw_tokens / global_batch_size), 1)
    max_tokens = steps * global_batch_size

    warmup = max(round(steps * 0.10), 100)

    eval_interval = max(round(steps / 10), 100)
    eval_interval = round(eval_interval / 100) * 100
    eval_interval = max(eval_interval, 100)

    model_config = spec["config_fn"](vocab_size=TOKENIZER.padded_vocab_size())

    return {
        "lr": lr,
        "global_batch_size": global_batch_size,
        "max_tokens": max_tokens,
        "steps": steps,
        "warmup": warmup,
        "eval_interval": eval_interval,
        "gpus": gpus,
        "model_config": model_config,
    }


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: TransformerConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = SEED
    global_batch_size: int = 0
    # New composable pipeline (populated when --pipeline=new)
    instance_sources: Optional[List[InstanceSourceConfig]] = None
    composable_data_loader: Optional[ComposableDataLoaderConfig] = None
    # Old NumpyFSL pipeline (populated when --pipeline=old)
    dataset: Optional[NumpyFSLDatasetConfig] = None
    numpy_data_loader: Optional[NumpyDataLoaderConfig] = None


def _build_composable_data(treatment: str, icl_overlap_paths: list):
    if treatment == "baseline":
        sources = [
            ConcatAndChunkInstanceSourceConfig(
                sources=[NumpyDocumentSourceConfig(source_paths=DOLMA2_BASELINE_PATHS, tokenizer=TOKENIZER)],
                sequence_length=SEQ_LEN,
            ),
        ]
    else:
        sources = [
            MixingInstanceSourceConfig(
                source_specs=[
                    MixingInstanceSourceSpecConfig(
                        source=ConcatAndChunkInstanceSourceConfig(
                            sources=[NumpyDocumentSourceConfig(source_paths=icl_overlap_paths, tokenizer=TOKENIZER)],
                            sequence_length=SEQ_LEN,
                        ),
                        ratio=0.5,
                        label="icl-overlap",
                    ),
                    MixingInstanceSourceSpecConfig(
                        source=ConcatAndChunkInstanceSourceConfig(
                            sources=[NumpyDocumentSourceConfig(source_paths=DOLMA2_BASELINE_PATHS, tokenizer=TOKENIZER)],
                            sequence_length=SEQ_LEN,
                        ),
                        ratio=0.5,
                        label="baseline",
                    ),
                ],
            ),
        ]
    loader = ComposableDataLoaderConfig(num_workers=8, instance_filter_config=InstanceFilterConfig())
    return sources, loader


def _build_numpy_fsl_data(treatment: str, token_budget: int, global_batch_size: int, work_dir: str, icl_overlap_paths: list):
    """Matches the cookbook's NumpyFSL pipeline with chunk_based_mixture for mixtures."""
    if treatment == "baseline":
        dataset = NumpyFSLDatasetConfig.glob(
            *DOLMA2_BASELINE_PATHS,
            sequence_length=SEQ_LEN,
            tokenizer=TOKENIZER,
            work_dir=work_dir,
        )
    else:
        src_mix = SourceMixtureDatasetConfig(
            source_list=SourceMixtureList(sources=[
                SourceMixtureConfig(
                    source_name="icl-overlap",
                    paths=icl_overlap_paths,
                    target_ratio=0.5,
                ),
                SourceMixtureConfig(
                    source_name="baseline",
                    paths=DOLMA2_BASELINE_PATHS,
                    target_ratio=0.5,
                ),
            ]),
            requested_tokens=token_budget,
            global_batch_size=global_batch_size,
            seed=SEED,
        )
        dataset = NumpyFSLDatasetConfig.from_src_mix(
            src_mix,
            sequence_length=SEQ_LEN,
            tokenizer=TOKENIZER,
            work_dir=work_dir,
            chunk_based_mixture=True,
        )
    loader = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=SEED,
        num_workers=4,
        work_dir=work_dir,
    )
    return dataset, loader


def build_config(
    script: str, run_name: str, overrides: List[str],
    treatment: str, pipeline: str, budget: str,
    size: str, icl_data: str = "500b",
) -> ExperimentConfig:
    hp = compute_hyperparams(size, budget)

    root_dir = get_root_dir(CLUSTER)
    work_dir = get_work_dir(root_dir)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    save_folder = f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}"
    final_step = hp["steps"]

    icl_overlap_paths = ICL_OVERLAP_PATHS_3B if icl_data == "3b" else ICL_OVERLAP_PATHS_500B

    instance_sources = None
    composable_data_loader = None
    dataset = None
    numpy_data_loader = None

    if pipeline == "new":
        instance_sources, composable_data_loader = _build_composable_data(treatment, icl_overlap_paths)
    else:
        dataset, numpy_data_loader = _build_numpy_fsl_data(
            treatment, hp["max_tokens"], hp["global_batch_size"], work_dir, icl_overlap_paths,
        )

    # Optimizer: SkipStepAdamW matching cookbook's overlap-pretrain branch
    # step_increment_bugfix=False reproduces the bug where adam step counter never increments
    optimizer = SkipStepAdamWConfig(
        lr=hp["lr"],
        weight_decay=0.033,
        betas=(0.9, 0.95),
        step_increment_bugfix=False,
        group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
    )

    # Scheduler: CosWithWarmupAndLinearDecay matching cookbook default (COS_LINEAR)
    scheduler = CosWithWarmupAndLinearDecay(warmup_steps=hp["warmup"])

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQ_LEN,
        optim=optimizer,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )

    # Checkpointer: save at intervals matching 1xC steps for this size
    hp_1xc = compute_hyperparams(size, "1xC")
    save_interval = hp_1xc["steps"]

    trainer_config = TrainerConfig(
        save_folder=save_folder,
        work_dir=work_dir,
        save_overwrite=True,
        metrics_collect_interval=10,
        cancel_check_interval=5,
        max_duration=Duration.tokens(hp["max_tokens"]),
        callbacks={
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "config_saver": ConfigSaverCallback(),
            "garbage_collector": GarbageCollectorCallback(),
            "beaker": BeakerCallback(),
            "checkpointer": CheckpointerCallback(
                save_interval=save_interval,
                ephemeral_save_interval=250,
                save_async=True,
            ),
            "wandb": WandBCallback(
                name=run_name,
                group=f"cookbook-sweep-repro-{size}",
                project="suffix-train-cookbook-sweep",
                cancel_check_interval=10,
                enabled=True,
            ),
            "lm_evaluator": LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=root_dir,
                    sequence_length=SEQ_LEN,
                    tokenizer=TOKENIZER,
                    work_dir=work_dir,
                ),
                eval_interval=hp["eval_interval"],
                fixed_steps=[final_step],
            ),
            "downstream_evaluator": DownstreamEvaluatorCallbackConfig(
                tokenizer=TOKENIZER,
                tasks=sorted(task_groups.FULL_TASKS),
                eval_interval=hp["eval_interval"],
                fixed_steps=[final_step],
            ),
            "metric_saver": MetricSaverCallback(
                save_interval=hp["eval_interval"],
                fixed_steps=[final_step],
            ),
        },
    )

    # Script path relative to OLMo-core root (gantry clones OLMo-core, not suffix-train)
    script_rel = "src/scripts/train/ladder/2026Q1/cookbook_sweep_repro.py"

    config = ExperimentConfig(
        model=hp["model_config"],
        train_module=train_module_config,
        trainer=trainer_config,
        global_batch_size=hp["global_batch_size"],
        instance_sources=instance_sources,
        composable_data_loader=composable_data_loader,
        dataset=dataset,
        numpy_data_loader=numpy_data_loader,
        launch=build_launch_config(
            name=run_name,
            root_dir=root_dir,
            cmd=[
                script_rel, "train", run_name,
                f"--treatment={treatment}", f"--pipeline={pipeline}", f"--budget={budget}",
                f"--size={size}", f"--icl-data={icl_data}",
                *overrides,
            ],
            cluster=CLUSTER,
            workspace="ai2/oe-t-ladder",
            budget="ai2/oe-base",
            num_nodes=NUM_NODES,
        ),
    )
    # Override num_gpus (build_launch_config hardcodes 8)
    config.launch.num_gpus = hp["gpus"]
    config.launch.priority = "normal"
    config.launch.preemptible = True
    config.launch.allow_dirty = True
    # Override git to clone OLMo-core directly (not suffix-train).
    # The script lives in the ianm/overlap-pretrain-ladder branch of OLMo-core.
    # Must push changes there via `git subtree push` before launching.
    if OLMO_CORE_REF:
        config.launch.git = GitRepoState(
            repo="allenai/OLMo-core",
            repo_url="https://github.com/allenai/OLMo-core.git",
            ref=OLMO_CORE_REF,
            branch=OLMO_CORE_BRANCH,
        )
    config.launch.env_secrets.append(
        BeakerEnvSecret(name="HF_TOKEN", secret="HF_TOKEN", required=False)
    )
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
            global_batch_size=config.global_batch_size,
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
  --size=190M|370M|600M             Model size (required)
  --treatment=baseline|icl_overlap  Data treatment (required)
  --budget=0.5xC|1xC|2xC           Token budget (default: 1xC)
  --pipeline=old|new                Data pipeline (default: old)
  --icl-data=500b|3b                ICL overlap data variant (default: 500b)
  --olmo-core-ref=COMMIT_SHA        OLMo-core commit to clone (required for launch)
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script = sys.argv[0]
    cmd = sys.argv[1]
    run_name = sys.argv[2]

    size = ""
    treatment = "baseline"
    pipeline = "old"
    budget = "1xC"
    icl_data = "500b"
    overrides = []
    for arg in sys.argv[3:]:
        if arg.startswith("--size="):
            size = arg.split("=", 1)[1]
        elif arg.startswith("--treatment="):
            treatment = arg.split("=", 1)[1]
        elif arg.startswith("--pipeline="):
            pipeline = arg.split("=", 1)[1]
        elif arg.startswith("--budget="):
            budget = arg.split("=", 1)[1]
        elif arg.startswith("--icl-data="):
            icl_data = arg.split("=", 1)[1]
        elif arg.startswith("--olmo-core-ref="):
            OLMO_CORE_REF = arg.split("=", 1)[1]
        else:
            overrides.append(arg)

    if not size:
        print(f"Error: --size is required. Must be one of: {list(MODEL_SPECS.keys())}")
        sys.exit(1)
    if size not in MODEL_SPECS:
        print(f"Unknown size: {size}. Must be one of: {list(MODEL_SPECS.keys())}")
        sys.exit(1)
    if treatment not in ("baseline", "icl_overlap"):
        print(f"Unknown treatment: {treatment}. Must be 'baseline' or 'icl_overlap'")
        sys.exit(1)
    if pipeline not in ("new", "old"):
        print(f"Unknown pipeline: {pipeline}. Must be 'new' or 'old'")
        sys.exit(1)
    if budget not in BUDGET_MULTS:
        print(f"Unknown budget: {budget}. Must be one of: {list(BUDGET_MULTS.keys())}")
        sys.exit(1)
    if icl_data not in ("500b", "3b"):
        print(f"Unknown icl-data: {icl_data}. Must be '500b' or '3b'")
        sys.exit(1)
    if cmd == "launch" and not OLMO_CORE_REF:
        print("Error: --olmo-core-ref=COMMIT_SHA is required for launch mode.")
        print("Push changes to OLMo-core first: git subtree push --prefix=olmo-core ...")
        sys.exit(1)

    if cmd == "train":
        prepare_training_environment()
    else:
        prepare_cli_environment()

    config = build_config(script, run_name, overrides, treatment, pipeline, budget, size, icl_data)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        import rich
        hp = compute_hyperparams(size, budget)
        rich.print(f"\n[bold]Hyperparameters for {size} @ {budget}:[/bold]")
        rich.print(f"  LR:               {hp['lr']}")
        rich.print(f"  Global batch size: {hp['global_batch_size']}")
        rich.print(f"  Max tokens:        {hp['max_tokens']}")
        rich.print(f"  Steps:             {hp['steps']}")
        rich.print(f"  Warmup:            {hp['warmup']}")
        rich.print(f"  Eval interval:     {hp['eval_interval']}")
        rich.print(f"  GPUs:              {hp['gpus']}")
        rich.print(f"  Treatment:         {treatment}")
        rich.print(f"  Pipeline:          {pipeline}")
        rich.print(f"  ICL data:          {icl_data}")
        rich.print()
        rich.print(config)
    else:
        print(usage)
        sys.exit(1)
