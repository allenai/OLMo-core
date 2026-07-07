"""
MoE ladder midtraining stage.

This script reuses the MoE ladder architecture definitions and launch flags, but
switches the training recipe from pretraining to the OLMo 3 midtraining mixture:

- load an existing OLMo-core checkpoint from ``--load-path``
- start a fresh trainer/optimizer/scheduler state by default
- train on the source-mixture midtraining data for a fixed token budget
- use the dense scaling-ladders midtraining convention of constant LR after a
  fixed warmup

It is intentionally separate from ``moe_a0_ladder.py`` so midtraining runs have
an explicit, reproducible stage script.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, cast

REPO_SRC = Path(__file__).resolve().parents[5]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

LADDER_DIR = Path(__file__).resolve().parents[2]
if str(LADDER_DIR) not in sys.path:
    sys.path.insert(0, str(LADDER_DIR))

from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.optim import SchedulerUnits
from olmo_core.optim.scheduler import (
    ComposableScheduler,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
)
from olmo_core.script_utils import main
from olmo_core.train import Duration, LoadStrategy
from olmo_core.train.callbacks import WandBCallback

import moe_a0_ladder as base  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_MIDTRAIN_TOKENS = 100_000_000_000
DEFAULT_MIDTRAIN_SOURCE_MIXTURE = (
    "src/olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
)
DEFAULT_MIDTRAIN_SEQUENCE_LENGTH = 8192
DEFAULT_MIDTRAIN_WARMUP_STEPS = 2000


def get_parser() -> argparse.ArgumentParser:
    parser = base.get_parser()
    parser.set_defaults(
        sequence_length=DEFAULT_MIDTRAIN_SEQUENCE_LENGTH,
        ladder_evals=False,
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Pretrained OLMo-core checkpoint/folder to initialize midtraining from.",
    )
    parser.add_argument(
        "--midtrain-max-tokens",
        type=int,
        default=DEFAULT_MIDTRAIN_TOKENS,
        help="Fixed midtraining token budget.",
    )
    parser.add_argument(
        "--midtrain-source-mixture-yaml",
        type=str,
        default=DEFAULT_MIDTRAIN_SOURCE_MIXTURE,
        help="Source-mixture YAML for midtraining data.",
    )
    parser.add_argument(
        "--midtrain-warmup-steps",
        type=int,
        default=DEFAULT_MIDTRAIN_WARMUP_STEPS,
        help="Warmup steps for the constant-with-warmup midtraining scheduler.",
    )
    parser.add_argument(
        "--midtrain-source-processes",
        type=int,
        default=16,
        help="Processes used to materialize the source mixture.",
    )
    parser.add_argument(
        "--midtrain-load-optim-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load optimizer state from --load-path. Defaults to false for LR "
            "searches and weight-only continuation."
        ),
    )
    parser.add_argument(
        "--midtrain-instance-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the standard repetition instance filter to midtraining data.",
    )
    return parser


def finalize_midtraining_config(
    config: base.ExperimentConfig, opts: argparse.Namespace, max_duration_tokens: int
) -> None:
    total_params_in_b = config.model.num_params / 1_000_000_000
    active_params_in_b = config.model.num_active_params / 1_000_000_000
    active_non_embedding_params_in_b = config.model.num_active_non_embedding_params / 1_000_000_000
    log.info(
        "Total params: %.2fB, Active params: %.2fB, Active non-embedding params: %.2fB",
        total_params_in_b,
        active_params_in_b,
        active_non_embedding_params_in_b,
    )
    log.info(
        "Midtraining for %.2fB tokens from %s",
        max_duration_tokens / 1_000_000_000,
        opts.load_path,
    )

    wandb_cb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_original_name = wandb_cb.name
    wandb_cb.name += f"_{active_params_in_b:.2f}@{total_params_in_b:.2f}B"
    wandb_cb.name += (
        f"_{opts.model_size}_{base.EXPERT_GEOMETRY_TAG}_{base.NUM_LAYERS}L"
        f"{base.TOP_K}K{base.NUM_EXPERTS}N{base.NUM_SHARED_EXPERTS}S_{base.TAG}"
    )
    wandb_cb.group = wandb_original_name[:120]


def build_midtraining_dataset_config(
    opts: argparse.Namespace,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
) -> NumpyFSLDatasetConfig:
    source_list = SourceMixtureList.from_yaml(opts.midtrain_source_mixture_yaml)
    source_list.validate()

    return NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=opts.midtrain_max_tokens,
            global_batch_size=base.GLOBAL_BATCH_SIZE,
            processes=opts.midtrain_source_processes,
            seed=base.SEED,
        ),
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
        sequence_length=sequence_length,
        instance_filter_config=(
            InstanceFilterConfig(
                repetition_max_period=13,
                repetition_min_period=1,
                repetition_max_count=32,
            )
            if opts.midtrain_instance_filter
            else None
        ),
    )


def build_config(opts: argparse.Namespace, overrides: List[str]) -> base.ExperimentConfig:
    in_eval_mode = bool(getattr(opts, "eval_checkpoints", None))
    if not opts.load_path and not in_eval_mode:
        raise ValueError("midtraining_ladder.py requires --load-path=<pretrained checkpoint>")
    if opts.midtrain_max_tokens <= 0:
        raise ValueError("--midtrain-max-tokens must be > 0")
    if opts.midtrain_warmup_steps < 0:
        raise ValueError("--midtrain-warmup-steps must be >= 0")

    base.prepare_s3_environment(opts)
    overrides = base.consume_script_overrides(opts, overrides)
    sequence_length = opts.sequence_length or DEFAULT_MIDTRAIN_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    base.configure_model_size(opts)
    model_config = base.build_model_config(tokenizer_config)
    max_duration_tokens = opts.midtrain_max_tokens
    base.configure_sweep_hparams(opts, sequence_length, max_duration_tokens)

    # Eval-only backfills never consume the training dataloader, but script_utils
    # still builds one before Trainer.eval_checkpoints(). Use the cheaper
    # baseline data config in eval mode instead of materializing a 100B-token
    # midtraining source mixture just to satisfy the trainer constructor.
    dataset_config = (
        base.build_dataset_config(opts, tokenizer_config, sequence_length)
        if in_eval_mode
        else build_midtraining_dataset_config(opts, tokenizer_config, sequence_length)
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=base.GLOBAL_BATCH_SIZE,
        seed=base.SEED,
        num_workers=8,
    )
    train_module_config = base.build_train_module_config(
        sequence_length,
        max_duration_tokens=max_duration_tokens,
        in_eval_mode=in_eval_mode,
    )
    train_module_config.scheduler = ComposableScheduler(
        units=SchedulerUnits.steps,
        stages=[
            ComposableSchedulerStage(
                duration=max(1, opts.midtrain_warmup_steps),
                shape=ComposableSchedulerStageType.linear,
                start_lr_fraction=0.0,
                end_lr_fraction=1.0,
            )
        ],
    )

    trainer_config = base.build_trainer_config(
        opts,
        tokenizer_config,
        sequence_length,
        max_duration_tokens=max_duration_tokens,
        in_eval_mode=in_eval_mode,
    )
    if opts.load_path:
        trainer_config.load_path = opts.load_path
        trainer_config.load_strategy = LoadStrategy.always
    trainer_config.load_trainer_state = False
    trainer_config.load_optim_state = opts.midtrain_load_optim_state
    trainer_config.max_duration = Duration.tokens(max_duration_tokens)
    # base.build_trainer_config() attaches ladder eval callbacks when
    # in_eval_mode=True or opts.ladder_evals=True.
    wandb_cb = trainer_config.callbacks.get("wandb")
    if wandb_cb is not None:
        wandb_cb.tags = [*getattr(wandb_cb, "tags", []), "midtraining"]

    config = base.ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=base.SEED,
    ).merge(overrides)

    finalize_midtraining_config(config, opts, max_duration_tokens)
    log.info("Fresh optimizer: %s", not opts.midtrain_load_optim_state)
    return config


if __name__ == "__main__":
    main(build_config, parser=get_parser())
