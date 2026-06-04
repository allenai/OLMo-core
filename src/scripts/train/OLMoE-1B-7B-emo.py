"""
Train an OLMoE-1B-7B mixture-of-experts model, optionally with **EMO** routing
(:class:`~olmo_core.nn.moe.emo_router.EmoRouter`): two-level document routing with a random
per-document expert pool, shared experts, and an optional data-parallel-global load-balancing loss.

Whether EMO routing is used is controlled by ``--model-type`` (``moe`` for the stock OLMoE router,
``emo`` for the EMO router). The EOS token id used to derive document boundaries is taken
automatically from the tokenizer config, so you never have to specify it by hand.

Launch with torchrun, e.g. (single-node, 8 GPUs):

    torchrun --nproc-per-node=8 src/scripts/train/OLMoE-1B-7B-emo.py RUN_NAME \\
        --model-type=emo \\
        --save-folder=/path/to/checkpoints \\
        --work-dir=/path/to/dataset-cache \\
        --data-root=/path/to/ai2-llm

To reproduce the ``emo_1b14b_1t`` configuration, pass the same config overrides the original launch
used (these are plain dotted-path overrides applied after the base config is built)::

    --model.block.feed_forward_moe.num_experts=128 \\
    --model.block.name=moe \\
    --model.block.sequence_mixer.qk_norm=null \\
    --model.block.sequence_mixer.backend=flash_2 \\
    --model.block.feed_forward_moe.lb_loss_weight=1e-1 \\
    --dataset.generate_doc_lengths=true \\
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}'

Use ``--dry-run`` to print the fully-resolved config and exit (no GPUs or data required) — handy for
checking that EMO routing is wired up the way you expect.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.mixes import DataMix
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe import MoERouterType
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

SEQUENCE_LENGTH = 4096


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    init_seed: int = 12536


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder or f"/tmp/{opts.run_name}"
    work_dir = opts.work_dir or "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmoe_1B_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    # Swap in EMO routing if requested. Everything else (num_experts, block type, lb weight, ...)
    # is left at the OLMoE defaults and can be tweaked via config overrides (see module docstring).
    if opts.model_type == "emo":
        assert isinstance(model_config.block, TransformerBlockConfig)
        assert model_config.block.feed_forward_moe is not None
        router = model_config.block.feed_forward_moe.router
        router.name = MoERouterType.emo
        # The EOS token id is taken from the tokenizer so document boundaries always match how the
        # data was tokenized — no need to specify it on the command line.
        router.emo_eos_token_id = tokenizer_config.eos_token_id
        router.emo_min_document_expert_pool = opts.min_document_expert_pool
        router.emo_max_document_expert_pool = opts.max_document_expert_pool
        router.emo_eval_document_expert_pool = opts.eval_document_expert_pool
        router.emo_num_shared_experts = opts.num_shared_experts
        router.emo_global_load_balancing = opts.global_load_balancing
        log.info(
            "Using EMO routing (eos_token_id=%s, pool=[%s, %s], eval_pool=%s, "
            "shared_experts=%s, global_lb=%s)",
            router.emo_eos_token_id,
            opts.min_document_expert_pool,
            opts.max_document_expert_pool,
            router.emo_eval_document_expert_pool,
            opts.num_shared_experts,
            opts.global_load_balancing,
        )
    elif opts.model_type == "moe":
        log.info("Using the stock OLMoE router (no EMO routing).")
    else:
        raise ValueError(f"Unknown --model-type: {opts.model_type!r} (expected 'moe' or 'emo')")

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size * SEQUENCE_LENGTH,  # specified in tokens
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * SEQUENCE_LENGTH,  # specified in tokens
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=opts.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
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
            CometCallback(name=opts.run_name, cancel_check_interval=10, enabled=False),
        )
        .with_callback(
            "wandb",
            WandBCallback(name=opts.run_name, cancel_check_interval=10, enabled=False),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    "hellaswag",
                    "arc_challenge",
                    "piqa",
                    "copa",
                ],
                tokenizer=tokenizer_config,
                eval_interval=2500,
            ),
        )
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
        train_module=train_module_config,
    )

    # Apply any dotted-path config overrides (e.g. --model.block.feed_forward_moe.num_experts=128).
    config = config.merge(overrides)
    return config


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"torchrun ... {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="The name of the run.")
    parser.add_argument("--save-folder", type=str, help="Local or remote dir for checkpoints.")
    parser.add_argument("--work-dir", type=str, help="Local dir for dataset preprocessing cache.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/weka/oe-training-default/ai2-llm",
        help="Root directory for the data mix (mix_base_dir).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the config and exit.")

    parser.add_argument(
        "--model-type",
        choices=["moe", "emo"],
        default="emo",
        help="'moe' for the stock OLMoE router, 'emo' for EMO routing.",
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=1024,
        help="Global batch size in instances (multiplied by the sequence length).",
    )

    # EMO router options (only used when --model-type=emo).
    emo = parser.add_argument_group("EMO routing options (used when --model-type=emo)")
    emo.add_argument("--min-document-expert-pool", type=int, default=8)
    emo.add_argument("--max-document-expert-pool", type=int, default=128)
    emo.add_argument(
        "--eval-document-expert-pool",
        type=int,
        default=32,
        help="Fixed pool size at eval time. Defaults to the midpoint of min/max if unset.",
    )
    emo.add_argument("--num-shared-experts", type=int, default=1)
    emo.add_argument(
        "--global-load-balancing",
        dest="global_load_balancing",
        action="store_true",
        default=True,
        help="Reduce the load-balancing statistics across the DP group (default).",
    )
    emo.add_argument(
        "--no-global-load-balancing",
        dest="global_load_balancing",
        action="store_false",
        help="Use rank-local load-balancing statistics instead of DP-global.",
    )

    return parser.parse_known_args()


def main():
    # We read data from S3 (e.g. --data-root=s3://ai2-llm) using the AWS_* env-var credentials
    # injected via Beaker secrets, not a named AWS profile. The Beaker launcher sets S3_PROFILE=S3
    # by default, which would make boto look for an '[S3]' profile that doesn't exist here; clearing
    # it falls back to the default (env-var) credential chain.
    os.environ.pop("S3_PROFILE", None)

    opts, overrides = parse_args()
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
