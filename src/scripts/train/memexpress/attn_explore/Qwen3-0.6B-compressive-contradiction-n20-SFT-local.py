"""
LOCAL (torchrun, no Beaker/weka) COMPRESSIVE-landmark variant of the Qwen3-0.6B contradiction-n20 SFT.

Sibling of ``Qwen3-0.6B-fast-landmark-contradiction-n20-SFT-local.py``. Two things differ, both
forced by the compressive attention kernel (mirrors the 4B compressive script):
  1. Model: ``fast_compressive_landmark=True`` (+ ``nonselected_landmark_mass=0.1`` -- each past
     block's landmark token contributes its value as a compressed block summary; alpha reserves 10%
     of attention mass for the non-selected blocks) instead of ``fast_landmark=True``.
  2. Init from the COMPRESSIVE-pretrained 0.6B 64k base (its landmark-token embedding + compressive
     grouped-softmax attention were trained during compressive CPT), NOT the dense/plain-landmark base.

Data pipeline is identical to the fast-landmark launcher: the single-doc n20 instances go through
``PadToLengthInstanceSource`` (pad each doc to ``content_len``, a multiple of ``mem_freq``) then
``LandmarkInstanceSource`` inserts a landmark token every ``mem_freq`` positions, so the kernel's
positional ``is_mem`` (``pos % block_size == block_size - 1``) holds. No packing needed at n20.

Run::

    PYTHONPATH=.../OLMo-core/src torchrun --nproc_per_node=8 \\
      src/scripts/train/memexpress/attn_explore/Qwen3-0.6B-compressive-contradiction-n20-SFT-local.py \\
      --run-name q06b-comp-contra-n20-sft-local
"""

import argparse
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkInstanceSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
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
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

DEFAULT_MEM_FREQ = 63
SEQUENCE_LENGTH = 2048
LANDMARK_TOKEN_ID = 151860
NONSELECTED_LANDMARK_MASS = 0.1  # alpha for compressive attention

# ---- LOCAL paths ----
DATA_PATH = "/scratch/users/prasann/longctx_sft_qwen/contradiction_n20"
# COMPRESSIVE-pretrained 0.6B 64k base (weka -> s3 -> mooney /data). Override with --base-checkpoint.
BASE_CHECKPOINT = "/data/prasann/olmo_ckpts/q06_bases/comp/model_and_optim"
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache"

GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8
LR = 5e-5
NUM_EPOCHS = 3


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"

    mem_freq = opts.mem_freq
    block_size = mem_freq + 1
    if SEQUENCE_LENGTH % block_size != 0:
        raise SystemExit(
            f"SEQUENCE_LENGTH={SEQUENCE_LENGTH} not divisible by block_size={block_size}"
        )
    content_sequence_length = SEQUENCE_LENGTH // block_size * mem_freq
    print(f"[geometry] mem_freq={mem_freq} block_size={block_size} "
          f"seq_len={SEQUENCE_LENGTH} content_len={content_sequence_length}", flush=True)

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    model_config = TransformerConfig.qwen3_0_6B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_compressive_landmark=True,
        mem_freq=mem_freq,
        nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    instance_source_config = LandmarkInstanceSourceConfig(
        source=PadToLengthInstanceSourceConfig.from_npy(
            f"{DATA_PATH}/token_ids_part_*.npy",
            tokenizer=doc_tokenizer_config,
            sequence_length=content_sequence_length,
            label_mask_paths=[f"{DATA_PATH}/labels_mask_*.npy"],
            expand_glob=True,
        ),
        mem_freq=mem_freq,
        mem_id=LANDMARK_TOKEN_ID,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=WORK_DIR,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            load_path=opts.base_checkpoint,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(opts.epochs),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )
    if opts.wandb:
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=run_name,
                entity=opts.wandb_entity,
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    seed_all(12536)
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(
        source, dp_process_group=train_module.dp_process_group
    )
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="q06b-comp-contra-n20-sft-local")
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument("--base-checkpoint", default=BASE_CHECKPOINT,
                    help="olmo distcp model_and_optim subdir to init from")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full; smoke)")
    ap.add_argument("--mem-freq", type=int, default=DEFAULT_MEM_FREQ,
                    help="landmark spacing; block_size = mem_freq+1 (63->64)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-entity", default=None)
    opts = ap.parse_args()

    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
