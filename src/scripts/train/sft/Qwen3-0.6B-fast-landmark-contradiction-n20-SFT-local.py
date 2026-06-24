"""
LOCAL (torchrun, no Beaker/weka) variant of Qwen3-0.6B-fast-landmark-contradiction-n20-SFT.py.

Same model + data pipeline as the Beaker script (qwen3_0_6B + fast_landmark attention, composable
LandmarkInstanceSource over EOS-terminated SFT shards) but everything reads/writes LOCAL paths so it
can run on a Berkeley GPU node (e.g. an idle jsteinhardt H200 like mooney) without weka access.

It does not go through ``olmo_core.internal.experiment`` (whose paths are ai2/weka-specific) -- it
builds the configs directly and calls ``trainer.fit()``, mirroring the build order in
``internal/experiment.py::train`` / ``_build_data_loader`` for the composable path::

    model       = model_config.build(init_device="meta")
    train_module= train_module_config.build(model)
    sources     = [src.build(work_dir) for src in dataset]      # InstanceSourceConfig.build
    data_loader = data_loader_config.build(*sources, dp_process_group=train_module.dp_process_group)
    trainer     = trainer_config.build(train_module, data_loader)
    trainer.fit()

Prereqs (all local / shared NFS, no weka):
  * Data shards (regen locally with convert_longctx_tasks_to_sft.py):
      {DATA_PATH}/token_ids_part_*.npy + labels_mask_*.npy
  * Base checkpoint -- an olmo-core distcp conversion of Qwen3-0.6B-Base. NOTE load_path must point
    at the ``model_and_optim`` SUBDIR (the parent has no .metadata marker -> trainer silently trains
    from scratch). Produce it with corpus-reasoning's scripts/train/convert_hf_to_olmo.py.

Run with torchrun, using THIS working tree's olmo_core (it has fast_landmark + the composable
landmark sources)::

    PYTHONPATH=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src \\
    torchrun --nproc_per_node=8 \\
      src/scripts/train/sft/Qwen3-0.6B-fast-landmark-contradiction-n20-SFT-local.py \\
      --run-name q06b-fast-contra-n20-sft-local
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

# ---- landmark / sequence geometry (identical to the Beaker script) ----
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 2048
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 2016
LANDMARK_TOKEN_ID = 151860

# ---- LOCAL paths (shared NFS, readable from any Berkeley GPU node) ----
DATA_PATH = "/scratch/users/prasann/longctx_sft_qwen/contradiction_n20"
# Base checkpoint: point at the model_and_optim SUBDIR (see module docstring). Lives on mooney's
# node-local /data (the conversion writes it there to dodge an NFS is_dir flake on /scratch), so
# this launcher is mooney-pinned -- move the base to /scratch if you need other nodes to read it.
BASE_CHECKPOINT = "/data/prasann/olmo_ckpts/qwen3_0_6B_base_olmo/model_and_optim"
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache"

GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8  # 8 instances/step (assumes ~8 ranks; accumulates otherwise)
LR = 5e-5
NUM_EPOCHS = 3


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"

    tokenizer_config = TokenizerConfig.qwen3()
    # Shards are separated by single EOS tokens; qwen3 ties bos==eos, so drop BOS for segmentation.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    model_config = TransformerConfig.qwen3_0_6B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_landmark=True,
        mem_freq=MEM_FREQ,
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
            sequence_length=CONTENT_SEQUENCE_LENGTH,
            label_mask_paths=[f"{DATA_PATH}/labels_mask_*.npy"],
            expand_glob=True,
        ),
        mem_freq=MEM_FREQ,
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
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(opts.epochs),
            # --max-steps N -> stop early after N optimizer steps (for smoke tests).
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
                entity="prasanns-allen-institute-for-ai",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    # ---- build + fit (mirrors internal/experiment.py::train for the composable path) ----
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
    ap.add_argument("--run-name", default="q06b-fast-contra-n20-sft-local")
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full; smoke)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    opts = ap.parse_args()

    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
