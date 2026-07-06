"""
LOCAL (torchrun, no Beaker/weka) **SINGLE-TASK** length-ladder SFT for Qwen3-4B, dense OR compressive.

This is the per-task counterpart of the 5-task ``*-cptmix-5task-32k-SFT.py`` launchers: instead of
mixing all 5 tasks + CPT, it fine-tunes on **one task's whole length ladder** (rungs 4k/8k/16k/32k)
with **NO CPT data** (pure task-specific). It still INITIALISES from the matched CPT'd base (the
landmark/compressive attentions need their CPT base; dense uses it for apples-to-apples), but no CPT
text is in the SFT mix.

Motivation: the 5-task MIX had weak 32k numbers; this isolates whether single-task training does
better, reported per task per rung.

Variants:
  * dense       -> flash-attn 2 + YaRN(factor2), ConcatAndChunk packing. Local dense CPT base.
  * compressive -> fast_compressive_landmark + LandmarkPacking. Compressive CPT base (usually weka;
                   pass --base-ckpt for a local copy). [Prefer running compressive on Beaker where the
                   base lives -- see Qwen3-4B-compressive-singletask-ladder-32k-SFT.py.]

Data: per-task ladder shards (one dir per task, all rungs concatenated) at
``<data-root>/<task>/{token_ids_part_*.npy,labels_mask_*.npy}`` -- built by the single-task tokenizer
step (convert_longctx_tasks_to_sft.py / convert_unified_to_sft.py over all rungs of one task).

Budget: 1-2 EPOCHS over the (~2000-example, ~30M-token) task ladder. At GBS = one 40960-window/step
that is ~700 steps/epoch, so 2 epochs ~= the proven ~1465-step recipe.

Run (8x H200, CP=8)::

    PYTHONPATH=src torchrun --nproc_per_node=8 \\
      src/scripts/train/sft/singletask_ladder/Qwen3-4B-singletask-ladder-SFT-local.py \\
      --variant dense --task contra --data-root /scratch/users/prasann/single_task_ladders \\
      --epochs 2 --run-name q4b-dense-contra-ladder32k-local
"""

import argparse
import os
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    LandmarkPackingInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
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
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

# ---- Landmark / window geometry (matches the cptmix-32k launchers) ----
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 40960  # "32k + a bit": fits the 32k rung's ~40k-token examples; divisible by 64
LANDMARK_TOKEN_ID = 151860
NONSELECTED_LANDMARK_MASS = 0.1
CP_DEGREE = 8

# Local CPT'd bases (model-only olmo-native distcp). Compressive base is usually weka-only -> pass
# --base-ckpt or run the Beaker launcher for compressive.
BASE_CKPTS = {
    "dense": "/scratch/users/prasann/stable_bases/q4b-dense-cpt-step2385-modelonly/model_and_optim",
    "compressive": "/scratch/users/prasann/stable_bases/q4b-fast-compressive-landmark-step2385-modelonly/model_and_optim",
}

_TASK_DIR = {
    "contra": "contradiction", "nq": "nq", "oolong": "oolong",
    "rerank": "rerank", "outlier": "outlier",
}
_TASK_LABEL = {
    "contra": "contradiction", "nq": "nq_retrieval", "oolong": "oolong",
    "rerank": "rerank", "outlier": "outlier",
}

SAVE_ROOT = "/data/prasann/singletask_ladder_runs"
WORK_DIR = "/data/prasann/singletask_ladder_runs/dataset-cache"


def build_and_fit(opts: argparse.Namespace) -> None:
    variant = opts.variant
    task = opts.task
    seq_len = opts.seq_len
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    base_checkpoint = opts.base_ckpt or BASE_CKPTS[variant]
    task_root = f"{opts.data_root.rstrip('/')}/{_TASK_DIR[task]}"

    if variant == "compressive" and seq_len % BLOCK_SIZE != 0:
        raise SystemExit(f"--seq-len must be a multiple of {BLOCK_SIZE} for compressive.")

    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    cp_degree = opts.cp_degree or min(CP_DEGREE, world_size)
    global_batch_size = opts.batch_tokens or seq_len  # one window per optimizer step
    print(f"[cfg] variant={variant} task={task} seq_len={seq_len} ws={world_size} cp={cp_degree} "
          f"gbs={global_batch_size}\n[cfg] base={base_checkpoint}\n[cfg] data={task_root}", flush=True)

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model ----
    if variant == "compressive":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            fast_compressive_landmark=True,
            nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
            mem_freq=MEM_FREQ,
        )
    else:  # dense -- flash-attn 2 + YaRN context extension (native 32k -> 64k)
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_2,
        ).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
        )
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq_len,
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr, weight_decay=0.0, betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full, shard_degree=1,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=cp_degree),
        # FFN-only AC (NOT budget mode -- budget requires torch.compile; we default compile off for
        # fast local startup). flash-attn is memory-efficient so 40960 fits on a 141 GiB H200 with
        # FFN-only recompute (matches the docchunk-local launcher that fit dense at 40960).
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Data: single task, NO CPT (ratio 1.0) ----
    task_source = NumpyDocumentSourceConfig(
        source_paths=[f"{task_root}/token_ids_part_*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{task_root}/labels_mask_*.npy"],
        expand_glob=True,
    )
    mixing = MixingDocumentSourceConfig(source_specs=[
        MixingDocumentSourceSpecConfig(
            source=task_source, ratio=1.0, max_repetition_factor=8.0, label=_TASK_LABEL[task],
        )
    ])
    if variant == "compressive":
        instance_source_config = LandmarkPackingInstanceSourceConfig(
            source=mixing, sequence_length=seq_len, mem_freq=MEM_FREQ,
            mem_id=LANDMARK_TOKEN_ID, pad_id=tokenizer_config.pad_token_id,
        )
        generate_doc_lengths = False
    else:
        instance_source_config = ConcatAndChunkInstanceSourceConfig(
            sources=[mixing], sequence_length=seq_len,
        )
        generate_doc_lengths = True

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config, work_dir=WORK_DIR,
        global_batch_size=global_batch_size, seed=34521, num_workers=4,
        generate_doc_lengths=generate_doc_lengths,
    )

    max_duration = (
        Duration.steps(opts.max_steps) if opts.max_steps > 0 else Duration.epochs(opts.epochs)
    )
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder, save_overwrite=True, load_path=base_checkpoint,
            load_strategy=LoadStrategy.always, load_trainer_state=False, load_optim_state=False,
            metrics_collect_interval=10, cancel_check_interval=10, max_duration=max_duration,
            async_bookkeeping=False,
        )
        .with_callback("checkpointer", CheckpointerCallback(
            save_interval=opts.save_interval, ephemeral_save_interval=None,
            max_checkpoints=2, save_async=False))
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )
    if opts.wandb:
        trainer_config = trainer_config.with_callback("wandb", WandBCallback(
            name=run_name_with_ts, group=opts.wandb_group or run_name,
            entity=opts.wandb_entity, project="memory-networks", enabled=True, cancel_check_interval=10))

    seed_all(12536)
    print("[stage] building model...", flush=True)
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)
    trainer.callbacks["config_saver"].config = {
        "model": model_config.as_config_dict(),
        "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
    }
    print(f"[stage] loading base from {base_checkpoint} and starting fit()...", flush=True)
    trainer.fit()


def main() -> None:
    import faulthandler
    faulthandler.dump_traceback_later(900, repeat=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=["dense", "compressive"])
    ap.add_argument("--task", required=True, choices=["contra", "nq", "oolong", "rerank", "outlier"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-root", default="/scratch/users/prasann/single_task_ladders",
                    help="dir holding <task>/{token_ids_part_*,labels_mask_*}.npy subdirs")
    ap.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 overrides --epochs")
    ap.add_argument("--save-interval", type=int, default=400,
                    help="persist a checkpoint every N steps to node-local /data (preemption safety net)")
    ap.add_argument("--batch-tokens", type=int, default=0)
    ap.add_argument("--cp-degree", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (slower start)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", default=None)
    # Local wandb login is the PERSONAL entity, NOT the ai2 one (which gives CommError: permission
    # denied). Default to the personal entity; override with --wandb-entity for ai2/gantry runs.
    ap.add_argument("--wandb-entity",
                    default="prasann-uc-berkeley-electrical-engineering-computer-sciences")
    opts = ap.parse_args()
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
