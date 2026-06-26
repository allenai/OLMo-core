"""
LOCAL (torchrun, no Beaker/weka) document-chunked OOLONG SFT for Qwen3-4B, in two variants:

  * ``--variant dense``    -> :class:`DocumentChunkedAttention` (dense full attention restricted by the
                              chunked-document mask), initialized from the dense CPT base.
  * ``--variant landmark`` -> :class:`DocumentLandmarkAttention` (grouped-softmax landmark + chunked
                              mask), initialized from the fast-landmark CPT base.

Both reconstruct per-token ``chunk_ids`` at runtime from the ``<|box_start|>`` / ``<|box_end|>``
special tokens (151648 / 151649) in the data (built by
``src/scripts/data/convert_unified_to_document_landmark.py`` with the matching ``--emit``). Each
OOLONG item line is one document, so the document count scales with context length. Data is one
already-chunked instance per example, right-padded to ``--seq-len`` (PadToLength; NO packing, NO CP,
eager attention).

Mirrors the standalone build/fit of ``Qwen3-4B-dense-cptmix-contra-local.py`` (local paths,
``trainer.fit()``, ``async_bookkeeping=False`` for jsteinhardt nodes).

Run (8x H200, full FSDP shard)::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=8 \\
      src/scripts/train/sft/Qwen3-4B-document-chunked-longctx-SFT-local.py \\
      --variant dense --run-name q4b-docdense-oolong-ctx2048 --max-steps 120
"""

import argparse
import os
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

# Special-token / reserved ids (match the converter + document_chunk_landmark defaults).
EOS_TOKEN_ID = 151643
LANDMARK_TOKEN_ID = 151860
DOC_START_ID = 151648  # <|box_start|>
DOC_END_ID = 151649  # <|box_end|>
PAD_TOKEN_ID = 151863  # interior window-fill padding (landmark only)

MEM_FREQ = 63  # landmark block size = 64

# ---- LOCAL paths (shared NFS, readable from any Berkeley GPU node) ----
# "full" = full-attention baseline: standard causal attention on the SAME document-chunked data
# (box markers present but NOT masked), to isolate the effect of the chunked mask.
DATA_ROOTS = {
    "dense": "/scratch/users/prasann/longctx_sft_qwen/oolong_ctx2048_docdense",
    "landmark": "/scratch/users/prasann/longctx_sft_qwen/oolong_ctx2048_doclandmark",
    "full": "/scratch/users/prasann/longctx_sft_qwen/oolong_ctx2048_docdense",
}
# Matched CPT bases (model-only olmo-native distcp; point at the model_and_optim SUBDIR).
BASE_CKPTS = {
    "dense": "/scratch/users/prasann/cpt_mix_ckpts/q4b-dense-cpt-step2385-modelonly/model_and_optim",
    "landmark": "/scratch/users/prasann/cpt_mix_ckpts/q4b-fast-landmark-step2385/model_and_optim",
    "full": "/scratch/users/prasann/cpt_mix_ckpts/q4b-dense-cpt-step2385-modelonly/model_and_optim",
}
SAVE_ROOT = "/data/prasann/doc_oolong_runs"  # node-local ZFS (fast distcp; eval runs on same node)
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-docchunk"

DEFAULT_SEQ_LEN = 4096
DEFAULT_LR = 5e-5
DEFAULT_MAX_STEPS = 0  # 0 -> use --epochs
DEFAULT_EPOCHS = 3


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    seq_len = opts.seq_len
    variant = opts.variant

    if variant == "landmark" and seq_len % (MEM_FREQ + 1) != 0:
        raise SystemExit(
            f"--seq-len must be a multiple of {MEM_FREQ + 1} for the landmark variant."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    shard_degree = opts.shard_degree or world_size
    global_batch_size = opts.batch_tokens or (seq_len * world_size)
    per_step = seq_len * world_size
    if global_batch_size % per_step != 0:
        global_batch_size = max(per_step, (global_batch_size // per_step) * per_step)

    base_checkpoint = opts.base_ckpt or BASE_CKPTS[variant]
    data_root = opts.data_root or DATA_ROOTS[variant]
    print(
        f"[cfg] variant={variant} seq_len={seq_len} lr={opts.lr} world_size={world_size} "
        f"shard_degree={shard_degree} gbs_tokens={global_batch_size}\n"
        f"[cfg] base={base_checkpoint}\n[cfg] data={data_root}",
        flush=True,
    )

    tokenizer_config = TokenizerConfig.qwen3()
    # Instances are EOS-separated; qwen3 ties bos==eos, so drop BOS for document-boundary detection.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model: document-chunked attention (dense or landmark), or the full-attention baseline ----
    if variant == "full":
        # Standard causal attention (no chunked mask) on the same document-chunked data -> baseline.
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
        )
        # NB: no document_chunk_attention -> chunk_ids are never reconstructed; full attention.
    elif variant == "dense":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_chunked=True,
            cross_doc_mode="chunked",
        )
        model_config.document_chunk_attention = {
            "doc_start_id": DOC_START_ID,
            "doc_end_id": DOC_END_ID,
            "eos_id": EOS_TOKEN_ID,
            "mode": "chunked",
        }
    else:  # landmark
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_landmark=True,
            mem_freq=MEM_FREQ,
        )
        model_config.document_chunk_attention = {
            "doc_start_id": DOC_START_ID,
            "doc_end_id": DOC_END_ID,
            "eos_id": EOS_TOKEN_ID,
            "mode": "chunked",
            "pad_id": PAD_TOKEN_ID,
        }
    # Fused linear cross-entropy (avoid materializing the full seq x vocab logits).
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq_len,
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        # The chunked mask / grouped softmax are eager (@torch.compiler.disable); keep compile off.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=shard_degree,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Data: one already-chunked instance per example, padded to seq_len (no packing, no CP) ----
    instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
        f"{data_root}/token_ids_part_*.npy",
        tokenizer=doc_tokenizer_config,
        sequence_length=seq_len,
        label_mask_paths=[f"{data_root}/labels_mask_part_*.npy"],
        expand_glob=True,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=WORK_DIR,
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=4,
        # chunk roles are reconstructed from boundary tokens, not EOS-derived doc lengths.
    )

    if opts.max_steps > 0:
        max_duration = Duration.steps(opts.max_steps)
    else:
        max_duration = Duration.epochs(opts.epochs)

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            load_path=base_checkpoint,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=max_duration,
            async_bookkeeping=False,  # the async-bookkeeping process group hangs on jsteinhardt nodes
        )
        .with_callback(
            "checkpointer",
            # Only the final checkpoint matters here (post_train always saves one); no mid-run saves.
            CheckpointerCallback(
                save_interval=100000,
                ephemeral_save_interval=None,
                max_checkpoints=2,
                save_async=False,
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
                group=opts.wandb_group or f"q4b-doc{variant}-oolong",
                entity=opts.wandb_entity,
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    seed_all(12536)
    print("[stage] building model...", flush=True)
    model = model_config.build(init_device="meta")
    print("[stage] building train_module (FSDP+optim)...", flush=True)
    train_module = train_module_config.build(model)
    print("[stage] building instance source...", flush=True)
    source = instance_source_config.build(data_loader_config.work_dir)
    print("[stage] building data loader...", flush=True)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    print("[stage] building trainer...", flush=True)
    trainer = trainer_config.build(train_module, data_loader)
    # Record the model + tokenizer config into each saved checkpoint's config.json so the native eval
    # (TransformerGenerationModule.from_checkpoint) can reconstruct the document-chunked model. Must be
    # set after the trainer is built (mirrors internal/experiment.py).
    trainer.callbacks["config_saver"].config = {
        "model": model_config.as_config_dict(),
        "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
    }
    print(
        f"[stage] trainer built; loading base from {base_checkpoint} and starting fit()...",
        flush=True,
    )
    trainer.fit()


def main() -> None:
    import faulthandler

    faulthandler.dump_traceback_later(600, repeat=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=["dense", "landmark", "full"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument(
        "--base-ckpt", default=None, help="override base CPT checkpoint model_and_optim dir"
    )
    ap.add_argument(
        "--data-root", default=None, help="override the OOLONG document-chunked shard dir"
    )
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=">0 overrides --epochs (smoke)"
    )
    ap.add_argument(
        "--batch-tokens", type=int, default=0, help="global batch in tokens; 0 = seq_len*world_size"
    )
    ap.add_argument("--shard-degree", type=int, default=0, help="0 = full shard across WORLD_SIZE")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-entity", default=None)
    opts = ap.parse_args()

    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
