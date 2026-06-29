"""
LOCAL (torchrun, no Beaker/weka) document-chunked **5-task** (contradiction / nq / oolong / rerank /
outlier) SFT for Qwen3-4B -- the doc-chunked rows of the no-CPT 32k matrix, run on Berkeley H200
(141 GiB) because doc-chunked attention cannot use CP, so the full window's activations live on one
GPU (~81 GiB for dense/hier at 40960 -> needs >80 GiB).

Variants (mirror src/scripts/train/sft/_docchunk_5task_32k_nocpt_common.py):
  * dense        -> DocumentChunkedAttention (cross_doc_mode="chunked"),  dense base,   FlexAttention.
  * hierarchical -> DocumentChunkedAttention (hierarchical_dilated n=4,m=2), dense base, FlexAttention.
  * landmark     -> DocumentLandmarkAttention (eager grouped-softmax),    fast-landmark base.
  * compressive  -> DocumentCompressiveLandmarkAttention (eager),         compressive base.

dense/hier run at the full 40960 window (FlexAttention is block-sparse -> fits 141 GiB). landmark/
compressive are EAGER grouped-softmax which materialises the full TxT scores (O(seq^2)); at 40960
that is ~100 GiB and OOMs, so they MUST run at a reduced ``--seq-len`` (memory-limited, preliminary).

Data: regenerate locally with docchunk_run/convert_local.sh (writes
``/scratch/users/prasann/docchunk_local<SEQ>/{task}_{dense,landmark}``). One already-chunked example
per window, PadToLength, 5-task mix at the headline weights; NO packing, NO CP.

Run (8x H200, full FSDP shard)::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=8 \\
      src/scripts/train/sft/Qwen3-4B-docchunk-5task-local.py \\
      --variant dense --seq-len 40960 --data-root /scratch/users/prasann/docchunk_local40960 \\
      --run-name q4b-docchunk-dense-5task-32k-nocpt-local --max-steps 1465
"""

import argparse
import os
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
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
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

EOS_TOKEN_ID = 151643
DOC_START_ID = 151648
DOC_END_ID = 151649
PAD_TOKEN_ID = 151863
MEM_FREQ = 63
NONSELECTED_LANDMARK_MASS = 0.1

# Mix weights -- IDENTICAL to the packed / Beaker 32k no-CPT rows (sum 7).
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())
_TASK_LABEL = {
    "contra": "contradiction", "nq": "nq_retrieval", "oolong": "oolong",
    "rerank": "rerank", "outlier": "outlier",
}

# Local matched CPT bases (model-only olmo-native distcp; same step2385 checkpoints as the weka rows).
BASE_CKPTS = {
    "dense": "/scratch/users/prasann/stable_bases/q4b-dense-cpt-step2385-modelonly/model_and_optim",
    "hierarchical": "/scratch/users/prasann/stable_bases/q4b-dense-cpt-step2385-modelonly/model_and_optim",
    "landmark": "/scratch/users/prasann/stable_bases/q4b-fast-landmark-step2385-modelonly/model_and_optim",
    "compressive": "/scratch/users/prasann/stable_bases/q4b-fast-compressive-landmark-step2385-modelonly/model_and_optim",
}
SAVE_ROOT = "/scratch/users/prasann/docchunk_local_runs"
WORK_DIR = "/scratch/users/prasann/docchunk_local_runs/dataset-cache"


def _task_source(data_root, emit, name, doc_tok):
    r = f"{data_root}/{name}_{emit}"
    return NumpyDocumentSourceConfig(
        source_paths=[f"{r}/token_ids_part_*.npy"],
        tokenizer=doc_tok,
        label_mask_paths=[f"{r}/labels_mask_*.npy"],
        expand_glob=True,
    )


def build_and_fit(opts: argparse.Namespace) -> None:
    variant = opts.variant
    emit = "landmark" if variant in ("landmark", "compressive") else "dense"
    seq_len = opts.seq_len
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    data_root = opts.data_root
    base_checkpoint = opts.base_ckpt or BASE_CKPTS[variant]

    if variant in ("landmark", "compressive") and seq_len % (MEM_FREQ + 1) != 0:
        raise SystemExit(f"--seq-len must be a multiple of {MEM_FREQ + 1} for {variant}.")

    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    shard_degree = opts.shard_degree or world_size
    global_batch_size = opts.batch_tokens or (seq_len * world_size)
    per_step = seq_len * world_size
    if global_batch_size % per_step != 0:
        global_batch_size = max(per_step, (global_batch_size // per_step) * per_step)
    print(f"[cfg] variant={variant} emit={emit} seq_len={seq_len} ws={world_size} "
          f"gbs={global_batch_size}\n[cfg] base={base_checkpoint}\n[cfg] data={data_root}", flush=True)

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model ----
    # YaRN factor 2 only when extending past the 32768 native window (dense/hier at 40960). landmark/
    # compressive run reduced (<=32k) so they keep native RoPE (matches the packed landmark row).
    use_yarn = seq_len > 32768
    if variant == "landmark":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_landmark=True, mem_freq=MEM_FREQ, landmark_use_kernel=False,
        )
        cca = {"doc_start_id": DOC_START_ID, "doc_end_id": DOC_END_ID, "eos_id": EOS_TOKEN_ID,
               "mode": "chunked", "pad_id": PAD_TOKEN_ID}
    elif variant == "compressive":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_compressive=True, mem_freq=MEM_FREQ,
            nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
        )
        cca = {"doc_start_id": DOC_START_ID, "doc_end_id": DOC_END_ID, "eos_id": EOS_TOKEN_ID,
               "mode": "chunked", "pad_id": PAD_TOKEN_ID}
    elif variant == "hierarchical":
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_chunked=True, cross_doc_mode="hierarchical_dilated",
            dilation_n=opts.dilation_n, dilation_m=opts.dilation_m,
        )
        cca = {"doc_start_id": DOC_START_ID, "doc_end_id": DOC_END_ID, "eos_id": EOS_TOKEN_ID,
               "mode": "chunked"}
    else:  # dense
        model_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_chunked=True, cross_doc_mode="chunked",
        )
        cca = {"doc_start_id": DOC_START_ID, "doc_end_id": DOC_END_ID, "eos_id": EOS_TOKEN_ID,
               "mode": "chunked"}
    if use_yarn:
        model_config = model_config.with_rope_scaling(
            YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
        )
    model_config.document_chunk_attention = cca
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq_len,
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr, weight_decay=0.0, betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full, shard_degree=shard_degree,
        ),
        # FFN-only AC (NOT full): on 141 GiB H200 dense/hier fit at full speed without recomputing the
        # FlexAttention block (which is also not recompute-stable -> CheckpointError).
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Data: 5-task PadToLength mix ----
    specs = [
        MixingDocumentSourceSpecConfig(
            source=_task_source(data_root, emit, name, doc_tokenizer_config),
            ratio=_W[name] / _WSUM, max_repetition_factor=8.0, label=_TASK_LABEL[name],
        )
        for name in ("contra", "nq", "oolong", "rerank", "outlier")
    ]
    instance_source_config = PadToLengthInstanceSourceConfig(
        sources=[MixingDocumentSourceConfig(source_specs=specs)],
        sequence_length=seq_len, tokenizer=doc_tokenizer_config,
    )
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config, work_dir=WORK_DIR,
        global_batch_size=global_batch_size, seed=34521, num_workers=4,
    )

    max_duration = Duration.steps(opts.max_steps) if opts.max_steps > 0 else Duration.epochs(opts.epochs)
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder, save_overwrite=True, load_path=base_checkpoint,
            load_strategy=LoadStrategy.always, load_trainer_state=False, load_optim_state=False,
            metrics_collect_interval=10, cancel_check_interval=10, max_duration=max_duration,
            async_bookkeeping=False,
        )
        .with_callback("checkpointer", CheckpointerCallback(
            save_interval=100000, ephemeral_save_interval=None, max_checkpoints=2, save_async=False))
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
    ap.add_argument("--variant", required=True, choices=["dense", "hierarchical", "landmark", "compressive"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-root", required=True, help="dir holding {task}_{dense,landmark} shard subdirs")
    ap.add_argument("--seq-len", type=int, default=40960)
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--dilation-n", type=int, default=4)
    ap.add_argument("--dilation-m", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 overrides --epochs")
    ap.add_argument("--batch-tokens", type=int, default=0)
    ap.add_argument("--shard-degree", type=int, default=0)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-entity", default="prasanns-allen-institute-for-ai")
    opts = ap.parse_args()
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
