"""
LOCAL (torchrun, no Beaker/weka) **fast-landmark** Qwen3-4B SFT that mixes 5 long-context SFT tasks
(contradiction, nq, oolong, rerank, outlier) + Tulu chat + raw continued-pretraining (CPT) text,
to lift task accuracy WITHOUT degrading RULER long-context retrieval.

This is the LANDMARK counterpart of ``Qwen3-4B-dense-cptmix-contra-local.py``. The recipe, token
mixing, weighting, hyperparameters and the sbatch train->export->eval wrapper are kept identical so
the landmark result is directly comparable to the dense baseline (RULER 0.799 / contra 0.76 / nq
0.98 / oolong 0.62 / rerank 0.40 / outlier 0.41 at cpt0.85, weighted 5-task, lr1e-5, 8k, 48M tok).

Two things differ from the dense launcher, both forced by landmark attention:

  1. **Attention**: ``fast_landmark=True, mem_freq=63`` (block_size = 64). The model is initialised
     from the **fast-landmark** CPT base (q4b-fast-landmark-dolma3longmino/step2385), NOT the dense
     CPT base -- the landmark-token (151860) embedding and the grouped-softmax attention were trained
     during landmark CPT, so loading a dense base would be wrong.

  2. **Data pipeline**: ``MixingDocumentSource -> LandmarkPackingInstanceSource`` instead of
     ``MixingDocumentSource -> ConcatAndChunkInstanceSource``. ``LandmarkPackingInstanceSource``
     inserts a landmark token after every ``mem_freq`` content tokens *per document*, so every
     document occupies a whole number of landmark blocks, then greedily packs whole documents into
     ``seq_len`` windows and emits block-aligned ``doc_lens``. The model masks attention
     block-diagonally from those ``doc_lens`` (a query never attends across a document boundary) --
     the landmark analogue of the dense recipe's ``generate_doc_lengths=True`` EOS masking. So SFT
     examples are isolated exactly as in the dense run.

     ``generate_doc_lengths`` is left False: document boundaries come from the packing source, not
     from EOS (EOS-derived boundaries are not block-aligned and the landmark attention rejects them).

     NOTE: ``LandmarkPackingInstanceSource`` DROPS any document longer than one ``seq_len`` window
     (in landmark-token space, ~``seq_len * mem_freq / block_size`` content tokens). The 5 SFT tasks
     have ~0 such documents, but the CPT sample (dolma3longmino) has a few very long documents that
     hold a non-trivial token mass; those are dropped, so the *realised* CPT fraction is somewhat
     below the nominal ``--cpt-frac``. The MixingDocumentSource token-count log prints the realised
     mix; bump ``--cpt-frac`` if you need to match the dense CPT mass exactly.

Build order mirrors ``internal/experiment.py::train`` for the composable path::

    model        = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    source       = instance_source_config.build(work_dir)
    data_loader  = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    trainer      = trainer_config.build(train_module, data_loader)
    trainer.fit()

Run (8x H200, full FSDP shard, no CP -- 8k fits)::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=8 \\
      src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-local.py \\
      --run-name q4b-lm-5task-c85 --cpt-frac 0.85 --nq-frac 0.0375 --oolong-frac 0.0375 \\
      --rerank-frac 0.0225 --outlier-frac 0.0225 --lr 1e-5 --target-tokens 48000000
"""

import argparse
import os
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
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

# ---- landmark geometry ----
# mem_freq = landmark spacing; block_size = mem_freq + 1 (mem_freq content toks + 1 landmark).
# seq_len (landmark-token space) must be divisible by block_size.
DEFAULT_MEM_FREQ = 63
LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

# ---- LOCAL paths (shared NFS, readable from any Berkeley GPU node) ----
SFT_DATA_ROOT = "/scratch/users/prasann/longctx_sft_qwen/contradiction_8k"
CPT_DATA_ROOT = "/scratch/users/prasann/cpt_data/dolma3_longmino_qwen3_sample"
# Model-only olmo-native distcp of the FAST-LANDMARK CPT base (point at the model_and_optim SUBDIR).
# Fetched from weka (q4b-fast-landmark-dolma3longmino/step2385) via s3 -- see landmark_base_sync.yaml
# / [[weka-s3-checkpoint-transfer]]. On real /scratch (oz NFS), readable from every node.
BASE_CKPTS = {
    "qwen3_4B": "/scratch/users/prasann/cpt_mix_ckpts/q4b-fast-landmark-step2385/model_and_optim",
}
TULU_DATA_ROOT = "/scratch/users/prasann/cpt_data/tulu3_sft_qwen"  # diverse chat SFT (25M tok)
NQ_DATA_ROOT = "/scratch/users/prasann/cpt_data/nq_aligned_k20_qwen"  # fixed (aligned) NQ retrieval k20
OOLONG_DATA_ROOT = "/scratch/users/prasann/cpt_data/oolong_qwen"
RERANK_DATA_ROOT = "/scratch/users/prasann/cpt_data/rerank_qwen"
OUTLIER_DATA_ROOT = "/scratch/users/prasann/cpt_data/outlier_qwen"
SAVE_ROOT = "/data/prasann/cpt_mix_runs"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-lmcptmix"

DEFAULT_SEQ_LEN = 8192
DEFAULT_CPT_FRAC = 0.85
DEFAULT_LR = 1e-5
DEFAULT_MAX_STEPS = 150


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    seq_len = opts.seq_len
    cpt_frac = opts.cpt_frac

    mem_freq = opts.mem_freq
    block_size = mem_freq + 1
    if seq_len % block_size != 0:
        raise SystemExit(
            f"--seq-len {seq_len} not divisible by block_size={block_size} (mem_freq+1); "
            f"pick a seq_len that is a multiple of {block_size}."
        )
    content_cap = seq_len // block_size * mem_freq
    print(f"[geometry] mem_freq={mem_freq} block_size={block_size} seq_len={seq_len} "
          f"content_cap_per_doc={content_cap}", flush=True)

    # Full FSDP shard across all ranks (4B + AdamW does NOT fit replicated on one GPU).
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    shard_degree = opts.shard_degree or world_size
    global_batch_size = opts.batch_tokens or (seq_len * world_size)
    per_step = seq_len * world_size
    if global_batch_size % per_step != 0:
        global_batch_size = max(per_step, (global_batch_size // per_step) * per_step)
    if global_batch_size > per_step:
        print(f"[cfg] grad-accum: global_batch={global_batch_size} tok "
              f"({global_batch_size // per_step} microbatch-steps/rank)", flush=True)

    if opts.target_tokens > 0:
        opts.max_steps = max(1, round(opts.target_tokens / global_batch_size))
        print(f"[cfg] target_tokens={opts.target_tokens} -> max_steps={opts.max_steps}", flush=True)

    print(
        f"[cfg] seq_len={seq_len} cpt_frac={cpt_frac} lr={opts.lr} max_steps={opts.max_steps} "
        f"world_size={world_size} shard_degree={shard_degree} gbs_tokens={global_batch_size}",
        flush=True,
    )

    tokenizer_config = TokenizerConfig.qwen3()
    # Shards are separated by single EOS tokens; qwen3 ties bos==eos, so drop BOS for doc splitting.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    base_checkpoint = opts.base_ckpt or BASE_CKPTS[opts.model]
    model_factory = {
        "qwen3_4B": TransformerConfig.qwen3_4B,
    }[opts.model]
    model_config = model_factory(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_landmark=True,
        mem_freq=mem_freq,
    )
    _used = opts.tulu_frac + opts.nq_frac + opts.oolong_frac + opts.rerank_frac + opts.outlier_frac
    print(f"[cfg] model={opts.model} base={base_checkpoint} tulu={opts.tulu_frac} nq={opts.nq_frac} "
          f"oolong={opts.oolong_frac} rerank={opts.rerank_frac} outlier={opts.outlier_frac} "
          f"contra={max(0.0, 1.0-cpt_frac-_used):.3f}", flush=True)

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
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=shard_degree,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- N-way mixed document source: up to 5 SFT tasks + Tulu chat + CPT (identical to dense) ----
    cpt = CPT_DATA_ROOT.rstrip("/")

    def _sft_source(root):  # completion-only masked SFT shards
        return NumpyDocumentSourceConfig(
            source_paths=[f"{root.rstrip('/')}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{root.rstrip('/')}/labels_mask_*.npy"],
            expand_glob=True,
        )

    cpt_doc_source = NumpyDocumentSourceConfig(
        source_paths=[f"{cpt}/part-*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{cpt}/mask-*.npy"],  # explicit all-True => full-sequence CPT loss
        expand_glob=True,
    )

    task_specs = [
        ("tulu3_chat", TULU_DATA_ROOT, opts.tulu_frac, 4.0),
        ("nq_retrieval", NQ_DATA_ROOT, opts.nq_frac, 8.0),
        ("oolong", OOLONG_DATA_ROOT, opts.oolong_frac, 8.0),
        ("rerank", RERANK_DATA_ROOT, opts.rerank_frac, 8.0),
        ("outlier", OUTLIER_DATA_ROOT, opts.outlier_frac, 8.0),
    ]
    used = sum(f for _, _, f, _ in task_specs)
    contra_frac = max(0.0, 1.0 - cpt_frac - used)

    specs = []
    if contra_frac > 1e-6:
        specs.append(MixingDocumentSourceSpecConfig(
            source=_sft_source(SFT_DATA_ROOT), ratio=contra_frac, max_repetition_factor=8.0,
            label="contradiction"))
    for label, root, frac, rep in task_specs:
        if frac > 1e-6:
            specs.append(MixingDocumentSourceSpecConfig(
                source=_sft_source(root), ratio=frac, max_repetition_factor=rep, label=label))
    if cpt_frac > 1e-6:
        specs.append(MixingDocumentSourceSpecConfig(
            source=cpt_doc_source, ratio=cpt_frac, max_repetition_factor=3.0, label="cpt_longmino"))

    if len(specs) == 1:
        mixed_doc_source: object = specs[0].source
    else:
        mixed_doc_source = MixingDocumentSourceConfig(source_specs=specs)

    # Per-document landmark insertion + greedy packing into seq_len windows + block-aligned doc_lens.
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=mixed_doc_source,
        sequence_length=seq_len,
        mem_freq=mem_freq,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=WORK_DIR,
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=4,
        # Document boundaries come from LandmarkPackingInstanceSource (block-aligned doc_lens);
        # EOS-derived boundaries would NOT be block-aligned and the landmark attention rejects them.
        generate_doc_lengths=False,
    )

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
            max_duration=Duration.steps(opts.max_steps),
            # async-bookkeeping's 2nd NCCL process group hangs on these jsteinhardt nodes.
            async_bookkeeping=False,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100000,  # only save at the end (ephemeral handles mid-run)
                ephemeral_save_interval=opts.max_steps,
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
                group=opts.wandb_group or "q4b-lm-cptmix-5task",
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
    print("[stage] building instance source (landmark packing over mixed docs)...", flush=True)
    source = instance_source_config.build(data_loader_config.work_dir)
    print("[stage] building data loader...", flush=True)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    print("[stage] building trainer...", flush=True)
    trainer = trainer_config.build(train_module, data_loader)
    # Write a self-describing config.json into every saved step dir so the NATIVE landmark eval
    # (TransformerGenerationModule.build(checkpoint_dir)) can reconstruct the model + tokenizer.
    # The standalone launcher does not go through internal/experiment (which normally sets this), so
    # without it ConfigSaverCallback warns "Config not set ... doing nothing" and writes no config.json.
    config_saver = trainer.callbacks.get("config_saver")
    if config_saver is not None:
        config_saver.config = {
            "model": model_config.as_config_dict(),
            "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
        }
    print(f"[stage] trainer built; loading base from {base_checkpoint} and starting fit()...", flush=True)
    trainer.fit()


def main() -> None:
    import faulthandler
    faulthandler.dump_traceback_later(600, repeat=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument("--model", default="qwen3_4B", choices=["qwen3_4B"],
                    help="model size (also selects the default fast-landmark CPT base checkpoint)")
    ap.add_argument("--base-ckpt", default=None, help="override base CPT checkpoint model_and_optim dir")
    ap.add_argument("--tulu-frac", type=float, default=0.0)
    ap.add_argument("--nq-frac", type=float, default=0.0)
    ap.add_argument("--oolong-frac", type=float, default=0.0)
    ap.add_argument("--rerank-frac", type=float, default=0.0)
    ap.add_argument("--outlier-frac", type=float, default=0.0)
    ap.add_argument("--cpt-frac", type=float, default=DEFAULT_CPT_FRAC,
                    help="token fraction of CPT data in the mix (contradiction = remainder)")
    ap.add_argument("--mem-freq", type=int, default=DEFAULT_MEM_FREQ,
                    help="landmark spacing; block_size = mem_freq+1 (63->64). seq_len must be a "
                         "multiple of block_size.")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="if >0, overrides --max-steps to hit this many training tokens")
    ap.add_argument("--batch-tokens", type=int, default=0,
                    help="global batch size in tokens (grad-accum); 0 = seq_len*world_size")
    ap.add_argument("--shard-degree", type=int, default=0, help="0 = full shard across WORLD_SIZE")
    ap.add_argument("--compile", action="store_true", help="enable torch.compile (slower startup)")
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
