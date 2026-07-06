"""
LOCAL (torchrun, no Beaker/weka) dense Qwen3-4B SFT that mixes the contradiction task with raw
continued-pretraining (CPT) text, to improve contradiction accuracy WITHOUT degrading RULER
long-context retrieval. This is the local counterpart of
``Qwen3-4B-dense-noruler-5k-CPTmix-SFT.py`` (Beaker), specialised to a single SFT task
(contradiction) + CPT mix, runnable on a Berkeley H200 node (mooney/cubbins/sneetches).

Everything reads/writes LOCAL paths (shared NFS /scratch so any node can read):
  * SFT data  : contradiction shards (token_ids_part_*.npy + labels_mask_*.npy), completion-masked.
  * CPT data  : dolma3longmino Qwen3-tokenized sample (part-*.npy + all-True mask-*.npy) -> full loss.
  * base ckpt : MODEL-ONLY olmo-native distcp of the dense CPT base q4b-dense-dolma3longmino/step2385
                (load_path points at the ``model_and_optim`` subdir; optim/trainer state not loaded).

Mix: a ``--cpt-frac`` token-fraction MixingDocumentSource (CPT all-True loss + SFT completion-only),
then ConcatAndChunk into ``--seq-len`` windows. Sweep ``--cpt-frac`` up (and ``--lr`` down) until
RULER stops degrading while contradiction improves.

Build order mirrors ``internal/experiment.py::train`` for the composable path::

    model        = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    sources      = [src.build(work_dir) for src in dataset]
    data_loader  = data_loader_config.build(*sources, dp_process_group=train_module.dp_process_group)
    trainer      = trainer_config.build(train_module, data_loader)
    trainer.fit()

Run (8x H200, full FSDP shard)::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=8 \\
      src/scripts/train/sft/Qwen3-4B-dense-cptmix-contra-local.py \\
      --run-name q4b-cptmix-contra-f70-lr5e5 --cpt-frac 0.70 --lr 5e-5 --max-steps 150
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
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
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

# ---- LOCAL paths (shared NFS, readable from any Berkeley GPU node) ----
SFT_DATA_ROOT = "/scratch/users/prasann/longctx_sft_qwen/contradiction_8k"
CPT_DATA_ROOT = "/scratch/users/prasann/cpt_data/dolma3_longmino_qwen3_sample"
# Model-only olmo-native distcp of the dense CPT base (point at the model_and_optim SUBDIR).
# NOTE: must be on REAL shared NFS, NOT /scratch/users/prasann/olmo_ckpts which is a SYMLINK to
# node-local /data (invisible to other nodes). cpt_mix_ckpts is a real /scratch (oz NFS) dir.
# Base CPT checkpoints by model size (model-only olmo-native distcp, model_and_optim subdir).
# Both on real /scratch (oz NFS), readable from every node.
BASE_CKPTS = {
    "qwen3_4B": "/scratch/users/prasann/cpt_mix_ckpts/q4b-dense-cpt-step2385-modelonly/model_and_optim",
    "qwen3_0_6B": "/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly/model_and_optim",
}
TULU_DATA_ROOT = "/scratch/users/prasann/cpt_data/tulu3_sft_qwen"  # diverse chat SFT (25M tok)
NQ_DATA_ROOT = "/scratch/users/prasann/cpt_data/nq_aligned_k20_qwen"  # fixed (aligned) NQ retrieval k20
OOLONG_DATA_ROOT = "/scratch/users/prasann/cpt_data/oolong_qwen"
RERANK_DATA_ROOT = "/scratch/users/prasann/cpt_data/rerank_qwen"
OUTLIER_DATA_ROOT = "/scratch/users/prasann/cpt_data/outlier_qwen"
# Save trained checkpoints (model+optim) to node-local /data (fast ZFS): writing distcp to
# /scratch NFS is slow and intermittently flaky. The combined sweep job exports->evals on the same
# node, so node-local is fine; only the small eval-summary JSON is written to shared NFS.
SAVE_ROOT = "/data/prasann/cpt_mix_runs"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-cptmix"

DEFAULT_SEQ_LEN = 8192
DEFAULT_CPT_FRAC = 0.70
DEFAULT_LR = 5e-5
DEFAULT_MAX_STEPS = 150


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    seq_len = opts.seq_len
    cpt_frac = opts.cpt_frac
    sft_data_root = opts.sft_data_root or SFT_DATA_ROOT

    # Full FSDP shard across all ranks (4B + AdamW does NOT fit replicated on one GPU).
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    shard_degree = opts.shard_degree or world_size
    # Global batch in *tokens*. Default = one rank_microbatch (= seq_len) per rank per step (no
    # accumulation). The CPT base trained with a MUCH larger batch (~4.2M tokens); --batch-tokens
    # lets us match that scale via gradient accumulation (fewer, more stable steps -> less forgetting).
    global_batch_size = opts.batch_tokens or (seq_len * world_size)
    # Must be a whole number of per-rank microbatches.
    per_step = seq_len * world_size
    if global_batch_size % per_step != 0:
        global_batch_size = max(per_step, (global_batch_size // per_step) * per_step)
    if global_batch_size > per_step:
        print(f"[cfg] grad-accum: global_batch={global_batch_size} tok "
              f"({global_batch_size // per_step} microbatch-steps/rank)", flush=True)

    # Optionally derive max_steps from a token budget so runs are comparable across GPU counts
    # (different world_size => different tokens/step). The mixed dataset is ~47.9M tokens/epoch.
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
        "qwen3_0_6B": TransformerConfig.qwen3_0_6B,
    }[opts.model]
    model_config = model_factory(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
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

    # ---- N-way mixed document source: up to 5 SFT tasks + Tulu chat + CPT ----
    # token fractions for each named SFT task come from --<task>-frac; contradiction = remainder.
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

    # (label, data_root, frac, max_repetition_factor); contradiction gets the leftover fraction.
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
            source=_sft_source(sft_data_root), ratio=contra_frac, max_repetition_factor=8.0,
            label="contradiction"))
    for label, root, frac, rep in task_specs:
        if frac > 1e-6:
            specs.append(MixingDocumentSourceSpecConfig(
                source=_sft_source(root), ratio=frac, max_repetition_factor=rep, label=label))
    if cpt_frac > 1e-6:
        # allow up to 3 passes of the 40M-token CPT sample so a high CPT ratio still holds when the
        # total token budget is scaled up (e.g. 85% of 96M = 81.6M > 2x40M).
        specs.append(MixingDocumentSourceSpecConfig(
            source=cpt_doc_source, ratio=cpt_frac, max_repetition_factor=3.0, label="cpt_longmino"))

    if len(specs) == 1:
        instance_source_config = ConcatAndChunkInstanceSourceConfig(
            sources=[specs[0].source], sequence_length=seq_len)
    else:
        instance_source_config = ConcatAndChunkInstanceSourceConfig(
            sources=[MixingDocumentSourceConfig(source_specs=specs)], sequence_length=seq_len)

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=WORK_DIR,
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=4,
        generate_doc_lengths=True,  # block-diagonal masking at EOS doc boundaries (flash varlen)
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
            # The async-bookkeeping CPU process group (dist.new_group()) hangs on these jsteinhardt
            # nodes (a 2nd NCCL communicator never establishes); use synchronous bookkeeping.
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
                group=opts.wandb_group or "q4b-cptmix-contra",
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
    # Write a self-describing config.json into every saved step dir so the NATIVE eval
    # (TransformerGenerationModule.build(checkpoint_dir)) can reconstruct the model + tokenizer
    # without an olmo->HF export. Mirrors the fast-landmark launcher.
    config_saver = trainer.callbacks.get("config_saver")
    if config_saver is not None:
        config_saver.config = {
            "model": model_config.as_config_dict(),
            "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
        }
    print(f"[stage] trainer built; loading base from {base_checkpoint} and starting fit()...", flush=True)
    trainer.fit()


def main() -> None:
    # Dump all-thread stacks every 10min if still alive, so a real hang shows exactly where
    # (long interval keeps normal logs clean).
    import faulthandler
    faulthandler.dump_traceback_later(600, repeat=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument("--model", default="qwen3_4B", choices=["qwen3_4B", "qwen3_0_6B"],
                    help="model size (also selects the default CPT base checkpoint)")
    ap.add_argument("--base-ckpt", default=None, help="override base CPT checkpoint model_and_optim dir")
    ap.add_argument("--sft-data-root", default=None,
                    help=f"override contradiction SFT shard dir (default {SFT_DATA_ROOT})")
    ap.add_argument("--tulu-frac", type=float, default=0.0,
                    help="token fraction of Tulu3 diverse chat SFT in the mix")
    ap.add_argument("--nq-frac", type=float, default=0.0,
                    help="token fraction of (fixed/aligned) NQ retrieval in the mix")
    ap.add_argument("--oolong-frac", type=float, default=0.0, help="token fraction of OOLONG")
    ap.add_argument("--rerank-frac", type=float, default=0.0, help="token fraction of msmarco rerank")
    ap.add_argument("--outlier-frac", type=float, default=0.0, help="token fraction of outlier")
    ap.add_argument("--cpt-frac", type=float, default=DEFAULT_CPT_FRAC,
                    help="token fraction of CPT data in the mix (0.70 = 70%% CPT / 30%% contradiction)")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="if >0, overrides --max-steps to hit this many training tokens "
                         "(comparable across GPU counts); dataset is ~47.9M tok/epoch")
    ap.add_argument("--batch-tokens", type=int, default=0,
                    help="global batch size in tokens (grad-accum to match CPT-scale batches); "
                         "0 = seq_len*world_size (1 microbatch/rank, current default)")
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
