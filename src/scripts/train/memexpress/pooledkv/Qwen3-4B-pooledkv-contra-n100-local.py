"""
LOCAL (torchrun, Berkeley H200) three-arm pooled-KV experiment on contradiction n100, Qwen3-4B.

The experiment: can training with most context documents COMPRESSED match full-attention training,
while the checkpoint is evaluated with ordinary full attention (zero-shot transfer)? See
``records/pooled-doc-kv-attention.md``.

Arms (``--arm``):
  * full      -> plain causal attention, the accuracy + wall-clock baseline.
  * pooledkv  -> v1: PooledDocKVAttention (exact per-layer mean-KV slots; matched wall-clock arm).
  * softtoken -> B1: enable_pooled_soft_tokens (pooled docs collapse to ONE projected soft token at
                 the input; the whole stack runs on the ~10x shorter compacted sequence -- the
                 speedup arm). Needs the projector-baked base (bake_projector_into_base.py).

All arms share the shard, base weights, batch geometry (tokens are counted PRE-compaction, so every
arm sees identical data), LR, and epochs. The keep set (gold + ``--n-random`` negatives) comes from
the shard's ``gold_fingerprints.json`` unless ``--gold-blind``.

Run (8x H200)::

    PYTHONPATH=<repo>/src $PY -m torch.distributed.run --nproc_per_node=8 \\
      src/scripts/train/memexpress/pooledkv/Qwen3-4B-pooledkv-contra-n100-local.py \\
      --arm softtoken --run-name q4b-pooledkv-b1-contra-n100
"""

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    NumpyDocumentSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionType
from olmo_core.nn.lm_head import LMLossImplementation
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

# query-position BOTH: matches the layout eval_lc_native_docchunk_contra.py hardcodes, and the
# empirically-sensitive regime for contradiction (see the querypos-three-regime record).
DATA_DIR = "/data/prasann/pooledkv_exp/contra_n100_qboth_train"
BASE_PLAIN = "/data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark/model_and_optim"
BASE_B1 = "/data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark-b1/model_and_optim"
SAVE_ROOT = "/data/prasann/pooledkv_exp/runs"
WORK_DIR = "/data/prasann/pooledkv_exp/dataset-cache"


def build_and_fit(opts: argparse.Namespace) -> None:
    arm = opts.arm
    seq_len = opts.seq_len
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    base_checkpoint = opts.base_ckpt or (BASE_B1 if arm == "softtoken" else BASE_PLAIN)

    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    global_batch_size = opts.batch_tokens or (seq_len * world_size)
    print(
        f"[cfg] arm={arm} seq_len={seq_len} ws={world_size} gbs={global_batch_size} "
        f"lr={opts.lr} epochs={opts.epochs}\n[cfg] base={base_checkpoint}\n[cfg] data={opts.data}",
        flush=True,
    )

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model ----
    from olmo_core.nn.attention import AttentionBackendName

    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName(opts.attn_backend),
    )
    if arm == "pooledkv":
        # v1: swap the attention mixer for exact-mean pooled-doc-KV; chunk_ids threaded via markers.
        mixer = model_config.block.sequence_mixer  # type: ignore[union-attr]
        assert mixer is not None
        mixer.name = AttentionType.pooled_doc_kv
        mixer.pooled_keep_prob = opts.keep_prob
        mixer.pooled_keep_seed = opts.seed
        model_config.document_chunk_attention = {
            "doc_start_id": DOC_START_ID,
            "doc_end_id": DOC_END_ID,
            "eos_id": EOS_TOKEN_ID,
            "mode": "chunked",
        }
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_tokens or seq_len,
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
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=opts.shard_degree or world_size,
        ),
        ac_config=(
            TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.full
            )
            if opts.ac_mode == "full"
            else TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.selected_modules,
                modules=["blocks.*.feed_forward"],
            )
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Data: the main shard, optionally mixed with a SHORT full-attention anchor shard ----
    # Length-stratified anchoring: anchor rows come from a short-context shard whose fingerprints
    # are NOT in the gold sidecar -> the keep-fn leaves all their docs real (full attention), and
    # the softtoken compaction drops their padding so they run at true (short) length. Continuous
    # full-attention anchoring at a fraction of the anchor cost of long uncompressed rows.
    def _src(d):
        return NumpyDocumentSourceConfig(
            source_paths=[f"{d}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{d}/labels_mask_*.npy"],
            expand_glob=True,
        )

    if opts.anchor_data:
        from olmo_core.data.composable import (
            MixingDocumentSourceConfig,
            MixingDocumentSourceSpecConfig,
        )

        sources = [
            MixingDocumentSourceConfig(
                source_specs=[
                    MixingDocumentSourceSpecConfig(
                        source=_src(opts.data),
                        ratio=1.0 - opts.anchor_ratio,
                        max_repetition_factor=8.0,
                        label="main",
                    ),
                    MixingDocumentSourceSpecConfig(
                        source=_src(opts.anchor_data),
                        ratio=opts.anchor_ratio,
                        max_repetition_factor=8.0,
                        label="anchor",
                    ),
                ]
            )
        ]
    else:
        sources = [_src(opts.data)]
    instance_source_config = PadToLengthInstanceSourceConfig(
        sources=sources,
        sequence_length=seq_len,
        tokenizer=doc_tokenizer_config,
    )
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=WORK_DIR,
        global_batch_size=global_batch_size,
        seed=34521 + opts.seed,
        num_workers=4,
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
            max_duration=(
                Duration.steps(opts.max_steps)
                if opts.max_steps > 0
                else Duration.epochs(opts.epochs)
            ),
            async_bookkeeping=False,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                max_checkpoints=1,
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
                group=opts.wandb_group or run_name,
                entity=opts.wandb_entity,
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    seed_all(12536 + opts.seed)
    print("[stage] building model...", flush=True)
    model = model_config.build(init_device="meta")
    if arm == "softtoken":
        oracle_cache = None
        if opts.oracle_slot_cache:
            from olmo_core.nn.oracle_slot import OracleSlotCache

            oracle_cache = OracleSlotCache(opts.oracle_slot_cache)
            print(
                f"[oracle-slot] cache loaded: {oracle_cache.n_docs} docs, "
                f"{oracle_cache.n_layers} layers from {opts.oracle_slot_cache}",
                flush=True,
            )
        # B1: soft-token pooling on the plain-causal model (projector keys come from the baked base).
        model.enable_pooled_soft_tokens(
            DOC_START_ID,
            DOC_END_ID,
            EOS_TOKEN_ID,
            placeholder_id=LANDMARK_TOKEN_ID,
            keep_prob=opts.keep_prob,
            keep_seed=opts.seed,
            aux_match_weight=opts.aux_weight,
            detach_soft_kv=opts.detach_soft_kv,
            distill_prob=opts.distill_prob,
            distill_weight=opts.distill_weight,
            oracle_cache=oracle_cache,
        )
    if opts.ffn_gate_start_layer >= 0:
        # Flexible-compute FFN: context-doc tokens skip the full FFN from this layer on (both
        # the full arm and the softtoken arm's kept-doc tokens). No new params.
        model.enable_role_gated_ffn(
            DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID, start_layer=opts.ffn_gate_start_layer
        )
    train_module = train_module_config.build(model)

    if arm in ("pooledkv", "softtoken") and not opts.gold_blind:
        from olmo_core.nn.attention.pooled_doc_kv import (
            install_pooled_doc_keep,
            make_fingerprint_keep_docs_fn,
        )

        sidecar = opts.gold_sidecar or f"{opts.data}/gold_fingerprints.json"
        with open(sidecar) as f:
            gold_table = json.load(f)
        # Forwards per rank = steps * grad-accum: the anneal horizon for the mixing curriculum.
        n_examples = 2000
        gbs_examples = max(1, global_batch_size // seq_len)
        steps = -(-n_examples * opts.epochs // gbs_examples)
        micro = opts.rank_microbatch_tokens or seq_len
        accum = max(1, global_batch_size // (world_size * micro))
        keep_fn = make_fingerprint_keep_docs_fn(
            gold_table,
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
            eos_id=EOS_TOKEN_ID,
            n_random=opts.n_random,
            n_random_range=(
                tuple(int(x) for x in opts.n_random_range.split(","))
                if opts.n_random_range
                else None
            ),
            mode=opts.keep_mode,
            n_gold=opts.n_gold,
            seed=opts.seed,
            mix_start_p=opts.mix_start_p,
            mix_end_p=opts.mix_end_p,
            mix_total_calls=int(steps * accum * opts.mix_anneal_frac),
        )
        holder = install_pooled_doc_keep(train_module.model, keep_fn)
        if holder.n_attached == 0:
            raise SystemExit("[pooledkv] gold keep hook attached to nothing (arm mismatch?)")
        print(
            f"[pooledkv] gold keep hook: {holder.n_attached} module(s), "
            f"{len(gold_table)} fingerprints, n_random={opts.n_random}",
            flush=True,
        )

    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)
    trainer.callbacks["config_saver"].config = {
        "model": model_config.as_config_dict(),
        "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
    }
    print(f"[stage] loading base from {base_checkpoint} and starting fit()...", flush=True)
    trainer.fit()

    # Model-only export in the eval loader's layout. Extra pooled_projector.* keys (softtoken arm)
    # are ignored by loads into plain models; the config.json records the PLAIN architecture, which
    # is exactly what full-attention eval should build.
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from olmo_core.distributed.utils import get_rank

    save_model_and_optim_state(
        f"{save_folder}/model_and_optim", train_module.model, save_overwrite=True
    )
    if get_rank() == 0:
        export_config = TransformerConfig.qwen3_4B(
            vocab_size=tokenizer_config.padded_vocab_size()
        )
        export_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
        with open(f"{save_folder}/config.json", "w") as f:
            json.dump(
                {
                    "model": export_config.as_config_dict(),
                    "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
                },
                f,
            )
        print(f"[stage] saved model-only checkpoint -> {save_folder}", flush=True)


def main() -> None:
    import faulthandler

    faulthandler.dump_traceback_later(900, repeat=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["full", "pooledkv", "softtoken"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument(
        "--anchor-data", default=None,
        help="short-context shard mixed in as ALWAYS-FULL-ATTENTION anchor rows (its fingerprints "
        "must be absent from the gold sidecar); the cheap continuous anchor for long-context runs",
    )
    ap.add_argument("--anchor-ratio", type=float, default=0.3, help="fraction of rows from anchor")
    ap.add_argument(
        "--distill-prob", type=float, default=0.0,
        help="softtoken arm: probability a forward is PAIRED -- full pass with LM gradient "
        "(protects the full-attention pathway) + detached hidden-state targets for the "
        "compressed pass at the divergence layers (consistency distillation)",
    )
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument(
        "--detach-soft-kv", action="store_true",
        help="softtoken arm: pooled slots act as STATIC KV (per-layer detached from the LM "
        "backward); the projector trains only via --aux-weight",
    )
    ap.add_argument(
        "--oracle-slot-cache", type=str, default=None,
        help="softtoken arm: directory of precomputed oracle log-mass slots "
        "(build_oracle_slot_cache.py); pooled slots' per-layer K/V are overridden with the "
        "cached slots (max-fidelity static KV, no full-context forwards)",
    )
    ap.add_argument(
        "--ffn-gate-start-layer", type=int, default=-1,
        help="enable role-gated FFN (context-doc tokens skip the full FFN) from this layer on; "
        "-1 disables. The matching eval must pass the same flag.",
    )
    ap.add_argument(
        "--aux-weight", type=float, default=0.0,
        help="softtoken arm: weight of the per-layer attention-contribution matching loss "
        "(shadow soft tokens for kept docs matched against their real tokens' softmax mass and "
        "weighted value). The no-anchor, no-mixing fidelity objective.",
    )
    ap.add_argument("--seq-len", type=int, default=6144)
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--gold-sidecar", default=None)
    ap.add_argument("--gold-blind", action="store_true", help="random keep set (control arm)")
    ap.add_argument("--n-random", type=int, default=2, help="random negatives kept real")
    ap.add_argument(
        "--n-random-range",
        type=str,
        default="",
        help="lo,hi: per-row log-uniform negative count (overrides --n-random)",
    )
    ap.add_argument(
        "--keep-mode",
        default="gold_plus_random",
        help="select_keep_docs policy. gold_plus_random leaks (real docs ~= gold -> f1 0.16 under "
        "full-attn eval); gold_subsample --n-gold 2 preserves the base rate.",
    )
    ap.add_argument("--n-gold", type=int, default=0, help="gold docs kept (gold_subsample mode)")
    ap.add_argument(
        "--mix-start-p", type=float, default=0.0,
        help="compression-mixing curriculum: P(train row UNCOMPRESSED) at step 0",
    )
    ap.add_argument("--mix-end-p", type=float, default=0.0, help="curriculum end probability")
    ap.add_argument(
        "--mix-anneal-frac", type=float, default=1.0,
        help="fraction of training over which the curriculum anneals (rest trains at mix-end-p); "
        "front-loading (e.g. 0.25) recovers most of the compression speedup",
    )
    ap.add_argument("--keep-prob", type=float, default=0.1, help="fallback keep prob (gold-blind)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 overrides --epochs (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=250)
    ap.add_argument("--batch-tokens", type=int, default=0)
    ap.add_argument(
        "--rank-microbatch-tokens",
        type=int,
        default=0,
        help="tokens (PRE-compaction) per rank per forward; 0 -> seq_len (1 example). The "
        "softtoken arm can afford several examples per forward (compaction shrinks them ~10x), "
        "which is where its wall-clock win comes from -- small forwards are launch-latency-bound.",
    )
    ap.add_argument("--shard-degree", type=int, default=0)
    ap.add_argument(
        "--attn-backend", default="flash_2", choices=["flash_2", "torch"],
        help="torch = SDPA; the escape hatch for flash-attn 2.8.2's stochastic padded-bwd SIGSEGV "
        "(3 crashes observed across arms/lengths on mooney H200s)",
    )
    ap.add_argument(
        "--ac-mode", default="ffn", choices=["ffn", "full"],
        help="activation checkpointing: ffn-only (fast, fits <=8k on 2 GPUs) or full-block "
        "(needed at 32k on 2 GPUs; ~30%% recompute cost, applied to ALL arms for fairness)",
    )
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument(
        "--wandb-entity",
        default=None,
        help="None -> the logged-in account's default entity (the local .netrc key is the "
        "Berkeley account, NOT the AI2 one -- a mismatched entity fails with CommError).",
    )
    opts = ap.parse_args()
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
