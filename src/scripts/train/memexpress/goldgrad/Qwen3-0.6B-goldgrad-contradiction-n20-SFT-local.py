"""
LOCAL (torchrun) **gold-gradient (O(1)-backward) contradiction-n20 SFT** launcher.

Tests the hypothesis: *keep the full forward over all 20 documents, but back-propagate only through the
ground-truth contradiction documents (+ a few random distractors) -- treating the rest of the KV as
static.* The forward (and therefore the loss/logits) is bit-identical to normal document-chunked
training; only the backward graph changes, so weight-update compute scales with the number of
**selected** documents, not the 20 in context. See
:mod:`olmo_core.nn.attention.gold_grad_mask` for the mechanism (detach non-selected docs' K/V).

Model: ``qwen3_0_6B(document_chunked=True, cross_doc_mode="chunked")`` -- the ``"chunked"`` mask makes
each context document an isolated tower (only FREE query/answer tokens bridge across docs), which is
what lets the K/V detach fully sever a non-selected doc's backward path.

Three arms (share the SAME base checkpoint + data + eval -- only ``--grad-mode`` differs):

  * ``full``             -- normal document-chunked training (every doc gets gradient). Upper-bound
                            reference. The gold-grad hook is NOT installed.
  * ``gold_plus_random`` -- the O(1) arm: keep gradient for every gold contradiction doc + ``--n-random``
                            random non-gold docs (default 2); detach the rest.
  * ``random_only``      -- control: keep the SAME NUMBER of docs as ``gold_plus_random``
                            (``|gold| + n_random``) but chosen entirely at random (gold identity
                            hidden). Isolates whether knowing the true contradiction matters vs. any
                            sparse gradient.

The two selective arms need the gold sidecar ``gold_fingerprints.json`` produced by
``convert_unified_to_document_landmark.py --emit dense --emit-gold-sidecar`` (each example's gold docs
are looked up by a content fingerprint of ``input_ids`` -- nothing gold-specific enters the token
stream). Run with ``--no-compile`` (the per-forward Python fingerprint/RNG is not compile-capturable).

Run::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=8 \\
      src/scripts/train/memexpress/goldgrad/Qwen3-0.6B-goldgrad-contradiction-n20-SFT-local.py \\
      --data-dir /scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense_nocot_gold \\
      --grad-mode gold_plus_random --n-random 2 --save-checkpoint \\
      --run-name q06b-goldgrad-contra-n20-gpr2
"""

import argparse
import json
from dataclasses import replace

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
    PAD_TOKEN_ID,
    REAL_VOCAB_SIZE,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
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

# ---- LOCAL paths (shared /scratch, readable from any Berkeley GPU node) ----
# Shared dense CPT base so all attention-pattern / grad-mode variants init from the SAME weights.
# MUST be a marker-REPAIRED (-fixmark) base: Qwen3 never trains the <|box_start|>/<|box_end|> rows, so
# on a stock base they are bit-identical (cos=1.0) and the model cannot perceive document structure.
# With 100 docs (200 markers) that silently produces low train CE but ~0 held-out f1 -- a broken-base
# artifact that reads exactly like a modeling result. See records/document-chunked-marker-embeddings.md.
BASE_CHECKPOINT = (
    "/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-fixmark/model_and_optim"
)
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-goldgrad"

SEQUENCE_LENGTH = 2048
LR = 5e-5
NUM_EPOCHS = 3


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    base_checkpoint = opts.base_checkpoint or BASE_CHECKPOINT

    seq_len = opts.seq_len
    global_batch_size = opts.grad_accum * seq_len

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    model_factory = {
        "0.6B": TransformerConfig.qwen3_0_6B,
        "4B": TransformerConfig.qwen3_4B,
    }[opts.model_size]
    # ---- attention mask ----
    # DEFAULT is FULL (plain causal) attention: cross_doc_mode="random_doc" with doc_keep_prob=1.0 is
    # PROVABLY identical to a plain causal mask (allowed = causal & not_pad & (context_ok | q_free |
    # kv_free); at keep_prob=1.0 context_ok covers own + ALL earlier docs, and FREE tokens are always
    # visible -- verified: zero causal-allowed positions blocked, the only delta being benign PAD->PAD
    # self-attention). We go through the document-chunked path rather than a plain dense model ONLY so
    # the box markers, chunk_ids, gold fingerprints and the EXISTING docchunk eval keep working
    # unchanged -- the math is full attention.
    #
    # Why not "chunked": pure chunked (no mask-mixing) is the documented known-to-LOSE control, and it
    # parks every arm -- INCLUDING the full-grad baseline -- at the chance floor (CE 0.486 = format
    # learned, documents guessed), which cannot discriminate the gold-grad arms at all.
    # --plain-attention: a STOCK causal Qwen3 (no DocumentChunkedAttention at all). Legitimate here
    # because our default mask (random_doc @ doc_keep_prob=1.0) is provably plain causal anyway, so the
    # math is unchanged -- it just skips the document-chunked mask machinery. Gold-grad does NOT need
    # that machinery: install_gold_grad_mask patches any Attention.sdpa and reconstructs chunk_ids from
    # input_ids itself. Use this at seq 6144 (n100), where the doc-chunked dense-mask path SIGSEGVs
    # after ~6 minutes of training regardless of arm (the no-hook `full` arm dies identically).
    if opts.plain_attention:
        if opts.cross_doc_mode != "random_doc" or opts.doc_keep_prob != 1.0:
            raise SystemExit(
                "--plain-attention is only equivalent to the chunked path when the mask is already "
                "full causal (--cross-doc-mode random_doc --doc-keep-prob 1.0)."
            )
        model_config = model_factory(
            vocab_size=tokenizer_config.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_2,
        )
    else:
        qwen_kwargs: dict = dict(
            vocab_size=tokenizer_config.padded_vocab_size(),
            document_chunked=True,
            cross_doc_mode=opts.cross_doc_mode,
            # qwen3 factories default backend=None -> TorchAttentionBackend (no KV-cache at eval
            # decode). Pin flash_2 so the saved config.json supports KV-cached generation.
            attn_backend=AttentionBackendName.flash_2,
        )
        # doc_keep_prob is a MODEL-FACTORY knob (config.py asserts it is only valid with random_doc);
        # it is NOT accepted by enable_document_chunk_attention(), which takes only ids/mode/mix keys.
        if opts.cross_doc_mode == "random_doc":
            qwen_kwargs["doc_keep_prob"] = opts.doc_keep_prob
        model_config = model_factory(**qwen_kwargs)
        # Runtime chunk_ids reconstruction from the box markers.
        model_config.document_chunk_attention = {
            "doc_start_id": DOC_START_ID,
            "doc_end_id": DOC_END_ID,
            "eos_id": EOS_TOKEN_ID,
            "mode": opts.cross_doc_mode,
        }

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
            shard_degree=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
        f"{opts.data_dir}/token_ids_part_*.npy",
        tokenizer=doc_tokenizer_config,
        sequence_length=seq_len,
        label_mask_paths=[f"{opts.data_dir}/labels_mask_*.npy"],
        expand_glob=True,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir or WORK_DIR,
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=opts.num_workers,
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
            max_duration=Duration.epochs(opts.epochs),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )
    if opts.checkpoint_during_training:
        trainer_config = trainer_config.with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000, ephemeral_save_interval=250, max_checkpoints=2, save_async=True
            ),
        )
    if opts.wandb:
        from datetime import datetime

        run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=opts.wandb_group or "goldgrad-contra-n20",
                entity=opts.wandb_entity,
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    seed_all(12536)
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)

    # ---- Gold-gradient (O(1)-backward) install: detach non-selected docs' K/V ----
    if opts.grad_mode != "full":
        import os

        from olmo_core.nn.attention.gold_grad_mask import (
            install_gold_grad_mask,
            make_fingerprint_gold_mask_fn,
        )

        # The pair-aware modes need the PAIR-preserving sidecar (values are [[a, b], ...]); the
        # doc-level modes are happy with the flat one. gold_pairs.json is built by
        # debug/build_gold_pairs.py from the source JSONL's gold_doc_indices.
        needs_pairs = opts.grad_mode in ("gold_pair", "gold_halves")
        default_sidecar = "gold_pairs.json" if needs_pairs else "gold_fingerprints.json"
        gold_path = opts.gold_path or f"{opts.data_dir}/{default_sidecar}"
        if not os.path.exists(gold_path):
            raise SystemExit(
                f"--grad-mode={opts.grad_mode} needs a gold sidecar; not found at {gold_path}. "
                + (
                    "Build it with debug/build_gold_pairs.py (the flat sidecar cannot express which "
                    "doc contradicts which)."
                    if needs_pairs
                    else "Re-tokenize with convert_unified_to_document_landmark.py --emit-gold-sidecar."
                )
            )
        gold_table = json.load(open(gold_path))
        gm_fn = make_fingerprint_gold_mask_fn(
            gold_table,
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
            eos_id=EOS_TOKEN_ID,
            n_random=opts.n_random,
            mode=opts.grad_mode,
            seed=opts.grad_seed,
            n_gold=opts.n_gold,
            n_pairs=opts.n_pairs,
        )
        holder = install_gold_grad_mask(train_module.model, gm_fn)
        if get_rank() == 0:
            print(
                f"[goldgrad] mode={opts.grad_mode} n_random={opts.n_random} n_gold={opts.n_gold} "
                f"n_pairs={opts.n_pairs} sidecar={os.path.basename(gold_path)} "
                f"patched {holder.n_patched} attention modules; {len(gold_table)} gold examples",
                flush=True,
            )
    elif get_rank() == 0:
        print("[goldgrad] mode=full -> no gold-grad install (full-gradient baseline)", flush=True)

    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    if opts.save_checkpoint:
        from olmo_core.distributed.checkpoint import save_model_and_optim_state

        model_dir = f"{save_folder}/model_and_optim"
        save_model_and_optim_state(model_dir, train_module.model, save_overwrite=True)
        if get_rank() == 0:
            experiment = {
                "model": model_config.as_config_dict(),
                "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
            }
            with open(f"{save_folder}/config.json", "w") as f:
                json.dump(experiment, f)
            print(f"[goldgrad] saved model-only checkpoint -> {save_folder}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        default="/scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense_nocot_gold",
        help="docdense-gold shard dir (token_ids_part_*.npy + labels_mask_*.npy + gold_fingerprints.json)",
    )
    ap.add_argument(
        "--n-gold",
        type=int,
        default=1,
        help="gold_subsample / gold_halves only: how many gold docs to keep (constant in "
        "N -> O(1)). For gold_halves these are drawn from DISTINCT pairs.",
    )
    ap.add_argument(
        "--n-pairs",
        type=int,
        default=1,
        help="gold_pair only: how many COMPLETE contradicting pairs to keep (2 docs each)",
    )
    ap.add_argument(
        "--plain-attention",
        action="store_true",
        help="stock causal Qwen3 (no DocumentChunkedAttention). Math-identical to the "
        "default random_doc@1.0 mask, but skips the doc-chunked dense-mask path that "
        "SIGSEGVs at seq 6144. gold-grad still works (it reads chunk_ids from ids).",
    )
    ap.add_argument(
        "--grad-mode",
        choices=[
            "full",
            "gold_plus_random",
            "gold_subsample",
            "random_only",
            "random_nongold",
            "gold_pair",
            "gold_halves",
        ],
        default="gold_plus_random",
        help="full = baseline (all docs get grad); gold_plus_random = O(1) (gold + n_random); "
        "random_only = same-sparsity control drawn from ALL docs (NB: keeps ~40%% of "
        "gold BY CHANCE -- not gold-free); random_nongold = STRICT gold-free control "
        "(same sparsity, sampled only from non-gold docs); gold_pair = keep n_pairs "
        "COMPLETE contradicting pairs; gold_halves = its matched control (same gold "
        "COUNT, but orphaned halves from distinct pairs -- never a complete pair)",
    )
    ap.add_argument(
        "--n-random",
        type=int,
        default=2,
        help="extra random docs kept beyond the forced set (see --grad-mode)",
    )
    ap.add_argument(
        "--grad-seed", type=int, default=0, help="seed for the per-example random doc choice"
    )
    ap.add_argument(
        "--gold-path",
        default=None,
        help="override gold sidecar path (default <data-dir>/gold_fingerprints.json)",
    )
    ap.add_argument("--run-name", default="q06b-goldgrad-contra-n20")
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help=f"model_and_optim distcp subdir (default {BASE_CHECKPOINT})",
    )
    ap.add_argument("--work-dir", default=None, help=f"data-loader cache dir (default {WORK_DIR})")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--model-size", choices=["0.6B", "4B"], default="0.6B")
    # DEFAULT random_doc + doc_keep_prob=1.0 == plain FULL causal attention (see the model_config
    # comment). "chunked" is available but is the known-to-lose control -- every arm floors out.
    ap.add_argument(
        "--cross-doc-mode",
        choices=["random_doc", "chunked"],
        default="random_doc",
        help="random_doc(+--doc-keep-prob 1.0) = FULL causal attention (default)",
    )
    ap.add_argument(
        "--doc-keep-prob",
        type=float,
        default=1.0,
        help="random_doc keep prob; 1.0 = full causal, 0.0 = pure chunked",
    )
    ap.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    ap.add_argument(
        "--grad-accum", type=int, default=8, help="instances per optimizer step (mbs = seq_len)"
    )
    ap.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="after fit, save model-only checkpoint for eval",
    )
    ap.add_argument(
        "--checkpoint-during-training",
        action="store_true",
        help="also periodic-checkpoint during fit",
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full; smoke)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="torch.compile the model (default OFF; the per-forward gold fingerprint is not compile-capturable)",
    )
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-entity", default=None)
    opts = ap.parse_args()

    # NOTE: this launcher deliberately omits `faulthandler.dump_traceback_later(...)` (the other
    # -SFT-local templates arm it). I once blamed that watchdog for the intermittent `exitcode: -11` on
    # these runs; that was WRONG -- it defaults to exit=False, and the `full` arm fired it 4x and still
    # completed 750/750 steps. Leaving it out only because nothing here needs it, NOT as a bug fix, and
    # it should NOT be stripped from the other launchers.
    #
    # The intermittent -11 remains UNEXPLAINED (it is intermittent, not deterministic: the same script
    # on the same data both crashes and completes). Prime suspect is the known flash-attn 2.8.3
    # varlen-backward SIGSEGV -- pin 2.8.2. Do not add a new theory without a controlled test that
    # survives WITH the suspect present.
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
