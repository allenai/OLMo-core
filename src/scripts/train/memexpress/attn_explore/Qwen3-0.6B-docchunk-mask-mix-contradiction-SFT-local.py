"""LOCAL (torchrun, no Beaker/weka) docchunk + **mask-mixing** smoke vehicle.

Adapts ``Qwen3-0.6B-fast-landmark-contradiction-n20-SFT-local.py`` to the *document-chunked* path to
exercise the native mask-mixing schedule (see
:meth:`olmo_core.nn.transformer.Transformer.enable_document_chunk_attention`):

  * model: ``qwen3_0_6B(document_chunked=True, cross_doc_mode="chunked")`` (dense chunked mask);
  * data: a docdense contradiction shard (``<|box_start|>`` / ``<|box_end|>`` markers present),
    one example per instance right-padded to ``--seq-len`` (``PadToLengthInstanceSource``);
  * mixing: ``--mix-mode {none,static,curriculum}`` wires the five mix keys into the
    ``document_chunk_attention`` config. ``mix_total_forwards`` is auto-computed as
    ``n_examples * epochs`` (global batch == seq len -> 1 instance/step -> forwards == steps).

Run with ``--no-compile`` (the python-seeded mix coin + counter are not torch.compile-capturable)::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=1 \\
      src/scripts/train/memexpress/attn_explore/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py \\
      --data-dir /scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense \\
      --mix-mode curriculum --mix-start-p 0.8 --mix-end-p 0.0 --mix-log-interval 5 \\
      --no-compile --run-name smoke-curriculum
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
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.chunked_mask import mask_mix_standard_prob
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

# ---- LOCAL paths (shared NFS / readable from any Berkeley GPU node) ----
# Base: an existing qwen3_0_6B dense olmo-core distcp (d_model 1024 / n_layers 28 / vocab 151936 ==
# TransformerConfig.qwen3_0_6B). Point at the model_and_optim SUBDIR (parent has no .metadata marker).
# Shared dense CPT base (on shared /scratch, so all attention-pattern variants -- full / chunked /
# hierarchical -- init from the SAME weights for a fair comparison). Point at the model_and_optim subdir.
BASE_CHECKPOINT = "/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly/model_and_optim"
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-docchunk-mix"

LR = 5e-5
NUM_EPOCHS = 3


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    base_checkpoint = opts.base_checkpoint or BASE_CHECKPOINT

    meta = json.load(open(f"{opts.data_dir}/metadata.json"))
    n_examples = int(meta["num_instances"])
    seq_len = opts.seq_len
    if meta.get("max_example_len", 0) > seq_len:
        raise SystemExit(
            f"--seq-len={seq_len} < max example length {meta['max_example_len']} "
            f"(PadToLength would SKIP the long examples). Raise --seq-len."
        )
    # global batch = grad_accum instances; each rank holds grad_accum/world_size of them and chews
    # through them micro_batch_instances at a time.
    #
    # ⚠ mix_total_forwards must be the number of FORWARDS a rank actually performs, because that is
    # what the curriculum's counter counts. Get the divisor wrong and p_standard silently stops short
    # of mix_end_p, leaving the model training on PLAIN CAUSAL for a large fraction of forwards
    # forever -- which quietly voids any experiment about attention masks, since the mask it is
    # supposed to be learning under is only partly in force. Two divisors, both learned the hard way:
    #   * world_size    -- data is sharded across DP ranks (the original mask-mix-ngpu-anneal bug:
    #                      p stalled at mix_start_p*(1-1/world_size), worse the more GPUs).
    #   * micro_batch_instances -- a forward carries this many instances, so raising it CUTS the
    #                      forward count by the same factor. Observed 2026-07-16: with
    #                      micro_batch_instances=4 on 2 GPUs, p_standard ended at 0.601 instead of
    #                      0.0 (927 real forwards vs an assumed 3710 => only 25% of the way down).
    # The invariant to hold onto: forwards_per_rank == number of optimizer steps.
    global_batch_size = opts.grad_accum * seq_len
    world_size = max(1, get_world_size())
    # A rank only ever holds global_batch/world_size instances; asking for more per forward would
    # make rank_microbatch_size exceed the rank's share. Fail loudly rather than let olmo-core
    # silently reinterpret the batch (which would change the effective global batch, and with it the
    # optimization -- exactly the kind of quiet drift that makes runs non-comparable).
    _per_rank = opts.grad_accum // world_size
    if _per_rank < 1 or opts.micro_batch_instances > _per_rank:
        raise SystemExit(
            f"--micro-batch-instances={opts.micro_batch_instances} invalid: with --grad-accum="
            f"{opts.grad_accum} and world_size={world_size} each rank holds {_per_rank} instance(s). "
            f"Use a value in [1, {_per_rank}]."
        )
    mix_total_forwards = max(
        1, (n_examples * opts.epochs) // (world_size * opts.micro_batch_instances)
    )

    # ---- mask-mix config keys threaded into document_chunk_attention ----
    mix_keys: dict = {}
    if opts.mix_mode == "static":
        mix_keys = dict(
            standard_mix_prob=opts.standard_mix_prob,
            mix_seed=opts.mix_seed,
            mix_log_interval=opts.mix_log_interval,
        )
    elif opts.mix_mode == "curriculum":
        mix_keys = dict(
            mix_start_p=opts.mix_start_p,
            mix_end_p=opts.mix_end_p,
            mix_total_forwards=mix_total_forwards,
            mix_seed=opts.mix_seed,
            mix_log_interval=opts.mix_log_interval,
        )
    # Independently re-derive the forward count from the STEP count and check the curriculum will
    # actually land on mix_end_p. A silent shortfall here does not crash and does not look wrong in
    # the loss -- it just means the model never trained predominantly on its target mask, which
    # invalidates the run while it still reports a plausible CE. Assert instead of trusting.
    if opts.mix_mode == "curriculum":
        steps = max(1, (n_examples * opts.epochs) // opts.grad_accum)
        forwards_per_rank = steps * (_per_rank // opts.micro_batch_instances)
        final_p = mask_mix_standard_prob(
            forwards_per_rank - 1,
            mix_start_p=opts.mix_start_p,
            mix_end_p=opts.mix_end_p,
            mix_total_forwards=mix_total_forwards,
        )
        print(
            f"[docchunk-mix] steps={steps} forwards_per_rank={forwards_per_rank} "
            f"predicted final p_standard={final_p:.4f}",
            flush=True,
        )
        # Tolerance is 0.01, not 1e-3: integer-division rounding legitimately leaves the last forward
        # a hair off (p=0.0009 at world=2/micro=4). We are catching gross shortfalls like the observed
        # 0.601, not rounding dust.
        if abs(final_p - opts.mix_end_p) > 0.01:
            raise SystemExit(
                f"mask-mix curriculum would end at p_standard={final_p:.4f}, not mix_end_p="
                f"{opts.mix_end_p}: mix_total_forwards={mix_total_forwards} disagrees with the "
                f"{forwards_per_rank} forwards this rank will actually run. The model would keep "
                f"training on PLAIN CAUSAL for {final_p:.0%} of forwards forever, silently voiding "
                f"any mask experiment. Check the world_size / micro_batch_instances divisors."
            )
    print(
        f"[docchunk-mix] mode={opts.mix_mode} n_examples={n_examples} epochs={opts.epochs} "
        f"seq_len={seq_len} mix_total_forwards={mix_total_forwards} mix_keys={mix_keys}",
        flush=True,
    )

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- cross-document visibility policy (attention-pattern sweep) ----
    #   * "chunked"            -> plain document-chunked (each context doc attends itself only; the
    #                             free instruction/query/answer tokens attend the full context).
    #   * "hierarchical_dilated" -> a strict superset of "chunked": each context query additionally
    #                             attends dilation_n docs (incl. self) at a per-layer stride of
    #                             dilation_m**layer (chunk / whole-document granularity). The FREE
    #                             query + answer tokens still attend the full context. This is the
    #                             intended "hierarchical" variant (NOT the positional
    #                             dilated_sliding_window, which is token-granularity + no free/beginning).
    qwen_kwargs: dict = dict(
        vocab_size=tokenizer_config.padded_vocab_size(),
        document_chunked=True,
        cross_doc_mode=opts.cross_doc_mode,
        # qwen3_0_6B does NOT inject an attn_backend (unlike olmo2/qwen3_5 factories), so
        # llama_like defaults backend=None -> TorchAttentionBackend, which can't KV-cache during
        # eval decode. Pin flash_2 so the saved config.json supports KV-cached generation.
        attn_backend=AttentionBackendName.flash_2,
    )
    if opts.cross_doc_mode == "hierarchical_dilated":
        # dilation_n / dilation_m are ONLY valid with hierarchical_dilated (config.py asserts this).
        qwen_kwargs["dilation_n"] = opts.dilation_n
        qwen_kwargs["dilation_m"] = opts.dilation_m
        if opts.dilation_cycle is not None:
            # rotating "Hierarchical K": stride resets every L layers instead of saturating.
            qwen_kwargs["dilation_cycle"] = opts.dilation_cycle
    elif opts.cross_doc_mode == "random_doc":
        # random_doc: each context doc attends itself + a seeded-random ``doc_keep_prob`` subset of the
        # STRICTLY EARLIER docs (BigBird-style; RANDOM connectivity, not strided). FREE query/answer
        # tokens still attend the full context. keep_prob=1.0 == full cross-doc (approaches full attn).
        qwen_kwargs["doc_keep_prob"] = opts.doc_keep_prob
        qwen_kwargs["random_doc_seed"] = opts.random_doc_seed
        qwen_kwargs["random_doc_per_example"] = opts.random_doc_per_example
    elif opts.cross_doc_mode == "summary_attention":
        # summary_attention: documents are grouped into CELLS of summary_every_k docs, each followed by
        # its own SUMMARY SPAN chunk. Within a cell attention is full; the span reads its whole cell (it
        # sits at the cell END, so causally it has already seen every doc of the cell) and is then
        # attendable by every LATER cell -> any two docs in different cells are exactly 2 hops apart,
        # with no gold-aware term anywhere. summary_bandwidth throttles how many of each span's leading
        # tokens are visible: that is the ladder's dose axis, and it needs no separate shard.
        qwen_kwargs["summary_every_k"] = opts.summary_every_k
        qwen_kwargs["summary_bandwidth"] = opts.summary_bandwidth
        qwen_kwargs["summary_relay"] = opts.summary_relay
    # Hybrid full/chunked: these layer indices run plain full (causal) attention while the rest use
    # the chunked mask. Orthogonal to cross_doc_mode (applies to any document_chunked variant).
    full_attn_layers = None
    if opts.full_attention_layers:
        full_attn_layers = [
            int(x) for x in opts.full_attention_layers.split(",") if x.strip() != ""
        ]
        qwen_kwargs["full_attention_layers"] = full_attn_layers
    print(
        f"[docchunk-mix] cross_doc_mode={opts.cross_doc_mode} "
        f"full_attention_layers={full_attn_layers} "
        f"dilation_n={opts.dilation_n if opts.cross_doc_mode=='hierarchical_dilated' else None} "
        f"dilation_m={opts.dilation_m if opts.cross_doc_mode=='hierarchical_dilated' else None} "
        f"dilation_cycle={opts.dilation_cycle if opts.cross_doc_mode=='hierarchical_dilated' else None} "
        f"doc_keep_prob={opts.doc_keep_prob if opts.cross_doc_mode=='random_doc' else None} "
        f"summary_every_k={opts.summary_every_k if opts.cross_doc_mode=='summary_attention' else None} "
        f"summary_bandwidth={opts.summary_bandwidth if opts.cross_doc_mode=='summary_attention' else None} "
        f"summary_relay={opts.summary_relay if opts.cross_doc_mode=='summary_attention' else None}",
        flush=True,
    )
    model_factory = {
        "0.6B": TransformerConfig.qwen3_0_6B,
        "4B": TransformerConfig.qwen3_4B,
    }[opts.model_size]
    model_config = model_factory(**qwen_kwargs)
    model_config.document_chunk_attention = {
        "doc_start_id": DOC_START_ID,
        "doc_end_id": DOC_END_ID,
        "eos_id": EOS_TOKEN_ID,
        "mode": "chunked",
        **mix_keys,
    }

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.micro_batch_instances * seq_len,
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
        seed=34521 + opts.seed,
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
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.epochs(opts.epochs),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
            # no_checkpoints=True would ALSO skip the base-checkpoint LOAD block (trainer.fit gates
            # loading on `not no_checkpoints`) -> silent train-from-scratch (CE ~ln(vocab) ~= 12).
            # Keep it False so load_path (the dense CPT base) is loaded; there is no CheckpointerCallback
            # here so nothing is auto-saved during training -- the explicit --save-checkpoint below
            # writes the single model-only checkpoint for eval.
            no_checkpoints=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )

    # wandb. The docchunk path had NO wandb wiring at all, so every masked arm (chunked /
    # hierarchical / random_doc / the full-attention-layer hybrids) logged only to a slurm file --
    # while the `full` arm, which uses a different script, logged normally. Same group as the dense
    # script ("memory-networks") so masked and dense arms of one sweep land side by side.
    if opts.wandb:
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=opts.wandb_group or run_name,
                entity=opts.wandb_entity,
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
                config={
                    "cross_doc_mode": opts.cross_doc_mode,
                    "summary_every_k": opts.summary_every_k,
                    "summary_bandwidth": opts.summary_bandwidth,
                    "summary_relay": opts.summary_relay,
                    "full_attention_layers": opts.full_attention_layers or "none",
                    "dilation_n": opts.dilation_n,
                    "dilation_m": opts.dilation_m,
                    "dilation_cycle": opts.dilation_cycle,
                    "doc_keep_prob": opts.doc_keep_prob,
                    "mix_mode": opts.mix_mode,
                    "seq_len": seq_len,
                    "epochs": opts.epochs,
                    "seed": opts.seed,
                    "n_examples": n_examples,
                    "base_checkpoint": base_checkpoint,
                    "data_dir": opts.data_dir,
                },
            ),
        )

    seed_all(12536 + opts.seed)
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    # Save a model-only checkpoint in the eval loader's expected layout
    # (config.json + model_and_optim/, matching scripts/train/convert_hf_to_olmo.py) so
    # eval_lc_native_docchunk_contra.py --model-path <save_folder> loads it directly.
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
            print(f"[docchunk-mix] saved model-only checkpoint -> {save_folder}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        required=True,
        help="docdense shard dir (token_ids_part_*.npy + metadata.json)",
    )
    ap.add_argument("--run-name", default="q06b-docchunk-mix-smoke")
    ap.add_argument(
        "--model-size",
        choices=["0.6B", "4B"],
        default="0.6B",
        help="qwen3_0_6B vs qwen3_4B factory (docchunk kwargs thread through identically)",
    )
    ap.add_argument("--save-folder", default=None)
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help=f"model_and_optim distcp subdir (default {BASE_CHECKPOINT})",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help=f"data-loader cache dir (default {WORK_DIR}); use node-local /data for speed",
    )
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Dataloader worker processes per rank. Was hardcoded to 2; the goldgrad launcher's "
        "sbatch overrides its equivalent to 0. Use 0 when running several torchrun jobs on one "
        "node -- the whole dataset is a memory-mapped .npy already staged node-local, so the "
        "workers buy nothing here and only add processes that can wedge.",
    )
    ap.add_argument(
        "--micro-batch-instances",
        type=int,
        default=1,
        help="Instances per device per FORWARD (rank_microbatch_size = this * seq_len). Pure "
        "throughput knob: gradient accumulation is exact, so raising it is mathematically "
        "IDENTICAL -- same global batch, same steps, same result -- it just does the rank's share "
        "in fewer, larger forwards. Cannot exceed global_batch/world_size (= --grad-accum/NGPU), "
        "which is the rank's whole share; beyond that there is nothing left to batch. Default 1 "
        "preserves the historical setting every prior run used. Raise it when running on FEWER "
        "GPUs (e.g. --grad-accum 8 on 2 GPUs -> 4 instances/rank -> --micro-batch-instances 4), "
        "where 1 instance of a few-thousand tokens badly underuses an H200.",
    )
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="gradient accumulation = instances per optimizer step (mbs=1)",
    )
    ap.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="after fit, save model-only checkpoint (config.json + model_and_optim) for eval",
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument(
        "--cross-doc-mode",
        choices=["standard", "chunked", "hierarchical_dilated", "random_doc", "summary_attention"],
        default="chunked",
        help="document-chunked cross-document visibility policy (see build_and_fit). "
        "'standard' is plain causal attention through this module -- the control for "
        "separating a mask effect from a module/data problem",
    )
    ap.add_argument(
        "--dilation-n",
        type=int,
        default=3,
        help="hierarchical_dilated: #docs a context query attends per layer (incl. self)",
    )
    ap.add_argument(
        "--dilation-m",
        type=int,
        default=2,
        help="hierarchical_dilated: dilation base; per-layer stride = dilation_m**layer",
    )
    ap.add_argument(
        "--dilation-cycle",
        type=int,
        default=None,
        help="hierarchical_dilated: rotation period L. Stride ROTATES with depth as "
        "dilation_m**(layer_idx %% L) (deep layers revisit short strides) instead of the "
        "default pure-saturation dilation_m**layer_idx (stride grows monotonically). "
        "None = saturating (legacy); e.g. 3 = rotating 'Hierarchical K'.",
    )
    ap.add_argument(
        "--doc-keep-prob",
        type=float,
        default=0.5,
        help="random_doc: fraction of STRICTLY-EARLIER docs each context doc attends (0..1)",
    )
    ap.add_argument("--random-doc-seed", type=int, default=42, help="random_doc: RNG seed")
    ap.add_argument("--summary-every-k", type=int, default=10,
                    help="summary_attention: CELL SIZE -- documents per cell. MUST equal the shard's "
                         "--summary-every-k, and the shard must actually carry one summary span per "
                         "cell: the mask identifies a span purely as (chunk_id %% (k+1)) == k, so a "
                         "mismatch silently rebinds every chunk role rather than erroring.")
    ap.add_argument("--summary-bandwidth", type=int, default=0,
                    help="summary_attention: RELAY BANDWIDTH b -- how many of each earlier summary "
                         "span's leading tokens a later chunk may attend. The ladder's dose knob; the "
                         "DATA is identical across rungs, only visibility changes. b=0 removes the "
                         "relay entirely and reproduces a pure cell-blocks mask BIT-EXACTLY (the "
                         "ladder's floor control). Ladder: 0 / 1 / 4 / 16 / 42. b MUST be <= the "
                         "SHORTEST span (42 tokens, measured) or cell 0 silently exposes fewer keys "
                         "than the other cells and the dose axis is not uniform.")
    ap.add_argument("--no-summary-relay", dest="summary_relay", action="store_false",
                    help="summary_attention: PLACEBO -- forbid each summary span from reading its own "
                         "cell. The span keeps its position, its tokens and every edge into it, but "
                         "reads nothing, so it provably carries ZERO document content (verified in "
                         "src/test/nn/attention/summary_attention_test.py). Run at the TOP rung: it "
                         "must land on b=0; if it does not, information-free keys move the metric on "
                         "their own and every rung must be read against it. NOT a hop-inf control.")
    ap.set_defaults(summary_relay=True)
    ap.add_argument(
        "--random-doc-per-example",
        action="store_true",
        help="random_doc: give EVERY EXAMPLE its own random sparsity graph (deterministic per "
        "example) instead of one graph shared by all examples. Ablates whether a STABLE mask "
        "matters or whether sparse-but-varied connectivity suffices.",
    )
    ap.add_argument(
        "--full-attention-layers",
        type=str,
        default="",
        help="comma-separated layer indices that use PLAIN FULL (causal) attention instead "
        "of the chunked mask (hybrid full/chunked). Negative counts from the end "
        "(-1=last). E.g. '0' / '13' / '-1'. Empty = all layers chunked.",
    )
    ap.add_argument("--mix-mode", choices=["none", "static", "curriculum"], default="none")
    ap.add_argument("--standard-mix-prob", type=float, default=0.10)
    ap.add_argument("--mix-start-p", type=float, default=0.80)
    ap.add_argument("--mix-end-p", type=float, default=0.0)
    ap.add_argument("--mix-seed", type=int, default=42)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="run-to-run seed offset: shifts BOTH the data-order seed and the global torch seed. "
        "Vary it (with everything else held fixed) to measure the noise floor -- differences "
        "between attention masks are only meaningful above the spread across these seeds.",
    )
    ap.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="disable wandb (it is ON by default; needs WANDB_API_KEY)",
    )
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument(
        "--wandb-group",
        default=None,
        help="group several arms of one sweep together (default: the run name)",
    )
    ap.add_argument("--mix-log-interval", type=int, default=5)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    opts = ap.parse_args()

    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
