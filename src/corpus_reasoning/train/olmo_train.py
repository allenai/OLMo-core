"""Config-driven olmo-core full fine-tune for any Qwen3-dense model + any dataset.

Generalizes olmo_nq_sft.py (NQ/4B-only) and folds in the long-context knobs from
olmo_qwen3_cp_smoke.py (YaRN rope scaling, ring context parallelism, activation
checkpointing). The dispatcher (scripts/train/train.py, `backend: olmo-core`) resolves
the model, ensures a converted base checkpoint, tokenizes the dataset, computes the
batch/CP knobs, and invokes this script under torchrun. It can also be run by hand.

All shape/builder/vocab decisions go through scripts.lib.olmo_models so the dispatcher,
converter and exporter agree on what's supported (Qwen3 dense, full-FT only).

Launch with torchrun (see jobs/olmo_train.sh).
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import rich

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.optim import CosWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerContextParallelConfig,
)
from olmo_core.utils import seed_all

from corpus_reasoning.lib.olmo_models import build_transformer_config, resolve_olmo_model
from corpus_reasoning.lib.olmo_flex_attention import install_flex_attention

QWEN3_BASE_CTX = 40960  # Qwen3 max_position_embeddings; YaRN beyond this.


@dataclass
class LoRAInitCallback(Callback):
    """Initialize LoRA params (Kaiming A, zeros B) once before training.

    Runs in `pre_train` -- after the base distcp loads (the base ckpt has no LoRA
    keys, so they're never touched by the load) and after the train module's
    to_empty/reset_parameters left them uninitialized. This is the authoritative
    LoRA init. See scripts/lib/olmo_lora.py.
    """

    def pre_train(self):
        from corpus_reasoning.lib.olmo_lora import init_lora_params

        n = init_lora_params(self.trainer.train_module.model)
        if get_rank() == 0:
            print(f"[olmo] LoRA init: kaiming A / zeros B on {n} modules")


@dataclass
class FreezeGDNCallback(Callback):
    """Freeze the GatedDeltaNet (GDN linear-attention) layers before training (set via
    OLMO_FREEZE_GDN=1). Mimics axolotl's LoRA, which leaves the hybrid Qwen3.5 GDN at its
    pretrained init (its LoRA targets q/k/v/o, not the GDN's in_proj_*). Tests whether TRAINING
    the GDN is what drives the SFT loss into a bad basin (CE plateau ~0.83) vs frozen-GDN learning."""

    def pre_train(self):
        from olmo_core.nn.attention.recurrent import GatedDeltaNet
        model = self.trainer.train_module.model
        nmod = nparam = 0
        for m in model.modules():
            if isinstance(m, GatedDeltaNet):
                nmod += 1
                for p in m.parameters():
                    p.requires_grad_(False); nparam += 1
        if get_rank() == 0:
            print(f"[olmo] FROZE GDN: {nmod} GatedDeltaNet modules, {nparam} param-tensors")


def build(opts):
    # Precision arms (see the AdamW step in olmo-core: the weight update `p.add_(update)` happens
    # in the *master* param dtype):
    #   baseline      : fp32 master + fp32 moments. Compute in bf16 via FSDP param_dtype. Tested recipe.
    #   bf16          : bf16 master + bf16 moments. The lr*update is below bf16's 8-bit mantissa and
    #                   gets rounded away each step (olmo-core has NO stochastic rounding) -> big
    #                   accuracy loss. Off the tested path; here only for comparison.
    #   bf16_moments  : fp32 master + bf16 moments. Keeps the accuracy-critical fp32 weight
    #                   accumulation but halves optimizer-moment memory (cf. 8-bit Adam). Recommended saver.
    #   bf16_sr       : bf16 master + bf16 moments + STOCHASTIC ROUNDING on the weight update
    #                   (monkeypatched into olmo-core's AdamW). Recovers full-bf16 memory savings
    #                   without the round-to-nearest accuracy loss. See scripts/lib/sr_adamw.py.
    bf16 = opts.arm == "bf16"
    bf16_moments = opts.arm == "bf16_moments"
    bf16_sr = opts.arm == "bf16_sr"
    fp32 = opts.arm == "fp32"                   # full fp32 compute (no FSDP bf16 cast). For the
                                               # hybrid Qwen3.5: the GDN/SSM scan is bf16-fragile
                                               # (fp32 vs bf16 forward diverges ~13%), so bf16
                                               # compute corrupts the GDN during training. Diagnostic arm.
    model_bf16 = bf16 or bf16_sr               # full-bf16 and SR keep the master in bf16
    optim_bf16 = bf16 or bf16_moments or bf16_sr  # all bf16 arms use bf16 optimizer moments
    if bf16_sr:
        from corpus_reasoning.lib.sr_adamw import enable_stochastic_rounding
        enable_stochastic_rounding()  # patches olmo_core.optim.adamw.{adamw_step,foreach_adamw_step}
        if get_rank() == 0:
            print("[olmo] stochastic rounding ENABLED for bf16 master-weight updates")
    # Custom attention patterns (chunked / modified_swa) run via a FlexAttention
    # monkeypatch installed in main(); they own the full sequence per rank, so ring
    # context parallelism (which never materializes the full K/V on a rank) is not
    # supported. CP=1 only for now (see results/olmo_flex_attention.md).
    if opts.attn_pattern != "standard" and opts.cp_degree > 1:
        raise SystemExit(
            f"--attn-pattern={opts.attn_pattern} requires --cp-degree 1 "
            f"(got {opts.cp_degree}); FlexAttention has no ring-CP kernel.")

    spec = resolve_olmo_model(opts.base_model)

    model_config = build_transformer_config(
        spec, dtype=(DType.bfloat16 if model_bf16 else None))
    # Fused linear cross-entropy (liger): never materialize the full [tokens, vocab] logits
    # tensor — the big OOM at microbatch>1 with Qwen3's 151936 vocab, worse at long ctx.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    # Extend context past Qwen3's native window with YaRN (the OLMo3 long-ctx recipe knob).
    if opts.seq_len > QWEN3_BASE_CTX:
        factor = float(math.ceil(opts.seq_len / QWEN3_BASE_CTX))
        model_config = model_config.with_rope_scaling(
            YaRNRoPEScalingConfig(
                factor=factor, beta_fast=32, beta_slow=1,
                old_context_len=QWEN3_BASE_CTX,
            )
        )

    # Ring context parallelism is only supported by the flash_2 backend (torch SDPA and
    # flash_3 both reject ring CP) and additionally needs the ring-flash-attn package.
    if opts.cp_degree > 1:
        def _use_flash2(attn_cfg):
            attn_cfg.backend = AttentionBackendName.flash_2
            attn_cfg.use_flash = True
        _use_flash2(model_config.block.sequence_mixer)
        if model_config.block_overrides:
            for b in model_config.block_overrides.values():
                _use_flash2(b.sequence_mixer)

    # Use the eos/pad ids the data was actually written with (tokenizer sidecar).
    # bos_token_id=None -> documents are segmented purely on eos (matches our docs).
    meta = json.load(open(opts.meta))
    tok_cfg = TokenizerConfig(
        vocab_size=spec.vocab_size,
        eos_token_id=meta["eos_token_id"],
        pad_token_id=meta["pad_token_id"],
        bos_token_id=None,
        identifier=opts.base_model,
    )

    dataset_config = NumpyPaddedFSLDatasetConfig(
        paths=[opts.tokens],
        label_mask_paths=[opts.label_mask],
        sequence_length=opts.seq_len,
        tokenizer=tok_cfg,
        dtype=NumpyDatasetDType.uint32,   # Qwen vocab > 65535
        work_dir=opts.work_dir,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_tokens,  # TOKENS
        seed=0, num_workers=2,
    )

    # Optimizer (override OLMO_OPTIM=adamw for plain AdamW, matching axolotl's adamw_torch;
    # default SkipStepAdamW skips outlier steps). Testing whether SkipStep is what stalls hybrid SFT.
    import os as _os
    if _os.environ.get("OLMO_OPTIM") == "adamw":
        from olmo_core.optim import AdamWConfig
        optim = AdamWConfig(lr=opts.lr, **({"dtype": DType.bfloat16} if optim_bf16 else {}))
        if get_rank() == 0: print("[olmo] optimizer = plain AdamW (OLMO_OPTIM=adamw)")
    else:
        optim = SkipStepAdamWConfig(
            lr=opts.lr, **({"dtype": DType.bfloat16} if optim_bf16 else {}))

    # Activation checkpointing. `budget` needs compile; `full` is incompatible with
    # compile (CheckpointError). Default off — at <=8k it's unnecessary and slower.
    ac_config = None
    compile_model = False
    if opts.ac_mode == "full":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full)
    elif opts.ac_mode == "budget":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=opts.ac_budget)
        compile_model = True
    elif opts.ac_mode == "selected":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_blocks,
            block_interval=opts.ac_block_interval)

    cp_config = (TransformerContextParallelConfig.zig_zag(degree=opts.cp_degree)
                 if opts.cp_degree > 1 else None)

    # Data-parallel mode override (OLMO_DP_MODE = fsdp|ddp|none). 'none' = no FSDP/DDP wrapping
    # (plain single-GPU module) — used to isolate whether FSDP is what breaks hybrid Qwen3.5 training.
    import os as _os
    _dp_mode = _os.environ.get("OLMO_DP_MODE", "fsdp")
    if _dp_mode == "none":
        _dp_config = None
        if get_rank() == 0: print("[olmo] dp_config=None (no FSDP/DDP wrapping)")
    else:
        _dp_config = TransformerDataParallelConfig(
            name=(DataParallelType.ddp if _dp_mode == "ddp" else DataParallelType.fsdp),
            # Compute in bf16 via FSDP cast, EXCEPT when the master is already bf16 (full-bf16 arm)
            # or the fp32 arm (no cast -> fp32 compute; needed for the bf16-fragile GDN/SSM scan).
            param_dtype=(None if (model_bf16 or fp32) else DType.bfloat16),
            reduce_dtype=DType.float32,
        )
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_tokens,
        max_sequence_length=opts.seq_len,
        optim=optim,
        compile_model=compile_model,
        dp_config=_dp_config,
        cp_config=cp_config,
        ac_config=ac_config,
        # Grad clip (override via OLMO_MAX_GRAD_NORM). Default 1.0 — but the hybrid Qwen3.5 trains
        # with grad-norm ~13, so clipping to 1.0 shrinks the update ~13x and the SFT loss plateaus
        # while axolotl (lighter clipping) converges. Set high (e.g. 1e9) to disable.
        max_grad_norm=float(_os.environ.get("OLMO_MAX_GRAD_NORM", "1.0")),
        # warmup_steps takes precedence when >0 (exact step count, e.g. to reproduce a prior run);
        # otherwise warmup_fraction (relative to total steps).
        scheduler=(CosWithWarmup(warmup_steps=opts.warmup_steps) if opts.warmup_steps > 0
                   else CosWithWarmup(warmup_fraction=opts.warmup_fraction)),
    )
    if opts.lora:
        # The LoRA adapter params (lora_A/lora_B) are NOT in the converted base distcp.
        # A strict model load would error on those extra params; non-strict prunes the
        # absent keys and loads the base weights cleanly (they keep their original names).
        # flatten_optimizer_state_dict mirrors olmo-core's own default for load opts.
        train_module_config.state_dict_load_opts = {
            "strict": False, "flatten_optimizer_state_dict": True}

    # load_path MUST point at the model_and_optim subdir, NOT the converted-ckpt parent.
    # The trainer's gate (Checkpointer.dir_is_checkpoint) only recognizes a dir as a
    # "model-only" checkpoint if `<dir>/.metadata` sits at its root; the converter writes
    # that marker inside `<ckpt>/model_and_optim/.metadata`. Pointing at the parent makes the
    # gate return False and the trainer silently trains FROM SCRATCH (12.5 == ln(vocab) loss).
    # Append defensively so both the dispatcher (passes the parent) and manual torchrun
    # callers (may pass either) resolve correctly.
    load_path = opts.load_path
    if load_path is not None and os.path.isdir(os.path.join(load_path, "model_and_optim")):
        load_path = os.path.join(load_path, "model_and_optim")

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            load_path=load_path,             # converted base weights (model_and_optim subdir)
            load_trainer_state=False,        # weights only -- fresh optimizer/data state
            load_optim_state=False,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            # max_steps > 0 caps training (smoke/debug); otherwise run the full epochs.
            max_duration=(Duration.steps(opts.max_steps) if opts.max_steps > 0
                          else Duration.epochs(opts.epochs)),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                enabled=True,
                project=opts.wandb_project,
                name=opts.wandb_name,
                tags=["olmo-core", spec.builder, opts.arm],
            ),
        )
        # pre_train_checkpoint=False: skip the wasteful step-0 save (it fires because
        # load_path doesn't set checkpoint_loaded).
        # enabled=opts.save_optim: by DEFAULT we do NOT use the trainer's checkpointer at all,
        # because its only save (post_train) writes model+optimizer together -- and the AdamW
        # moments are ~2x the model size (a 4B fp32 ckpt is 16GB model + 32GB optim = 48GB).
        # We never resume these runs (we only export for eval), so when save_optim is False we
        # disable this callback and write a MODEL-ONLY checkpoint after fit() instead (see main).
        # Pass --save-optim to get a full resumable model+optim checkpoint.
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=1_000_000, pre_train_checkpoint=False,
                                 save_async=False, enabled=opts.save_optim),
        )
    )
    if opts.lora:
        trainer_config = trainer_config.with_callback("lora_init", LoRAInitCallback())
    import os as _os2
    if _os2.environ.get("OLMO_FREEZE_GDN"):
        trainer_config = trainer_config.with_callback("freeze_gdn", FreezeGDNCallback())
    return spec, model_config, dataset_config, data_loader_config, train_module_config, trainer_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    # Optional: only needed for real training; the --dry-run dataset gate never loads weights.
    ap.add_argument("--load-path", default=None, help="converted olmo-core base ckpt")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--label-mask", required=True)
    ap.add_argument("--meta", required=True, help="tokenizer sidecar (_meta.json)")
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--global-batch-tokens", type=int, required=True)
    ap.add_argument("--rank-microbatch-tokens", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 caps training (smoke/debug)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-fraction", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=0, help=">0 overrides warmup-fraction")
    ap.add_argument("--cp-degree", type=int, default=1)
    ap.add_argument("--attn-pattern", default="standard",
                    help="standard | chunked | modified_swa (FlexAttention; needs "
                         "--wrap-docs data and --cp-degree 1)")
    ap.add_argument("--attn-window", type=int, default=512,
                    help="token-window width W for modified_swa")
    ap.add_argument("--compile-flex", action="store_true",
                    help="torch.compile the flex BlockMask build + attention kernel")
    # Mix curriculum: per-forward prob of collapsing an example to plain causal anneals
    # linearly from --mix-start-p to --mix-end-p over training. Runtime counterpart of the
    # tokenize-time standard_mix_prob; needs marker-wrapped data tokenized with mix=0.
    ap.add_argument("--mix-start-p", type=float, default=0.0)
    ap.add_argument("--mix-end-p", type=float, default=0.0)
    ap.add_argument("--mix-seed", type=int, default=42)
    ap.add_argument("--ac-mode", choices=["none", "full", "budget", "selected"], default="none")
    ap.add_argument("--ac-budget", type=float, default=0.5)
    ap.add_argument("--ac-block-interval", type=int, default=2)
    # bf16_sr is the default: empirically at parity with baseline on NQ-k50 (recall 0.500 vs 0.498)
    # and contradiction n=100/k=3 (f1 0.955 vs 0.948), while delivering full-bf16 memory savings
    # (bf16 master + bf16 moments via the SR monkeypatch). See scripts/lib/sr_adamw.py.
    ap.add_argument("--arm", choices=["baseline", "bf16", "bf16_moments", "bf16_sr", "fp32"], default="bf16_sr")
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--wandb-project", default="corpus-reasoning")
    ap.add_argument("--wandb-name", default="olmo_run")
    ap.add_argument("--work-dir", default="/tmp/olmo_cache")
    ap.add_argument("--save-optim", action="store_true",
                    help="save full model+optimizer checkpoint (resumable, ~3x larger); "
                         "default is model-only (we only export for eval, never resume)")
    ap.add_argument("--lora", action="store_true",
                    help="LoRA fine-tune: freeze the base, train only injected adapters")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", choices=["all_linear", "attn_only"], default="all_linear",
                    help="all_linear = attn q/k/v/o + MLP gate/up/down; attn_only = q/k/v/o")
    ap.add_argument("--gold-grad-mask", action="store_true",
                    help="gradient-mask to ground-truth documents: detach K/V of "
                         "distractor-doc tokens so only gold docs + free tokens drive "
                         "the weight update (needs --wrap-docs data + a gold sidecar)")
    ap.add_argument("--gold-path", default=None,
                    help="{content_fingerprint -> gold chunk idx} sidecar (meta.gold_path)")
    ap.add_argument("--dry-run", action="store_true")
    opts = ap.parse_args()

    spec, mc, dc, dlc, tmc, tc = build(opts)
    if get_rank() == 0:
        summary = {"base_model": opts.base_model, "builder": spec.builder,
                   "arm": opts.arm, "seq_len": opts.seq_len, "epochs": opts.epochs,
                   "lr": opts.lr, "cp_degree": opts.cp_degree, "ac_mode": opts.ac_mode,
                   "global_batch_tokens": opts.global_batch_tokens,
                   "rank_microbatch_tokens": opts.rank_microbatch_tokens,
                   "save_folder": opts.save_folder}
        if opts.lora:
            summary["lora"] = {"r": opts.lora_r, "alpha": opts.lora_alpha,
                               "dropout": opts.lora_dropout, "target": opts.lora_target}
        rich.print(summary)

    if opts.dry_run:
        ds = dc.build()
        ds.prepare()  # builds the instance-index cache (data loader does this at train time)
        if opts.lora and get_rank() == 0:
            # Cheap sanity check: build a meta model, inject LoRA, freeze, and report the
            # trainable fraction (no GPU / no checkpoint needed -- meta tensors).
            from corpus_reasoning.lib.olmo_lora import freeze_base, inject_lora
            _m = mc.build(init_device="meta")
            patched = inject_lora(_m, r=opts.lora_r, alpha=opts.lora_alpha,
                                  dropout=opts.lora_dropout, target=opts.lora_target)
            n_train, n_frozen = freeze_base(_m)
            pct = 100.0 * n_train / (n_train + n_frozen)
            print(f"[dry-run][lora] target={opts.lora_target} r={opts.lora_r}: patched "
                  f"{len(patched)} linears; trainable={n_train:,} ({pct:.3f}%) "
                  f"frozen={n_frozen:,}")
        if get_rank() == 0:
            print(f"[dry-run] dataset built OK: {len(ds)} examples, seq_len={opts.seq_len}")
        return

    prepare_training_environment()
    try:
        seed_all(12536)
        model = mc.build(init_device="meta")
        # LoRA must be injected BEFORE the train-module build so FSDP shards the adapter
        # params and the optimizer (built inside tmc.build from requires_grad params)
        # captures only them. The adapter values are (re)initialized in pre_train by
        # LoRAInitCallback, after the base distcp loads. Base param names are preserved.
        if opts.lora:
            from corpus_reasoning.lib.olmo_lora import freeze_base, inject_lora
            patched = inject_lora(model, r=opts.lora_r, alpha=opts.lora_alpha,
                                  dropout=opts.lora_dropout, target=opts.lora_target)
            n_train, n_frozen = freeze_base(model)
            if get_rank() == 0:
                pct = 100.0 * n_train / (n_train + n_frozen)
                print(f"[olmo] LoRA: injected {len(patched)} adapters (target="
                      f"{opts.lora_target}, r={opts.lora_r}, alpha={opts.lora_alpha}); "
                      f"trainable={n_train:,} ({pct:.3f}%) frozen={n_frozen:,}")
        train_module = tmc.build(model)
        # Install custom attention masking on the built model (after FSDP wrapping in
        # tmc.build). Reads the marker ids the data was tokenized with from the meta
        # sidecar so train-time chunk boundaries match what --wrap-docs emitted.
        if opts.attn_pattern != "standard":
            _m = json.load(open(opts.meta))
            if not _m.get("wrap_docs"):
                raise SystemExit(
                    f"--attn-pattern={opts.attn_pattern} needs doc-wrapped data; "
                    "re-tokenize the dataset with --wrap-docs.")
            # Mix curriculum: total microbatch-forwards over the run = examples processed
            # / examples-per-forward / world (capped by --max-steps for smoke runs). Each
            # forward consumes rank_microbatch_tokens // seq_len examples (1 here).
            mix_total_forwards = 0
            if opts.mix_start_p > 0.0 or opts.mix_end_p > 0.0:
                ex_per_fwd = max(1, opts.rank_microbatch_tokens // opts.seq_len)
                world = max(1, get_world_size())
                n_ex = int(_m.get("n_examples", 0))
                total = math.ceil(n_ex * opts.epochs / (ex_per_fwd * world))
                if opts.max_steps > 0:
                    accum = max(1, opts.global_batch_tokens // (opts.rank_microbatch_tokens * world))
                    total = min(total, opts.max_steps * accum)
                mix_total_forwards = total
            holder = install_flex_attention(
                train_module.model,
                pattern=opts.attn_pattern,
                doc_start_id=_m["doc_start_id"],
                doc_end_id=_m["doc_end_id"],
                eos_id=_m["eos_token_id"],
                window=opts.attn_window,
                compile_flex=opts.compile_flex,
                mix_start_p=opts.mix_start_p,
                mix_end_p=opts.mix_end_p,
                mix_total_forwards=mix_total_forwards,
                mix_seed=opts.mix_seed,
            )
            if get_rank() == 0:
                print(f"[olmo] FlexAttention pattern={opts.attn_pattern} "
                      f"window={opts.attn_window} -> patched {holder.n_patched} "
                      f"attention modules (cp=1)")
                if mix_total_forwards > 0:
                    print(f"[olmo] mix curriculum: p_standard {opts.mix_start_p} -> "
                          f"{opts.mix_end_p} (linear) over {mix_total_forwards} forwards, "
                          f"seed={opts.mix_seed}")
        # Gold-document gradient masking: detach K/V of distractor-doc tokens so only
        # the ground-truth documents (+ free/instruction/answer tokens) drive the
        # weight update. Composes with the flex install above (wraps whatever sdpa is
        # patched). Per-example gold sets are looked up by content fingerprint from the
        # tokenize-time sidecar -- nothing gold-specific enters the token stream.
        if opts.gold_grad_mask:
            from corpus_reasoning.lib.olmo_gold_grad_mask import (
                install_gold_grad_mask, make_fingerprint_gold_mask_fn)
            _m = json.load(open(opts.meta))
            if not _m.get("wrap_docs"):
                raise SystemExit("--gold-grad-mask needs doc-wrapped data (re-tokenize "
                                 "with --wrap-docs / emit_gold_mask).")
            gold_path = opts.gold_path or _m.get("gold_path")
            if not gold_path or not os.path.exists(gold_path):
                raise SystemExit(f"--gold-grad-mask: gold sidecar not found ({gold_path})")
            gold_table = json.load(open(gold_path))
            gm_fn = make_fingerprint_gold_mask_fn(
                gold_table, doc_start_id=_m["doc_start_id"], doc_end_id=_m["doc_end_id"],
                eos_id=_m["eos_token_id"])
            holder = install_gold_grad_mask(train_module.model, gm_fn)
            if get_rank() == 0:
                print(f"[olmo] gold-grad mask: patched {holder.n_patched} attention "
                      f"modules; {len(gold_table)} examples in gold table "
                      f"(of {_m.get('n_examples', '?')})")
        dataset = dc.build()
        data_loader = dlc.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = tc.build(train_module, data_loader)
        trainer.fit()
        if not opts.save_optim:
            # Model-only final checkpoint (no optimizer state). save_model_and_optim_state with
            # optim=None writes just the model shards + .metadata -- symmetric with the
            # load_model_and_optim_state(model) the exporter uses. Written to step<N>/model_and_optim
            # so export_olmo_to_hf.py's latest_step_dir() picks it up unchanged.
            from olmo_core.distributed.checkpoint import save_model_and_optim_state
            mao = os.path.join(opts.save_folder, f"step{trainer.global_step}", "model_and_optim")
            if get_rank() == 0:
                print(f"[olmo] saving MODEL-ONLY checkpoint (no optimizer state) -> {mao}")
            save_model_and_optim_state(mao, train_module.model, optim=None, save_overwrite=True)
        # The saved checkpoint contains base + LoRA params. Write a sidecar so the exporter
        # can rebuild a matching skeleton (inject same r/target) to load + merge / emit PEFT.
        if opts.lora and get_rank() == 0:
            from corpus_reasoning.lib.olmo_lora import write_lora_sidecar
            path = write_lora_sidecar(
                opts.save_folder, base_model=opts.base_model, r=opts.lora_r,
                alpha=opts.lora_alpha, dropout=opts.lora_dropout, target=opts.lora_target)
            print(f"[olmo] wrote LoRA sidecar -> {path}")
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
