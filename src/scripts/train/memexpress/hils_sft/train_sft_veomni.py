"""
SFT task for HiLS-Attention-7B and Olmo-3-1025-7B under **veomni**.

One task file drives **both arms** — the model is the only thing that differs — because a
comparison across two trainers measures the trainers as much as the architectures. The HiLS repo
has no SFT task of its own (``tasks/pretrain_with_ruler.py`` is CPT with RULER synthesis), so this
mirrors that file's veomni usage and swaps in our pre-packed SFT data.

Run it with the HiLS repo and our scripts both importable::

    source src/scripts/train/memexpress/hils_eval/hils_env_setup.sh   # exports $HILS_REPO
    PYTHONPATH=$HILS_REPO:src/scripts torchrun --nproc_per_node=8 \\
        src/scripts/train/memexpress/hils_sft/train_sft_veomni.py \\
        --model-path <weka hf ckpt> --data-dir <weka packed_32k> --out-dir <weka save folder> ...

Data comes from ``sft_shard_dataset.SFTShardDataset(prepacked=True)`` — one materialized pack that
this and the olmo_core bridge arm both read, so "same data" is literal rather than a recipe run
twice. See the family README.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_shard_dataset import IGNORE_INDEX, SFTShardDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="weka HF checkpoint dir (weights).")
    ap.add_argument("--config-path", default=None, help="defaults to --model-path.")
    ap.add_argument("--data-dir", required=True, help="materialized pack (prepacked windows).")
    ap.add_argument("--out-dir", required=True, help="save folder; MUST be unique per run.")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--eos-token-id", type=int, default=100257)
    ap.add_argument("--pad-token-id", type=int, default=100277)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--global-batch-size", type=int, default=8, help="windows per optimizer step.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from epochs.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=34521)
    ap.add_argument("--save-steps", type=int, default=0, help="0 = only at the end.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--attn-impl", default="", help="empty -> probe fa3, fa2, sdpa.")
    ap.add_argument("--wandb-project", default="memory-networks")
    ap.add_argument("--wandb-name", default="")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build data + print the plan, no model.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # BEFORE anything allocates: without this every rank defaults to cuda:0 and eight 7B models
    # land on one card. It does not fail cleanly either -- the attention-implementation probe
    # catches the resulting OOM and reports "flash_attention_2 unavailable", which reads as a
    # missing dependency rather than a device-placement bug.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    is_main = rank == 0
    torch.manual_seed(args.seed)

    # ---- data ----------------------------------------------------------------------------------
    # prepacked=True: the windows were mixed, shuffled and packed once by
    # sft_shard_dataset.materialize(). Re-deriving them here would give this arm different data
    # from the olmo_core bridge arm, which is exactly what the pack exists to prevent.
    dataset = SFTShardDataset(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
        prepacked=True,
    )
    if is_main:
        manifest_path = os.path.join(args.data_dir, "pack_manifest.json")
        print(f"[data] {len(dataset)} windows x {args.max_seq_len} tokens from {args.data_dir}")
        if os.path.exists(manifest_path):
            # Log the realized mixture with the run, so a results row can be traced to the data.
            print("[data] pack_manifest:", json.dumps(json.load(open(manifest_path)), indent=2))

    if args.global_batch_size % (args.micro_batch_size * world):
        raise SystemExit(
            f"global_batch_size {args.global_batch_size} is not divisible by "
            f"micro_batch_size {args.micro_batch_size} x world {world}; the realized batch would "
            f"differ from the configured one."
        )
    accum = args.global_batch_size // (args.micro_batch_size * world)

    sampler: Optional[DistributedSampler] = None
    if world > 1:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                     seed=args.seed, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = max(1, len(loader) // accum)
    max_steps = args.max_steps or steps_per_epoch * args.epochs
    if is_main:
        content = len(dataset) * args.max_seq_len
        print(f"[plan] world={world} micro={args.micro_batch_size} accum={accum} "
              f"global_batch={args.global_batch_size} windows steps_per_epoch={steps_per_epoch} "
              f"max_steps={max_steps}")
        print(f"[plan] window tokens seen = {args.global_batch_size * args.max_seq_len * max_steps:,} "
              f"(one epoch of the pack = {content:,})")
    if args.dry_run:
        return 0

    # ---- model ---------------------------------------------------------------------------------
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from ctc_eval.lib.hils_loader import init_veomni_parallel_state, is_hils_checkpoint, register_hils

    from veomni.distributed.parallel_state import init_parallel_state  # noqa: F401
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_foundation_model
    from veomni.optim import build_lr_scheduler, build_optimizer

    if is_hils_checkpoint(args.model_path):
        # Registers the out-of-tree modeling code AND initializes veomni's parallel state, which
        # HiLS's forward reads on every step (it asserts the parallel sizes multiply to world_size,
        # so it passes at world=1 and fails under torchrun if left alone).
        register_hils(args.model_path)
    # shard=True: TRAINING. Replicating a 7B puts weights + grads + fp32 Adam moments on every
    # rank (~84 GB) and OOMs an 80 GB card; the eval path's default (replicate) is right there and
    # wrong here.
    init_veomni_parallel_state(shard=True)

    attn_candidates = [args.attn_impl] if args.attn_impl else [
        "flash_attention_3", "flash_attention_2", "sdpa"
    ]
    model = None
    for attn in attn_candidates:
        try:
            model = build_foundation_model(
                config_path=args.config_path or args.model_path,
                weights_path=args.model_path,
                torch_dtype="bfloat16",
                attn_implementation=attn,
                # meta, like the HiLS repo's own task: build_parallelize_model materializes
                # sharded weights from weights_path, so no rank ever holds the full 7B. With
                # "cuda" each rank first builds a complete model on its card.
                init_device="meta",
            )
            if is_main:
                print(f"[model] built with attn_implementation={attn}")
            break
        except torch.cuda.OutOfMemoryError:
            # An OOM is NOT evidence that this attention implementation is unsupported, and
            # silently falling through to the next one turns a memory/placement bug into a
            # misleading "unavailable" message. Fail here instead.
            raise
        except Exception as e:  # noqa: BLE001 -- probing what this runtime supports
            if is_main:
                print(f"[model] attn={attn} unavailable ({type(e).__name__}: {e})")
    if model is None:
        raise SystemExit("could not build the model with any attention implementation")

    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=args.model_path,
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        basic_modules=getattr(model, "_no_split_modules", []) or [],
    )
    optimizer = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, fused=True, optimizer_type="adamw"
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=max_steps,
        lr=args.lr,
        lr_min=args.min_lr,
        lr_warmup_ratio=args.warmup_ratio,
        lr_decay_style="cosine",
    )

    use_wandb = is_main and not args.no_wandb
    if use_wandb:
        # A missing logger must never kill a multi-hour training run. Every smoke test passed
        # --no-wandb, so the import was first exercised by the real launch -- which died at it.
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or os.path.basename(args.out_dir),
                config=vars(args),
            )
        except Exception as e:  # noqa: BLE001 -- logging is not worth losing the run over
            print(f"[wandb] disabled ({type(e).__name__}: {e})", flush=True)
            use_wandb = False

    # ---- train ---------------------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    step, t0 = 0, time.time()
    last_t = t0
    done = False
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            (out.loss / accum).backward()
            if (micro + 1) % accum:
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            now = time.time()
            # Instantaneous, NOT elapsed/step. The cumulative average buries steady-state
            # throughput under the one-off startup cost (model load + tilelang JIT is ~90 s, so a
            # 7 s/step run reads as 22 s/step over its first few steps) -- which is exactly the
            # number someone extrapolates a multi-hour ETA from.
            step_s, last_t = now - last_t, now
            if is_main and step % args.log_every == 0:
                # Loss is over ASSISTANT tokens only (labels are IGNORE_INDEX elsewhere), so it is
                # not comparable to a pretraining CE on the same data.
                ntrain = int((batch["labels"] != IGNORE_INDEX).sum())
                msg = (f"step {step}/{max_steps} loss {out.loss.item():.4f} "
                       f"lr {lr_scheduler.get_last_lr()[0]:.2e} gnorm {float(gnorm):.2f} "
                       f"trainable_tok/micro {ntrain} {step_s:.2f}s/step "
                       f"(avg {(now - t0) / step:.2f})")
                print(msg, flush=True)
                if use_wandb:
                    wandb.log({"loss": out.loss.item(), "lr": lr_scheduler.get_last_lr()[0],
                               "grad_norm": float(gnorm), "step": step,
                               "seconds_per_step": step_s})
            if args.save_steps and step % args.save_steps == 0:
                _save(model, args.out_dir, step, rank, is_main, args.model_path)
            if step >= max_steps:
                done = True
                break
        if done:
            break

    _save(model, args.out_dir, step, rank, is_main, args.model_path)
    if is_main:
        print(f"[done] {step} steps in {(time.time()-t0)/60:.1f} min -> {args.out_dir}")
    return 0


#: Non-weight files copied from the source checkpoint so the saved dir is a COMPLETE HF model.
#: Without them the output has safetensors and nothing else, and `eval_lc_native.py --backend hf`
#: cannot load it (no config.json -> no model_type -> the HiLS classes are never registered).
_ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def _save(model, out_dir: str, step: int, rank: int, is_main: bool, source_model_path: str) -> None:
    """
    Save a directly-loadable HF checkpoint.

    ``save_model_weights`` takes a **state dict**, not a model, and gathers DTensors itself
    (``tensor.data.full_tensor()``), so FSDP sharding needs no special handling here. It must be
    called on every rank when ``global_rank`` is passed.

    Weights land as safetensors, so copying the config/tokenizer beside them makes this a complete
    HF directory -- the eval path reads it as-is, with no DCP->HF conversion step.

    :param model: The (parallelized) model.
    :param out_dir: Run save folder; the step dir is created under it.
    :param step: Step number, used for the step dir name.
    :param rank: Global rank -- passed through so veomni knows this runs on all ranks.
    :param is_main: Whether this rank does the rank-0-only asset copy and logging.
    :param source_model_path: The HF checkpoint this run started from, for the asset files.
    """
    import shutil

    from veomni.models import save_model_weights

    dest = os.path.join(out_dir, f"step{step}")
    save_model_weights(dest, model.state_dict(), global_rank=rank)
    if is_main:
        for name in _ASSET_FILES:
            src = os.path.join(source_model_path, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, name))
        print(f"[save] {dest} (weights + config/tokenizer; loadable by --backend hf)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
