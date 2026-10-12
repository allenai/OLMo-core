"""A step-time partition that actually sums to 100%.

Single-component ablation does NOT partition a step: the deltas are superadditive (removing
attention and GDN together saved 58 ms MORE at 4B/16k than removing each alone, because whichever
one is left still contends for the same bandwidth), and the "head" figure previously quoted was the
SAVING from gating rather than the head's cost. Together those left ~8-9% unexplained and made the
components look like a partition when they were not.

Here components are removed CUMULATIVELY, so each delta is a marginal cost in the presence of what
remains, and the deltas sum exactly to (full - floor). `floor` is everything never ablated:
embedding lookup, residual adds, the final norm, zero_grad and inter-kernel gaps.

    python research/kv_route/flexcompute_40x/bench_step_partition.py --scale 4B --tokens 8192,16384

Scales 9B and 27B were added later so the wall-clock partition exists at more than two model
sizes. They need two accommodations that 0.8B/4B did not:

* **CPU init.** The original path built fp32 on the GPU and only then cast to bf16, which peaks at
  6 bytes/param -- 162 GB for 27B, i.e. OOM on a 141 GB H200 before a single step. Weights are now
  initialised on the host and cast there, so the GPU only ever sees bf16.
* **``--ac``** wraps every block in non-reentrant activation checkpointing. This CHANGES the thing
  being measured (each block's forward runs twice), so AC and non-AC rows are not comparable to
  each other. Run the AC arm at every scale you want to compare across, and use non-AC rows only
  against other non-AC rows.
"""

import argparse
import json
import os
import time

import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320
OUT = os.path.dirname(os.path.abspath(__file__))
SCALES = {
    "0.8B": TransformerConfig.qwen3_5_0_8B,
    "2B": TransformerConfig.qwen3_5_2B,
    "4B": TransformerConfig.qwen3_5_4B,
    "9B": TransformerConfig.qwen3_5_9B,
    "27B": TransformerConfig.qwen3_5_27B,
}


class Identity(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class AcBlock(torch.nn.Module):
    """Non-reentrant activation checkpointing around one block.

    Deliberately not the model's own :meth:`apply_activation_checkpointing`: that wraps the block
    in a ``CheckpointWrapper`` whose ``__getattr__`` forwards reads to the inner module but whose
    ``__setattr__`` does not, so ``block.feed_forward = Identity()`` would register a new submodule
    on the wrapper and leave the real FFN running. Holding ``inner`` explicitly keeps the ablation
    pointed at the module that actually executes.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, **kwargs):
        return torch.utils.checkpoint.checkpoint(self.inner, x, use_reentrant=False, **kwargs)


def timed(fn, warmup=2, iters=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="4B", choices=list(SCALES))
    ap.add_argument("--tokens", default="8192,16384")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument(
        "--ac", action="store_true", help="wrap every block in activation checkpointing"
    )
    ap.add_argument("--suffix", default="", help="appended to the output filename")
    args = ap.parse_args()
    lengths = [int(x) for x in args.tokens.split(",")]

    device = torch.device("cuda")
    cfg = SCALES[args.scale](vocab_size=VOCAB)
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    torch.manual_seed(0)
    # Host-side init + cast, then one bf16 copy to the GPU: peaks at 2 bytes/param on device
    # instead of 6, which is what makes 27B reachable at all on a single card.
    model = cfg.build(init_device="meta")
    model.init_weights(max_seq_len=max(lengths), device=torch.device("cpu"))
    model = model.to(torch.bfloat16).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"# {torch.cuda.get_device_name(0)} scale={args.scale} params={n_params/1e9:.3f}B "
        f"ac={args.ac} weights={torch.cuda.memory_allocated()/2**30:.1f} GiB",
        flush=True,
    )

    blocks = list(model.blocks.values())
    if args.ac:
        for key, blk in list(model.blocks.items()):
            model.blocks[key] = AcBlock(blk)
    real_mix = [b.attention for b in blocks]
    real_ff = [b.feed_forward for b in blocks]
    is_attn = [type(m) is Attention for m in real_mix]
    rows = []

    for T in lengths:
        labels = torch.full((1, T), -100, dtype=torch.long, device=device)
        k = max(1, T // 512)
        labels[:, -k:] = torch.randint(0, VOCAB, (1, k), device=device)
        input_ids = torch.randint(0, VOCAB, (1, T), device=device)

        def step():
            model.zero_grad(set_to_none=True)
            out = model(input_ids, labels=labels, loss_reduction="sum", return_logits=False)
            out.loss.backward()

        def restore():
            for b, ff, mx in zip(blocks, real_ff, real_mix):
                b.feed_forward, b.attention = ff, mx
            model.lm_head._sparse_loss_positions = False

        row = {"scale": args.scale, "tokens": T, "ac": args.ac, "params": n_params}
        torch.cuda.reset_peak_memory_stats()
        try:
            restore()
            stages = [("full", None)]
            row["full_s"] = timed(step, iters=args.iters)

            # 1. gate the head (keeps the head, removes only the wasted rows)
            model.lm_head._sparse_loss_positions = True
            t_head = timed(step, iters=args.iters)

            # 2. + drop the FFN
            for b in blocks:
                b.feed_forward = Identity()
            t_ffn = timed(step, iters=args.iters)

            # 3. + drop full-attention blocks
            for b, keep in zip(blocks, is_attn):
                if keep:
                    b.attention = Identity()
            t_attn = timed(step, iters=args.iters)

            # 4. + drop GDN blocks -> only embeddings/residual/norm/head remain
            for b, keep in zip(blocks, is_attn):
                if not keep:
                    b.attention = Identity()
            t_floor = timed(step, iters=args.iters)
            restore()
            del stages

            f = row["full_s"]
            row.update(
                head_waste_s=f - t_head,
                ffn_s=t_head - t_ffn,
                attn_s=t_ffn - t_attn,
                gdn_s=t_attn - t_floor,
                floor_s=t_floor,
            )
            for kk in ("head_waste", "ffn", "attn", "gdn", "floor"):
                row[f"{kk}_pct"] = 100 * row[f"{kk}_s"] / f
            row["sum_pct"] = sum(
                row[f"{kk}_pct"] for kk in ("head_waste", "ffn", "attn", "gdn", "floor")
            )
            row["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        except torch.cuda.OutOfMemoryError:
            restore()
            row["error"] = "OOM"
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)

    name = f"bench_step_partition_{args.scale}{'_ac' if args.ac else ''}{args.suffix}.json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"# wrote {name}", flush=True)


if __name__ == "__main__":
    main()
