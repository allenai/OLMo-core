"""Split the mixer's wall-clock into full-attention layers vs GatedDeltaNet layers.

The component table says the mixer costs 1.22-1.29x its FLOP share at 4B but ~0.9x at 0.8B. Qwen3.5
is a hybrid -- 8 full-attention and 24 GDN blocks at 4B -- and the two have very different
arithmetic intensity: attention is one flash kernel, GDN is a chunked recurrence with a conv, gates,
per-chunk states and an fp32 output norm. This ablates each family separately so the excess can be
attributed, and prices each against its own FLOPs to give an achieved TFLOP/s.

    python research/kv_route/flexcompute_40x/bench_mixer_split.py --scale 4B --tokens 8192,16384
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
SCALES = {"0.8B": TransformerConfig.qwen3_5_0_8B, "4B": TransformerConfig.qwen3_5_4B}


class Identity(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


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
    args = ap.parse_args()
    lengths = [int(x) for x in args.tokens.split(",")]

    device = torch.device("cuda")
    cfg = SCALES[args.scale](vocab_size=VOCAB)
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    torch.manual_seed(0)
    model = cfg.build(init_device="cpu").to(device)
    model.init_weights(max_seq_len=max(lengths), device=device)
    model = model.to(torch.bfloat16)
    model.train()

    blocks = list(model.blocks.values())
    real_mix = [b.attention for b in blocks]
    is_attn = [type(m) is Attention for m in real_mix]
    print(
        f"# {torch.cuda.get_device_name(0)} {args.scale}: "
        f"{sum(is_attn)} full-attention / {len(is_attn) - sum(is_attn)} GDN blocks",
        flush=True,
    )

    rows = []
    for T in lengths:
        labels = torch.full((1, T), -100, dtype=torch.long, device=device)
        k = max(1, T // 512)
        labels[:, -k:] = torch.randint(0, VOCAB, (1, k), device=device)
        input_ids = torch.randint(0, VOCAB, (1, T), device=device)

        # Per-family FLOPs for one sequence, from the model's own formulas.
        f_attn = f_gdn = 0.0
        for b, a in zip(blocks, real_mix):
            f = a.num_flops_per_token(T) * T
            if type(a) is Attention:
                f_attn += f
            else:
                f_gdn += f

        def step():
            model.zero_grad(set_to_none=True)
            out = model(input_ids, labels=labels, loss_reduction="sum", return_logits=False)
            out.loss.backward()

        def restore():
            for b, m in zip(blocks, real_mix):
                b.attention = m

        row = {"scale": args.scale, "tokens": T}
        try:
            restore()
            full = timed(step, iters=args.iters)

            for b, keep in zip(blocks, is_attn):
                if keep:
                    b.attention = Identity()
            no_attn = timed(step, iters=args.iters)
            restore()

            for b, keep in zip(blocks, is_attn):
                if not keep:
                    b.attention = Identity()
            no_gdn = timed(step, iters=args.iters)
            restore()

            row.update(
                full_s=full,
                attn_s=full - no_attn,
                gdn_s=full - no_gdn,
                attn_share=(full - no_attn) / full,
                gdn_share=(full - no_gdn) / full,
                attn_gflops=f_attn / 1e9,
                gdn_gflops=f_gdn / 1e9,
                attn_tflops_s=f_attn / 1e12 / max(full - no_attn, 1e-9),
                gdn_tflops_s=f_gdn / 1e12 / max(full - no_gdn, 1e-9),
            )
        except torch.cuda.OutOfMemoryError:
            restore()
            row["error"] = "OOM"
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(os.path.join(OUT, f"bench_mixer_split_{args.scale}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
