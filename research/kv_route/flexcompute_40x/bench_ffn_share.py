"""Where does a training step's WALL-CLOCK actually go, and how much of it is the FFN?

The FLOP anatomy says FFN is 55-62% of a Qwen3.5-4B token. The routed-FFN work measured something
very different: deleting the routed FFNs entirely moved a step only 603 -> 497 ms, a 1.21x ceiling,
implying FFN is ~17% of wall-clock. Before optimising an FFN router it is worth knowing which
number is right and why they differ, because no FLOP reduction can beat the ablation bound.

Method: ablate one component at a time by replacing its forward with an identity, and time a full
fwd+bwd step. The share attributable to a component is (full - ablated) / full. Ablation also
removes that component's parameters from the backward, which is the point: it is the "if this were
free" bound, the most any router targeting it could ever buy.

Swept over tokens-per-step, because the bandwidth-vs-compute hypothesis predicts the FFN share
GROWS with batch: small steps leave the GEMMs memory-bound, where FLOPs do not predict time.

    python research/kv_route/flexcompute_40x/bench_ffn_share.py --scale 4B --tokens 2048,8192,32768
"""

import argparse
import json
import os
import time

import torch

from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320
OUT = os.path.dirname(os.path.abspath(__file__))
SCALES = {
    "0.8B": TransformerConfig.qwen3_5_0_8B,
    "4B": TransformerConfig.qwen3_5_4B,
    "qwen3-4B": TransformerConfig.qwen3_4B,
}


class Identity(torch.nn.Module):
    """Stands in for an ablated submodule: same shape out, no work, no parameters in the graph."""

    def forward(self, x, *args, **kwargs):
        return x


def timed(fn, warmup, iters):
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
    ap.add_argument(
        "--tokens", default="2048,8192,32768", help="tokens per step (seq len, batch 1)"
    )
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lengths = [int(x) for x in args.tokens.split(",")]

    device = torch.device("cuda")
    print(
        f"# {torch.cuda.get_device_name(0)} torch {torch.__version__} scale={args.scale}",
        flush=True,
    )

    cfg = SCALES[args.scale](vocab_size=VOCAB)
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    torch.manual_seed(0)
    model = cfg.build(init_device="cpu").to(device)
    model.init_weights(max_seq_len=max(lengths), device=device)
    model = model.to(torch.bfloat16)
    model.train()

    blocks = list(model.blocks.values())
    real_ff = [b.feed_forward for b in blocks]
    real_mix = [b.attention for b in blocks]
    results = []

    for T in lengths:
        labels = torch.full((1, T), -100, dtype=torch.long, device=device)
        labels[:, -max(1, T // 512) :] = torch.randint(
            0, VOCAB, (1, max(1, T // 512)), device=device
        )
        input_ids = torch.randint(0, VOCAB, (1, T), device=device)

        def step():
            model.zero_grad(set_to_none=True)
            out = model(input_ids, labels=labels, loss_reduction="sum", return_logits=False)
            out.loss.backward()

        def restore():
            for b, ff, mx in zip(blocks, real_ff, real_mix):
                b.feed_forward, b.attention = ff, mx

        row = {"scale": args.scale, "tokens": T}
        try:
            restore()
            row["full_s"] = timed(step, 2, args.iters)

            for b in blocks:
                b.feed_forward = Identity()
            row["no_ffn_s"] = timed(step, 2, args.iters)
            restore()

            for b in blocks:
                b.attention = Identity()
            row["no_mixer_s"] = timed(step, 2, args.iters)
            restore()

            # Head gated to its supervised rows -- the fix already landed, shown for comparison.
            model.lm_head._sparse_loss_positions = True
            row["gated_head_s"] = timed(step, 2, args.iters)
            model.lm_head._sparse_loss_positions = False

            row["ffn_share"] = (row["full_s"] - row["no_ffn_s"]) / row["full_s"]
            row["mixer_share"] = (row["full_s"] - row["no_mixer_s"]) / row["full_s"]
            row["head_share"] = (row["full_s"] - row["gated_head_s"]) / row["full_s"]
            row["ffn_ceiling"] = row["full_s"] / row["no_ffn_s"]
        except torch.cuda.OutOfMemoryError:
            restore()
            row["error"] = "OOM"
            torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps(row), flush=True)

    out = args.out or os.path.join(OUT, f"bench_ffn_share_{args.scale}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"# wrote {out}", flush=True)


if __name__ == "__main__":
    main()
