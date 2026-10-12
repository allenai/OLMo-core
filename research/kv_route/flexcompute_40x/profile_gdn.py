"""Where does a GDN block's wall-clock go?

95.2% of the FLOPs `num_flops_per_token` counts for a 4B GDN block are linear projections -- the
same GEMM work the FFN runs at ~595 TF/s -- yet the block measures ~115 TF/s. Either the non-GEMM
parts (short conv, chunked delta rule, the GVA `repeat_interleave`, the fp32 gated norm) dominate
time despite being ~5% of counted FLOPs, or the formula badly undercounts the chunked kernel.
A kernel-level profile distinguishes them.

Note the GVA expansion: n_heads=16 vs n_v_heads=32 means q and k are `repeat_interleave`d 2x
(key_dim 2048 -> 4096) -- pure memory traffic, zero FLOPs, paid in forward and backward.

    python research/kv_route/flexcompute_40x/profile_gdn.py --tokens 8192
"""

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--scale", default="4B")
    args = ap.parse_args()
    device = torch.device("cuda")

    cfg = {"4B": TransformerConfig.qwen3_5_4B, "0.8B": TransformerConfig.qwen3_5_0_8B}[args.scale]
    torch.manual_seed(0)
    model = cfg(vocab_size=VOCAB).build(init_device="cpu").to(device)
    model.init_weights(max_seq_len=args.tokens, device=device)
    model = model.to(torch.bfloat16)

    gdn = [b.attention for b in model.blocks.values() if type(b.attention) is not Attention][0]
    ffn = list(model.blocks.values())[0].feed_forward
    d = model.d_model
    x = torch.randn(1, args.tokens, d, device=device, dtype=torch.bfloat16, requires_grad=True)

    def run(mod):
        mod.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        mod(x).sum().backward()

    for name, mod in (("GDN", gdn), ("FFN", ffn)):
        for _ in range(3):
            run(mod)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(5):
                run(mod)
            torch.cuda.synchronize()

        tot = defaultdict(float)
        for e in prof.key_averages():
            if e.self_device_time_total > 0:
                tot[e.key] += e.self_device_time_total
        total = sum(tot.values())
        print(f"\n=== {name}: {total/5/1000:.2f} ms/iter of CUDA time, top kernels")
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:14]:
            print(f"   {100*v/total:5.1f}%  {v/5/1000:8.3f} ms  {k[:88]}")


if __name__ == "__main__":
    main()
