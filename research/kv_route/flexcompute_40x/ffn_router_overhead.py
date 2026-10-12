"""What does the ROUTER cost, on top of the ladder kernel?

Every speed number reported so far timed `_NestedLadderFn.apply(...)` with the rung index tensors
PRECOMPUTED outside the timed region. The deployed path (nested_ffn_moe.py:785-802) also runs, per
layer per forward:

    probs = router(x)                      a d_model x n_rungs GEMM (tiny -> launch-bound)
    choice = probs.argmax(-1)
    counts = bincount(choice).tolist()     <-- an explicit HOST SYNC, one per layer
    order  = argsort(choice, stable=True)  a full sort over T
    idxs   = split(order, counts)

The host sync drains the GPU queue, and the GDN work established that a 4B step already sits near
the CPU/GPU crossover -- so 36 of these per step may not be free. This times the routing prologue
alone, the ladder alone, and both together.

    python research/kv_route/flexcompute_40x/ffn_router_overhead.py --tokens 32768
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from olmo_core.nn.nested_ffn_moe import _NestedLadderFn
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))


def timed(fn, warmup=5, iters=15):
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
    return ts[len(ts) // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--scale", default="4B")
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens
    fac = {"4B": TransformerConfig.qwen3_5_4B, "0.8B": TransformerConfig.qwen3_5_0_8B}[args.scale]
    m = fac(vocab_size=248320).build(init_device="meta")
    d = m.d_model
    Fh = list(m.blocks.values())[0].feed_forward.w1.out_features
    n_layers = len(m.blocks)
    del m

    torch.manual_seed(0)
    x = torch.randn(T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w3 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(d, Fh, device=dev, dtype=torch.bfloat16, requires_grad=True)

    # 7-rung trained geometry (Qwen3-4B divisors 1,16,64,256,1024,9728 + null)
    widths = [Fh, Fh // 16, Fh // 64, Fh // 256, Fh // 1024, 1, 0]
    n_rungs = len(widths)
    wr = torch.randn(n_rungs, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    print(
        f"# {torch.cuda.get_device_name(0)} {args.scale} d_model={d} hidden={Fh} "
        f"tokens={T} rungs={n_rungs} n_layers={n_layers}"
    )

    def routing_prologue():
        """Exactly the deployed prologue: router GEMM, argmax, bincount().tolist(), argsort, split."""
        probs = F.linear(x, wr).float().softmax(-1)
        choice = probs.argmax(dim=-1)
        counts = torch.bincount(choice, minlength=n_rungs).tolist()  # HOST SYNC
        order = torch.argsort(choice, stable=True)
        return torch.split(order, counts), counts

    # a fixed assignment for the ladder-only arm, matched to what the router happens to produce
    with torch.no_grad():
        idxs0, counts0 = routing_prologue()
    idxs0 = [i.contiguous() for i in idxs0]
    mean_c = sum(c * w / Fh for w, c in zip(widths, counts0)) / T
    print(f"# router's own split -> counts {counts0}  mean c = {mean_c:.5f}\n")

    def ladder_only():
        for t in (x, w1, w3, w2):
            t.grad = None
        _NestedLadderFn.apply(widths, x, w1, w3, w2, *idxs0).sum().backward()

    def prologue_only():
        wr.grad = None
        routing_prologue()

    def full_path():
        for t in (x, w1, w3, w2, wr):
            t.grad = None
        idxs, _ = routing_prologue()
        _NestedLadderFn.apply(
            widths, x, w1, w3, w2, *[i.contiguous() for i in idxs]
        ).sum().backward()

    def sync_only():
        choice = torch.randint(0, n_rungs, (T,), device=dev)
        torch.bincount(choice, minlength=n_rungs).tolist()  # the sync in isolation

    t_l, t_p, t_f, t_s = (
        timed(ladder_only),
        timed(prologue_only),
        timed(full_path),
        timed(sync_only),
    )
    print(f"{'ladder only (what I reported)':<34} {t_l:8.3f} ms")
    print(f"{'routing prologue only':<34} {t_p:8.3f} ms")
    print(f"{'  of which the host sync':<34} {t_s:8.3f} ms")
    print(f"{'full path (router + ladder)':<34} {t_f:8.3f} ms")
    print(f"\nrouter overhead = {t_f - t_l:.3f} ms = {100*(t_f-t_l)/t_l:.1f}% on top of the ladder")
    print(f"across {n_layers} layers that is {(t_f-t_l)*n_layers:.1f} ms per step")
    json.dump(
        dict(
            tokens=T,
            scale=args.scale,
            widths=widths,
            counts=counts0,
            mean_c=mean_c,
            ladder_ms=t_l,
            prologue_ms=t_p,
            sync_ms=t_s,
            full_ms=t_f,
        ),
        open(os.path.join(OUT, f"ffn_router_overhead_{args.scale}_{T}.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
