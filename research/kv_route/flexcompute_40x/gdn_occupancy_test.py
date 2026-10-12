"""Direct test of the GDN occupancy diagnosis.

`resolve_gdn_gap.py` concluded "occupancy/latency-bound" by ELIMINATION: 2.5% of compute peak,
15.8% of bandwidth peak, so neither roofline binds. That is an inference, not a measurement. If the
kernel genuinely leaves the GPU idle during its sequential inter-chunk scan, then adding INDEPENDENT
parallel work should be nearly free -- doubling the batch should cost far less than 2x.

Two sweeps at fixed sequence length:
  batch  -- B = 1,2,4,8 with everything else fixed. Perfectly occupancy-bound => time flat.
            Perfectly saturated => time proportional to B.
  heads  -- H = 8,16,32,64. Same logic along the other independent axis.

`scaling` is time(x)/time(smallest); `free` is how much of the added work came for free
(1 - scaling/x). Near 1.0 = idle capacity, near 0.0 = saturated.

    python research/kv_route/flexcompute_40x/gdn_occupancy_test.py --tokens 8192
"""

import argparse
import time

import torch

VOCAB = 248320


def timed(fn, warmup=4, iters=12):
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
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    args = ap.parse_args()
    dev = torch.device("cuda")
    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_gated_delta_rule

    def run(B, H, T):
        q = torch.randn(B, T, H, args.dk, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, args.dk, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, args.dv, device=dev, dtype=torch.bfloat16)
        g = torch.rand(B, T, H, device=dev, dtype=torch.float32) * -0.1
        beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16)

        def step():
            qi, ki, vi = (t.detach().requires_grad_(True) for t in (q, k, v))
            gi, bi = (t.detach().requires_grad_(True) for t in (g, beta))
            o, _ = dispatch_chunk_gated_delta_rule(
                q=qi,
                k=ki,
                v=vi,
                g=gi,
                beta=bi,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            o.sum().backward()

        return timed(step)

    print(f"# T={args.tokens} dk={args.dk} dv={args.dv}  (H200)")
    for label, axis, base, values, fixed in (
        ("batch", "B", 1, [1, 2, 4, 8], dict(H=32)),
        ("heads", "H", 8, [8, 16, 32, 64], dict(B=1)),
    ):
        print(
            f"\n{label} sweep ({', '.join(f'{k}={v}' for k, v in fixed.items())}, T={args.tokens})"
        )
        print(f"  {axis:>4} {'ms':>9} {'scaling':>9} {'ideal':>7} {'free':>7}")
        t0 = None
        for x in values:
            kw = dict(fixed)
            kw[axis] = x
            try:
                t = run(kw.get("B", 1), kw.get("H", 32), args.tokens)
            except torch.cuda.OutOfMemoryError:
                print(f"  {x:>4}  OOM")
                torch.cuda.empty_cache()
                continue
            if t0 is None:
                t0 = t
            scaling = t / t0
            ideal = x / base
            free = 1 - scaling / ideal
            print(f"  {x:>4} {t*1e3:>8.3f} {scaling:>8.2f}x {ideal:>6.0f}x {free:>6.1%}")


if __name__ == "__main__":
    main()
