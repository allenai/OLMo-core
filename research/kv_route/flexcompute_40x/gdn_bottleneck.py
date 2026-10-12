"""Resolve the contradictory GDN occupancy sweeps, and split CPU-launch cost from GPU time.

`gdn_occupancy_test.py` ran a batch sweep and then a heads sweep SEQUENTIALLY and got opposite
answers (batch: 13% of the added work free => saturated; heads: 84% free => idle), and measured the
SAME config 58% apart across the two. Sequential sweeps cannot distinguish a real axis effect from
drift, so everything here is INTERLEAVED in one process with a repeated control point, and the
control's spread is reported so drift is quantified rather than absorbed.

Four sections:

  1. warmup   -- every point is run to convergence first, so Triton autotune (keyed on H, K, V, BT)
                 never lands inside a timed window.
  2. sweep    -- rounds over a rotated point order, control re-measured inside every round. Reports
                 p10/median/p90 per point and the control's per-round medians.
  3. cpu/gpu  -- per point: time the CPU to ENQUEUE k iterations without syncing, then sync. If the
                 enqueue time is close to the total, the kernel sequence is CPU-launch-bound, not
                 occupancy-bound -- a completely different fix (fewer/graphed launches vs more
                 parallel work).
  4. profile  -- per-kernel CUDA time and launch count, so the cost is attributed to the sequential
                 inter-chunk scan (`*_kernel_h_blockdim64` / `*_kernel_dhu_blockdim64`) or not.

    python research/kv_route/flexcompute_40x/gdn_bottleneck.py --tokens 8192
"""

import argparse
import json
import time
from collections import defaultdict

import torch

DK = DV = 128


def percentiles(ts):
    ts = sorted(ts)
    n = len(ts)

    def q(p):
        return ts[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return q(0.10), q(0.50), q(0.90)


def make_step(B, H, T, dev):
    """A closure holding its own tensors, so allocator state is fixed before timing starts."""
    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_gated_delta_rule

    q = torch.randn(B, T, H, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, DV, device=dev, dtype=torch.bfloat16)
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

    return step


def time_once(step):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4, help="timed iters per point per round")
    ap.add_argument("--out", default="research/kv_route/flexcompute_40x/gdn_bottleneck.json")
    args = ap.parse_args()
    T = args.tokens
    dev = torch.device("cuda")
    print(f"# GPU: {torch.cuda.get_device_name(0)}   T={T}  DK={DK} DV={DV}")

    # (label, B, H). "CTL" duplicates B=1,H=32 with its OWN tensors: if CTL and B1H32 disagree the
    # cause is allocator/tensor state, not the config.
    points = [
        ("B1H32", 1, 32),
        ("B2H32", 2, 32),
        ("B4H32", 4, 32),
        ("B8H32", 8, 32),
        ("B1H8", 1, 8),
        ("B1H16", 1, 16),
        ("B1H64", 1, 64),
    ]
    ctl = ("CTL", 1, 32)
    steps = {lbl: make_step(B, H, T, dev) for lbl, B, H in points + [ctl]}

    # ---- 1. warmup: every point, so autotune is settled for every (H, K, V) key.
    for lbl in steps:
        for _ in range(8):
            steps[lbl]()
    torch.cuda.synchronize()
    print(f"# warmup done, {torch.cuda.max_memory_allocated()/2**30:.1f} GiB peak")

    # ---- 2. interleaved sweep
    samples = defaultdict(list)
    ctl_round_median = []
    for r in range(args.rounds):
        order = points[r % len(points) :] + points[: r % len(points)]
        ctl_this = []
        for i, (lbl, _, _) in enumerate(order):
            if i % 3 == 0:  # control woven through, not bracketing
                ctl_this += [time_once(steps["CTL"]) for _ in range(args.samples)]
            samples[lbl] += [time_once(steps[lbl]) for _ in range(args.samples)]
        ctl_this += [time_once(steps["CTL"]) for _ in range(args.samples)]
        samples["CTL"] += ctl_this
        ctl_round_median.append(sorted(ctl_this)[len(ctl_this) // 2])

    print("\n## Control (B=1,H=32) per round -- this is the drift floor")
    print("  round " + " ".join(f"{i:>7}" for i in range(args.rounds)))
    print("  ms    " + " ".join(f"{t*1e3:>7.3f}" for t in ctl_round_median))
    cp10, cmed, cp90 = percentiles(samples["CTL"])
    print(
        f"  all {len(samples['CTL'])} samples: p10 {cp10*1e3:.3f} / med {cmed*1e3:.3f} / "
        f"p90 {cp90*1e3:.3f} ms   spread p90/p10 = {cp90/cp10:.3f}x   "
        f"round-median spread {max(ctl_round_median)/min(ctl_round_median):.3f}x"
    )

    res = {}
    print(f"\n## Interleaved points ({args.rounds} rounds x {args.samples} samples)")
    print(f"  {'point':>7} {'p10':>8} {'med':>8} {'p90':>8} {'p90/p10':>8}")
    for lbl in list(dict.fromkeys([p[0] for p in points])) + ["CTL"]:
        p10, med, p90 = percentiles(samples[lbl])
        res[lbl] = dict(p10=p10, median=med, p90=p90, n=len(samples[lbl]))
        print(f"  {lbl:>7} {p10*1e3:>7.3f} {med*1e3:>7.3f} {p90*1e3:>7.3f} {p90/p10:>7.3f}x")

    def scaling(rows, base):
        b = res[base]["median"]
        print(f"\n  {'x':>5} {'point':>7} {'med ms':>8} {'scaling':>8} {'ideal':>7} {'free':>7}")
        for x, lbl in rows:
            t = res[lbl]["median"]
            s = t / b
            ideal = x / rows[0][0]
            print(f"  {x:>5} {lbl:>7} {t*1e3:>7.3f} {s:>7.2f}x {ideal:>6.0f}x {1-s/ideal:>6.1%}")

    print("\n### batch axis (H=32)")
    scaling([(1, "B1H32"), (2, "B2H32"), (4, "B4H32"), (8, "B8H32")], "B1H32")
    print("\n### heads axis (B=1)")
    scaling([(8, "B1H8"), (16, "B1H16"), (32, "B1H32"), (64, "B1H64")], "B1H8")

    # ---- 3. CPU enqueue vs GPU execute
    print("\n## CPU enqueue vs total (K=20 iters, no sync inside)")
    print(f"  {'point':>7} {'cpu ms/it':>10} {'wall ms/it':>11} {'cpu frac':>9}")
    K = 20
    for lbl, _, _ in points:
        s = steps[lbl]
        for _ in range(3):
            s()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(K):
            s()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        cpu, wall = (t1 - t0) / K, (t2 - t0) / K
        res[lbl]["cpu_enqueue"] = cpu
        res[lbl]["wall_pipelined"] = wall
        print(f"  {lbl:>7} {cpu*1e3:>9.3f} {wall*1e3:>10.3f} {cpu/wall:>8.1%}")

    # ---- 4. kernel attribution
    from torch.profiler import ProfilerActivity, profile

    print("\n## Kernel attribution (5 iters)")
    for lbl in ("B1H32", "B1H64", "B8H32"):
        s = steps[lbl]
        for _ in range(3):
            s()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                s()
            torch.cuda.synchronize()
        tot, cnt = defaultdict(float), defaultdict(int)
        for e in prof.key_averages():
            if e.self_device_time_total > 0:
                tot[e.key] += e.self_device_time_total
                cnt[e.key] += e.count
        total = sum(tot.values())
        clean = res[lbl]["median"]
        print(
            f"\n  === {lbl}: {total/5/1000:.3f} ms GPU-busy/iter vs {clean*1e3:.3f} ms clean wall"
            f"  -> {100*(1-total/5/1e6/clean):.1f}% idle gap; "
            f"{sum(cnt.values())/5:.0f} kernel launches/iter"
        )
        res[lbl]["gpu_busy"] = total / 5 / 1e6
        res[lbl]["launches"] = sum(cnt.values()) / 5
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {100*v/total:5.1f}%  {v/5/1000:7.3f} ms  x{cnt[k]/5:>4.0f}  {k[:70]}")

    with open(args.out, "w") as f:
        json.dump(dict(tokens=T, control_round_median=ctl_round_median, points=res), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
