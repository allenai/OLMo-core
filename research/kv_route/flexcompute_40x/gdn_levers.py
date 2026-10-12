"""Act on the `gdn_bottleneck.py` diagnosis: the delta-rule sequence is CPU-launch-bound at B=1.

That run measured, interleaved and drift-controlled at T=8192, H=32, B=1: 2.125 ms of GPU-busy
time against a 4.418 ms wall (51.9% idle), with the CPU taking 4.667 ms to ENQUEUE one iteration
(99.9% of the wall). So the GPU is waiting on Python/Triton, not the other way round.

This script tests the consequences:

  1. drift      -- is the 1.96x within-process control drift Python-side (GC) or hardware? Control
                   arms with gc on and gc off, interleaved.
  2. cpu attrib -- torch profiler with CPU activity on, to name what is spending the ~4.6 ms.
  3. graph      -- capture fwd+bwd in a CUDA graph and replay. This removes ALL per-launch CPU cost,
                   so it measures the ceiling available from fixing the launch path.
  4. varlen     -- the same T tokens as N packed documents (`cu_seqlens`), which is what training
                   actually passes. Changes the scan grid without changing the launch count.
  5. block      -- the same, one level up: a real 4B GDN block (projections + convs + delta rule +
                   gated norm), stock vs CUDA-graphed, so the ceiling is quoted in situ.
  6. bandwidth  -- an honest roofline for the dominant kernel. `resolve_gdn_gap.py` priced traffic
                   as q/k/v/o only and concluded 15.8% of HBM peak; the chunked algorithm also
                   materialises `h` = (B, T/64, H, K, V), 134 MB at this shape, which that model
                   omitted entirely. Byte counts here are CALCULATED from real tensor shapes and
                   compared against a MEASURED device-copy bandwidth, not the spec sheet.

    python research/kv_route/flexcompute_40x/gdn_levers.py --tokens 8192
"""

import argparse
import gc
import json
import time
from collections import defaultdict

import torch

DK = DV = 128
VOCAB = 248320


def percentiles(ts):
    ts = sorted(ts)
    n = len(ts)
    return (
        ts[int(round(0.10 * (n - 1)))],
        ts[int(round(0.50 * (n - 1)))],
        ts[int(round(0.90 * (n - 1)))],
    )


def time_once(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def make_step(B, H, T, dev, cu_seqlens=None):
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
            cu_seqlens=cu_seqlens,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        o.sum().backward()

    return step


def capture(step, warmups=5):
    """Capture one fwd+bwd into a CUDA graph. Returns a replay callable, or None if capture fails."""
    for _ in range(warmups):
        step()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        step()
    return gr.replay


def cpu_vs_wall(fn, K=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(K):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / K, (t2 - t0) / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", default="research/kv_route/flexcompute_40x/gdn_levers.json")
    args = ap.parse_args()
    T, H = args.tokens, args.heads
    dev = torch.device("cuda")
    print(f"# GPU: {torch.cuda.get_device_name(0)}  T={T} H={H} DK={DK} DV={DV}")
    out = {}

    # ---------------- 1 + 4: kernel-level arms, interleaved ----------------
    arms = {"base": make_step(1, H, T, dev), "CTL": make_step(1, H, T, dev)}
    for n in (4, 8, 32):
        cu = torch.arange(0, T + 1, T // n, device=dev, dtype=torch.int32)
        arms[f"varlen{n}"] = make_step(1, H, T, dev, cu_seqlens=cu)
    for lbl in list(arms):
        for _ in range(8):
            arms[lbl]()
    graph_replay = None
    try:
        graph_replay = capture(make_step(1, H, T, dev))
        arms["graph"] = graph_replay
        print("# CUDA graph captured OK")
    except Exception as e:  # noqa: BLE001
        print(f"# CUDA graph capture FAILED: {type(e).__name__}: {e}")
    torch.cuda.synchronize()

    labels = list(arms)
    samples = {lbl: [] for lbl in labels}
    ctl_rounds, gc_state = [], []
    for r in range(args.rounds):
        # Alternate GC on/off by round so a GC-driven drift shows up as an alternating control.
        gc_on = r % 2 == 0
        (gc.enable if gc_on else gc.disable)()
        order = labels[r % len(labels) :] + labels[: r % len(labels)]
        c = [time_once(arms["CTL"]) for _ in range(args.samples)]
        for lbl in order:
            samples[lbl] += [time_once(arms[lbl]) for _ in range(args.samples)]
        c += [time_once(arms["CTL"]) for _ in range(args.samples)]
        samples["CTL"] += c
        ctl_rounds.append(sorted(c)[len(c) // 2])
        gc_state.append("gc-on" if gc_on else "gc-off")
    gc.enable()

    print("\n## Control per round (drift), alternating gc on/off")
    for st, t in zip(gc_state, ctl_rounds):
        print(f"  {st:>7} {t*1e3:>8.3f} ms")
    print(f"  round-median spread {max(ctl_rounds)/min(ctl_rounds):.3f}x")
    out["control_rounds"] = list(zip(gc_state, ctl_rounds))

    ctl_med = percentiles(samples["CTL"])[1]
    out["arms"] = {}
    print(f"\n{'arm':>10} {'p10':>8} {'med':>8} {'p90':>8} {'vs CTL':>8} {'cpu ms':>8} {'cpu%':>6}")
    for lbl in labels:
        p10, med, p90 = percentiles(samples[lbl])
        cpu, wall = cpu_vs_wall(arms[lbl])
        out["arms"][lbl] = dict(p10=p10, median=med, p90=p90, cpu=cpu, wall_pipelined=wall)
        print(
            f"{lbl:>10} {p10*1e3:>7.3f} {med*1e3:>7.3f} {p90*1e3:>7.3f} {ctl_med/med:>7.3f}x "
            f"{cpu*1e3:>7.3f} {100*cpu/wall:>5.1f}%"
        )

    # ---------------- 2: what is the CPU doing? ----------------
    from torch.profiler import ProfilerActivity, profile

    print("\n## CPU-side attribution of one base iteration (5 iters)")
    s = arms["base"]
    for _ in range(3):
        s()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            s()
        torch.cuda.synchronize()
    cpu_tot = defaultdict(float)
    cnt = defaultdict(int)
    for e in prof.key_averages():
        if e.self_cpu_time_total > 0:
            cpu_tot[e.key] += e.self_cpu_time_total
            cnt[e.key] += e.count
    tot = sum(cpu_tot.values())
    print(f"  total self-CPU {tot/5/1000:.3f} ms/iter (profiler adds overhead; use for ranking)")
    for k, v in sorted(cpu_tot.items(), key=lambda kv: -kv[1])[:14]:
        print(f"    {100*v/tot:5.1f}%  {v/5/1000:7.3f} ms  x{cnt[k]/5:>5.1f}  {k[:66]}")

    # ---------------- 5: full 4B GDN block, stock vs graphed ----------------
    print("\n## Full 4B GDN block, stock vs CUDA-graphed")
    from olmo_core.nn.attention import Attention
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    model = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB).build(init_device="cpu").to(dev)
    model.init_weights(max_seq_len=T, device=dev)
    model = model.to(torch.bfloat16)
    gdn = [b.attention for b in model.blocks.values() if type(b.attention) is not Attention][0]
    x = torch.randn(1, T, model.d_model, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def block_step():
        gdn.zero_grad(set_to_none=False)
        x.grad = None
        gdn(x).sum().backward()

    for _ in range(5):
        block_step()
    torch.cuda.synchronize()
    block_graph = None
    try:
        block_graph = capture(block_step)
        print("# block graph captured OK")
    except Exception as e:  # noqa: BLE001
        print(f"# block graph capture FAILED: {type(e).__name__}: {e}")

    blk = {"block": block_step}
    if block_graph is not None:
        blk["block-graph"] = block_graph
    bsamples = {k: [] for k in blk}
    for _ in range(args.rounds * 2):
        for k, fn in blk.items():
            bsamples[k] += [time_once(fn) for _ in range(args.samples)]
    out["block"] = {}
    print(f"\n{'arm':>12} {'p10':>8} {'med':>8} {'p90':>8} {'cpu ms':>8} {'cpu%':>6}")
    for k in blk:
        p10, med, p90 = percentiles(bsamples[k])
        cpu, wall = cpu_vs_wall(blk[k], K=10)
        out["block"][k] = dict(p10=p10, median=med, p90=p90, cpu=cpu)
        print(
            f"{k:>12} {p10*1e3:>7.3f} {med*1e3:>7.3f} {p90*1e3:>7.3f} {cpu*1e3:>7.3f} "
            f"{100*cpu/wall:>5.1f}%"
        )
    if "block-graph" in out["block"]:
        sp = out["block"]["block"]["median"] / out["block"]["block-graph"]["median"]
        print(f"  block speedup from graphing: {sp:.3f}x (median)")

    # ---------------- 6: honest bandwidth roofline ----------------
    print("\n## Bandwidth: MEASURED achievable copy rate vs CALCULATED kernel traffic")
    nbytes = 128 * H * DK * DV * 2  # the `h` tensor at this shape
    a = torch.empty(nbytes // 2, device=dev, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    ts = [time_once(lambda: b.copy_(a)) for _ in range(20)]
    tcopy = percentiles(ts)[1]
    copy_bw = 2 * nbytes / tcopy
    print(f"  device copy of {nbytes/2**20:.0f} MiB: {tcopy*1e6:.1f} us -> {copy_bw/1e12:.2f} TB/s")
    print("  (this is the achievable rate on this GPU; HBM3e spec is 4.8 TB/s)")
    out["copy_bw"] = copy_bw

    # exact traffic of chunk_gated_delta_rule_fwd_h at this shape, from the tensors it touches
    tok = T * H * DK * 2
    h_bytes = dict(
        read_k=tok, read_w=tok, read_u=tok, read_g=T * H * 4, write_h=nbytes, write_vnew=tok
    )
    print(
        f"  chunk_gated_delta_rule_fwd_h traffic (calculated): {sum(h_bytes.values())/2**20:.0f} MiB"
    )
    for k, v in h_bytes.items():
        print(f"     {k:>11} {v/2**20:>7.1f} MiB")
    out["h_kernel_bytes"] = h_bytes

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
