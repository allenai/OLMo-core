"""Does halving the GDN value head dim convert its traffic cut into wall-clock?

The delta rule was shown (see README) to run at ~68% of this GPU's measured achievable copy
bandwidth, so it is memory-bound and TRAFFIC is the only lever with headroom. The value head dim
sets the size of every V-shaped tensor (v, u, v_new, o, do, dv, dv2) AND of the materialised
per-chunk state h = (B, T/C, H, K, V), the single largest tensor in the block.

Naming, because the earlier README note got it wrong: the 4B config reaches value_dim 4096 via
`linear_num_value_heads=32` x `linear_value_head_dim=128`, and its `expand_v` is already
1.0 (= linear_value_head_dim / linear_key_head_dim), NOT 2.0. The knob is
`linear_value_head_dim` 128 -> 64, i.e. expand_v 1.0 -> 0.5, DV 128 -> 64, value_dim 4096 -> 2048.

Three levels, each an interleaved A/B in this one process with a repeated control:

  kernel  `chunk_gated_delta_rule` fwd+bwd at DV=128 vs DV=64, identical in every other respect.
          Also CUDA-graph captured, which the README established removes the ~2x eager CPU-side
          drift and yields the clean GPU number.
  block   a real 4B GDN block at both settings. NOT a like-for-like traffic test: w_v/w_g/w_out
          also halve, so this mixes the traffic win with a GEMM/parameter reduction.
  step    a full 4B fwd+bwd at both settings, since kernel wins mostly did not reach a 4B step.

The byte model below is the CORRECTED one -- it includes h and dh, whose omission produced the
earlier wrong "not bandwidth-bound" verdict. Bytes are CALCULATED from the tensors each fla
launcher passes; times are MEASURED; the achievable bandwidth it is compared against is MEASURED
in-process, not taken from the spec sheet.

    python research/kv_route/flexcompute_40x/gdn_expandv.py --tokens 8192
"""

import argparse
import json
import time

import torch

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


def traffic_bytes(T, H, DK, DV, C=64):
    """CALCULATED DRAM traffic of one fwd+bwd `chunk_gated_delta_rule`, per fla launcher.

    Each entry lists the tensors that launcher reads and writes; sizes come from the real shapes.
    Returns (total_bytes, per_kernel_dict). Approximate in the sense that it assumes no cache
    reuse between kernels (true: the tensors are far larger than L2) and full reuse inside one.
    """
    uk = T * H * DK * 2  # q, k, w, dq, dk, dw
    uv = T * H * DV * 2  # v, u, v_new, o, do, dv, dv2
    hh = (T // C) * H * DK * DV * 2  # h, dh
    a32, a16 = T * H * C * 4, T * H * C * 2
    gs, bs = T * H * 4, T * H * 2
    k = {
        "l2norm_fwd x2": 2 * (uk + uk),
        "l2norm_bwd x2": 2 * (3 * uk),
        "chunk_local_cumsum x2": 4 * gs,
        "chunk_scaled_dot_kkt_fwd": uk + gs + bs + a32,
        "solve_tril": a32 + a16,
        "recompute_w_u_fwd x2": 2 * (uk + uv + bs + a16 + gs + uk + uv),
        "chunk_..._fwd_h x2": 2 * (uk + uk + uv + gs + hh + uv),
        "chunk_fwd_o": uk + uk + uv + hh + gs + uv,
        "chunk_bwd_dv_local": uk + uk + gs + uv + uv,
        "chunk_..._bwd_dhu": 3 * uk + gs + uv + uv + hh + uv,
        "chunk_bwd_dqkwg": 3 * uk + uv + gs + hh + uv + uv + hh + 3 * uk + gs,
        "prepare_wy_repr_bwd": 2 * (uk + uv) + 2 * bs + 2 * gs + a32 + uk + uv,
    }
    return sum(k.values()), k


def make_step(B, H, T, DK, DV, dev):
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


def capture(step, warmups=5):
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


def interleave(arms, rounds, samples, control):
    """Round-robin over a rotated arm order, control re-measured inside every round."""
    labels = list(arms)
    samples_out = {lbl: [] for lbl in labels}
    ctl_rounds = []
    for r in range(rounds):
        order = labels[r % len(labels) :] + labels[: r % len(labels)]
        c = [time_once(arms[control]) for _ in range(samples)]
        for lbl in order:
            samples_out[lbl] += [time_once(arms[lbl]) for _ in range(samples)]
        c += [time_once(arms[control]) for _ in range(samples)]
        samples_out[control] += c
        ctl_rounds.append(sorted(c)[len(c) // 2])
    return samples_out, ctl_rounds


def report(name, samples_out, ctl_rounds, ref):
    print(f"\n## {name}: control per round (drift) " + " ".join(f"{t*1e3:.3f}" for t in ctl_rounds))
    print(f"   round-median spread {max(ctl_rounds)/min(ctl_rounds):.3f}x")
    base = percentiles(samples_out[ref])[1]
    print(f"   {'arm':>16} {'p10':>9} {'med':>9} {'p90':>9} {'p90/p10':>8} {'vs ' + ref:>10}")
    out = {}
    for lbl, ts in samples_out.items():
        p10, med, p90 = percentiles(ts)
        out[lbl] = dict(p10=p10, median=med, p90=p90, n=len(ts))
        print(
            f"   {lbl:>16} {p10*1e3:>8.3f} {med*1e3:>8.3f} {p90*1e3:>8.3f} "
            f"{p90/p10:>7.3f}x {base/med:>9.3f}x"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", default="research/kv_route/flexcompute_40x/gdn_expandv.json")
    args = ap.parse_args()
    T, H, DK = args.tokens, args.heads, args.dk
    dev = torch.device("cuda")
    print(f"# GPU: {torch.cuda.get_device_name(0)}  T={T} H={H} DK={DK}")
    out = {}

    # ---- measured achievable bandwidth, for the roofline denominators ----
    a = torch.empty(64 * 2**20, device=dev, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    tcopy = percentiles([time_once(lambda: b.copy_(a)) for _ in range(20)])[1]
    copy_bw = 2 * a.numel() * 2 / tcopy
    print(f"# MEASURED achievable copy bandwidth: {copy_bw/1e12:.2f} TB/s")
    out["copy_bw"] = copy_bw
    a = b = None  # the lambda above has already run

    print("\n## CALCULATED traffic per fwd+bwd (corrected model: includes h and dh)")
    tr = {}
    for DV in (128, 64):
        total, per = traffic_bytes(T, H, DK, DV)
        tr[DV] = total
        print(f"  DV={DV:>4}: {total/2**20:>8.1f} MiB total")
        for kk, vv in sorted(per.items(), key=lambda x: -x[1])[:5]:
            print(f"            {vv/2**20:>8.1f} MiB  {kk}")
    print(
        f"  traffic ratio DV64/DV128 = {tr[64]/tr[128]:.3f}  -> a perfectly bandwidth-bound "
        f"kernel would go {tr[128]/tr[64]:.3f}x faster"
    )
    out["traffic_mib"] = {str(k): v / 2**20 for k, v in tr.items()}

    # ---- level 1: isolated kernel, eager interleaved ----
    arms = {
        "DV128": make_step(1, H, T, DK, 128, dev),
        "DV64": make_step(1, H, T, DK, 64, dev),
        "CTL(DV128)": make_step(1, H, T, DK, 128, dev),
    }
    for lbl in arms:
        for _ in range(8):
            arms[lbl]()
    torch.cuda.synchronize()
    s, c = interleave(arms, args.rounds, args.samples, "CTL(DV128)")
    out["kernel_eager"] = report("kernel, eager", s, c, "DV128")

    # ---- level 1b: same, CUDA-graphed (clean GPU time) ----
    graphed = {}
    for DV in (128, 64):
        try:
            graphed[f"g-DV{DV}"] = capture(make_step(1, H, T, DK, DV, dev))
        except Exception as e:  # noqa: BLE001
            print(f"# graph capture failed for DV={DV}: {type(e).__name__}: {e}")
    if len(graphed) == 2:
        graphed["g-CTL"] = graphed["g-DV128"]
        gs, gc = interleave(graphed, args.rounds, args.samples, "g-CTL")
        gs.pop("g-CTL")
        out["kernel_graphed"] = report("kernel, CUDA-graphed", gs, gc, "g-DV128")
        print("\n   achieved bandwidth on the graphed (pure GPU) times:")
        for DV in (128, 64):
            t = out["kernel_graphed"][f"g-DV{DV}"]["median"]
            bw = tr[DV] / t
            print(
                f"     DV={DV:>4}  {t*1e3:>7.3f} ms  {tr[DV]/2**20:>7.1f} MiB  "
                f"-> {bw/1e12:>5.2f} TB/s = {100*bw/copy_bw:>5.1f}% of achievable"
            )
            out["kernel_graphed"][f"g-DV{DV}"]["achieved_bw"] = bw

    # ---- level 2 + 3: real 4B block and full step at both settings ----
    from olmo_core.nn.attention import Attention
    from olmo_core.nn.transformer import TransformerConfig

    models, blocks, steps = {}, {}, {}
    for DV in (128, 64):
        torch.manual_seed(0)
        m = (
            TransformerConfig.qwen3_5_4B(vocab_size=VOCAB, linear_value_head_dim=DV)
            .build(init_device="cpu")
            .to(dev)
        )
        m.init_weights(max_seq_len=T, device=dev)
        m = m.to(torch.bfloat16)
        models[DV] = m
        gdn = [b.attention for b in m.blocks.values() if type(b.attention) is not Attention][0]
        nparam = sum(p.numel() for p in m.parameters())
        print(
            f"\n# DV={DV}: head_v_dim={gdn.head_v_dim} value_dim={gdn.value_dim} "
            f"expand_v={gdn.expand_v} | model params {nparam/1e9:.3f}B"
        )
        out[f"params_DV{DV}"] = nparam
        x = torch.randn(1, T, m.d_model, device=dev, dtype=torch.bfloat16, requires_grad=True)
        ids = torch.randint(0, VOCAB, (1, T), device=dev)

        def mk_block(gdn=gdn, x=x):
            def f():
                gdn.zero_grad(set_to_none=True)
                x.grad = None
                gdn(x).sum().backward()

            return f

        def mk_step(m=m, ids=ids):
            def f():
                m.zero_grad(set_to_none=True)
                m(ids).sum().backward()

            return f

        blocks[f"blk-DV{DV}"] = mk_block()
        steps[f"step-DV{DV}"] = mk_step()

    blocks["blk-CTL"] = blocks["blk-DV128"]
    for f in list(blocks.values()) + list(steps.values()):
        for _ in range(4):
            f()
    torch.cuda.synchronize()
    bs, bc = interleave(blocks, args.rounds, args.samples, "blk-CTL")
    bs.pop("blk-CTL")
    out["block"] = report(
        "4B GDN block (mixes traffic win with a GEMM/param cut)", bs, bc, "blk-DV128"
    )

    steps["step-CTL"] = steps["step-DV128"]
    ss, sc = interleave(steps, max(3, args.rounds // 2), 2, "step-CTL")
    ss.pop("step-CTL")
    out["step"] = report("full 4B fwd+bwd step (NOT iso-parameter)", ss, sc, "step-DV128")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
