"""Does the fla chunk size C move GDN time the way a TRAFFIC model predicts?

`gdn_levers.py` measured the dominant delta-rule kernels at 49-85% of this GPU's achievable copy
bandwidth (3.55 TB/s measured), so the chunked delta rule is memory-bound, not scan-bound. The
largest single tensor is the materialised per-chunk state `h`, shape (B, T/C, H, K, V) -- 128 MiB
at C=64, T=8192, H=32, K=V=128. Its size is INVERSELY proportional to C, while the intra-chunk
work grows with C. So a bandwidth-bound kernel should get FASTER with larger C and markedly slower
with smaller C.

C > 64 is not reachable: `fla.ops.utils.solve_tril` only ships 16x16 / 32x32 / 64x64 merge kernels
(`assert A.shape[-1] in [16, 32, 64]`), so raising the chunk size is a kernel-authoring job, not
the config knob an earlier note in the README claimed. What IS reachable is the other direction --
C = 16, 32, 64 -- which tests the same slope and tells us what C = 128 would be worth.

Forward only: `chunk_gated_delta_rule_bwd_dhu` and `prepare_wy_repr_bwd` hardcode BT = 64 and
ignore their `chunk_size` argument, so the backward cannot be swept without forking those kernels.

Each arm is captured in a CUDA graph and timed by replay, which removes the CPU launch cost that
dominates this sequence in eager mode (see the README) and makes the GPU time nearly noise-free.

    python research/kv_route/flexcompute_40x/gdn_chunksize.py --tokens 8192
"""

import argparse
import json
import time

import torch

DK = DV = 128


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


def build_fwd(C):
    """fla's `chunk_gated_delta_rule_fwd`, with the chunk size threaded through every call."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum, solve_tril

    def fwd(q, k, v, g, beta, scale):
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        g = chunk_local_cumsum(g, chunk_size=C)
        A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, chunk_size=C, output_dtype=torch.float32)
        A = solve_tril(A=A, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g, initial_state=None, output_final_state=False, chunk_size=C
        )
        return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, chunk_size=C)

    return fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--chunks", default="16,32,64,128")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="research/kv_route/flexcompute_40x/gdn_chunksize.json")
    args = ap.parse_args()
    T, H = args.tokens, args.heads
    dev = torch.device("cuda")
    print(f"# GPU: {torch.cuda.get_device_name(0)}  T={T} H={H} DK={DK} DV={DV}  FORWARD ONLY")

    torch.manual_seed(0)
    q = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
    g = torch.rand(1, T, H, device=dev, dtype=torch.float32) * -0.1
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
    scale = DK**-0.5

    ref, res = None, {}
    for C in [int(x) for x in args.chunks.split(",")]:
        fwd = build_fwd(C)
        try:
            with torch.no_grad():
                o = fwd(q, k, v, g, beta, scale)
                for _ in range(5):
                    fwd(q, k, v, g, beta, scale)
        except Exception as e:  # noqa: BLE001
            print(f"  C={C:>4}  UNSUPPORTED: {type(e).__name__}: {str(e)[:110]}")
            continue
        if ref is None:
            ref = o.float()
            rel = 0.0
        else:
            rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()

        replay = None
        try:
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    fwd(q, k, v, g, beta, scale)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr), torch.no_grad():
                fwd(q, k, v, g, beta, scale)
            replay = gr.replay
        except Exception as e:  # noqa: BLE001
            print(f"  C={C:>4}  graph capture failed ({type(e).__name__}), timing eagerly")

        fn = replay if replay is not None else (lambda: fwd(q, k, v, g, beta, scale))
        ts = [time_once(fn) for _ in range(args.iters)]
        p10, med, p90 = percentiles(ts)
        h_mib = (T // C) * H * DK * DV * 2 / 2**20
        res[C] = dict(
            p10=p10, median=med, p90=p90, h_mib=h_mib, rel_vs_first=rel, graphed=replay is not None
        )
        print(
            f"  C={C:>4}  p10 {p10*1e6:>8.1f} med {med*1e6:>8.1f} p90 {p90*1e6:>8.1f} us   "
            f"h tensor {h_mib:>6.1f} MiB   rel-vs-C{args.chunks.split(',')[0]} {rel:.2e}"
        )

    if 64 in res:
        b = res[64]["median"]
        print("\n  relative to C=64 (median):")
        for C, r in sorted(res.items()):
            print(f"    C={C:>4}  {b/r['median']:>6.3f}x   h {r['h_mib']:>6.1f} MiB")

    with open(args.out, "w") as f:
        json.dump({str(c): r for c, r in res.items()}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
