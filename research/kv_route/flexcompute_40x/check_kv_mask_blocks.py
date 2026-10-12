"""Is the FlexAttention kept-key mask still exact after the full/partial block split?

``_compacted_block_mask`` now marks a (query block, compacted key block) pair as FULL when every
entry in it is allowed, so FlexAttention can skip ``mask_mod`` there. Over-claiming a full block
would silently let a query attend to a key it must not see, so this checks the flex output against
the reference path in the same function (a materialised boolean mask + SDPA), which is independent
of the block bookkeeping.

Also times the two block-mask builders so the speed claim is measured, not assumed.

    python research/kv_route/flexcompute_40x/check_kv_mask_blocks.py
"""

import argparse
import time

import torch

from olmo_core.nn.attention import kv_route


def reference(q, k, v, keep, doc, leftpad, scale):
    """Force the non-flex branch of ``_masked_attention``."""
    saved = kv_route._HAS_FLEX
    kv_route._HAS_FLEX = False
    try:
        return kv_route._masked_attention(q, k, v, keep, doc, leftpad, scale)
    finally:
        kv_route._HAS_FLEX = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    B, T, Hq, Hk, D = 1, args.T, args.heads, args.kv_heads, args.dim
    q = torch.randn(B, T, Hq, D, device=dev, dtype=torch.float32)
    k = torch.randn(B, T, Hk, D, device=dev, dtype=torch.float32)
    v = torch.randn(B, T, Hk, D, device=dev, dtype=torch.float32)
    scale = D**-0.5

    for n_docs in (1, 8):
        doc = None
        if n_docs > 1:
            per = T // n_docs
            doc = (torch.arange(T, device=dev) // per).clamp(max=n_docs - 1)
        for frac in (1.0, 0.5, 0.25, 0.05):
            keep = torch.zeros(B, T, dtype=torch.bool, device=dev)
            kk = max(1, int(round(frac * T)))
            keep[:, torch.linspace(0, T - 1, kk, device=dev).round().long().unique()] = True
            ref = reference(q, k, v, keep, doc, None, scale)
            got = kv_route._masked_attention(q, k, v, keep, doc, None, scale)
            rel = (ref - got).abs().max() / ref.abs().max().clamp(min=1e-9)
            print(f"docs={n_docs:<3} keep={frac:<5} rel max-err {float(rel):.3e}")

    # Timing in bf16 (what training runs in) against the flash baseline the routed layer replaces.
    # Arms interleaved with the flash control re-measured every round, per the directory's
    # measurement discipline.
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    try:
        from flash_attn import flash_attn_func

        def flash():
            return flash_attn_func(qb, kb, vb, causal=True, softmax_scale=scale)

    except Exception:

        def flash():
            n_rep = Hq // Hk
            return torch.nn.functional.scaled_dot_product_attention(
                qb.transpose(1, 2),
                kb.repeat_interleave(n_rep, dim=2).transpose(1, 2),
                vb.repeat_interleave(n_rep, dim=2).transpose(1, 2),
                is_causal=True,
                scale=scale,
            )

    def timeit(fn, iters=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    print(f"\n# bf16, T={T}, one attention layer, median of 10; control = plain causal flash")
    print(f"{'round':>5} {'arm':>12} {'ms':>9} {'vs flash':>9}")
    for rnd in range(3):
        base = timeit(flash)
        print(f"{rnd:>5} {'flash':>12} {base * 1e3:>9.3f} {1.0:>9.2f}")
        for frac in (1.0, 0.5, 0.25, 0.05):
            keep = torch.zeros(B, T, dtype=torch.bool, device=dev)
            kk = max(1, int(round(frac * T)))
            keep[:, torch.linspace(0, T - 1, kk, device=dev).round().long().unique()] = True
            t = timeit(lambda: kv_route._masked_attention(qb, kb, vb, keep, None, None, scale))
            print(f"{rnd:>5} {'keep=' + str(frac):>12} {t * 1e3:>9.3f} {t / base:>9.2f}")

    breakdown(qb, kb, vb, scale, T, Hq, Hk, dev)


def breakdown(q, k, v, scale, T, Hq, Hk, dev):
    """Split the kept-key path's fixed cost into gather/cat, block-mask build, and the flex call."""
    from torch.profiler import ProfilerActivity, profile

    keep = torch.zeros(1, T, dtype=torch.bool, device=dev)
    kk = max(1, int(round(0.25 * T)))
    keep[:, torch.linspace(0, T - 1, kk, device=dev).round().long().unique()] = True

    counts = keep.sum(1)
    Kp = min(T, kv_route._round_up(max(1, int(counts.max().item())), 1024))
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :Kp]
    valid_slot = torch.arange(Kp, device=dev)[None, :] < counts[:, None]
    pos_k = torch.where(valid_slot, order, torch.full_like(order, T + 1))

    def prep():
        gidx = order[:, :, None, None].expand(1, Kp, Hk, k.shape[-1])
        k_c = torch.gather(k, 1, gidx)
        v_c = torch.gather(v, 1, gidx)
        kc = torch.cat([k_c, k], dim=1).transpose(1, 2).contiguous()
        vc = torch.cat([v_c, v], dim=1).transpose(1, 2).contiguous()
        return kc, vc, q.transpose(1, 2).contiguous()

    def mask_mod(b, h, qi, kj):
        in_c = kj < Kp
        pk = pos_k[b, torch.clamp(kj, max=Kp - 1)]
        c_ok = in_c & (pk <= qi)
        d_ok = (~in_c) & ((kj - Kp) == qi) & (~keep)[b, qi]
        return c_ok | d_ok

    def build():
        return kv_route._compacted_block_mask(mask_mod, pos_k, counts, None, T, Kp, dev)

    kc, vc, q_ = prep()
    bm = build()

    def flexcall():
        return kv_route._flex_attention(
            q_, kc, vc, block_mask=bm, scale=scale, enable_gqa=(Hq != Hk)
        )

    def timeit(fn, iters=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    print(f"\n# breakdown at T={T}, keep=0.25, bf16 (median of 10)")
    for name, fn in (("gather+cat", prep), ("block-mask build", build), ("flex call", flexcall)):
        print(f"{name:>18} {timeit(fn) * 1e3:>8.3f} ms")

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            flexcall()
        torch.cuda.synchronize()
    rows = sorted(
        ((e.self_device_time_total / 3 / 1e3, e.count / 3, e.key) for e in prof.key_averages()),
        reverse=True,
    )
    print("  flex-call GPU kernels (ms, calls, name):")
    for ms, n, key in rows[:6]:
        if ms > 0:
            print(f"    {ms:>8.3f} {n:>6.0f}  {key[:70]}")


if __name__ == "__main__":
    main()
