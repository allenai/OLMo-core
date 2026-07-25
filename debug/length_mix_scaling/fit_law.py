#!/usr/bin/env python
"""Fit the length-mix scaling law from the per-arm vLLM eval results.

Model (long pool held FIXED, short data added on top):

    f1@32k(N_S) = f_inf - (f_inf - f0) * exp(-N_S / tau)

  f0    = f1 with zero short data (the A0/B0 anchor)
  f_inf = asymptote as short data grows
  tau   = short tokens needed to close 1-1/e of the gap  -> "where short data stops paying"

Reported per row:
  * marginal value of the first short tokens  df1/dN_S at N_S=0 = (f_inf-f0)/tau
  * f1 gained per GPU-minute of short data (wall-clock ~ tokens, since 4B throughput is flat
    8k->40k), which is the number that actually decides a production mix
  * Row A vs Row B: if short data helps MORE when long data is scarce, it substitutes for long
    data; if it helps equally, it adds something long data never provided.
  * Row C at matched token budget: does piling on short data beat the uniform production mix at
    equal cost? That is the original question.

Refuses to over-claim: with <4 usable points in a row it reports the raw points and a linear
slope instead of a 3-parameter fit, and every f1 carries its binomial SE.
"""
import argparse
import glob
import json
import math
import os

ARM_TOKENS = {  # (long_tokens, short_tokens) as composed; filled from arm metadata when available
    "A0": (35.2, 0.0), "A1": (35.2, 20.0), "A2": (35.2, 41.5), "A3": (35.2, 84.5),
    "A4": (35.2, 148.7), "B0": (20.2, 0.0), "B2": (20.2, 21.5), "B4": (20.2, 84.5),
    "A25": (35.2, 105.9), "A30": (35.2, 105.9), "A35": (35.2, 127.4),
    "A4e": (35.2, 148.7),   # A4 data, 200 steps instead of 351
    "C3": (None, None), "C4": (None, None),
}


def load(results_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        arm = os.path.basename(p)[:-5]
        if "_rung" in arm:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        r = d.get("rungs", {}).get("32768", {})
        if isinstance(r.get("f1"), (int, float)):
            out[arm] = {"f1": r["f1"], "se": r.get("binomial_se"),
                        "parse": r.get("parse_rate"), "n": r.get("eval_size"),
                        "all": {k: v.get("f1") for k, v in d.get("rungs", {}).items()}}
    return out


def fit_exp(xs, ys):
    """Grid-search f_inf/tau (2 params; f0 pinned to the N_S=0 point). Tiny data -> no optimizer."""
    if len(xs) < 4:
        return None
    f0 = ys[0]
    best = None
    lo, hi = min(ys), max(ys)
    for f_inf in [lo + (hi - lo) * i / 200 + (hi - lo) * 0.5 * j
                  for i in range(201) for j in (0, 1)]:
        if f_inf <= f0:
            continue
        for tau in [2 ** (k / 4) for k in range(0, 45)]:
            sse = sum((y - (f_inf - (f_inf - f0) * math.exp(-x / tau))) ** 2
                      for x, y in zip(xs, ys))
            if best is None or sse < best[0]:
                best = (sse, f_inf, tau, f0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline-f1", type=float, default=0.335,
                    help="ctc-s5-contra-full-4b uniform mix @32k, vLLM")
    args = ap.parse_args()
    res = load(args.results)
    if not res:
        raise SystemExit(f"no usable arm results in {args.results}")

    print(f"=== length-mix results (f1@32k, vLLM; production uniform baseline = {args.baseline_f1}) ===")
    print(f"{'arm':<4} {'long(M)':>8} {'short(M)':>9} {'f1@32k':>8} {'SE':>7} {'parse':>6}  rungs 2k/8k/32k")
    for arm in sorted(res):
        L, S = ARM_TOKENS.get(arm, (None, None))
        d = res[arm]
        a = d["all"]
        rungs = "/".join(f"{a.get(k):.3f}" if isinstance(a.get(k), float) else "  -  "
                         for k in ("2048", "8192", "32768"))
        se = f"{d['se']:.3f}" if d["se"] else "  -  "
        print(f"{arm:<4} {('%.1f'%L) if L else '  unif':>8} {('%.1f'%S) if S is not None else '   -':>9} "
              f"{d['f1']:>8.3f} {se:>7} {d['parse'] if d['parse'] is not None else '-':>6}  {rungs}")
        if d["parse"] is not None and d["parse"] < 0.5:
            print(f"     !! {arm} parse_rate {d['parse']:.2f} -- DUMP GENERATIONS before believing this")

    for row, arms in (("A (full long pool)", ["A0", "A1", "A2", "A3", "A30", "A35", "A4"]),
                      ("B (half long pool)", ["B0", "B2", "B4"])):
        pts = [(ARM_TOKENS[a][1], res[a]["f1"]) for a in arms if a in res]
        if len(pts) < 2:
            print(f"\n{row}: only {len(pts)} point(s) -- cannot fit"); continue
        pts.sort()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        print(f"\n--- Row {row} ---")
        print("   short_tokens(M) -> f1@32k : " + ", ".join(f"{x:.0f}->{y:.3f}" for x, y in pts))
        # The round-1 curve is NON-MONOTONIC (rises to an optimum, then collapses), so a
        # saturating-exponential is the wrong model for the whole row -- fitting it across the
        # turnover reports a meaningless tau. Locate the argmax first, fit only the RISING part,
        # and report the degradation past the peak separately.
        peak_i = max(range(len(ys)), key=lambda i: ys[i])
        print(f"   PEAK: f1={ys[peak_i]:.3f} at {xs[peak_i]:.0f}M short tokens")
        if peak_i < len(ys) - 1:
            drop = ys[peak_i] - ys[-1]
            se = math.sqrt(max(ys[peak_i], 1e-9) * (1 - ys[peak_i]) / 500) + \
                 math.sqrt(max(ys[-1], 1e-9) * (1 - ys[-1]) / 500)
            print(f"   DEGRADATION past peak: {ys[peak_i]:.3f} -> {ys[-1]:.3f} "
                  f"({drop:+.3f}, ~{drop/se:.1f} SE) by {xs[-1]:.0f}M short tokens")
            print("   => monotone-saturation model does NOT apply; this row has an interior optimum")
        rise_x, rise_y = xs[:peak_i + 1], ys[:peak_i + 1]
        f = fit_exp(rise_x, rise_y)
        if f is None:
            if len(rise_x) >= 2 and rise_x[-1] != rise_x[0]:
                slope = (rise_y[-1] - rise_y[0]) / (rise_x[-1] - rise_x[0])
                print(f"   rising part ({len(rise_x)} pts, too few to fit): linear slope "
                      f"= {slope*100:.4f} f1 per 100M short tokens")
            continue
        sse, f_inf, tau, f0 = f
        marg = (f_inf - f0) / tau
        print(f"   rising-part fit: f0={f0:.3f}  f_inf={f_inf:.3f}  tau={tau:.1f}M  (SSE={sse:.5f})")
        print(f"   marginal value at N_S=0 : {marg*100:.4f} f1 per 100M short tokens")

    # Compare rows at their PEAKS, not their endpoints. Row A's endpoint is the collapsed arm, so an
    # endpoint comparison pits a healthy arm against a broken one and reports nonsense.
    def peak_of(arms):
        pts = [(ARM_TOKENS[a][1], res[a]["f1"], a) for a in arms if a in res]
        return max(pts, key=lambda p: p[1]) if pts else None
    pa = peak_of(["A0", "A1", "A2", "A3", "A30", "A35", "A4"])
    pb = peak_of(["B0", "B2", "B4"])
    if pa and pb:
        print(f"\n--- substitute vs complement (compared at each row's PEAK) ---")
        print(f"   Row A peak: f1={pa[1]:.3f} ({pa[2]}, {pa[0]:.0f}M short on the FULL long pool)")
        print(f"   Row B peak: f1={pb[1]:.3f} ({pb[2]}, {pb[0]:.0f}M short on HALF the long pool)")
        se = math.sqrt(pa[1] * (1 - pa[1]) / 500) + math.sqrt(pb[1] * (1 - pb[1]) / 500)
        d = pb[1] - pa[1]
        if abs(d) <= 2 * se:
            print(f"   delta={d:+.3f} +/-{se:.3f} -> TIE: halving the long pool costs nothing once "
                  "short data is plentiful => short data SUBSTITUTES for long data at the optimum")
        elif d < 0:
            print(f"   delta={d:+.3f} +/-{se:.3f} -> long data still needed: halving it hurts even "
                  "at the optimum => COMPLEMENT")
        else:
            print(f"   delta={d:+.3f} +/-{se:.3f} -> less long data is BETTER at the optimum")
        # the low-short end tells the opposite story, and that contrast is the real finding
        if "A1" in res and "B2" in res:
            print(f"   BUT at ~20M short: A1 (full long) {res['A1']['f1']:.3f} vs "
                  f"B2 (half long) {res['B2']['f1']:.3f} -> long data is CRITICAL when short is scarce")

    print("\n--- Row C: equal-token uniform reference (the 'is it cheaper?' test) ---")
    for c, a in (("C3", "A3"), ("C4", "A4")):
        if c in res and a in res:
            d = res[a]["f1"] - res[c]["f1"]
            se = math.sqrt((res[a]["se"] or 0) ** 2 + (res[c]["se"] or 0) ** 2)
            verdict = "fixed-long+short WINS" if d > 2 * se else \
                      "uniform WINS" if d < -2 * se else "TIE (within 2 SE)"
            print(f"   {a} {res[a]['f1']:.3f} vs {c} {res[c]['f1']:.3f} (uniform)  "
                  f"delta={d:+.3f} +/-{se:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
