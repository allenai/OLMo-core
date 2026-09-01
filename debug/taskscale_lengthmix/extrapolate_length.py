"""Extrapolate each task's token budget from the measured rungs out to 64k / 128k / 256k / 1M.

Two stages, both fitted from measured points only:

  1. Per rung, the data law f(B) = fmax*B^g/(B^g+K^g) gives B(target) -- the tokens needed to hit a
     target score at that context length. A rung with only two budgets is fitted with fmax pinned
     to a supplied ceiling, since two points cannot identify three parameters.
  2. Across rungs, log B(target) is regressed on log L. The slope is the task's LENGTH EXPONENT
     beta: B(L) proportional to L^beta. beta ~ 0 means length is nearly free once the task is
     learned; beta ~ 2 means every doubling of context costs 4x the data.

The extrapolation is then B(L) = B(L_max_measured) * (L / L_max)^beta.

Read the output as an ORDER OF MAGNITUDE, not a number. Every prediction past the largest measured
rung compounds two extrapolations -- a data law fitted on three budgets and a length law fitted on
three or four rungs -- and 1M is five doublings past our 32k ceiling. The r2 column says how well
the length law fits the rungs we DID measure; a low r2 means the beta itself is not trustworthy.
"""
import argparse
import json

import numpy as np
from scipy.optimize import curve_fit

RUNG_TOK = {"2k": 2048, "3k": 3072, "4k": 4096, "8k": 8192, "16k": 16384,
            "32k": 32768, "64k": 65536}
TARGETS_OUT = (65536, 131072, 262144, 1048576)


def hill(B, fmax, g, K):
    return fmax * B**g / (B**g + K**g)


def fit_rung(budgets, scores, ceiling):
    B, f = np.asarray(budgets, float), np.asarray(scores, float)
    if len(B) >= 3:
        # clamp the fmax seed into the bounds: a rung sitting at the floor (textgroups scores 0.00)
        # otherwise seeds fmax at 0 and scipy rejects the guess outright.
        p0 = [min(max(max(f) * 1.05, 0.06), 1.04), 1.0, float(np.median(B))]
        p, _ = curve_fit(hill, B, f, p0=p0,
                         bounds=([0.05, 0.1, 1e5], [1.05, 4.0, 1e11]), maxfev=400000)
        return p
    # two points: pin fmax, fit (g, K) only
    def h2(B_, g, K):
        return hill(B_, ceiling, g, K)
    p, _ = curve_fit(h2, B, f, p0=[1.0, float(np.median(B))],
                     bounds=([0.1, 1e5], [4.0, 1e11]), maxfev=400000)
    return np.array([ceiling, p[0], p[1]])


def budget_for(p, t):
    fmax, g, K = p
    return None if t >= fmax else K * (t / (fmax - t)) ** (1.0 / g)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points")
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--ceiling", type=float, default=1.0,
                    help="fmax to pin when a rung has only two budgets")
    a = ap.parse_args()
    data = json.load(open(a.points))

    print(f"target score = {a.target}\n")
    for task, variants in sorted(data.items()):
        for variant in ("dense", "sparse"):
            rungs = variants.get(variant, {})
            # If the task never reaches the target anywhere we measured, a "budget to reach it" is
            # pure fiction -- the fit is extrapolating a curve that has not approached the value.
            best = max((v for r in rungs.values() for v in r.values()), default=0.0)
            if best < a.target:
                print(f"{task:14s} {variant:6s}  target {a.target} never reached in the measured "
                      f"range (best {best:.3f}) -- no prediction")
                continue
            pts = []
            for lab, budgets in rungs.items():
                if lab not in RUNG_TOK or len(budgets) < 2:
                    continue
                bs = sorted(float(b) for b in budgets)
                p = fit_rung(bs, [budgets[f"{int(b)}"] for b in bs], a.ceiling)
                need = budget_for(p, a.target)
                # A rung whose implied budget is far beyond anything we ran is not a measurement of
                # that rung's cost, it is the fit running away: drop it rather than let it set the
                # length exponent. (qdmatch's 64k rung did exactly this, forcing beta to 5.5.)
                if need is not None and need > 20 * max(bs):
                    need = None
                pts.append((RUNG_TOK[lab], need, max(bs), p[0]))
            usable = [(L, b) for L, b, _, _ in pts if b is not None]
            dropped = [L for L, b, _, _ in pts if b is None]
            if len(usable) < 2:
                why = ("target above every rung's fitted ceiling"
                       if pts else "fewer than two budgets per rung")
                print(f"{task:14s} {variant:6s}  no length law -- {why}")
                continue
            L = np.log(np.array([x[0] for x in usable], float))
            B = np.log(np.array([x[1] for x in usable], float))
            beta, c = np.polyfit(L, B, 1)
            r2 = 1 - np.sum((B - (beta * L + c)) ** 2) / max(np.sum((B - B.mean()) ** 2), 1e-12)
            Lmax = max(x[0] for x in usable)
            Bmax = [b for l, b in usable if l == Lmax][0]
            # was the anchor itself inside the budgets we ran at that rung?
            ranmax = max(bm for L_, b_, bm, _ in pts if L_ == Lmax and b_ is not None)
            anchor_note = "" if Bmax <= ranmax else f" [anchor is {Bmax / ranmax:.0f}x past measurement]"
            preds = "  ".join(
                f"{t // 1024}k:{Bmax * (t / Lmax) ** beta / 1e9:.2f}B" for t in TARGETS_OUT)
            drop = f"  [dropped {','.join(str(x // 1024) + 'k' for x in sorted(dropped))}]" if dropped else ""
            print(f"{task:14s} {variant:6s}  beta={beta:5.2f} r2={r2:4.2f}  rungs="
                  f"{len(usable)}  anchor {Lmax // 1024}k={Bmax / 1e6:.0f}M  ->  {preds}{drop}"
                  f"{anchor_note}")


if __name__ == "__main__":
    main()
