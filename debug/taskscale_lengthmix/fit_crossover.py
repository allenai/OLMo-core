"""Fit per-task data-scaling laws and ask when (if ever) sparse overtakes dense.

Input is a JSON of measured points, one entry per (task, variant, rung):

  {"outlier": {"dense": {"16k": {"64000000": 0.453, ...}, ...}, "sparse": {...}}, ...}

For each (task, rung, variant) it fits the Hill law f(B) = fmax * B^g / (B^g + K^g) -- the same
form the outlier/qdmatch/nq campaign used, so the numbers stay comparable -- and reports:

  * B(f) for a few target scores, i.e. the token bill each variant pays for that score,
  * the token RATIO sparse/dense at the highest score both variants actually reach,
  * whether the two fitted curves cross inside a plausible budget range, and where.

Three points per curve exactly determine three parameters, so the fit interpolates rather than
generalizes: it pins the ceiling well and K/g poorly. Treat any B(f) above the largest measured
budget as an extrapolation, which is why every such figure is printed with a >= marker.
"""
import argparse
import json

import numpy as np
from scipy.optimize import curve_fit

TARGETS = (0.7, 0.8, 0.9)
MAX_EXTRAP = 1e11
G_MAX = 4.0          # upper bound on the Hill exponent; a fit that reaches it did not converge


def hill(B, fmax, g, K):
    return fmax * B**g / (B**g + K**g)


FLOOR = 0.02  # a ladder whose best score is under this carries no scaling signal to fit


def fit(budgets, scores):
    """Hill fit, or None for a ladder that never leaves the floor.

    reorder@16k is 0.000 at all three budgets and reorder@4k sparse is slightly NEGATIVE (kendall
    tau, not f1). Both made p0[0] = max(f) * 1.05 fall outside the fmax lower bound, and curve_fit
    raises "Initial guess is outside of provided bounds" -- which aborted the whole script partway
    through the task list, so every task sorting after `reorder` silently went unfitted. Clamp the
    guess, and refuse to fit a flat-zero ladder rather than reporting a law for it.
    """
    B = np.asarray(budgets, float)
    f = np.asarray(scores, float)
    if float(np.max(f)) < FLOOR:
        return None
    p0 = [min(max(float(np.max(f)) * 1.05, 0.05), 1.05), 1.0, float(np.median(B))]
    p, _ = curve_fit(
        hill, B, f, p0=p0,
        bounds=([0.05, 0.1, 1e5], [1.05, G_MAX, MAX_EXTRAP]), maxfev=400000,
    )
    return p, float(np.sqrt(np.mean((hill(B, *p) - f) ** 2)))


def budget_for(p, target):
    fmax, g, K = p
    if target >= fmax:
        return None
    return K * (target / (fmax - target)) ** (1.0 / g)


def crossover(pd, ps, lo, hi):
    """Smallest budget in [lo, hi] where sparse >= dense, or None."""
    grid = np.logspace(np.log10(lo), np.log10(hi), 4000)
    d, s = hill(grid, *pd), hill(grid, *ps)
    hits = np.nonzero(s >= d)[0]
    return float(grid[hits[0]]) if len(hits) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points", help="JSON of measured points")
    ap.add_argument("--max-budget", type=float, default=1e10,
                    help="upper end of the crossover search (default 10B tokens)")
    a = ap.parse_args()
    data = json.load(open(a.points))

    for task, variants in sorted(data.items()):
        print(f"\n=== {task}")
        rungs = sorted({r for v in variants.values() for r in v},
                       key=lambda r: int(r.rstrip("k")) * (1024 if r.endswith("k") else 1))
        for rung in rungs:
            fits = {}
            for variant in ("dense", "sparse"):
                pts = variants.get(variant, {}).get(rung)
                if not pts or len(pts) < 3:
                    continue
                budgets = sorted(float(b) for b in pts)
                res = fit(budgets, [pts[f"{int(b)}"] for b in budgets])
                if res is None:                       # flat-zero ladder, nothing to fit
                    print(f"  {rung:>4s}  {variant[0]}[flat at the floor -- no fit]")
                    continue
                fits[variant] = (res, max(budgets), min(budgets))
            if not fits:
                continue
            line = [f"  {rung:>4s}"]
            unreliable = set()
            for variant, ((p, rmse), bmax, _bmin) in fits.items():
                bits = []
                for t in TARGETS:
                    b = budget_for(p, t)
                    if b is None:
                        bits.append(f"{t}:above ceiling {p[0]:.2f}")
                    else:
                        bits.append(f"{t}:{'>=' if b > bmax else ''}{b / 1e6:.0f}M")
                # A Hill exponent pinned at the g bound with a large residual means the ladder is
                # a STEP, not a smooth curve: sparse-landmark arms sit near zero until a takeoff
                # budget and then jump (nq 2k .137 -> .912 between 32M and 48M). The fitter answers
                # with the steepest curve it is allowed, so both the law and any crossover read off
                # it are artifacts of the wrong functional form. Flag rather than quote.
                bad = p[1] >= 0.99 * G_MAX or rmse > 0.05
                line.append(f"{variant[:1]}[fmax {p[0]:.2f} g {p[1]:.2f} K {p[2] / 1e6:.0f}M "
                            f"rmse {rmse:.3f}]{' !STEP-NOT-HILL' if bad else ''} " + " ".join(bits))
                if bad:
                    unreliable.add(variant)
            if len(fits) == 2 and unreliable:
                line.append(f"CROSSOVER suppressed -- {'/'.join(sorted(unreliable))} is a step, "
                            f"not a Hill curve")
            elif len(fits) == 2:
                (pd, _), bd, bdlo = fits["dense"]
                (ps, _), bs, bslo = fits["sparse"]
                # Search from the smallest MEASURED budget, not below it: two 3-point fits diverge
                # freely outside their data, and starting the search an order of magnitude low
                # manufactured a "crossover" at 8M on a rung where sparse trails at every measured
                # point.
                lo = max(bdlo, bslo)
                x = crossover(pd, ps, lo, a.max_budget)
                bmax = max(bd, bs)
                if x is None:
                    line.append(f"CROSSOVER none below {a.max_budget / 1e9:.0f}B")
                elif x <= bmax:
                    line.append(f"CROSSOVER {x / 1e6:.0f}M (inside measured range)")
                else:
                    # A crossover past the last measured point is usually an artifact of the
                    # loser's fmax being unconstrained: a curve still in its rising phase gets a
                    # high fitted ceiling, while the leader's bends and pins a low one. Say so
                    # rather than printing a number that reads like a prediction.
                    line.append(f"CROSSOVER {x / 1e6:.0f}M = {x / bmax:.1f}x PAST the largest "
                                f"measured budget ({bmax / 1e6:.0f}M) -- EXTRAPOLATION, and the "
                                f"fitted ceilings driving it are dense {pd[0]:.2f} / sparse "
                                f"{ps[0]:.2f}")
            print("\n        ".join(line))


if __name__ == "__main__":
    main()
