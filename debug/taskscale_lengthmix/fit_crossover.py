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


def hill(B, fmax, g, K):
    return fmax * B**g / (B**g + K**g)


def fit(budgets, scores):
    B = np.asarray(budgets, float)
    f = np.asarray(scores, float)
    p, _ = curve_fit(
        hill, B, f, p0=[max(f) * 1.05, 1.0, float(np.median(B))],
        bounds=([0.05, 0.1, 1e5], [1.05, 4.0, MAX_EXTRAP]), maxfev=400000,
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
                fits[variant] = (fit(budgets, [pts[f"{int(b)}"] for b in budgets]), max(budgets))
            if not fits:
                continue
            line = [f"  {rung:>4s}"]
            for variant, ((p, rmse), bmax) in fits.items():
                bits = []
                for t in TARGETS:
                    b = budget_for(p, t)
                    if b is None:
                        bits.append(f"{t}:above ceiling {p[0]:.2f}")
                    else:
                        bits.append(f"{t}:{'>=' if b > bmax else ''}{b / 1e6:.0f}M")
                line.append(f"{variant[:1]}[fmax {p[0]:.2f} g {p[1]:.2f} K {p[2] / 1e6:.0f}M "
                            f"rmse {rmse:.3f}] " + " ".join(bits))
            if len(fits) == 2:
                (pd, _), bd = fits["dense"]
                (ps, _), bs = fits["sparse"]
                x = crossover(pd, ps, min(bd, bs) / 10, a.max_budget)
                line.append("CROSSOVER " + (f"{x / 1e6:.0f}M" if x else
                                            f"none below {a.max_budget / 1e9:.0f}B"))
            print("\n        ".join(line))


if __name__ == "__main__":
    main()
