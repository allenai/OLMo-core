"""
Fit FLOP- and data-scaling trends per (task, arm) from results/flop_scaling/results.csv and
write results/flop_scaling/fits.csv + fits.md (+ overlay plots).

Model: saturating power law  f1(x) = A - B * x^(-alpha)   (x = training PFLOPs or tokens/1e6),
fit by least squares with A in [max f1, 1], B > 0, alpha > 0. With 4-5 points per curve this
is a 3-parameter fit, so the table also reports a 2-parameter log-linear slope on the
pre-saturation points (f1 vs log x) and the compute needed to reach a target f1 per arm,
i.e. the FLOP multiplier of each method vs dense at equal accuracy -- the compute-optimality
number the study is for.

    python debug/flop_scaling/fit_scaling.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

import numpy as np

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
OUT = f"{REPO}/results/flop_scaling"
TARGETS = {"contradiction": 0.85, "outlier": 0.7, "nq": 0.8, "oolong": 0.5}


def satpow(x, A, B, a):
    return A - B * np.power(x, -a)


def fit_curve(xs, ys):
    """Return (A, B, alpha, rmse) or None."""
    from scipy.optimize import curve_fit

    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        return None
    try:
        p0 = [min(1.0, max(ys) + 0.05), max(0.01, max(ys) - min(ys)), 0.5]
        popt, _ = curve_fit(satpow, xs, ys, p0=p0, bounds=([max(ys), 1e-6, 0.01], [1.0, 50.0, 5.0]), maxfev=20000)
        rmse = float(np.sqrt(np.mean((satpow(xs, *popt) - ys) ** 2)))
        return (*popt, rmse)
    except Exception:
        return None


def hill(x, fmax, g, K):
    """The prior campaigns' law (debug/taskscale_lengthmix/fit_crossover.py): f = fmax x^g/(x^g+K^g)."""
    return fmax * np.power(x, g) / (np.power(x, g) + np.power(K, g))


def fit_hill(xs, ys):
    """-> (fmax, g, K, rmse) or None; same bounds style as fit_crossover.py (x in PFLOPs or Mtokens)."""
    from scipy.optimize import curve_fit

    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        return None
    try:
        p0 = [min(1.0, max(ys) + 0.05), 1.0, float(np.median(xs))]
        popt, _ = curve_fit(hill, xs, ys, p0=p0, bounds=([0.05, 0.1, min(xs) / 100], [1.05, 4.0, max(xs) * 100]), maxfev=400000)
        rmse = float(np.sqrt(np.mean((hill(xs, *popt) - ys) ** 2)))
        return (*popt, rmse)
    except Exception:
        return None


def hill_x_for_target(fit, target):
    if fit is None:
        return None
    fmax, g, K, _ = fit
    if target >= fmax:
        return None
    return float(K * (target / (fmax - target)) ** (1.0 / g))


def loglin(xs, ys):
    xs, ys = np.log(np.asarray(xs, float)), np.asarray(ys, float)
    if len(xs) < 2:
        return None
    m, c = np.polyfit(xs, ys, 1)
    return float(m), float(c)


def x_for_target(fit, target):
    """Compute at which the fitted curve reaches `target` (None if never)."""
    if fit is None:
        return None
    A, B, a, _ = fit
    if target >= A:
        return None
    return float((B / (A - target)) ** (1.0 / a))


def main() -> None:
    rows = list(csv.DictReader(open(f"{OUT}/results.csv")))
    curves = defaultdict(list)
    for r in rows:
        if r["mean_f1"] in ("", "None") or r["actual_pflops"] in ("", "None"):
            continue
        curves[(r["task"], r["arm"])].append((float(r["actual_pflops"]), float(r["tokens"]) / 1e6, float(r["mean_f1"])))
    fits = []
    for (task, arm), pts in sorted(curves.items()):
        pts.sort()
        F = [p[0] for p in pts]; T = [p[1] for p in pts]; Y = [p[2] for p in pts]
        ff = fit_curve(F, Y); ft = fit_curve(T, Y)
        lf = loglin(F, Y); lt = loglin(T, Y)
        hf = fit_hill(F, Y); ht = fit_hill(T, Y)
        fits.append({
            "task": task, "arm": arm, "n_points": len(pts),
            "hill_flop_fmax": hf[0] if hf else None, "hill_flop_g": hf[1] if hf else None,
            "hill_flop_K": hf[2] if hf else None, "hill_flop_rmse": hf[3] if hf else None,
            "hill_pflops_to_target": hill_x_for_target(hf, TARGETS.get(task, 0.7)),
            "hill_data_fmax": ht[0] if ht else None, "hill_data_g": ht[1] if ht else None,
            "hill_data_K_mtok": ht[2] if ht else None,
            "hill_mtokens_to_target": hill_x_for_target(ht, TARGETS.get(task, 0.7)),
            "flop_A": ff[0] if ff else None, "flop_alpha": ff[2] if ff else None, "flop_rmse": ff[3] if ff else None,
            "flop_loglin_slope": lf[0] if lf else None,
            "pflops_to_target": x_for_target(ff, TARGETS.get(task, 0.7)),
            "data_A": ft[0] if ft else None, "data_alpha": ft[2] if ft else None, "data_rmse": ft[3] if ft else None,
            "data_loglin_slope": lt[0] if lt else None,
            "mtokens_to_target": x_for_target(ft, TARGETS.get(task, 0.7)),
            "best_f1": max(Y), "best_pflops": F[Y.index(max(Y))],
        })
    with open(f"{OUT}/fits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fits[0].keys())); w.writeheader(); w.writerows(fits)
    # markdown: per task, the FLOP multiplier of each arm vs dense at the target f1
    md = ["# FLOP-scaling fits\n", f"Primary law: Hill f1 = fmax x^g/(x^g+K^g) (the prior dense campaigns' form, debug/taskscale_lengthmix); secondary: saturating power law f1 = A - B x^-alpha. x = actual training PFLOPs. Target f1 per task: {TARGETS}. With 4-5 points per curve a 3-parameter fit interpolates; treat targets beyond the largest measured budget as extrapolations.\n"]
    by_task = defaultdict(dict)
    for ft in fits:
        by_task[ft["task"]][ft["arm"]] = ft
    for task, arms in by_task.items():
        tgt = TARGETS.get(task, 0.7)
        md.append(f"\n## {task}\n\n| arm | points | best f1 (PF) | Hill fmax | g | K (PF) | rmse | PF to f1={tgt} (Hill) | x dense | satpow A / alpha | PF to target (satpow) |\n|---|---|---|---|---|---|---|---|---|---|---|")
        dense = arms.get("dense", {}).get("hill_pflops_to_target")
        for arm, ft in sorted(arms.items()):
            pt = ft["hill_pflops_to_target"]
            mult = (pt / dense) if (pt and dense) else None
            fmt = lambda v, d=3: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
            md.append(f"| {arm} | {ft['n_points']} | {fmt(ft['best_f1'])} ({fmt(ft['best_pflops'],1)}) | {fmt(ft['hill_flop_fmax'])} | {fmt(ft['hill_flop_g'],2)} | {fmt(ft['hill_flop_K'],1)} | {fmt(ft['hill_flop_rmse'])} | {fmt(pt,1)} | {fmt(mult,2)} | {fmt(ft['flop_A'])} / {fmt(ft['flop_alpha'],2)} | {fmt(ft['pflops_to_target'],1)} |")
    open(f"{OUT}/fits.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for task, arms in by_task.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for arm, ft in sorted(arms.items()):
            pts = sorted(curves[(task, arm)])
            F = [p[0] for p in pts]; Y = [p[2] for p in pts]
            line, = ax.plot(F, Y, "o", label=arm)
            hf = fit_hill(F, Y)
            if hf is not None:
                xs = np.logspace(math.log10(min(F)) - 0.2, math.log10(max(F)) + 0.5, 60)
                ax.plot(xs, hill(xs, *hf[:3]), "-", color=line.get_color(), alpha=0.8)
        ax.set_xscale("log"); ax.set_xlabel("training PFLOPs (actual, method-aware)"); ax.set_ylabel("mean f1 over rungs")
        ax.set_title(f"{task}: FLOP scaling, Hill-law fits"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(f"{OUT}/{task}_flop_fit.png", dpi=130)
        print("plot:", f"{OUT}/{task}_flop_fit.png")


if __name__ == "__main__":
    main()
