"""Overnight results plots: 2k/8k data-scaling ladders, Hill fits, mix-vs-pure, qdmatch smoke."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"

hill = lambda n, fmax, K, g: fmax * n**g / (n**g + K**g)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ---- (a) pure-8k ladder + Hill fits ----
ax = axes[0][0]
N8 = np.array([250, 1000, 2000, 4000, 8000, 16000, 32000], float)
d8 = np.array([.422, .629, .720, .819, .915, .952, .980])
s8 = np.array([.051, .056, .051, .052, .224, .538, .923])
pd_, _ = curve_fit(hill, N8, d8, p0=[1.0, 1000, 0.8], maxfev=20000)
ps_, _ = curve_fit(hill, N8, s8, p0=[1.0, 16000, 3.0], maxfev=20000)
nn = np.logspace(np.log10(200), np.log10(60000), 200)
ax.plot(N8, d8, "o", color="tab:blue", ms=8, label="full attention")
ax.plot(nn, hill(nn, *pd_), "-", color="tab:blue", alpha=.6,
        label=f"Hill fit (K={pd_[1]:.0f}, γ={pd_[2]:.2f})")
ax.plot(N8, s8, "s", color="tab:red", ms=8, label="sparse landmark")
ax.plot(nn, np.clip(hill(nn, *ps_), 0, 1), "-", color="tab:red", alpha=.6,
        label=f"Hill fit (K={ps_[1]/1000:.0f}k, γ={ps_[2]:.2f})")
ax.axhline(.8, color="gray", lw=.7, ls=":")
ax.annotate("7.6× data gap at f1=0.8", xy=(9000, .81), fontsize=9, color="gray")
ax.set_xscale("log"); ax.set_ylim(0, 1.02)
ax.set_xlabel("training examples (8k-length)"); ax.set_ylabel("f1 @ 8k rung")
ax.set_title("(a) Pure-8k data scaling — smooth vs sigmoid  [lr: full 5e-6, sparse 1e-5]")
ax.legend(fontsize=9, loc="upper left")

# ---- (b) 2k ladder ----
ax = axes[0][1]
N2 = np.array([1250, 2500, 5000, 10000, 20000], float)
d2 = np.array([.670, .769, .703, .658, .791])
s2 = np.array([.138, .200, .529, .529, .498])
ax.plot(N2, d2, "o-", color="tab:blue", label="full attention")
ax.plot(N2, s2, "s-", color="tab:red", label="sparse landmark")
ax.axhspan(.70, .79, color="tab:blue", alpha=.08)
ax.axhspan(.50, .53, color="tab:red", alpha=.08)
ax.annotate("full plateau ≈ .75", xy=(1300, .72), fontsize=9, color="tab:blue")
ax.annotate("sparse plateau ≈ .52\n(asymptote gap, not data gap)", xy=(6000, .40),
            fontsize=9, color="tab:red")
ax.set_xscale("log"); ax.set_ylim(0, 1.02)
ax.set_xlabel("training examples (2k-length)"); ax.set_ylabel("f1 @ 3k rung")
ax.set_title("(b) 2k data scaling — sparse saturates 0.25 lower  [lr 2e-5]")
ax.legend(fontsize=9)

# ---- (c) check#2: mix vs pure at 8k ----
ax = axes[1][0]
runs = ["p8k_4000\n(4k×8k)", "m8k_mix\n(4k×8k+4k×2k)", "p8k_5000\n(5k×8k, FLOP≈mix)", "p8k_8000\n(8k×8k)"]
dvals = [.744, .887, .878, .924]
svals = [.052, .627, .054, .600]
x = np.arange(4); w = 0.38
ax.bar(x - w/2, dvals, w, color="tab:blue", label="full attention")
ax.bar(x + w/2, svals, w, color="tab:red", label="sparse landmark")
for xi, v in zip(x - w/2, dvals): ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=9)
for xi, v in zip(x + w/2, svals): ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=8.5)
ax.set_ylim(0, 1.05); ax.set_ylabel("f1 @ 8k rung")
ax.set_title("(c) Mix vs pure at 8k [lr 2e-5] — mixing ≈ additive for full (ρ≈0.25),\nqualitatively unblocking for sparse (.05 → .63)")
ax.legend(fontsize=9)

# ---- (d) length-generalization: eval rung profiles ----
ax = axes[1][1]
rungs = [3, 8, 16, 32]
prof = {
    "full 8k-trained (p8k_8000-lropt)": ([.392, .915, .424, .102], "tab:blue", "o-"),
    "full 32k-trained (p32k_2000)":     ([.248, .218, .150, .209], "tab:cyan", "o--"),
    "sparse 8k-trained (p8k_32000-lropt)": ([.228, .923, None, None], "tab:red", "s-"),
    "sparse 32k-trained (p32k_2000)":   ([.006, .007, .008, .015], "tab:orange", "s--"),
}
for lab, (ys, c, st) in prof.items():
    xs = [r for r, y in zip(rungs, ys) if y is not None]
    vs = [y for y in ys if y is not None]
    ax.plot(xs, vs, st, color=c, label=lab)
ax.set_xscale("log"); ax.set_xticks(rungs); ax.set_xticklabels(["3k", "8k", "16k", "32k"])
ax.set_ylim(0, 1.02)
ax.set_xlabel("eval rung (context length)"); ax.set_ylabel("f1")
ax.set_title("(d) Eval-length profiles — 32k needs in-length data;\n2000 ex @32k is above 8k-trained ceiling for full, floor for sparse")
ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Outlier data-scaling: full vs sparse landmark (Qwen3.5-4B, 600-example evals)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/lengthmix_scaling_overview.png", dpi=140)
print("wrote", f"{OUT}/lengthmix_scaling_overview.png")

# ---- second figure: qdmatch smoke ----
fig2, ax = plt.subplots(figsize=(7, 4.5))
labels = ["q2k_5000\n@3k rung", "q2k_5000\n@8k rung", "q8k_4000\n@3k rung", "q8k_4000\n@8k rung"]
dv = [.994, .286, .973, .948]
sv = [.031, .000, .000, .001]
x = np.arange(4); w = .38
ax.bar(x - w/2, dv, w, color="tab:blue", label="full attention")
ax.bar(x + w/2, sv, w, color="tab:red", label="sparse landmark")
for xi, v in zip(x - w/2, dv): ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=9)
for xi, v in zip(x + w/2, sv): ax.text(xi, v + .015, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("f1")
ax.set_title("qdmatch_nq smoke (rebuilt p10 data): sparse fails architecturally\n(well-formed answers, wrong pairs; train-CE gap 15× vs 2× on outlier)")
ax.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(f"{OUT}/qdmatch_smoke.png", dpi=140)
print("wrote", f"{OUT}/qdmatch_smoke.png")
