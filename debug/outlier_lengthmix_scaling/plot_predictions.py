"""N(f1=0.9) vs context length, extrapolated to 10M tokens, with scenario band."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"
# Dense Hill at 8k: fmax=1.089, K=572, gamma=0.58  -> N(.9) = K * 14.7
RATIO = (0.9 / (1.089 - 0.9)) ** (1 / 0.58)          # ≈14.7
K8 = 572.0
N9_8K = K8 * RATIO                                    # ≈8.4k
# K(32k) solved from the single measured point f1(2000 ex @32k)=0.209:
K32 = 2000 * ((1.089 - 0.209) / 0.209 * (2000 ** -0.58) ** 0) ** 0  # placeholder, computed below
K32 = 2000 * (((1.089 / 0.209) - 1)) ** (1 / 0.58)
ALPHA = np.log(K32 / K8) / np.log(32768 / 8192)       # ≈2.6

L = np.array([8192, 32768, 131072, 262144, 1048576, 10485760], float)
Llab = ["8k", "32k", "128k", "256k", "1M", "10M"]
# Scenario A: power-law K(L) through the (noisy, single-point) 32k anchor
nA = (K8 * (L / 8192) ** ALPHA) * RATIO
# Scenario B: N(.9) grows linearly with L (constant learning per token)
nB = N9_8K * (L / 8192)

fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(L, nB, nA, color="tab:blue", alpha=.15, label="dense: plausible range (scenario B–A)")
ax.plot(L, nA, "v--", color="tab:blue", label=f"A: fit through 32k point (K∝L^{ALPHA:.1f}) — pessimistic")
ax.plot(L, nB, "^--", color="tab:blue", alpha=.7, label="B: N ∝ context length — optimistic")
ax.plot([8192], [N9_8K], "o", color="black", ms=11, zorder=5, label="measured: dense N(.9)@8k ≈ 8.5k ex")
ax.plot([32768], [2000], "x", color="gray", ms=11, mew=3, label="measured: 2000 ex @32k → f1 .21 (anchors A)")
# sparse: measured N(.9)~31k at 8k; threshold grows ≥ as fast as dense scenario A
ax.plot([8192], [30500], "s", color="tab:red", ms=10, label="sparse N(.9)@8k ≈ 31k ex (pure-length)")
ax.annotate("sparse pure-length: ≥ dense curve ×4–8;\nmixing (m8kmix) collapses it — curriculum is the lever",
            xy=(40000, 45000), fontsize=9, color="tab:red")
for x, la, ya, yb in zip(L, Llab, nA, nB):
    ax.annotate(la, xy=(x, yb * 0.45), fontsize=9, ha="center", color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("context length (tokens)"); ax.set_ylabel("training examples needed for f1 = 0.9")
ax.set_title("Predicted data need for f1=0.9 vs context length (dense Qwen3.5-4B, outlier)\n"
             "⚠ scenario A hinges on ONE 32k point; 16k + larger-32k anchor runs launching now")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/n90_vs_context_prediction.png", dpi=140)
print("K32=%.0f  alpha=%.2f" % (K32, ALPHA))
rows = [("ctx", "A: examples", "A: tokens", "B: examples", "B: tokens")]
for x, la, a, b in zip(L, Llab, nA, nB):
    rows.append((la, f"{a:,.0f}", f"{a*x/1e9:,.1f}B", f"{b:,.0f}", f"{b*x/1e9:,.1f}B"))
for r in rows: print("%-6s %-14s %-12s %-14s %-12s" % r)
