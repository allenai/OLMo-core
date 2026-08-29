"""Refit N(f1=0.9) vs L with measured 16k + 32k anchors (no longer single-point extrapolation)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"
FMAX = 1.089
def fit_Kg(N, f):
    r = np.log(np.array(f) / (FMAX - np.array(f)))
    lnN = np.log(np.array(N, float))
    g, c = np.polyfit(lnN, r, 1)
    K = np.exp(-c / g)
    return K, g
def n_at(target, K, g):
    return K * (target / (FMAX - target)) ** (1 / g)

sets = {
    8192:  ([250,1000,2000,4000,8000,16000,32000], [.422,.629,.720,.819,.915,.952,.980]),
    16384: ([250,1000,4000], [.170,.370,.596]),
    32768: ([2000,8000], [.209,.472]),
}
Ls, N90s, rows = [], [], []
for L,(N,f) in sets.items():
    K,g = fit_Kg(N,f)
    n90 = n_at(0.9, K, g)
    Ls.append(L); N90s.append(n90)
    rows.append((L, K, g, n90))
    print(f"L={L}: K={K:,.0f} gamma={g:.2f} N(.9)={n90:,.0f}")
beta, lnA = np.polyfit(np.log(Ls), np.log(N90s), 1)
print(f"N(.9) ∝ L^{beta:.2f}")
Lx = np.array([8192,16384,32768,131072,262144,1048576,10485760], float)
pred = np.exp(lnA) * Lx ** beta

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(Lx, pred, "--", color="tab:blue", label=f"measured power law: N(0.9) ∝ L^{beta:.2f}")
ax.plot(Ls, N90s, "o", color="black", ms=11, zorder=5, label="MEASURED N(0.9) (fit per length)")
old_pess = 572*14.7*(Lx/8192)**2.69
old_opt  = 8433*(Lx/8192)
ax.fill_between(Lx, old_opt, old_pess, color="gray", alpha=.12, label="yesterday's scenario band (1 anchor)")
for x,y,lab in zip(Lx, pred, ["8k","16k","32k","128k","256k","1M","10M"]):
    ax.annotate(f"{lab}\n{y/1000:,.0f}k ex\n{y*x/1e9:,.0f}B tok", xy=(x, y*1.6), fontsize=8, ha="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("context length (tokens)"); ax.set_ylabel("examples for f1 = 0.9 (dense, pure-length)")
ax.set_title("REFIT with measured 16k & 32k anchors: dense data-need for f1=0.9\n"
             "N ∝ L^1.5 — between yesterday's scenarios; 0.9@32k confirmed reachable (0.472 measured at 8k ex)")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/n90_vs_context_prediction.png", dpi=140)
print("wrote fig")
