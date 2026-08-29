"""Dense: budget for f1=0.9 vs target context — pure-length vs short-heavy, tokens and GPU-hours."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"
# Measured B(0.9) anchors (tokens), from per-rung Hill fits
L_meas   = np.array([8192, 16384, 32768], float)
B_pure_m = np.array([71e6, 475e6, 2.29e9])        # pure-length: N90*L
L_sh_m   = np.array([3072, 8192, 16384, 32768], float)
B_sh_m   = np.array([55e6, 310e6, 1.04e9, 4.8e9]) # short-heavy per-rung fits (2-pt consistent)

def powfit(L, B):
    b, a = np.polyfit(np.log(L), np.log(B), 1)
    return lambda x: np.exp(a) * x**b, b
f_pure, b_pure = powfit(L_meas, B_pure_m)
f_sh, b_sh = powfit(L_sh_m[1:], B_sh_m[1:])   # fit on 8k+ rungs

L = np.logspace(np.log10(3072), np.log10(1_048_576), 200)
Lticks = [3072, 8192, 32768, 131072, 262144, 1048576]
Llabs = ["3k","8k","32k","128k","256k","1M"]

# GPU cost model (H200 s/token): dense at length ℓ: a0+b0*ℓ ; short-heavy mean ≈ flat ~1.0-1.2e-4
a0, b0 = 9.28e-5, 7.9e-10
cost_pure = a0 + b0 * L
cost_sh = 1.1e-4

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
ax.plot(L, f_pure(L), "-", color="tab:blue", label=f"pure-length (∝L^{b_pure:.1f})")
ax.plot(L_meas, B_pure_m, "o", color="tab:blue", ms=9)
ax.plot(L, f_sh(L), "-", color="tab:orange", label=f"short-heavy, tail extended (∝L^{b_sh:.1f})")
ax.plot(L_sh_m, B_sh_m, "s", color="tab:orange", ms=9)
ax.axvspan(32768, 1_100_000, color="gray", alpha=.08)
ax.annotate("extrapolated →", xy=(40000, 1e8), fontsize=9, color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(Lticks); ax.set_xticklabels(Llabs)
ax.set_xlabel("target context (eval rung)"); ax.set_ylabel("training tokens for f1 = 0.9")
ax.set_title("(a) DENSE token budget for 0.9 — short-heavy pays ~2–4× more tokens")
ax.legend(fontsize=9); ax.grid(alpha=.3, which="both")

ax = axes[1]
gh_pure = f_pure(L) * cost_pure / 3600
gh_sh   = f_sh(L) * cost_sh / 3600
ax.plot(L, gh_pure, "-", color="tab:blue", label="pure-length @ target context")
ax.plot(L, gh_sh, "-", color="tab:orange", label="short-heavy (cheap short tokens)")
ix = np.argmin(np.abs(np.log(gh_pure) - np.log(gh_sh))[L > 50000] )
Lx = L[L > 50000][ix]
ax.axvline(Lx, color="gray", ls=":")
ax.annotate(f"FLOP crossover ≈ {Lx/1024:.0f}k:\nbeyond here train mostly-short,\nbelow here train at target length",
            xy=(Lx*1.2, 30), fontsize=9, color="dimgray")
ax.axvspan(32768, 1_100_000, color="gray", alpha=.08)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(Lticks); ax.set_xticklabels(Llabs)
ax.set_xlabel("target context (eval rung)"); ax.set_ylabel("H200 GPU-hours for f1 = 0.9")
ax.set_title("(b) Same runs in COMPUTE — short-heavy wins past the crossover")
ax.legend(fontsize=9); ax.grid(alpha=.3, which="both")
fig.suptitle("Dense Qwen3.5-4B, outlier: cost of reaching f1=0.9 vs target context (measured ≤32k, extrapolated beyond)", fontsize=12)
fig.tight_layout(rect=[0,0,1,.94])
fig.savefig(f"{OUT}/dense_budget90_tokens_vs_flops.png", dpi=140)
print(f"pure slope {b_pure:.2f}, short-heavy slope {b_sh:.2f}, FLOP crossover ~{Lx:,.0f} tokens")
for l in [32768, 131072, 262144, 1048576]:
    print(f"L={l//1024}k: pure {f_pure(l)/1e9:.1f}B tok / {f_pure(l)*(a0+b0*l)/3600:,.0f} GPUh | sh {f_sh(l)/1e9:.1f}B tok / {f_sh(l)*cost_sh/3600:,.0f} GPUh")
