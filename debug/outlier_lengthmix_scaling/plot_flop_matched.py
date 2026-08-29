"""FLOP-matched sparse-vs-dense: (a) 8k ladder vs dense-equivalent compute, (b) crossover vs context."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"
hill = lambda n, fmax, K, g: fmax * n**g / (n**g + K**g)

# Measured Qwen3.5-4B train throughput (H200, AC on): dense 1/tok_s = a + b*L ; sparse flat.
# Fit through (16k,9457),(64k,6918); sparse ~11.9k tok/s.
b = (1/6918 - 1/9457) / (65536 - 16384)
a = 1/9457 - b * 16384
a_s = 1/11900
speedup = lambda L: (a + b * L) / a_s          # 16k→1.26 (meas 1.16), 32k→1.36 (1.37), 64k→1.75 (1.77)
SP8K = float(speedup(8192))                     # ~1.10

N8 = np.array([250,1000,2000,4000,8000,16000,32000], float)
d8 = np.array([.422,.629,.720,.819,.915,.952,.980])
s8 = np.array([.051,.056,.051,.052,.224,.538,.923])
pd_,_ = curve_fit(hill, N8, d8, p0=[1.0,1000,0.8], maxfev=20000)
ps_,_ = curve_fit(hill, N8, s8, p0=[1.0,16000,3.0], maxfev=20000)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

# (a) f1 vs dense-equivalent compute at 8k
ax = axes[0]
nn = np.logspace(np.log10(200), np.log10(60000), 300)
ax.plot(N8, d8, "o", color="tab:blue", ms=8)
ax.plot(nn, hill(nn,*pd_), "-", color="tab:blue", label="full attention")
ax.plot(N8/SP8K, s8, "s", color="tab:red", ms=8)
ax.plot(nn/SP8K, np.clip(hill(nn,*ps_),0,1), "-", color="tab:red",
        label=f"sparse landmark (cost/example ÷{SP8K:.2f} at 8k)")
ax.axhline(.9, color="gray", lw=.7, ls=":")
ax.annotate("at 8k the FLOP discount is only ~10%:\nsparse still needs ~3.3× dense compute for f1=0.9",
            xy=(320, .84), fontsize=9, color="dimgray")
ax.set_xscale("log"); ax.set_ylim(0,1.02)
ax.set_xlabel("dense-equivalent training compute (example-units @8k)")
ax.set_ylabel("f1 @ 8k rung")
ax.set_title("(a) FLOP-matched 8k ladder — discount barely moves sparse at short context")
ax.legend(fontsize=9, loc="lower right")

# (b) crossover: FLOP advantage vs data penalty across context length
ax = axes[1]
L = np.logspace(np.log10(8192), np.log10(10_485_760), 200)
ax.plot(L, speedup(L), "-", color="tab:green", label="sparse FLOP advantage (measured→extrapolated)")
meas_L = [16384, 32768, 65536]; meas_s = [1.16, 1.37, 1.77]
ax.plot(meas_L, meas_s, "ko", ms=7, label="measured speedups (H200)")
ax.axhline(3.6, color="tab:red", ls="--", lw=1.2,
           label="sparse data penalty at f1=0.9 (measured @8k: ~3.6×)")
ax.axhspan(3.0, 8.0, color="tab:red", alpha=.07)
Lx = (3.6 * a_s - a) / b
ax.axvline(Lx, color="gray", ls=":", lw=1)
ax.annotate(f"crossover ≈ {Lx/1024:.0f}k tokens\n(IF the data penalty stays ~flat in length —\nthe 16k anchors test exactly this)",
            xy=(Lx*1.15, 2.0), fontsize=9, color="dimgray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([8192,32768,131072,262144,1048576,10485760])
ax.set_xticklabels(["8k","32k","128k","256k","1M","10M"])
ax.set_xlabel("context length"); ax.set_ylabel("multiplier (×)")
ax.set_title("(b) Where sparse wins on FLOPs: advantage grows ∝ L,\npenalty measured flat-to-growing — crossover if penalty holds")
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/flop_matched_crossover.png", dpi=140)
print("speedup@8k=%.2f  crossover_L=%.0f tokens" % (SP8K, Lx))
