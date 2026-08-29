"""qdmatch_nq scaling laws: dense 2k/8k ladders + sparse probes, error-rate view."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"

N2 = np.array([1250,2500,5000,10000,20000]); T2 = N2*1925
d2_3k = [.993,.991,.994,.996,.999]
d2_8k = [.570,.617,.286,.023,.007]
N8 = np.array([1000,2000,4000,8000]); T8 = N8*7515
d8_3k = [.958,.891,.973,.961]
d8_8k = [.920,.947,.948,.964]

fig, axes = plt.subplots(1, 2, figsize=(13,5.2))
ax = axes[0]
ax.plot(T2, [1-x for x in d2_3k], "o-", color="tab:blue", label="2k-trained @3k rung")
ax.plot(T2, [1-x for x in d2_8k], "o--", color="tab:cyan", label="2k-trained @8k rung (OOD length)")
ax.plot(T8, [1-x for x in d8_8k], "s-", color="tab:orange", label="8k-trained @8k rung")
ax.plot(T8, [1-x for x in d8_3k], "s--", color="tab:red", label="8k-trained @3k rung (OOD length)")
ax.plot([20000*1925, 8000*7515], [1-.031, 1-.002], "^", color="black", ms=10, label="SPARSE probes (20k/8k ex): floor")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training tokens"); ax.set_ylabel("error rate (1 − f1)")
ax.set_title("qdmatch_nq: dense saturates by ~1000 ex; OOD-length transfer\nis NON-MONOTONIC — more short data destroys long generalization")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8.5)

ax = axes[1]
tasks = ["outlier\n(8k rung)", "qdmatch_nq\n(8k rung)", "nq\n(8k rung)"]
K_ex = [559, 300, None]
ax.bar([0], [559], .5, color="tab:blue")
ax.bar([1], [300], .5, color="tab:green")
ax.annotate("~559 ex\n(measured)", xy=(0, 600), ha="center", fontsize=9)
ax.annotate("<1000 ex — already .92 at 1000;\nK ≈ 200-400 (bracketed)", xy=(1, 350), ha="center", fontsize=9)
ax.annotate("training\nnow", xy=(2, 100), ha="center", fontsize=9, color="gray")
ax.set_xticks([0,1,2]); ax.set_xticklabels(tasks)
ax.set_ylabel("K (examples to half-max) at 8k length, dense")
ax.set_title("Task-generality: half-max data need at 8k\n(outlier is the HARD task of the three so far)")
fig.tight_layout()
fig.savefig(f"{OUT}/qdmatch_scaling_laws.png", dpi=140)
print("wrote")
