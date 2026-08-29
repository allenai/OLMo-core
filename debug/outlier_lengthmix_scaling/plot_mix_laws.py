"""Per-shape token-budget scaling laws, dense vs sparse, + rung profiles at 64M."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"
# f1 at [3k, 8k, 16k, 32k]; budgets in M tokens
dense = {
 "U": {16:[.658,.509,.284,.072], 32:[.769,.521,.307,.107], 64:[.891,.673,.453,.217], 128:[.954,.808,.590,.270]},
 "S": {16:[.748,.505,.243,.035], 64:[.925,.725,.444,.136]},
 "L": {32:[.637,.486,.361,.132], 64:[.765,.577,.417,.197], 128:[.905,.788,.603,.326]},
 "T": {16:[.597,.156,.068,.024], 32:[.697,.243,.174,.167], 64:[.779,.182,.158,.152]},
}
sparse = {
 "U": {16:[.178,.067,.023,.010], 32:[.281,.073,.030,.011], 64:[.752,.396,.169,.038], 128:[.865,.671,.394,.174]},
 "S": {16:[.561,.233,.094,.009], 32:[.676,.278,.119,.036], 64:[.832,.467,.201,.038]},
 "L": {16:[.143,.060,.029,.006], 32:[.144,.053,.025,.010], 64:[.626,.238,.099,.023], 128:[.800,.542,.355,.186]},
 "T": {16:[.190,.054,.016,.004], 32:[.480,.127,.038,.014], 64:[.563,.185,.144,.071]},
}
COLORS = {"U":"tab:blue","S":"tab:orange","L":"tab:green","T":"tab:purple"}
NAMES = {"U":"uniform","S":"short-heavy","L":"long-heavy","T":"two-point 2k+32k"}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
for ax, data, title in ((axes[0], dense, "DENSE: mean f1 over rungs vs budget"),
                        (axes[1], sparse, "SPARSE: mean f1 over rungs vs budget")):
    for k, pts in data.items():
        B = sorted(pts); m = [np.mean(pts[b]) for b in B]
        ax.plot(B, m, "o-", color=COLORS[k], label=NAMES[k])
    ax.set_xscale("log"); ax.set_xlabel("training budget (M tokens)")
    ax.set_ylabel("mean f1 (3k/8k/16k/32k rungs)"); ax.set_title(title)
    ax.set_ylim(0, .75); ax.grid(alpha=.3); ax.legend(fontsize=9)

ax = axes[2]
rungs = [3, 8, 16, 32]
for k in ("U","S","L","T"):
    if 64 in dense.get(k, {}):
        ax.plot(rungs, dense[k][64], "o-", color=COLORS[k], label=f"dense {NAMES[k]}")
for k in ("U","S"):
    if 64 in sparse.get(k, {}):
        ax.plot(rungs, sparse[k][64], "s--", color=COLORS[k], alpha=.6, label=f"sparse {NAMES[k]}")
ax.set_xscale("log"); ax.set_xticks(rungs); ax.set_xticklabels(["3k","8k","16k","32k"])
ax.set_xlabel("eval rung"); ax.set_ylabel("f1"); ax.set_title("Rung profiles at 64M tokens")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("Length-mix scaling laws: token-budget-matched shapes (Qwen3.5-4B, outlier)", fontsize=13)
fig.tight_layout(rect=[0,0,1,.95])
fig.savefig(f"{OUT}/mix_shape_laws.png", dpi=140)
print("wrote fig")
for arch, data in (("dense", dense), ("sparse", sparse)):
    for k, pts in data.items():
        for b in sorted(pts):
            print(f"{arch} {NAMES[k]:12} {b:4}M mean={np.mean(pts[b]):.3f} rungs={pts[b]}")
