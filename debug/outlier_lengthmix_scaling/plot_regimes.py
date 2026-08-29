"""One plot per data regime: error rate (1-f1) vs training tokens.
Color = eval context length; solid = dense, dashed = sparse."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"
RC = {"3k":"tab:blue","8k":"tab:orange","16k":"tab:green","32k":"tab:red"}
RUNGS = ["3k","8k","16k","32k"]

# ---------- pure-length: train length == eval rung ----------
pure = {  # length: (tokens_list, dense_f1_at_own_rung, sparse_f1_at_own_rung)
 "3k":  (np.array([1250,2500,5000,10000,20000])*2080,
         [.670,.769,.703,.658,.791], [.138,.200,.529,.529,.498]),
 "8k":  (np.array([250,1000,2000,4000,8000,16000,32000])*8300,
         [.422,.629,.720,.819,.915,.952,.980], [.051,.056,.051,.052,.224,.538,.923]),
 "16k": (np.array([250,1000,4000,8000,16000])*15600,
         [.170,.370,.596,None,None], [None,None,.029,.079,.157]),
 "32k": (np.array([2000,8000])*30500, [.209,.472], [.015,None]),
}
# ---------- mixes: budgets (tokens) x f1 per rung ----------
mix = {
 "uniform": dict(
   B=np.array([16,32,64,128])*1e6,
   dense={"3k":[.658,.769,.891,.954],"8k":[.509,.521,.673,.808],"16k":[.284,.307,.453,.590],"32k":[.072,.107,.217,.270]},
   sparse={"3k":[.178,.281,.752,.865],"8k":[.067,.073,.396,.671],"16k":[.023,.030,.169,.394],"32k":[.010,.011,.038,.174]}),
 "short-heavy": dict(
   B=np.array([16,32,64])*1e6,
   dense={"3k":[.748,None,.925],"8k":[.505,None,.725],"16k":[.243,None,.444],"32k":[.035,None,.136]},
   sparse={"3k":[.561,.676,.832],"8k":[.233,.278,.467],"16k":[.094,.119,.201],"32k":[.009,.036,.038]}),
 "long-heavy": dict(
   B=np.array([16,32,64,128])*1e6,
   dense={"3k":[None,.637,.765,.905],"8k":[None,.486,.577,.788],"16k":[None,.361,.417,.603],"32k":[None,.132,.197,.326]},
   sparse={"3k":[.143,.144,.626,.800],"8k":[.060,.053,.238,.542],"16k":[.029,.025,.099,.355],"32k":[.006,.010,.023,.186]}),
 "two-point 2k+32k": dict(
   B=np.array([16,32,64])*1e6,
   dense={"3k":[.597,.697,.779],"8k":[.156,.243,.182],"16k":[.068,.174,.158],"32k":[.024,.167,.152]},
   sparse={"3k":[.190,None,None],"8k":[.054,None,None],"16k":[.016,None,None],"32k":[.004,None,None]}),
}

def plot_curves(ax, x, ys, color, style, label):
    xv = [a for a,b in zip(x,ys) if b is not None]
    yv = [1-b for b in ys if b is not None]
    if xv:
        ax.plot(xv, yv, style, color=color, label=label, ms=6)

def finish(ax, title):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.015, 1.05); ax.axhline(0.1, color="gray", lw=.6, ls=":")
    ax.annotate("f1=0.9", xy=(ax.get_xlim()[0]*1.1, .105), fontsize=8, color="gray")
    ax.set_xlabel("training tokens"); ax.set_ylabel("error rate (1 − f1)")
    ax.set_title(title); ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="lower left")

figs = []
# pure-length plot
fig, ax = plt.subplots(figsize=(8.5,5.6))
for r in RUNGS:
    T, dn, sp = pure[r]
    plot_curves(ax, T, dn, RC[r], "o-", f"dense @{r}")
    plot_curves(ax, T, sp, RC[r], "s--", f"sparse @{r}")
fig_title = "PURE-LENGTH regime: trained at L, scored at L (color = context length)"
finish(ax, fig_title); fig.tight_layout()
fig.savefig(f"{OUT}/regime_pure.png", dpi=140); figs.append("regime_pure.png")

for name, d in mix.items():
    fig, ax = plt.subplots(figsize=(8.5,5.6))
    for r in RUNGS:
        plot_curves(ax, d["B"], d["dense"][r], RC[r], "o-", f"dense @{r}")
        plot_curves(ax, d["B"], d["sparse"][r], RC[r], "s--", f"sparse @{r}")
    finish(ax, f"{name.upper()} mix regime: error per eval rung vs total budget")
    fn = f"regime_{name.split()[0].replace('-','')}.png"
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}", dpi=140); figs.append(fn)
print("wrote:", figs)
