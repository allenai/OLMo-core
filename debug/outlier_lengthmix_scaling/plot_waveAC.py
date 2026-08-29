"""Wave A: complete short-heavy budget laws (5 points, seed-replicated).
Wave C: 3-task generality — dense data-need + sparse taxonomy."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"

B = np.array([16,32,64,96,160])*1e6
d8 = np.array([.505,.601,.725,.779,.887])   # dense short-heavy @8k rung
s8 = np.array([.233,.278,.467,.645,.754])   # sparse short-heavy @8k rung
dm = np.array([.383,.474,.558,.610,.688])   # dense mean over rungs
sm = np.array([.224,.277,.385,.513,.588])   # sparse mean
hill = lambda b,fmax,K,g: fmax*b**g/(b**g+K**g)
pd_,_ = curve_fit(hill,B,d8,p0=[1.0,2e7,0.6],bounds=([.5,1e6,.1],[1.05,1e10,4]),maxfev=30000)
ps_,_ = curve_fit(hill,B,s8,p0=[1.0,8e7,1.0],bounds=([.5,1e6,.1],[1.05,1e10,4]),maxfev=30000)
b90d = pd_[1]*(0.9/(pd_[0]-0.9))**(1/pd_[2]) if pd_[0]>0.9 else np.nan
b90s = ps_[1]*(0.9/(ps_[0]-0.9))**(1/ps_[2]) if ps_[0]>0.9 else np.nan

fig,axes = plt.subplots(1,2,figsize=(14,5.4))
ax=axes[0]
bb=np.logspace(np.log10(1e7),np.log10(1e9),200)
ax.plot(B,d8,"o",color="tab:blue",ms=9); ax.plot(bb,np.clip(hill(bb,*pd_),0,1),"-",color="tab:blue",label=f"dense @8k rung — B(0.9)≈{b90d/1e6:.0f}M tok")
ax.plot(B,s8,"s",color="tab:red",ms=9); ax.plot(bb,np.clip(hill(bb,*ps_),0,1),"-",color="tab:red",label=f"sparse @8k rung — B(0.9)≈{b90s/1e6:.0f}M tok")
ax.plot(B,dm,"o--",color="tab:blue",alpha=.4,label="dense mean-over-rungs")
ax.plot(B,sm,"s--",color="tab:red",alpha=.4,label="sparse mean-over-rungs")
ax.errorbar([64e6],[ .467],yerr=[.03],fmt="none",ecolor="black",capsize=4)
ax.annotate("seed replicate: ±0.03",xy=(6.7e7,.40),fontsize=8)
ax.axhline(.9,color="gray",lw=.6,ls=":")
ax.set_xscale("log"); ax.set_ylim(0,1.02)
ax.set_xlabel("total training tokens (short-heavy shape)"); ax.set_ylabel("f1")
ax.set_title("(A) SHORT-HEAVY scaling complete: sparse/dense gap shrinks\nfrom 1.7× to 1.2× (mean) as budget grows")
ax.legend(fontsize=8.5,loc="lower right"); ax.grid(alpha=.3,which="both")

ax=axes[1]
# 3-task dense @8k-rung curves (in-length training)
No = np.array([250,1000,2000,4000,8000,16000,32000])*8300
fo = [.422,.629,.720,.819,.915,.952,.980]
Nq = np.array([1000,2000,4000,8000])*7515
fq = [.920,.947,.948,.964]
Nn = np.array([1000,2000,4000,8000])*7515
fn = [.908,.928,.927,.950]
ax.plot(No,[1-x for x in fo],"o-",color="tab:blue",label="outlier (dense)")
ax.plot(Nq,[1-x for x in fq],"s-",color="tab:green",label="qdmatch_nq (dense)")
ax.plot(Nn,[1-x for x in fn],"^-",color="tab:purple",label="nq (dense)")
ax.annotate("SPARSE taxonomy @tested scales:\n outlier: late takeoff ✓ (needs ~4× data)\n nq: works @2k (.93), pre-takeoff @8k\n qdmatch: never (architectural)",
            xy=(2.5e6,.25),fontsize=8.5,
            bbox=dict(boxstyle="round",fc="mistyrose",alpha=.7))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training tokens (pure 8k-length)"); ax.set_ylabel("error @8k rung (1−f1)")
ax.set_title("(C) Task generality: same law shape, ~5-10× spread in K\n(outlier is the hard task; nq/qdmatch saturate by ~1k examples)")
ax.legend(fontsize=8.5); ax.grid(alpha=.3,which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/waveAC_shortheavy_and_3task.png",dpi=140)
print(f"dense sh: fmax={pd_[0]:.2f} K={pd_[1]/1e6:.0f}M g={pd_[2]:.2f} B90={b90d/1e6:.0f}M")
print(f"sparse sh: fmax={ps_[0]:.2f} K={ps_[1]/1e6:.0f}M g={ps_[2]:.2f} B90={b90s/1e6:.0f}M")
