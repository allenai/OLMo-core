"""Budget for f1=0.9 vs target context: dense pure / dense short-heavy / sparse short-heavy."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/outlier_lengthmix"
L_dp = np.array([8192,16384,32768]); B_dp = np.array([71e6,475e6,2.29e9])       # dense pure
L_ds = np.array([8192,16384,32768]); B_ds = np.array([310e6,1.04e9,4.8e9])      # dense short-heavy
# sparse short-heavy: fits on measured 3k/8k rung curves (16/32/64M): B(.9)@3k≈118M, @8k≈0.5-1.4B
L_ss = np.array([3072,8192]); B_ss_lo = np.array([118e6,455e6]); B_ss_hi = np.array([118e6,1.37e9])

def powfit(L,B):
    b,a = np.polyfit(np.log(L),np.log(B),1)
    return (lambda x: np.exp(a)*x**b), b
f_dp,b_dp = powfit(L_dp,B_dp)
f_ds,b_ds = powfit(L_ds,B_ds)
L = np.logspace(np.log10(3072), np.log10(1_048_576), 200)
# sparse extrapolation: dense-sh curve x measured sparse/dense ratio band (2.1x @3k, 1.5-4.4x @8k)
sp_lo, sp_hi = f_ds(L)*1.5, f_ds(L)*4.5
Lticks=[3072,8192,32768,131072,262144,1048576]; Llabs=["3k","8k","32k","128k","256k","1M"]
a0,b0 = 9.28e-5,7.9e-10; a_s = 8.4e-5

fig,axes = plt.subplots(1,2,figsize=(13.5,5.4))
ax=axes[0]
ax.plot(L,f_dp(L),"-",color="tab:blue",label=f"dense pure-length (∝L^{b_dp:.1f})")
ax.plot(L_dp,B_dp,"o",color="tab:blue",ms=8)
ax.plot(L,f_ds(L),"-",color="tab:orange",label=f"dense short-heavy (∝L^{b_ds:.1f})")
ax.plot(L_ds,B_ds,"s",color="tab:orange",ms=8)
ax.fill_between(L,sp_lo,sp_hi,color="tab:red",alpha=.18,label="sparse short-heavy (measured ratio band 1.5–4.5×)")
ax.plot(L_ss,B_ss_lo,"^",color="tab:red",ms=9)
ax.plot([L_ss[1]],[B_ss_hi[1]],"^",color="tab:red",ms=9,alpha=.5)
ax.axvspan(32768,1.1e6,color="gray",alpha=.08); ax.annotate("extrapolated →",xy=(40000,3e13),fontsize=9,color="gray")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(Lticks); ax.set_xticklabels(Llabs)
ax.set_xlabel("target context (eval rung)"); ax.set_ylabel("training tokens for f1=0.9")
ax.set_title("(a) Token budget for 0.9"); ax.legend(fontsize=8.5,loc="upper left"); ax.grid(alpha=.3,which="both")

ax=axes[1]
gh_dp = f_dp(L)*(a0+b0*L)/3600
gh_ds = f_ds(L)*1.1e-4/3600            # short-heavy mean token ~cheap for dense
gh_ss_lo = sp_lo*0.95e-4/3600; gh_ss_hi = sp_hi*0.95e-4/3600
ax.plot(L,gh_dp,"-",color="tab:blue",label="dense pure @ target length")
ax.plot(L,gh_ds,"-",color="tab:orange",label="dense short-heavy")
ax.fill_between(L,gh_ss_lo,gh_ss_hi,color="tab:red",alpha=.18,label="sparse short-heavy")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(Lticks); ax.set_xticklabels(Llabs)
ax.axvspan(32768,1.1e6,color="gray",alpha=.08)
ax.set_xlabel("target context (eval rung)"); ax.set_ylabel("H200 GPU-hours for f1=0.9")
ax.set_title("(b) Same in compute — sparse's cheap-long-token edge is neutralized by short-heavy mixes")
ax.legend(fontsize=8.5,loc="upper left"); ax.grid(alpha=.3,which="both")
fig.suptitle("Cost of f1=0.9 vs target context (outlier, Qwen3.5-4B) — measured ≤32k, shaded = extrapolation",fontsize=12)
fig.tight_layout(rect=[0,0,1,.94])
fig.savefig(f"{OUT}/budget90_dense_vs_sparse.png",dpi=140)
for l in [131072,1048576]:
    print(f"L={l//1024}k: dense-pure {f_dp(l)/1e9:.0f}B tok/{f_dp(l)*(a0+b0*l)/3600:,.0f}GPUh | dense-sh {f_ds(l)/1e9:.0f}B/{f_ds(l)*1.1e-4/3600:,.0f}GPUh | sparse-sh {sp_lo[np.argmin(abs(L-l))]/1e9:.0f}-{sp_hi[np.argmin(abs(L-l))]/1e9:.0f}B")
