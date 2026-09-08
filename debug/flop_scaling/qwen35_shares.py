"""
Where a Qwen3.5 token's training FLOPs and training memory go, as a function of context length,
at three model scales.

FLOPs: exact, from the model's own ``num_flops_per_token`` formulas (the same ones the FLOP meter
and the joint budget use), split into FFN / GDN mixer / attention projections / attention scores
(QK^T + PV, length-dependent) / embeddings + LM head.

Memory: an analytical per-GPU model for our training recipe -- FSDP2 over W GPUs, one packed row of
length T per GPU, bf16 compute with fp32 master weights + AdamW, FULL activation checkpointing
(one block's intermediates alive at a time), flash attention (scores never materialise), fused
linear cross-entropy (logits chunked). Components:
  * parameters + grads + Adam states: 16 B/param, sharded over W (length-independent);
  * retained block inputs: n_layers x d_model x 2 B per token (what full AC keeps);
  * block recompute transient: the larger of one attention block's or one GDN block's saved
    intermediates during its backward recompute, per token (fp32 norm copies included: the memory
    attribution of 2026-09-05 showed those dominate a block);
  * LM head / loss: final-norm fp32 copy per token + a fixed chunk of logits.
Cross-check: Qwen3-4B dense at 65k on 8 GPUs measured 26.5 GB live peak (mem-snapshot); this model
gives the same order for that configuration. Treat the memory panel as a first-order model.

    python debug/flop_scaling/qwen35_shares.py   -> visualizations/flop_scaling/qwen35_{flop,mem}_shares.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from olmo_core.nn.attention import Attention  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

VIZ = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/visualizations/flop_scaling"
SCALES = [("0.8B", TransformerConfig.qwen3_5_0_8B), ("4B", TransformerConfig.qwen3_5_4B), ("27B", TransformerConfig.qwen3_5_27B)]
VOCAB = 248320
LENGTHS = [2**k for k in range(10, 21)]  # 1k .. 1M
WORLD = 8
COLORS = {"FFN": "#0E6B66", "GDN mixer": "#7B2D8E", "attention projections": "#4C9BE0", "attention scores": "#1F3A93",
          "embeddings + LM head": "#B9552A"}
MCOLORS = {"params + grads + Adam (÷8 GPUs)": "#888888", "retained block inputs (full AC)": "#0E6B66",
           "block recompute transient": "#4C9BE0", "LM head / loss": "#B9552A"}


def flop_shares(model, L):
    dense = float(model.num_flops_per_token(L))
    ffn = gdn = proj = score = 0.0
    for b in model.blocks.values():
        ffn += b.feed_forward.num_flops_per_token(L)
        a = b.attention
        if type(a) is Attention:
            p = a.num_flops_per_token(0)
            proj += p
            score += a.num_flops_per_token(L) - p
        else:
            gdn += a.num_flops_per_token(L)
    fixed = dense - ffn - gdn - proj - score
    return {"FFN": ffn / dense, "GDN mixer": gdn / dense, "attention projections": proj / dense,
            "attention scores": score / dense, "embeddings + LM head": fixed / dense}, dense


def mem_model(model):
    """Bytes per token of the length-linear components, and fixed bytes per GPU."""
    n_params = sum(p.numel() for p in model.parameters())
    blocks = list(model.blocks.values())
    d = model.d_model
    F = blocks[0].feed_forward.w1.out_features
    ffn_t = (d * 4 + d * 2) + 2 * F * 2 + F * 2 + F * 2 + d * 2  # norm fp32+bf16, w1/w3 out, act, prod, w2 out
    attn_t = gdn_t = 0.0
    for b in blocks:
        a = b.attention
        if type(a) is Attention:
            qd, kd = a.w_q.out_features, a.w_k.out_features
            t = (d * 4 + d * 2)  # attention norm fp32 copy + bf16 out
            t += (qd + 2 * kd) * 2  # q, k, v
            t += (qd + kd) * 4 + (qd + kd) * 2  # qk-norm fp32 copies + rope'd copies
            t += qd * 2 + (qd // a.head_dim) * 4  # flash out + lse
            t += qd * 2 + d * 2 + d * 2  # gate, o-proj out, residual
            attn_t = max(attn_t, t + ffn_t)
        else:
            kdim, vdim = a.key_dim, a.value_dim
            t = (d * 4 + d * 2)
            t += (2 * kdim + vdim) * 2 * 2  # projections + conv outputs
            t += a.n_v_heads * (4 + 4)  # g (fp32), beta
            t += a.n_v_heads * a.head_k_dim * a.head_v_dim * 2 / 64  # chunk states (64-token chunks)
            t += vdim * 2 * 2  # w, u kernel intermediates
            t += vdim * 2 + vdim * 2 + vdim * 4 + d * 2  # o, gate, o_norm fp32, w_out out
            gdn_t = max(gdn_t, t + ffn_t)
    retained = len(blocks) * d * 2 + d * 2  # block inputs + embedding output
    lm_pt = d * 4  # final norm fp32 copy per token (logits chunked by fused linear CE)
    lm_fixed = 2.0 * 2**30  # a chunk of logits + CE workspace, ~2 GB
    fixed = 16 * n_params / WORLD
    return {"n_params": n_params, "fixed": fixed, "retained": retained, "attn_t": attn_t, "gdn_t": gdn_t,
            "lm_pt": lm_pt, "lm_fixed": lm_fixed}


def main():
    os.makedirs(VIZ, exist_ok=True)
    fig, axes = plt.subplots(2, len(SCALES), figsize=(5.2 * len(SCALES), 8.0), squeeze=False)
    fig2, axes2 = plt.subplots(2, len(SCALES), figsize=(5.2 * len(SCALES), 7.6), squeeze=False)
    rows = []
    for j, (name, fac) in enumerate(SCALES):
        model = fac(vocab_size=VOCAB).build(init_device="meta")
        shares = {k: [] for k in COLORS}
        absol = {k: [] for k in COLORS}
        for L in LENGTHS:
            s, dense = flop_shares(model, L)
            for k in COLORS:
                shares[k].append(100 * s[k])
                absol[k].append(s[k] * dense / 1e9)
            rows.append((name, L, s, dense))
        ax0 = axes[0][j]
        ax0.stackplot(LENGTHS, [absol[k] for k in COLORS], colors=[COLORS[k] for k in COLORS], alpha=0.9)
        ax0.set_xscale("log", base=2)
        ax0.set_xlim(LENGTHS[0], LENGTHS[-1])
        ax0.set_xticks([2**k for k in range(10, 21, 2)]); ax0.set_xticklabels(["1k", "4k", "16k", "64k", "256k", "1M"])
        ax0.set_title(f"Qwen3.5-{name}", fontsize=12)
        tot = [sum(absol[k][i] for k in COLORS) for i in range(len(LENGTHS))]
        for L, t in zip(LENGTHS, tot):
            if L in (2048, 32768, 262144, 1048576):
                ax0.annotate(f"{t:.0f}", (L, t), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        if j == 0:
            ax0.set_ylabel("training GFLOPs per token")
        ax0.grid(True, alpha=0.2)
        ax = axes[1][j]
        ax.stackplot(LENGTHS, [shares[k] for k in COLORS], labels=list(COLORS), colors=[COLORS[k] for k in COLORS], alpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xlim(LENGTHS[0], LENGTHS[-1]); ax.set_ylim(0, 100)
        ax.set_xticks([2**k for k in range(10, 21, 2)]); ax.set_xticklabels(["1k", "4k", "16k", "64k", "256k", "1M"])
        ax.set_xlabel("context length (tokens)")
        if j == 0:
            ax.set_ylabel("% of training FLOPs per token")
        ax.grid(True, alpha=0.2)
        # memory
        mm = mem_model(model)
        GB = 2**30
        comp = {k: [] for k in MCOLORS}
        for L in LENGTHS:
            T = L  # one row per GPU
            comp["params + grads + Adam (÷8 GPUs)"].append(mm["fixed"] / GB)
            comp["retained block inputs (full AC)"].append(mm["retained"] * T / GB)
            comp["block recompute transient"].append(max(mm["attn_t"], mm["gdn_t"]) * T / GB)
            comp["LM head / loss"].append((mm["lm_pt"] * T + mm["lm_fixed"]) / GB)
        ax2 = axes2[0][j]
        ax2.stackplot(LENGTHS, [comp[k] for k in MCOLORS], labels=list(MCOLORS), colors=[MCOLORS[k] for k in MCOLORS], alpha=0.9)
        ax2.axhline(80, color="#B9552A", ls="--", lw=1)
        ax2.text(LENGTHS[0] * 1.1, 83, "80 GB (H100)", fontsize=8, color="#B9552A")
        ax2.set_xscale("log", base=2)
        ax2.set_xlim(LENGTHS[0], LENGTHS[-1]); ax2.set_ylim(0, 160)
        ax2.set_xticks([2**k for k in range(10, 21, 2)]); ax2.set_xticklabels(["1k", "4k", "16k", "64k", "256k", "1M"])
        ax2.set_title(f"Qwen3.5-{name}  ({mm['n_params']/1e9:.1f}B params)", fontsize=12)
        ax2.text(0.03, 0.97, f"per token: retained {mm['retained']/1024:.0f} KB\nattn-block transient {mm['attn_t']/1024:.0f} KB\nGDN-block transient {mm['gdn_t']/1024:.0f} KB",
                 transform=ax2.transAxes, va="top", fontsize=8, color="#333333")
        if j == 0:
            ax2.set_ylabel("GB per GPU at peak (model)")
        ax2.grid(True, alpha=0.2)
        ax3 = axes2[1][j]
        tot = [sum(comp[k][i] for k in MCOLORS) for i in range(len(LENGTHS))]
        ax3.stackplot(LENGTHS, [[100 * comp[k][i] / tot[i] for i in range(len(LENGTHS))] for k in MCOLORS],
                      colors=[MCOLORS[k] for k in MCOLORS], alpha=0.9)
        ax3.set_xscale("log", base=2)
        ax3.set_xlim(LENGTHS[0], LENGTHS[-1]); ax3.set_ylim(0, 100)
        ax3.set_xticks([2**k for k in range(10, 21, 2)]); ax3.set_xticklabels(["1k", "4k", "16k", "64k", "256k", "1M"])
        ax3.set_xlabel("tokens per GPU (one packed row)")
        if j == 0:
            ax3.set_ylabel("% of peak memory")
        ax3.grid(True, alpha=0.2)
    handles, labels = axes[1][0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Qwen3.5: training FLOPs per token vs context length -- absolute (top) and shares (bottom); exact model FLOP formulas", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = f"{VIZ}/qwen35_flop_shares.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); print("wrote", p)
    h2, l2 = axes2[0][0].get_legend_handles_labels()
    fig2.legend(h2[::-1], l2[::-1], loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig2.suptitle("Qwen3.5: modelled peak training memory per GPU vs tokens per GPU (FSDP2 x8, full AC, flash attention, fused CE)", fontsize=11)
    fig2.tight_layout(rect=(0, 0.04, 1, 0.96))
    p = f"{VIZ}/qwen35_mem_shares.png"; fig2.savefig(p, dpi=140, bbox_inches="tight"); print("wrote", p)
    print(f"{'scale':6} {'L':>8} {'FFN':>6} {'GDN':>6} {'proj':>6} {'score':>6} {'fixed':>6} {'GF/tok':>8}")
    for name, L, s, dense in rows:
        if L in (2048, 8192, 32768, 131072, 1048576):
            print(f"{name:6} {L:8d} {100*s['FFN']:6.1f} {100*s['GDN mixer']:6.1f} {100*s['attention projections']:6.1f} {100*s['attention scores']:6.1f} {100*s['embeddings + LM head']:6.1f} {dense/1e9:8.1f}")


if __name__ == "__main__":
    main()
