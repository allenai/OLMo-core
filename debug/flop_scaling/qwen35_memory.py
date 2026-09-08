"""
Training memory of Qwen3.5 (0.8B / 4B / 27B) split two ways:

1. PARAMETRIC memory by part of the model (embeddings, LM head, attention weights, GDN weights,
   FFN weights, norms): parameter counts, bf16 inference bytes, and training-state bytes
   (fp32 master + fp32 grad + 2 fp32 Adam moments = 16 B/param, sharded over the FSDP world).

2. ACTIVATION memory vs tokens per GPU, WITH full activation checkpointing (one block's
   intermediates alive at a time + every block's input retained) and WITHOUT it (every block's
   intermediates saved for backward), split by what the tensors are: block inputs / residual
   stream, fp32 RMSNorm copies, attention intermediates, GDN intermediates, FFN intermediates,
   LM head + loss. Same recipe assumptions as qwen35_shares.py (flash attention: scores never
   materialise; fused linear CE: logits chunked; bf16 activations).

Per-token bytes come from the module shapes; the accounting is what autograd saves, not the
allocator's peak, so treat absolute numbers as a first-order model (measured peaks run 10-30% above).

    python debug/flop_scaling/qwen35_memory.py
      -> visualizations/flop_scaling/qwen35_param_memory.png, qwen35_act_memory.png, and a table on stdout
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
LENGTHS = [2**k for k in range(10, 19)]  # 1k .. 256k tokens per GPU
WORLD = 8
GB = 2**30

PCOLORS = {"FFN weights": "#0E6B66", "GDN weights": "#7B2D8E", "attention weights": "#4C9BE0",
           "embeddings": "#E08A5C", "LM head": "#B9552A", "norms + gates": "#999999"}
ACOLORS = {"params + grads + Adam (÷8 GPUs)": "#8a8a8a", "block inputs / residual": "#0E6B66", "fp32 norm copies": "#c9a227",
           "attention intermediates": "#4C9BE0", "GDN intermediates": "#7B2D8E", "FFN intermediates": "#2A9D8F",
           "LM head / loss": "#B9552A"}


def param_breakdown(model):
    out = {k: 0 for k in PCOLORS}
    gdn_blocks = {k for k, b in model.blocks.items() if type(b.attention) is not Attention}
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("embeddings"):
            out["embeddings"] += n
        elif name.startswith("lm_head"):
            out["LM head"] += n if "norm" not in name else 0
            out["norms + gates"] += n if "norm" in name else 0
        elif ".feed_forward." in name:
            out["FFN weights"] += n
        elif ".attention." in name:
            blk = name.split(".")[1]
            if "norm" in name:
                out["norms + gates"] += n
            elif blk in gdn_blocks:
                out["GDN weights"] += n
            else:
                out["attention weights"] += n
        else:
            out["norms + gates"] += n
    return out


def block_activation_bytes(model):
    """Per-token bytes autograd keeps for ONE block of each kind, by component; plus per-token
    bytes for the parts outside the blocks."""
    d = model.d_model
    blocks = list(model.blocks.values())
    F = blocks[0].feed_forward.w1.out_features
    per_kind = {}
    for b in blocks:
        a = b.attention
        kind = "attn" if type(a) is Attention else "gdn"
        if kind in per_kind:
            continue
        c = {"block inputs / residual": d * 2 + d * 2,  # block input (retained under AC) + post-attention residual
             "fp32 norm copies": 2 * d * 4,  # attention_norm + feed_forward_norm keep an fp32 copy of their input
             "FFN intermediates": d * 2 + 2 * F * 2 + F * 2 + F * 2 + d * 2}  # norm out, w1/w3 out, act, prod, w2 out
        if kind == "attn":
            qd, kd = a.w_q.out_features, a.w_k.out_features
            c["attention intermediates"] = (d * 2  # norm out
                                            + (qd + 2 * kd) * 2  # q, k, v
                                            + (qd + kd) * 4 + (qd + kd) * 2  # qk-norm fp32 copies + RoPE'd q, k
                                            + qd * 2 + (qd // a.head_dim) * 4  # flash out + logsumexp
                                            + qd * 2 + d * 2)  # gate, o-proj out
            c["GDN intermediates"] = 0
        else:
            kdim, vdim = a.key_dim, a.value_dim
            c["GDN intermediates"] = (d * 2
                                      + (2 * kdim + vdim) * 2 * 2  # projections + conv outputs
                                      + a.n_v_heads * 8  # g (fp32) + beta
                                      + a.n_v_heads * a.head_k_dim * a.head_v_dim * 2 / 64  # chunk states, 64-token chunks
                                      + vdim * 2 * 2  # kernel w, u
                                      + vdim * 2 + vdim * 2 + vdim * 4 + d * 2)  # o, gate, o_norm fp32, w_out out
            c["attention intermediates"] = 0
        per_kind[kind] = c
    n_attn = sum(1 for b in blocks if type(b.attention) is Attention)
    n_gdn = len(blocks) - n_attn
    outside = {"block inputs / residual": d * 2, "fp32 norm copies": d * 4, "LM head / loss": 0}  # embedding out, final norm
    return per_kind, n_attn, n_gdn, outside


def activation_curves(model, ac: bool):
    per_kind, n_attn, n_gdn, outside = block_activation_bytes(model)
    comps = {k: [] for k in ACOLORS}
    n_params = sum(p.numel() for p in model.parameters())
    for T in LENGTHS:
        per_tok = {k: 0.0 for k in ACOLORS}
        for k, v in outside.items():
            per_tok[k] += v
        if ac:
            # retained: every block's INPUT; transient: the larger block's full intermediates
            per_tok["block inputs / residual"] += (n_attn + n_gdn) * model.d_model * 2
            big = max(per_kind.values(), key=lambda c: sum(c.values()))
            for k, v in big.items():
                if k == "block inputs / residual":
                    v = model.d_model * 2  # the input is already counted above; the residual is the transient part
                per_tok[k] += v
        else:
            for kind, n in (("attn", n_attn), ("gdn", n_gdn)):
                for k, v in per_kind[kind].items():
                    per_tok[k] += n * v
        for k in ACOLORS:
            if k == "params + grads + Adam (÷8 GPUs)":
                comps[k].append(16 * n_params / WORLD / GB)
            elif k == "LM head / loss":
                comps[k].append((per_tok[k] * T + 2 * GB) / GB)
            else:
                comps[k].append(per_tok[k] * T / GB)
    return comps


def main():
    os.makedirs(VIZ, exist_ok=True)
    # ---- figure 1: parametric memory
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    names = [n for n, _ in SCALES]
    models = {n: fac(vocab_size=VOCAB).build(init_device="meta") for n, fac in SCALES}
    pb = {n: param_breakdown(m) for n, m in models.items()}
    print(f"{'scale':6} " + " ".join(f"{k:>18}" for k in PCOLORS) + f" {'total':>10}")
    for n in names:
        tot = sum(pb[n].values())
        print(f"{n:6} " + " ".join(f"{pb[n][k]/1e9:14.3f}B ({100*pb[n][k]/tot:2.0f}%)" for k in PCOLORS) + f" {tot/1e9:9.2f}B")
    for ax, (title, scale_b) in zip(axes, [("training state per GPU (fp32 weights + grads + Adam, sharded over 8 GPUs)", 16 / WORLD), ("training state, whole model (16 B/param) -- what one unsharded GPU would need", 16)]):
        bottoms = [0.0] * len(names)
        for k in PCOLORS:
            vals = [pb[n][k] * scale_b / GB for n in names]
            ax.bar(names, vals, bottom=bottoms, color=PCOLORS[k], label=k, width=0.6)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        for i, n in enumerate(names):
            ax.text(i, bottoms[i] * 1.01 + 0.5, f"{bottoms[i]:.0f} GB", ha="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("GB")
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Qwen3.5: parametric memory by part of the model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = f"{VIZ}/qwen35_param_memory.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); print("wrote", p)
    # ---- figure 2: activations with / without AC
    fig2, axes2 = plt.subplots(2, len(SCALES), figsize=(5.2 * len(SCALES), 8.0), squeeze=False)
    for j, n in enumerate(names):
        for i, ac in enumerate((False, True)):
            comps = activation_curves(models[n], ac)
            ax = axes2[i][j]
            ax.stackplot(LENGTHS, [comps[k] for k in ACOLORS], colors=[ACOLORS[k] for k in ACOLORS], labels=list(ACOLORS), alpha=0.92)
            ax.axhline(80, color="#B9552A", ls="--", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_xlim(LENGTHS[0], LENGTHS[-1]); ax.set_ylim(0, 160 if ac else 600)
            ax.set_xticks([2**k for k in range(10, 19, 2)]); ax.set_xticklabels(["1k", "4k", "16k", "64k", "256k"])
            ax.set_title(f"Qwen3.5-{n}  {'WITH full activation checkpointing' if ac else 'WITHOUT activation checkpointing'}", fontsize=9.5)
            tot64 = sum(comps[k][6] for k in ACOLORS)  # index 6 = 64k
            ax.text(0.03, 0.97, f"at 64k tokens/GPU: {tot64:.0f} GB", transform=ax.transAxes, va="top", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("GB per GPU at peak (model)")
            if i == 1:
                ax.set_xlabel("tokens per GPU (one packed row)")
            ax.grid(True, alpha=0.2)
    h, l = axes2[0][0].get_legend_handles_labels()
    fig2.legend(h[::-1], l[::-1], loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle("Qwen3.5: modelled peak training memory per GPU, without vs with activation checkpointing (FSDP2 x8, flash attention, fused CE)", fontsize=11)
    fig2.tight_layout(rect=(0, 0.05, 1, 0.95))
    p = f"{VIZ}/qwen35_act_memory.png"; fig2.savefig(p, dpi=140, bbox_inches="tight"); print("wrote", p)
    # table at 64k
    print(f"\n{'scale':6} {'AC':5} " + " ".join(f"{k[:22]:>22}" for k in ACOLORS) + "   total@64k")
    for n in names:
        for ac in (False, True):
            comps = activation_curves(models[n], ac)
            print(f"{n:6} {str(ac):5} " + " ".join(f"{comps[k][6]:22.1f}" for k in ACOLORS) + f"   {sum(comps[k][6] for k in ACOLORS):.1f} GB")


if __name__ == "__main__":
    main()
