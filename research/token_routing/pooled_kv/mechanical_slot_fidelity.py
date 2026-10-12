"""MECHANICAL validation: can static summary tokens reproduce the attention output?

No training anywhere. For real queries against real contexts through Qwen3-4B, replace each doc's
per-token KV with G summary slots and measure how close the attention OPERATION stays to the
per-token original:

  * G-cluster pooling (softmax-aware, fully realizable as static KV): k-means the doc's tokens by
    key; slot g = (mean K, mean V, +log L_g logit bias) == "L_g copies of the cluster mean". The
    doc's softmax mass becomes sum_g L_g exp(q.kbar_g), a G-piece log-linear approximation of the
    true convex logsumexp -- converging in G. G=1 is the plain mean+logL construction.
  * oracle-lsq (idealized ceiling, per-head free bias -- NOT directly realizable): per (doc, head)
    least-squares fit of (k*, b) to the true per-query log-mass on TRAIN queries, evaluated on
    HELD-OUT queries; value = mass-weighted mean.

Metrics per layer (vs the per-token forward): attention-output cosine, doc-mass total variation,
top-doc agreement. If G-cluster fidelity -> ~1 with small G, summaries are mechanically sufficient
and the training question is separate; if not, no projector can fix it.

Run:
    srun -p jsteinhardt -w mooney --gres=gpu:H200:1 --qos=preemptive_high -t 90 \
      env HF_HOME=/data/prasann/hf_cache \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
      research/token_routing/pooled_kv/mechanical_slot_fidelity.py
"""

import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_predictability_probe as probe  # noqa: E402
from fit_summarizer_and_fidelity import (  # noqa: E402
    QUERY_SUFFIX,
    compute_queries,
    rotate,
)

OUT_JSON = probe.OUT_JSON
probe.DATA = os.environ.get("POOL_JSONL", probe.DATA)
N_CTX = 3
N_QUERY = 24
G_LIST = (1, 2, 4, 8)


def log(m):
    print(f"[mech] {m}", flush=True)


def kmeans_tokens(k_doc: torch.Tensor, G: int, iters: int = 8):
    """Cluster a doc's tokens by flattened key; returns cluster assignment (L,)."""
    L = k_doc.shape[0]
    feats = k_doc.reshape(L, -1).float()
    G = min(G, L)
    cent = feats[torch.randperm(L)[:G]]
    for _ in range(iters):
        d = torch.cdist(feats, cent)
        assign = d.argmin(-1)
        for g in range(G):
            m = assign == g
            if m.any():
                cent[g] = feats[m].mean(0)
    return assign, G


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("loading Qwen/Qwen3-4B ...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    inv_freq = (
        1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    ).cuda()
    layers = list(range(0, cfg.num_hidden_layers, 4))
    if cfg.num_hidden_layers - 1 not in layers:
        layers.append(cfg.num_hidden_layers - 1)

    pool = probe.load_doc_pool(probe.DATA, tok)
    preamble_ids = tok(probe.PREAMBLE, add_special_tokens=False)["input_ids"]
    suffix_ids = tok(QUERY_SUFFIX, add_special_tokens=False)["input_ids"]

    results = {}
    for target_len in (8192, 32768):
        rng = random.Random(99 + target_len)
        acc = {f"g{g}": {li: {"cos": [], "tv": [], "top": []} for li in layers} for g in G_LIST}
        acc["oracle"] = {li: {"cos": [], "tv": [], "top": []} for li in layers}
        for ci in range(N_CTX):
            ids, spans = probe.build_context(
                pool, [], target_len - len(suffix_ids), preamble_ids, rng
            )
            ids = ids + suffix_ids
            x = torch.tensor([ids], device="cuda")
            T = x.shape[1]
            attn_inputs = {}
            hooks = [
                model.model.layers[li].self_attn.register_forward_pre_hook(
                    lambda m, a, kw, li=li: attn_inputs.__setitem__(
                        li, (a[0] if a else kw["hidden_states"]).detach()
                    ),
                    with_kwargs=True,
                )
                for li in layers
            ]
            try:
                out = model(x, use_cache=True, logits_to_keep=1)
            except TypeError:
                out = model(x, use_cache=True)
            for h in hooks:
                h.remove()
            out.logits = None
            qpos = torch.arange(T - N_QUERY, T, device="cuda")
            n_docs = len(spans)
            doc_col = torch.full((T,), -1, dtype=torch.long, device="cuda")
            for di, (_, s, e) in enumerate(spans):
                doc_col[s:e] = di
            is_doc = doc_col >= 0

            for li in layers:
                k, v = probe.get_layer_kv(out.past_key_values, li)
                k = k[0].transpose(0, 1).float()  # (T, H_kv, hd)
                v = v[0].transpose(0, 1).float()
                H_kv, hd = k.shape[1], k.shape[2]
                q = compute_queries(model, attn_inputs, li, qpos, inv_freq)  # (nq, Hq, hd)
                n_rep = q.shape[1] // H_kv
                scale = hd**-0.5
                kx = k.repeat_interleave(n_rep, dim=1)
                vx = v.repeat_interleave(n_rep, dim=1)
                logits_full = torch.einsum("qhd,thd->qht", q, kx) * scale
                causal = torch.arange(T, device="cuda")[None, :] > qpos[:, None]
                logits_full.masked_fill_(causal[:, None, :], float("-inf"))
                p_full = logits_full.softmax(-1)
                o_full = torch.einsum("qht,thd->qhd", p_full, vx)
                onehot = F.one_hot(doc_col.clamp(min=0), n_docs).float() * is_doc[:, None]
                mass_full = torch.einsum("qht,tD->qhD", p_full, onehot)
                base_logits = logits_full.masked_fill(is_doc[None, None, :], float("-inf"))

                # ---- build slot variants ----
                variants = {}
                for G in G_LIST:
                    sk, sv, slog, sdoc = [], [], [], []
                    for di, (_, s, e) in enumerate(spans):
                        assign, Ge = kmeans_tokens(k[s:e], G)
                        for g in range(Ge):
                            m = assign == g
                            if not m.any():
                                continue
                            sk.append(k[s:e][m].mean(0))
                            sv.append(v[s:e][m].mean(0))
                            slog.append(float(torch.log(m.sum().float())))
                            sdoc.append(di)
                    variants[f"g{G}"] = (
                        torch.stack(sk),
                        torch.stack(sv),
                        torch.tensor(slog, device="cuda"),
                        torch.tensor(sdoc, device="cuda"),
                    )
                # oracle: per-(doc, head) lsq of (k*, b) to true log-mass on half the queries.
                tr_q = torch.arange(0, N_QUERY, 2, device="cuda")
                sk, sv, sb, sdoc = [], [], [], []
                for di, (_, s, e) in enumerate(spans):
                    lg_d = torch.einsum("qhd,thd->qht", q[tr_q], kx[s:e]) * scale
                    lse = torch.logsumexp(lg_d, dim=-1)  # (ntr, Hq)
                    A = torch.cat(
                        [q[tr_q] * scale, torch.ones(len(tr_q), q.shape[1], 1, device="cuda")],
                        dim=-1,
                    )  # (ntr, Hq, hd+1)
                    # per-head lsq: solve A_h X_h = lse_h
                    X = torch.linalg.lstsq(
                        A.transpose(0, 1), lse.transpose(0, 1).unsqueeze(-1)
                    ).solution  # (Hq, hd+1, 1)
                    k_star = X[:, :hd, 0]  # (Hq, hd)
                    b_star = X[:, hd, 0]  # (Hq,)
                    w_loc = torch.softmax(lg_d, dim=-1)
                    v_star = torch.einsum("qht,thd->hd", w_loc, vx[s:e]) / len(tr_q)
                    sk.append(k_star)
                    sv.append(v_star)
                    sb.append(b_star)
                    sdoc.append(di)
                oracle = (
                    torch.stack(sk),
                    torch.stack(sv),
                    torch.stack(sb),
                    torch.tensor(sdoc, device="cuda"),
                )

                def eval_variant(sk, sv, sbias, sdoc, per_head_bias):
                    slot_logits = torch.einsum("qhd,Shd->qhS", q, sk) * (
                        1.0 if per_head_bias else scale
                    )
                    if per_head_bias:
                        # oracle keys already absorb the scale (fit on scaled queries)
                        slot_logits = torch.einsum("qhd,Shd->qhS", q * scale, sk) + sbias.T[None]
                    else:
                        slot_logits = slot_logits + sbias[None, None, :]
                    cat = torch.cat([base_logits, slot_logits], dim=-1)
                    p = cat.softmax(-1)
                    o = torch.einsum("qht,thd->qhd", p[..., :T], vx) + torch.einsum(
                        "qhS,Shd->qhd", p[..., T:], sv
                    )
                    n_s = sdoc.shape[0]
                    d_onehot = F.one_hot(sdoc, n_docs).float()
                    mass = torch.einsum("qhS,SD->qhD", p[..., T:], d_onehot)
                    cos = float(
                        F.cosine_similarity(o.flatten(0, 1), o_full.flatten(0, 1), dim=1).mean()
                    )
                    tv = float(0.5 * (mass - mass_full).abs().sum(-1).mean())
                    top = float((mass.argmax(-1) == mass_full.argmax(-1)).float().mean())
                    return cos, tv, top

                for name in (f"g{G}" for G in G_LIST):
                    sk, sv, slog, sdoc = variants[name]
                    skx = sk.repeat_interleave(n_rep, dim=1) if sk.shape[1] != q.shape[1] else sk
                    svx = sv.repeat_interleave(n_rep, dim=1) if sv.shape[1] != q.shape[1] else sv
                    cos, tv, top = eval_variant(skx, svx, slog, sdoc, per_head_bias=False)
                    a = acc[name][li]
                    a["cos"].append(cos)
                    a["tv"].append(tv)
                    a["top"].append(top)
                cos, tv, top = eval_variant(
                    oracle[0], oracle[1], oracle[2], oracle[3], per_head_bias=True
                )
                a = acc["oracle"][li]
                a["cos"].append(cos)
                a["tv"].append(tv)
                a["top"].append(top)
            del out
            log(f"len={target_len} ctx {ci + 1}/{N_CTX} done ({n_docs} docs, T={T})")

        res = {
            name: {
                li: {m: round(sum(vals[m]) / len(vals[m]), 4) for m in vals}
                for li, vals in per_layer.items()
            }
            for name, per_layer in acc.items()
        }
        results[target_len] = res
        mid = layers[len(layers) // 2]
        last = layers[-1]
        for name in list(f"g{G}" for G in G_LIST) + ["oracle"]:
            log(
                f"len={target_len} {name:7s} Lmid: cos {res[name][mid]['cos']:.3f} "
                f"tv {res[name][mid]['tv']:.3f} top {res[name][mid]['top']:.2f} | "
                f"Llast: cos {res[name][last]['cos']:.3f} tv {res[name][last]['tv']:.3f} "
                f"top {res[name][last]['top']:.2f}"
            )
        with open(f"{OUT_JSON}/mechanical_slot_fidelity.json", "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
